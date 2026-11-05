"""Subprocess-based local job manager (SPEC §3, §12): bounded concurrency, persistent status
in SQLite, cooperative cancellation (flag checked between stages and inside scoring loops)
plus SIGTERM as a fallback, CPU time and peak RSS recorded."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from aerodoppler.config import Settings
from aerodoppler.jobs.proc import Heartbeat, job_process_alive, process_start_time
from aerodoppler.jobs.store import Store
from aerodoppler.models import SearchRequest, SearchResult
from aerodoppler.util import new_id

log = logging.getLogger(__name__)


class SearchResourceExceeded(RuntimeError):
    """Raised from the progress callback when a search exceeds its memory budget."""


class JobRejected(RuntimeError):
    pass


def submit_search(store: Store, settings: Settings, req: SearchRequest) -> str:
    """Create the job row and launch the worker subprocess (if under the concurrency cap)."""
    reconcile_stale_jobs(store)
    run_id = new_id("run")
    out_dir = settings.runs_dir / run_id
    active = store.create_search_if_capacity(run_id, req, out_dir, settings.max_concurrent_jobs)
    if active is not None:
        raise JobRejected(
            f"{active} jobs already running (limit {settings.max_concurrent_jobs}); wait or cancel one"
        )
    env = dict(os.environ)
    env["AERODOPPLER_DATA_DIR"] = str(settings.data_dir)
    env["AERODOPPLER_OFFLINE"] = "true" if settings.offline else "false"
    log_fh = None
    proc = None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "request.json").write_text(req.model_dump_json(indent=2))
        log_fh = open(out_dir / "job.log", "ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", "aerodoppler.jobs.runner", run_id],
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        store.update_search(
            run_id,
            pid=proc.pid,
            pid_started_at=_process_start_time(proc.pid),
            status="running",
            stage="starting",
        )
    except Exception as e:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            except OSError:
                pass
        if log_fh is not None:
            log_fh.close()
        store.finish_if_active(
            run_id,
            status="failed",
            stage="launch_error",
            error=f"worker could not be started: {type(e).__name__}: {e}",
        )
        raise

    def _reap() -> None:  # keep the handle so the child never becomes a zombie
        returncode = proc.wait()
        log_fh.close()
        job = store.get_search(run_id)
        if job and job["status"] in ("queued", "running"):
            cancelled = job["cancel_requested"]
            store.finish_if_active(
                run_id,
                status="cancelled" if cancelled else "interrupted",
                stage="cancelled" if cancelled else "worker_exited",
                error=(
                    "cancelled by operator"
                    if cancelled
                    else f"worker exited before completing (exit code {returncode})"
                ),
            )

    threading.Thread(target=_reap, daemon=True).start()
    return run_id


_process_start_time = process_start_time
_job_process_alive = job_process_alive


def reconcile_stale_jobs(store: Store, queued_grace_s: float = 60.0) -> list[str]:
    """Fail active database rows whose worker no longer exists."""
    reconciled: list[str] = []
    now = time.time()
    for job in store.list_active_searches():
        stale = job["status"] == "running" and not _job_process_alive(job)
        if job["status"] == "queued":
            updated = datetime.fromisoformat(job["updated_at"].replace("Z", "+00:00"))
            stale = now - updated.astimezone(UTC).timestamp() > queued_grace_s
        if stale and store.finish_if_active(
            job["id"],
            status="interrupted",
            stage="worker_lost",
            error=(
                f"the search worker stopped without finishing (crash, kill, out of memory or "
                f"machine restart) during stage '{job.get('stage')}'; partial outputs are kept. "
                "Run the search again to retry."
            ),
        ):
            store.audit("search.reconciled", job["id"], {"previous_status": job["status"]})
            reconciled.append(job["id"])
    return reconciled


def cancel_search(store: Store, run_id: str, grace_s: float = 5.0) -> bool:
    job = store.get_search(run_id)
    if not job or job["status"] not in ("queued", "running"):
        return False
    store.request_cancel(run_id)
    pid = job.get("pid")
    if not _job_process_alive(job):
        store.finish_if_active(
            run_id,
            status="cancelled",
            stage="cancelled",
            error="cancelled by operator; worker was no longer running",
        )
        return True
    if pid:

        def _escalate() -> None:
            time.sleep(grace_s)
            j = store.get_search(run_id)
            if j and j["status"] == "running":
                if _job_process_alive(j):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                    except OSError:
                        pass
                store.update_search(
                    run_id,
                    status="cancelled",
                    stage="killed",
                    error="cancelled by operator (SIGTERM)",
                )

        threading.Thread(target=_escalate, daemon=True).start()
    return True


def run_search_inline(
    store: Store, settings: Settings, req: SearchRequest, run_id: str | None = None, progress=None
) -> SearchResult:
    """Run a search in the current process (CLI / tests). Same code path as the worker."""

    run_id = run_id or new_id("run")
    out_dir = settings.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    store.create_search(run_id, req, out_dir)
    store.update_search(
        run_id,
        status="running",
        stage="starting",
        pid=os.getpid(),
        pid_started_at=_process_start_time(os.getpid()),
    )
    try:
        return _execute(store, settings, req, run_id, out_dir, progress)
    except BaseException as e:
        store.finish_if_active(
            run_id,
            status="failed",
            stage="interrupted",
            error=f"inline search interrupted: {type(e).__name__}: {e}",
        )
        raise


def _execute(
    store: Store, settings: Settings, req: SearchRequest, run_id: str, out_dir: Path, progress=None
) -> SearchResult:
    from aerodoppler.resources import default_memory_budget_mb, recommend_workers, tree_pss_mb
    from aerodoppler.search.engine import SearchCancelled, SearchEngine, SearchInputError

    proc = psutil.Process(os.getpid())
    t0 = time.time()
    counters: dict[str, Any] = {}
    budget = settings.search_memory_budget_mb or default_memory_budget_mb(cap_mb=8192.0)
    if not req.quality_settings.workers:
        # forked scorers share the spectrogram copy-on-write; ~250 MB private each measured
        # on the dense-traffic benchmark (3.4 GB peak for 6 workers + 2.0 GB parent)
        req = req.model_copy(
            update={
                "quality_settings": req.quality_settings.model_copy(
                    update={"workers": recommend_workers(budget, 250.0, base_mb=2000.0)}
                )
            }
        )
    mem = {"last": 0.0, "peak": 0.0}

    def _progress(stage: str, info: dict[str, Any]) -> None:
        now = time.time()
        if now - mem["last"] >= 2.0:
            mem["last"] = now
            used = tree_pss_mb()
            mem["peak"] = max(mem["peak"], used)
            if used > 1.5 * budget:
                raise SearchResourceExceeded(
                    f"the search used {used:.0f} MB (workers included), above 1.5x its "
                    f"{budget:.0f} MB budget, during '{stage}'. Use a smaller search area, a "
                    "coarser grid or a narrower time window, or raise "
                    "AERODOPPLER_SEARCH_MEMORY_BUDGET_MB."
                )
        counters[stage] = info
        store.update_search(
            run_id,
            stage=stage,
            progress={
                "stages": counters,
                "elapsed_s": round(time.time() - t0, 1),
                "cpu_seconds": round(sum(proc.cpu_times()[:2]), 1),
                "rss_mb": round(proc.memory_info().rss / 1e6, 1),
                "tree_pss_peak_mb": round(mem["peak"], 1),
                "memory_budget_mb": round(budget),
                "workers": req.quality_settings.workers,
            },
        )
        if progress:
            progress(stage, info)

    def _cancelled() -> bool:
        return store.cancel_requested(run_id)

    engine = SearchEngine(store, settings, progress=_progress, cancel_check=_cancelled)
    try:
        with Heartbeat(lambda: store.update_search(run_id, heartbeat_at=time.time()), 5.0):
            result = engine.run(req, run_id, out_dir)
    except SearchCancelled:
        store.update_search(
            run_id, status="cancelled", stage="cancelled", error="cancelled by operator"
        )
        raise
    except SearchResourceExceeded as e:
        store.update_search(run_id, status="failed", stage="memory_budget", error=str(e))
        (out_dir / "error.json").write_text(json.dumps({"error": str(e), "type": "memory_budget"}))
        raise
    except SearchInputError as e:
        store.update_search(run_id, status="failed", stage="input_error", error=str(e))
        (out_dir / "error.json").write_text(json.dumps({"error": str(e), "type": "input"}))
        raise
    except Exception as e:  # unexpected
        log.exception("search %s failed", run_id)
        store.update_search(
            run_id, status="failed", stage="error", error=f"{type(e).__name__}: {e}"
        )
        (out_dir / "error.json").write_text(json.dumps({"error": str(e), "type": type(e).__name__}))
        raise
    cpu = sum(proc.cpu_times()[:2])
    rss = proc.memory_info().rss / 1e6
    store.update_search(
        run_id,
        status="abstained" if result.abstention.abstained else "succeeded",
        stage="finished",
        cpu_seconds=round(cpu, 1),
        peak_rss_mb=round(max(rss, mem["peak"]), 1),
    )
    store.audit(
        "search.finished",
        run_id,
        {
            "abstained": result.abstention.abstained,
            "reason": result.abstention.reason,
            "candidates": len(result.candidates),
            "cpu_s": round(cpu, 1),
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Worker entry point: ``python -m aerodoppler.jobs.runner <run_id>``."""
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m aerodoppler.jobs.runner <run_id>", file=sys.stderr)
        return 2
    run_id = args[0]
    from aerodoppler.config import get_settings

    settings = get_settings()
    store = Store(settings.db_path)
    job = store.get_search(run_id)
    if not job:
        print(f"unknown run {run_id}", file=sys.stderr)
        return 1
    req = SearchRequest.model_validate(job["request"])
    out_dir = Path(job["result_dir"])
    try:
        _execute(store, settings, req, run_id, out_dir)
    except Exception as e:
        print(f"search ended: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
