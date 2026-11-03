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
from pathlib import Path
from typing import Any

import psutil

from aerodoppler.config import Settings
from aerodoppler.jobs.store import Store
from aerodoppler.models import SearchRequest, SearchResult
from aerodoppler.util import new_id

log = logging.getLogger(__name__)


class JobRejected(RuntimeError):
    pass


def submit_search(store: Store, settings: Settings, req: SearchRequest) -> str:
    """Create the job row and launch the worker subprocess (if under the concurrency cap)."""
    active = store.count_active()
    if active >= settings.max_concurrent_jobs:
        raise JobRejected(
            f"{active} jobs already running (limit {settings.max_concurrent_jobs}); wait or cancel one"
        )
    run_id = new_id("run")
    out_dir = settings.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "request.json").write_text(req.model_dump_json(indent=2))
    store.create_search(run_id, req, out_dir)
    env = dict(os.environ)
    env["AERODOPPLER_DATA_DIR"] = str(settings.data_dir)
    env["AERODOPPLER_OFFLINE"] = "true" if settings.offline else "false"
    log_fh = open(out_dir / "job.log", "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "aerodoppler.jobs.runner", run_id],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    store.update_search(run_id, pid=proc.pid, status="running", stage="starting")

    def _reap() -> None:  # keep the handle so the child never becomes a zombie
        proc.wait()
        log_fh.close()

    threading.Thread(target=_reap, daemon=True).start()
    return run_id


def cancel_search(store: Store, run_id: str, grace_s: float = 5.0) -> bool:
    job = store.get_search(run_id)
    if not job or job["status"] not in ("queued", "running"):
        return False
    store.request_cancel(run_id)
    pid = job.get("pid")
    if pid:

        def _escalate() -> None:
            time.sleep(grace_s)
            j = store.get_search(run_id)
            if j and j["status"] == "running" and pid and psutil.pid_exists(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
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
    store.update_search(run_id, status="running", stage="starting", pid=os.getpid())
    return _execute(store, settings, req, run_id, out_dir, progress)


def _execute(
    store: Store, settings: Settings, req: SearchRequest, run_id: str, out_dir: Path, progress=None
) -> SearchResult:
    from aerodoppler.search.engine import SearchCancelled, SearchEngine, SearchInputError

    proc = psutil.Process(os.getpid())
    t0 = time.time()
    counters: dict[str, Any] = {}

    def _progress(stage: str, info: dict[str, Any]) -> None:
        counters[stage] = info
        store.update_search(
            run_id,
            stage=stage,
            progress={
                "stages": counters,
                "elapsed_s": round(time.time() - t0, 1),
                "cpu_seconds": round(sum(proc.cpu_times()[:2]), 1),
                "rss_mb": round(proc.memory_info().rss / 1e6, 1),
            },
        )
        if progress:
            progress(stage, info)

    def _cancelled() -> bool:
        return store.cancel_requested(run_id)

    engine = SearchEngine(store, settings, progress=_progress, cancel_check=_cancelled)
    try:
        result = engine.run(req, run_id, out_dir)
    except SearchCancelled:
        store.update_search(
            run_id, status="cancelled", stage="cancelled", error="cancelled by operator"
        )
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
        peak_rss_mb=round(rss, 1),
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
