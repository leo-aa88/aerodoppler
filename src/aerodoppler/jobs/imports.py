"""Persistent flight-archive import jobs.

An import (locate → download → verify → parse → segment → write) runs in a detached worker
process (``python -m aerodoppler.jobs.imports <id>``) and records its stage, byte/file
progress, heartbeat, PID + creation time and outcome in the ``imports`` table. The API only
creates and reads rows, so a page reload, an API restart or a closed browser never loses the
job, and a duplicate request returns the job already running for the same day and area.

What survives what:

* page reload / API restart: the worker keeps running; the row shows its live progress;
* worker crash, kill, OOM or machine restart: the row is reconciled to ``interrupted``
  (records and logs kept); completed archive parts and a partial ``.part`` download stay on
  disk, so a retry resumes the download and re-verifies cached parts instead of fetching
  3.5 GB again. Parsing is not checkpointed: it restarts from the cached archive.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aerodoppler.config import Settings
from aerodoppler.jobs.proc import Heartbeat, job_process_alive, process_start_time
from aerodoppler.jobs.store import Store
from aerodoppler.models import BBox
from aerodoppler.util import new_id

log = logging.getLogger(__name__)
QUEUED_GRACE_S = 60.0


class ImportRejected(RuntimeError):
    pass


def dedup_key(request: dict[str, Any]) -> str:
    bbox = request.get("bbox") or {}
    if isinstance(bbox, dict):
        bt = tuple(round(float(bbox[k]), 4) for k in ("west", "south", "east", "north"))
    else:
        bt = tuple(round(float(v), 4) for v in bbox)
    return "|".join(
        str(x)
        for x in (
            request.get("provider", "adsblol"),
            request["date_utc"],
            bt,
            round(float(request.get("margin_km", 15.0)), 3),
            bool(request.get("exclude_jets", False)),
            request.get("local_path") or "",
        )
    )


def reconcile_stale_imports(store: Store) -> list[str]:
    """Mark active rows whose worker no longer exists as ``interrupted`` (records kept)."""
    out: list[str] = []
    now = time.time()
    for job in store.list_imports(active_only=True):
        if job["status"] == "running":
            stale = not job_process_alive(job)
        else:  # queued: the API spawns the worker right after inserting the row
            created = datetime.fromisoformat(job["updated_at"].replace("Z", "+00:00"))
            stale = now - created.astimezone(UTC).timestamp() > QUEUED_GRACE_S
        if stale and store.finish_import_if_active(
            job["id"],
            status="interrupted",
            stage=job.get("stage") or "unknown",
            error=(
                f"the import worker stopped without finishing (crash, kill, out of memory or "
                f"machine restart) during stage '{job.get('stage')}'. Downloaded archive parts "
                "are kept; retry resumes the download and re-parses from the cached archive."
            ),
            error_kind="worker_lost",
        ):
            store.audit("import.reconciled", job["id"], {"previous_status": job["status"]})
            out.append(job["id"])
    return out


def submit_import(
    store: Store, settings: Settings, request: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Create (or reuse) an import job and start its worker. Returns ``(job, outcome)`` with
    outcome ``created`` or ``reused``; raises :class:`ImportRejected` when the import slot is
    taken by a different import."""
    reconcile_stale_imports(store)
    import_id = new_id("imp")
    outcome, jid = store.create_import_if_capacity(
        import_id, request, dedup_key(request), settings.max_concurrent_imports
    )
    if outcome == "reused":
        job = store.get_import(jid)  # type: ignore[arg-type]
        assert job is not None
        return job, "reused"
    if outcome == "full":
        active = store.list_imports(active_only=True)
        what = ", ".join(f"{j['request'].get('date_utc')} ({j['stage']})" for j in active)
        raise ImportRejected(
            f"another flight import is running ({what}); only {settings.max_concurrent_imports} "
            "at a time to bound memory and disk use. Wait for it or cancel it on the Jobs page."
        )
    log_dir = settings.data_dir / "imports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{import_id}.log"
    env = dict(os.environ)
    env["AERODOPPLER_DATA_DIR"] = str(settings.data_dir)
    env["AERODOPPLER_OFFLINE"] = "true" if settings.offline else "false"
    try:
        with open(log_path, "ab") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "aerodoppler.jobs.imports", import_id],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        store.update_import(
            import_id,
            pid=proc.pid,
            pid_started_at=process_start_time(proc.pid),
            status="running",
            stage="starting",
            log_path=str(log_path),
            heartbeat_at=time.time(),
        )
    except Exception as e:
        store.finish_import_if_active(
            import_id, "failed", "launch_error", f"worker could not be started: {e}", "launch"
        )
        raise
    _reap_in_background(store, import_id, proc)
    job = store.get_import(import_id)
    assert job is not None
    return job, "created"


def _reap_in_background(store: Store, import_id: str, proc: subprocess.Popen) -> None:
    import threading

    def _reap() -> None:
        code = proc.wait()
        job = store.get_import(import_id)
        if job and job["status"] in ("queued", "running"):
            store.finish_import_if_active(
                import_id,
                "cancelled" if job["cancel_requested"] else "interrupted",
                job.get("stage") or "unknown",
                "cancelled by operator"
                if job["cancel_requested"]
                else f"worker exited without a final state (exit code {code})",
                None if job["cancel_requested"] else "worker_lost",
            )

    threading.Thread(target=_reap, daemon=True).start()


def cancel_import(store: Store, import_id: str) -> bool:
    job = store.get_import(import_id)
    if not job or job["status"] not in ("queued", "running"):
        return False
    store.request_import_cancel(import_id)
    if not job_process_alive(job):
        store.finish_import_if_active(
            import_id, "cancelled", job.get("stage") or "cancelled", "cancelled by operator", None
        )
    # a live worker checks the flag between download chunks and parse batches (≤ ~1 s)
    return True


# ---- worker ---------------------------------------------------------------------------------


def run_import_job(store: Store, settings: Settings, import_id: str) -> int:
    from aerodoppler.adsb.adsblol import AdsbLolProvider, ProviderUnavailable
    from aerodoppler.adsb.ingest import (
        ImportCancelled,
        ImportRequest,
        MemoryBudgetExceeded,
        StorageBudgetExceeded,
        import_flights,
    )
    from aerodoppler.adsb.providers import LocalTraceProvider
    from aerodoppler.resources import tree_rss_mb

    job = store.get_import(import_id)
    if not job:
        print(f"unknown import {import_id}", file=sys.stderr)
        return 1
    rq = job["request"]
    stages: dict[str, Any] = {}
    peak = {"rss": 0.0}
    last_write = {"t": 0.0}

    def beat() -> None:
        peak["rss"] = max(peak["rss"], tree_rss_mb())
        store.update_import(import_id, heartbeat_at=time.time())

    def on_progress(stage: str, info: dict[str, Any]) -> None:
        stages[stage] = info
        now = time.time()
        if now - last_write["t"] < 1.0 and stage in ("download", "parse"):
            return
        last_write["t"] = now
        store.update_import(
            import_id,
            stage=stage,
            progress={"stage": stage, "stages": stages},
            heartbeat_at=now,
        )
        print(
            f"{datetime.now(UTC).isoformat()} {stage} {json.dumps(info, default=str)}", flush=True
        )

    def should_cancel() -> bool:
        return store.import_cancel_requested(import_id)

    if rq.get("provider") == "local":
        prov: Any = LocalTraceProvider(Path(rq["local_path"]))
    else:
        prov = AdsbLolProvider(offline=settings.offline)
    req = ImportRequest(
        date_utc=rq["date_utc"],
        bbox=BBox.model_validate(rq["bbox"]),
        margin_km=float(rq.get("margin_km", settings.adsb_bbox_margin_km)),
        storage_budget_gb=rq.get("storage_budget_gb"),
        confirm_download=bool(rq.get("confirm_download")),
        workers=int(rq.get("workers") or 0),
        memory_budget_mb=rq.get("memory_budget_mb") or settings.import_memory_budget_mb,
        exclude_jets=bool(rq.get("exclude_jets")),
    )
    try:
        with Heartbeat(beat, 5.0):
            ds = import_flights(
                prov,
                req,
                settings.raw_dir,
                settings.normalized_dir,
                progress=False,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
    except ImportCancelled as e:
        store.finish_import_if_active(import_id, "cancelled", "cancelled", str(e), None)
        return 0
    except ProviderUnavailable as e:
        status = "cancelled" if e.kind == "cancelled" else "failed"
        store.finish_import_if_active(
            import_id, status, job_stage(store, import_id), str(e), e.kind
        )
        store.update_import(
            import_id, progress={"stage": "error", "stages": stages, "details": e.details}
        )
        return 0 if status == "cancelled" else 1
    except StorageBudgetExceeded as e:
        store.finish_import_if_active(import_id, "failed", "locate", str(e), "storage_budget")
        return 1
    except MemoryBudgetExceeded as e:
        store.finish_import_if_active(import_id, "failed", "parse", str(e), "memory_budget")
        return 1
    except Exception as e:
        log.exception("import %s failed", import_id)
        store.finish_import_if_active(
            import_id, "failed", job_stage(store, import_id), f"{type(e).__name__}: {e}", "error"
        )
        return 1
    m = ds.manifest()
    store.update_import(
        import_id,
        dataset_id=ds.dataset_id,
        peak_rss_mb=round(
            max(peak["rss"], (m.get("resources") or {}).get("peak_tree_rss_mb_parse", 0)), 1
        ),
        progress={
            "stage": "done",
            "stages": stages,
            "summary": m.get("summary"),
            "storage_bytes": m.get("storage_bytes"),
            "resources": m.get("resources"),
        },
    )
    store.finish_import_if_active(import_id, "succeeded", "done", None, None)
    store.audit("import.finished", import_id, {"dataset_id": ds.dataset_id})
    return 0


def job_stage(store: Store, import_id: str) -> str:
    j = store.get_import(import_id)
    return (j or {}).get("stage") or "unknown"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m aerodoppler.jobs.imports <import_id>", file=sys.stderr)
        return 2
    from aerodoppler.config import get_settings

    settings = get_settings()
    return run_import_job(Store(settings.db_path), settings, args[0])


if __name__ == "__main__":
    sys.exit(main())
