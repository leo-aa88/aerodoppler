"""Job submission, inline execution and legacy migration.

* :func:`submit_job` — the one way work enters the queue (API and CLI). It computes the memory
  reservation and refuses up front a job that could never fit the shared budget.
* :func:`run_inline` — synchronous execution for the CLI and benchmarks. The job is a normal row
  (``mode='inline'``), goes through the same admission (it waits for a slot and says so), holds a
  lease renewed by a heartbeat thread, and is finalised in ``finally`` — a Ctrl-C, an exception or
  SIGTERM all end in a terminal state; a SIGKILL/OOM is reconciled by the scheduler's lease and
  process-identity check like any worker.
* Inside a running job (e.g. a benchmark), searches run as ``mode='child'`` rows covered by the
  parent's admission and reservation, so they neither deadlock on the analysis slot nor escape it.
* :func:`migrate_legacy_jobs` — one-time import of rows from the pre-queue ``searches`` and
  ``imports`` tables (kept untouched) into ``jobs``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aerodoppler.config import Settings
from aerodoppler.jobs.handlers import JOB_TYPES, JobContext, resource_class_for, shared_budget_mb
from aerodoppler.jobs.proc import process_start_time
from aerodoppler.jobs.queue import Job, JobCancelled, JobError, SqliteJobQueue
from aerodoppler.jobs.store import Store


class JobRejected(RuntimeError):
    def __init__(self, message: str, code: str = "rejected"):
        super().__init__(message)
        self.code = code


def queue_for(settings: Settings) -> SqliteJobQueue:
    q = SqliteJobQueue(settings.db_path)
    migrate_legacy_jobs(settings, q)
    return q


def submit_job(
    settings: Settings,
    job_type: str,
    params: dict[str, Any],
    *,
    queue: SqliteJobQueue | None = None,
    idempotency_key: str | None = None,
    mode: str = "worker",
) -> tuple[Job, bool]:
    if job_type not in JOB_TYPES:
        raise JobRejected(f"unknown job type {job_type}", "unknown_type")
    jt = JOB_TYPES[job_type]
    queue = queue or queue_for(settings)
    budget = shared_budget_mb(settings)
    mem = float(jt.memory_mb(settings, params, budget))
    if mem > budget:
        raise JobRejected(
            f"this {job_type.replace('_', ' ')} needs about {mem / 1e3:.1f} GB of memory but jobs may "
            f"use at most {budget / 1e3:.1f} GB on this machine (AERODOPPLER_JOBS_MEMORY_BUDGET_MB). "
            "Use a shorter segment, a smaller area, or raise the budget if the machine has the RAM.",
            "insufficient_memory",
        )
    rec, ds = jt.links(params)
    return queue.submit(
        job_type,
        params,
        resource_class=resource_class_for(job_type, params),
        memory_mb=mem,
        idempotency_key=idempotency_key or jt.idempotency(params),
        recording_id=rec,
        dataset_id=ds,
        max_attempts=jt.max_attempts,
        mode=mode,
    )


def run_inline(
    settings: Settings,
    job_type: str,
    params: dict[str, Any],
    *,
    fn: Callable[[JobContext], Any] | None = None,
    log: Callable[[str], None] | None = None,
    wait_s: float = 24 * 3600.0,
) -> tuple[Job, Any]:
    """Run a job in this process with full lifecycle guarantees. ``fn`` replaces the type's
    handler when the caller needs the in-memory return value (e.g. a SearchResult); its return
    value is passed back and its dict form (if any) stored as the result."""
    from aerodoppler.jobs.scheduler import capacity_for
    from aerodoppler.jobs.worker import CURRENT_JOB, Supervisor

    if log is None:
        # progress goes to stderr: stdout of CLI commands (demo, search, …) is machine-readable
        def log(m: str) -> None:
            print(m, file=sys.stderr, flush=True)

    queue = queue_for(settings)
    parent = CURRENT_JOB.get()
    jt = JOB_TYPES[job_type]
    pid, started = os.getpid(), process_start_time(os.getpid())
    if parent is not None:
        # a child of the running job: covered by the parent's admission and reservation
        job, _ = queue.submit(
            job_type,
            {**params, "_parent_job": parent.id},
            resource_class="child",
            memory_mb=0.0,
            mode="child",
            recording_id=jt.links(params)[0],
        )
        import secrets

        token = secrets.token_hex(8)
        with queue._tx() as c:
            c.execute(
                "UPDATE jobs SET status='running', stage='starting', started_at=?, claimed_at=?, "
                "worker_token=?, pid=?, pid_started_at=?, heartbeat_at=?, lease_expires_at=?, "
                "attempt=1, work_dir=? WHERE id=?",
                (
                    time.time(),
                    time.time(),
                    token,
                    pid,
                    started,
                    time.time(),
                    time.time() + 3600 * 24,
                    str(_work_dir(settings, job)),
                    job.id,
                ),
            )
    else:
        job, _ = submit_job(settings, job_type, params, queue=queue, mode="inline")
        deadline = time.time() + wait_s
        cap = capacity_for(settings)
        announced = 0.0
        while not queue.claim_inline(job.id, cap, pid, started):
            if time.time() > deadline:
                queue.request_cancel(job.id)
                raise JobRejected("no capacity became free in time", "timeout")
            if time.time() - announced > 30:
                busy = [
                    f"{j.type} {j.id} ({j.stage})"
                    for j in queue.list(statuses=["claimed", "running"], limit=10)
                ]
                log(
                    f"[jobs] waiting for a free {jt.resource_class} slot / memory: running {', '.join(busy) or 'nothing'}"
                )
                announced = time.time()
            time.sleep(2.0)
        with queue._conn() as c:
            token = c.execute("SELECT worker_token FROM jobs WHERE id=?", (job.id,)).fetchone()[
                "worker_token"
            ]
        queue.set_paths(job.id, None, str(_work_dir(settings, job)))
    job = queue.get(job.id)
    assert job is not None
    sup = Supervisor(queue, job, settings, token)
    work_dir = Path(job.work_dir or _work_dir(settings, job))
    work_dir.mkdir(parents=True, exist_ok=True)
    echo_state = {"stage": None, "t": 0.0}

    def report(stage: str, info: dict[str, Any]) -> None:
        queue.progress(job.id, token, stage, info)
        now = time.time()
        if parent is None and (stage != echo_state["stage"] or now - echo_state["t"] > 10.0):
            echo_state["stage"], echo_state["t"] = stage, now
            brief = ", ".join(f"{k}={v}" for k, v in list(info.items())[:5])
            log(f"[{job.id}] {stage}: {brief}")

    ctx = JobContext(job, settings, Store(settings.db_path), report, sup.cancel, work_dir, log)
    run = fn or jt.run
    prev_term = None
    if parent is None and threading.current_thread() is threading.main_thread():
        main_pid = os.getpid()

        def on_term(signum: int, frame: object) -> None:  # SIGTERM: stop cooperatively
            if os.getpid() != main_pid:  # an inherited handler in a forked pool worker
                os._exit(0)
            sup.reason = "cancelled by operator (SIGTERM)"
            sup.cancel.set()

        prev_term = signal.signal(signal.SIGTERM, on_term)
    tok = CURRENT_JOB.set(job)
    status, out = "failed", None
    try:
        with sup:
            out = run(ctx)
        res = out if isinstance(out, dict) else getattr(out, "job_result", None)
        status = (res or {}).get("status", "succeeded") if isinstance(res, dict) else "succeeded"
        if isinstance(res, dict):
            # "_"-prefixed keys are in-memory only (e.g. a full benchmark report)
            res = {k: v for k, v in res.items() if k != "status" and not k.startswith("_")}
        queue.finish(job.id, token, status, result=res, resources=sup.resources())
        return queue.get(job.id), out  # type: ignore[return-value]
    except JobCancelled as e:
        status = "cancelled"
        queue.finish(
            job.id,
            token,
            "cancelled",
            error_code="cancelled",
            error_message=sup.reason or str(e),
            resources=sup.resources(),
        )
        raise
    except JobError as e:
        queue.finish(
            job.id,
            token,
            "failed",
            error_code=e.code,
            error_message=str(e),
            error_details=e.details or None,
            resources=sup.resources(),
        )
        raise
    except BaseException as e:
        import traceback

        queue.finish(
            job.id,
            token,
            "interrupted" if isinstance(e, KeyboardInterrupt) else "failed",
            error_code="interrupted" if isinstance(e, KeyboardInterrupt) else "internal_error",
            error_message=f"{type(e).__name__}: {str(e)[:300]}",
            error_details={"traceback": traceback.format_exc()[-4000:]},
            resources=sup.resources(),
        )
        raise
    finally:
        CURRENT_JOB.reset(tok)
        if prev_term is not None:
            signal.signal(signal.SIGTERM, prev_term)


def _work_dir(settings: Settings, job: Job) -> Path:
    if job.type == "geolocation_search":
        return settings.runs_dir / job.id
    return settings.data_dir / "jobs" / job.id


# ---- legacy migration -------------------------------------------------------------------

_LEGACY_MIGRATION = 1


def migrate_legacy_jobs(settings: Settings, queue: SqliteJobQueue) -> int:
    """Copy pre-queue search and import rows into ``jobs`` once (ids, timestamps, status,
    results kept; nothing deleted). Rows that were still 'queued'/'running' are checked against
    their recorded worker process first: live workers keep running and are adopted; dead ones
    become ``interrupted``. Returns the number of rows copied."""
    if _LEGACY_MIGRATION in queue.applied_migrations():
        return 0
    from aerodoppler.jobs.proc import job_process_alive

    store = Store(settings.db_path)
    n = 0
    with queue._conn() as c:
        have = {r["name"] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        legacy_searches = (
            [dict(r) for r in c.execute("SELECT * FROM searches")] if "searches" in have else []
        )
        legacy_imports = (
            [dict(r) for r in c.execute("SELECT * FROM imports")] if "imports" in have else []
        )
    _ = store  # ensure the legacy schema exists (Store creates/migrates it)
    rows = []
    for r in legacy_searches:
        status = r["status"]
        if status in ("queued", "running"):
            status = "running" if status == "running" and job_process_alive(r) else "interrupted"
        rows.append(
            (
                r["id"],
                "geolocation_search",
                status,
                r["stage"],
                r["request_json"],
                "analysis",
                r["recording_id"],
                None,
                r["created_at"],
                r["updated_at"],
                r["error"],
                r["progress_json"] or "{}",
                r["result_dir"],
                r.get("pid"),
                r.get("pid_started_at"),
                json.dumps({"peak_rss_mb": r.get("peak_rss_mb"), "cpu_s": r.get("cpu_seconds")}),
            )
        )
    for r in legacy_imports:
        status = r["status"]
        if status in ("queued", "running"):
            status = "running" if status == "running" and job_process_alive(r) else "interrupted"
        rows.append(
            (
                r["id"],
                "flight_import",
                status,
                r["stage"],
                r["request_json"],
                "import",
                None,
                r.get("dataset_id"),
                r["created_at"],
                r["updated_at"],
                r["error"],
                r["progress_json"] or "{}",
                None,
                r.get("pid"),
                r.get("pid_started_at"),
                json.dumps({"peak_rss_mb": r.get("peak_rss_mb")}),
            )
        )

    def ts(v: Any) -> float:
        if isinstance(v, int | float):
            return float(v)
        from datetime import datetime

        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()

    with queue._tx() as c:
        for (
            jid,
            typ,
            status,
            stage,
            params,
            cls,
            rec,
            ds,
            created,
            updated,
            err,
            prog,
            wdir,
            pid,
            pst,
            res,
        ) in rows:
            if c.execute("SELECT 1 FROM jobs WHERE id=?", (jid,)).fetchone():
                continue
            c.execute(
                "INSERT INTO jobs(id, type, status, stage, params_json, resource_class, memory_mb, "
                "recording_id, dataset_id, mode, created_at, updated_at, finished_at, error_code, "
                "error_message, progress_json, resources_json, work_dir, pid, pid_started_at, attempt, "
                "max_attempts, legacy, lease_expires_at, heartbeat_at) "
                "VALUES (?,?,?,?,?,?,0,?,?,'worker',?,?,?,?,?,?,?,?,?,?,1,1,1,?,?)",
                (
                    jid,
                    typ,
                    status,
                    stage,
                    params,
                    cls,
                    rec,
                    ds,
                    ts(created),
                    ts(updated),
                    ts(updated) if status not in ("running",) else None,
                    "worker_lost" if status == "interrupted" else None,
                    (err or "")
                    + (
                        " [the worker had stopped before this version took over]"
                        if status == "interrupted"
                        else ""
                    )
                    or None,
                    prog,
                    res,
                    wdir,
                    pid,
                    pst,
                    time.time() + 3600 if status == "running" else None,
                    time.time() if status == "running" else None,
                ),
            )
            n += 1
    queue.record_migration(_LEGACY_MIGRATION, f"copied {n} legacy search/import rows into jobs")
    return n


def retry_job(settings: Settings, queue: SqliteJobQueue, job_id: str) -> Job | None:
    """Re-queue a terminal worker job with its memory reservation recomputed."""
    j = queue.get(job_id)
    if j is None:
        return None
    jt = JOB_TYPES.get(j.type)
    budget = shared_budget_mb(settings)
    mem = jt.memory_mb(settings, j.params, budget) if jt else None
    if mem is not None and mem > budget:
        # same admission rule as submit_job: a job that can never fit is refused, not queued
        raise JobRejected(
            f"this {j.type.replace('_', ' ')} needs about {mem / 1e3:.1f} GB of memory but jobs may "
            f"use at most {budget / 1e3:.1f} GB on this machine (AERODOPPLER_JOBS_MEMORY_BUDGET_MB); "
            "it was not re-queued",
            "insufficient_memory",
        )
    return queue.requeue(job_id, memory_mb=mem)
