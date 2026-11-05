"""Worker process for one job: ``python -m aerodoppler.jobs.worker <job_id> <token>``.

The scheduler claims a job (status ``claimed``, a fresh random token, a lease) and spawns this
process in its own session. The worker:

1. moves the job to ``running`` only if the token still matches (no double execution);
2. renews the lease every few seconds from a heartbeat thread, recording memory (PSS of the
   whole process tree), CPU time and free disk; it stops (cooperatively) when the job is
   cancelled, when its lease was taken away, or when it exceeds its CPU-time or disk budget;
3. runs the handler and writes exactly one terminal state, guarded by the token;
4. on SIGTERM sets the cancellation flag (cooperative stop); SIGKILL/OOM leave the row to the
   scheduler's reconciliation (lease expiry + process identity).
"""

from __future__ import annotations

import contextvars
import logging
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from aerodoppler.config import Settings
from aerodoppler.jobs.handlers import JOB_TYPES, JobContext
from aerodoppler.jobs.proc import process_start_time
from aerodoppler.jobs.queue import Job, JobCancelled, JobError, SqliteJobQueue, host_id
from aerodoppler.jobs.store import Store

log = logging.getLogger(__name__)

# the job executing in this process (searches started inside a benchmark run as its children)
CURRENT_JOB: contextvars.ContextVar[Job | None] = contextvars.ContextVar(
    "CURRENT_JOB", default=None
)

HEARTBEAT_S = 5.0
SIGNAL_POLL_S = 0.25

# Set by the worker's SIGTERM handler. A signal handler runs between two bytecodes of the main
# thread, possibly inside a SQLite write transaction of that same thread: it must not touch
# the database (a second connection would wait on the lock its own thread holds — up to the
# 30 s busy timeout, stalling the scheduler too). The supervisor thread acts on this flag.
_term_requested = False
RETRY_BASE_S = 30.0
RETRY_MAX_S = 1800.0


def backoff_s(attempt: int, base_s: float = RETRY_BASE_S) -> float:
    return min(RETRY_MAX_S, base_s * 2 ** max(0, attempt - 1))


class Supervisor:
    """Heartbeat + budget watchdog for a running job (used by workers and inline runs)."""

    def __init__(self, queue: SqliteJobQueue, job: Job, settings: Settings, token: str):
        self.queue, self.job, self.settings, self.token = queue, job, settings, token
        self.cancel = threading.Event()
        self.reason: str | None = None  # why the job was asked to stop
        self.lost = False
        self.kill_soon = False
        self._flagged_at = 0.0
        self.peak_pss = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        params = job.params or {}
        self.max_cpu_s = float(params.get("max_cpu_s") or settings.jobs_max_cpu_hours * 3600.0)

    def resources(self) -> dict[str, Any]:
        import psutil

        from aerodoppler.resources import disk_free_bytes, tree_pss_mb

        pss = tree_pss_mb()
        self.peak_pss = max(self.peak_pss, pss)
        cpu = 0.0
        try:
            p = psutil.Process()
            for q in [p, *p.children(recursive=True)]:
                try:
                    t = q.cpu_times()
                    cpu += t.user + t.system
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        return {
            "pss_mb": round(pss, 1),
            "peak_pss_mb": round(self.peak_pss, 1),
            "memory_reserved_mb": round(self.job.memory_mb),
            "cpu_s": round(cpu, 1),
            "disk_free_gb": round(disk_free_bytes(self.settings.data_dir) / 1e9, 2),
        }

    def beat(self) -> None:
        if self.kill_soon and self.job.mode == "worker" and time.time() - self._flagged_at > 10.0:
            # the handler did not stop cooperatively after a memory overrun: record why and end
            # this worker's whole process group before the machine runs out of memory
            self.queue.finish(
                self.job.id,
                self.token,
                "failed",
                error_code="memory_budget",
                error_message=str(self.reason)
                + " (the worker did not stop in time and was terminated)",
            )
            os.killpg(os.getpgid(0), signal.SIGKILL)
        res = self.resources()
        owner, cancel = self.queue.heartbeat(self.job.id, self.token, res)
        if not owner:
            # the scheduler reconciled this job away (lease expired): stop, write nothing
            self.lost = True
            self.reason = self.reason or "lease lost"
            self.cancel.set()
        elif cancel:
            self.reason = self.reason or "cancelled by operator"
            self.cancel.set()
        elif self.job.memory_mb and res["pss_mb"] > 1.5 * self.job.memory_mb + 200.0:
            # generic backstop; imports and searches also throttle/stop themselves earlier
            self.reason = (
                f"memory budget exceeded: the job used {res['pss_mb']:.0f} MB, its reservation is "
                f"{self.job.memory_mb:.0f} MB"
            )
            self.cancel.set()
            if not self.kill_soon:
                self.kill_soon, self._flagged_at = True, time.time()
        elif res["cpu_s"] > self.max_cpu_s:
            self.reason = (
                f"CPU-time budget exceeded ({res['cpu_s']:.0f} s > {self.max_cpu_s:.0f} s)"
            )
            self.cancel.set()
        elif res["disk_free_gb"] * 1e3 < self.settings.jobs_min_free_disk_mb:
            self.reason = (
                f"free disk fell below {self.settings.jobs_min_free_disk_mb / 1e3:.1f} GB "
                f"({res['disk_free_gb']} GB left)"
            )
            self.cancel.set()

    def on_term_flag(self) -> None:
        """Turn a SIGTERM (seen as ``_term_requested``) into a recorded cancellation request
        and the handler's cancel flag, from this thread, not from the signal handler."""
        global _term_requested
        if not _term_requested:
            return
        _term_requested = False
        self.reason = self.reason or "cancelled by operator (SIGTERM)"
        self.cancel.set()
        self.queue.request_cancel(self.job.id)

    def _loop(self) -> None:
        last = time.time()
        while not self._stop.wait(SIGNAL_POLL_S):
            try:
                self.on_term_flag()
                if time.time() - last >= HEARTBEAT_S:
                    last = time.time()
                    self.beat()
            except Exception:  # never let the watchdog kill the job
                log.exception("heartbeat failed")

    def __enter__(self) -> Supervisor:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()


def execute(
    queue: SqliteJobQueue, settings: Settings, job: Job, token: str, logger=None
) -> tuple[str, Any]:
    """Run a started job to exactly one terminal state. Returns ``(status, result)``."""
    jt = JOB_TYPES[job.type]
    store = Store(settings.db_path)
    work_dir = Path(job.work_dir) if job.work_dir else settings.data_dir / "jobs" / job.id
    work_dir.mkdir(parents=True, exist_ok=True)
    sup = Supervisor(queue, job, settings, token)

    def report(stage: str, info: dict[str, Any]) -> None:
        queue.progress(job.id, token, stage, info)

    ctx = JobContext(job, settings, store, report, sup.cancel, work_dir, logger)
    tok = CURRENT_JOB.set(job)
    try:
        with sup:
            sup.beat()
            result = jt.run(ctx)
    except JobCancelled as e:
        if sup.lost:
            return "lost", None
        reason = sup.reason or str(e)
        status = "cancelled" if reason.startswith("cancelled") else "failed"
        code = "cancelled" if status == "cancelled" else "budget_exceeded"
        _cleanup_tmp(work_dir)
        queue.finish(
            job.id, token, status, error_code=code, error_message=reason, resources=sup.resources()
        )
        return status, None
    except JobError as e:
        if sup.lost:
            return "lost", None
        retry = e.transient and job.attempt < job.max_attempts
        retry_at = (
            time.time() + backoff_s(job.attempt, settings.jobs_retry_base_s) if retry else None
        )
        msg = str(e) + (
            f" — retrying automatically (attempt {job.attempt + 1} of {job.max_attempts})"
            if retry
            else ""
        )
        _cleanup_tmp(work_dir)
        queue.finish(
            job.id,
            token,
            "failed",
            error_code=e.code,
            error_message=msg,
            error_details=e.details or None,
            resources=sup.resources(),
            retry_at=retry_at,
        )
        return ("queued" if retry else "failed"), None
    except BaseException as e:  # unexpected: keep the traceback as a detail, not the message
        if sup.lost:
            return "lost", None
        _cleanup_tmp(work_dir)
        queue.finish(
            job.id,
            token,
            "failed",
            error_code="internal_error",
            error_message=f"unexpected {type(e).__name__}: {str(e)[:300]}",
            error_details={"traceback": traceback.format_exc()[-4000:]},
            resources=sup.resources(),
        )
        if isinstance(e, KeyboardInterrupt | SystemExit):
            raise
        return "failed", None
    finally:
        CURRENT_JOB.reset(tok)
    if sup.lost:
        return "lost", None
    status = result.pop("status", "succeeded") if isinstance(result, dict) else "succeeded"
    queue.finish(
        job.id,
        token,
        status,
        result=result,
        resources=sup.resources(),
        dataset_id=(result or {}).get("dataset_id"),
    )
    if isinstance(result, dict) and result.get("recording_id"):
        queue.set_links(job.id, recording_id=result["recording_id"])
    return status, result


def term_handler(main_pid: int):
    def on_term(signum: int, frame: object) -> None:
        global _term_requested
        # forked helpers (the search engine's scoring pool) inherit this handler, and
        # multiprocessing stops its pool with SIGTERM: a child just exits
        if os.getpid() != main_pid:
            os._exit(0)
        _term_requested = True  # no I/O, no locks here (see _term_requested)

    return on_term


def _cleanup_tmp(work_dir: Path) -> None:
    """Remove only this job's scratch space (never shared downloads or datasets)."""
    shutil.rmtree(work_dir / "tmp", ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m aerodoppler.jobs.worker <job_id> <token>", file=sys.stderr)
        return 2
    job_id, token = args
    from aerodoppler.config import get_settings

    settings = get_settings()
    queue = SqliteJobQueue(settings.db_path)
    job = queue.start(job_id, token, os.getpid(), process_start_time(os.getpid()), host_id())
    if job is None:
        print(
            f"{job_id}: not claimed with this token (cancelled or reassigned); exiting",
            file=sys.stderr,
        )
        return 3

    # SIGTERM = please stop: the supervisor thread records it as a cancellation request and
    # sets the handler's cancel flag within SIGNAL_POLL_S; the scheduler follows with SIGKILL
    signal.signal(signal.SIGTERM, term_handler(os.getpid()))
    log.info("job %s (%s) started, attempt %d", job.id, job.type, job.attempt)
    status, _ = execute(queue, settings, job, token)
    log.info("job %s finished: %s", job.id, status)
    return 0 if status in ("succeeded", "abstained", "cancelled", "queued") else 1


if __name__ == "__main__":
    code = main()
    # the final state is written: leave at once instead of a slow interpreter teardown (freeing
    # GBs of arrays), so a finished job never overlaps with the next one admitted in its slot
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(code)
