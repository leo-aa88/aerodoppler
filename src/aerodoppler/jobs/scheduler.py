"""Scheduler: admission, dispatch, supervision and crash recovery for the job queue.

Runs as a background thread in the API process (``aerodoppler serve``) or headless
(``aerodoppler jobs scheduler``); a lease in ``scheduler_lock`` makes sure only one dispatches.
Every tick it:

1. **reconciles** live rows against reality: a ``claimed``/``running`` job whose worker process
   is gone (PID missing, a different process with the same PID — creation time or command line
   differ — or a zombie) is *lost*; resume-safe types go back to the queue while attempts
   remain, others become ``interrupted`` with the reason. A live worker that stopped renewing
   its lease for ``stall_kill_s`` is terminated and interrupted. Rows are never deleted and a
   lost job never keeps a concurrency slot;
2. **escalates cancellations**: a running job asked to cancel gets SIGTERM after
   ``cancel_grace_s`` and SIGKILL after twice that, then is finalised as cancelled;
3. **claims and spawns** queued jobs while the per-class limits and the shared memory budget
   allow (see :meth:`SqliteJobQueue.claim_next`).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from aerodoppler.config import Settings
from aerodoppler.jobs.handlers import JOB_TYPES, shared_budget_mb
from aerodoppler.jobs.proc import process_start_time
from aerodoppler.jobs.queue import LIVE, Capacity, Job, SqliteJobQueue

log = logging.getLogger(__name__)


def capacity_for(settings: Settings) -> Capacity:
    return Capacity(
        class_limits={
            "import": settings.jobs_max_import,
            "analysis": settings.jobs_max_analysis,
            "light": settings.jobs_max_light,
        },
        memory_budget_mb=shared_budget_mb(settings),
    )


def worker_alive(job: Job) -> bool:
    """True only if the recorded process exists, is not a zombie, has the recorded creation
    time and (for spawned workers) carries this job's id on its command line."""
    if not job.pid:
        return False
    try:
        p = psutil.Process(job.pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return False
        if job.pid_started_at is not None and abs(p.create_time() - job.pid_started_at) > 1.0:
            return False  # PID reused by another process
        if job.mode == "worker":
            try:
                if job.id not in " ".join(p.cmdline()):
                    return False
            except psutil.AccessDenied:
                pass
        return True
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # cannot inspect: safer to assume alive than to run the job twice


class Scheduler:
    def __init__(
        self,
        settings: Settings,
        queue: SqliteJobQueue | None = None,
        poll_s: float = 1.0,
        cancel_grace_s: float = 20.0,
        stall_kill_s: float = 900.0,
        spawn_grace_s: float = 60.0,
    ):
        self.settings = settings
        self.queue = queue or SqliteJobQueue(settings.db_path)
        self.poll_s = poll_s
        self.cancel_grace_s = cancel_grace_s
        self.stall_kill_s = stall_kill_s
        self.spawn_grace_s = spawn_grace_s
        self.owner = f"scheduler-{os.getpid()}-{int(time.time() * 1000)}"
        self._procs: dict[str, subprocess.Popen] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- dispatch ----------------------------------------------------------------------
    def spawn(self, job: Job) -> None:
        work_dir = self.settings.data_dir / "jobs" / job.id
        if job.type == "geolocation_search":
            work_dir = self.settings.runs_dir / job.id  # results keep their historic location
        work_dir.mkdir(parents=True, exist_ok=True)
        log_path = work_dir / "job.log"
        self.queue.set_paths(job.id, str(log_path), str(work_dir))
        env = dict(os.environ)
        # the worker must run with the scheduler's settings (budgets, retry policy, offline…),
        # not whatever its environment happens to say
        for k, v in self.settings.model_dump().items():
            if v is None or isinstance(v, dict | list):
                continue
            env[f"AERODOPPLER_{k.upper()}"] = (
                ("true" if v else "false") if isinstance(v, bool) else str(v)
            )
        try:
            with open(log_path, "ab") as fh:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "aerodoppler.jobs.worker",
                        job.id,
                        str(job.worker_token),
                    ],
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except OSError as e:
            self.queue.finish(
                job.id,
                None,
                "failed",
                error_code="launch_error",
                error_message=f"the worker process could not be started: {e}",
            )
            return
        self.queue.set_process(
            job.id, str(job.worker_token), proc.pid, process_start_time(proc.pid)
        )
        self._procs[job.id] = proc

    def _reap(self) -> None:
        for jid, p in list(self._procs.items()):
            if p.poll() is not None:
                del self._procs[jid]

    # ---- reconciliation ----------------------------------------------------------------
    def reconcile(self, now: float | None = None) -> list[str]:
        """Finalise or requeue live rows whose worker is gone or hung. Returns affected ids."""
        now = now or time.time()
        affected = []
        for job in self.queue.list(statuses=list(LIVE), limit=1000):
            if job.mode == "child":
                continue  # owned by its parent's process; reconciled with the parent below
            alive = worker_alive(job)
            if job.status == "claimed" and not job.pid:
                # spawned row whose process was never recorded (scheduler crashed in between)
                if job.claimed_at and now - job.claimed_at > self.spawn_grace_s:
                    alive = False
                else:
                    continue
            if alive:
                hb = job.heartbeat_at or job.claimed_at or job.updated_at
                if hb and now - hb > self.stall_kill_s:
                    self._kill(job, signal.SIGKILL)
                    self._lost(
                        job,
                        f"the worker stopped sending heartbeats for {now - hb:.0f} s (hung); it was terminated",
                    )
                    affected.append(job.id)
                continue
            legacy_final = self._legacy_final(job) if job.legacy else None
            if legacy_final is not None:
                # a pre-queue worker adopted at upgrade time: it wrote its outcome to the old table
                self.queue.finish(job.id, None, legacy_final[0], error_message=legacy_final[1])
            elif job.cancel_requested:
                self.queue.finish(
                    job.id,
                    None,
                    "cancelled",
                    error_code="cancelled",
                    error_message="cancelled by operator; the worker had already stopped",
                )
            else:
                self._lost(
                    job,
                    f"the worker process ended without finishing during stage '{job.stage}' (crash, kill, out of memory or machine restart)",
                )
            affected.append(job.id)
        # children of jobs that are no longer running
        for job in self.queue.list(statuses=list(LIVE), limit=1000):
            if job.mode != "child":
                continue
            parent = (
                self.queue.get(str(job.params.get("_parent_job")))
                if job.params.get("_parent_job")
                else None
            )
            if parent is None or parent.status not in LIVE or not worker_alive(job):
                self.queue.finish(
                    job.id,
                    None,
                    "interrupted",
                    error_code="parent_lost",
                    error_message="the job that ran this search stopped",
                )
                affected.append(job.id)
        return affected

    def _legacy_final(self, job: Job) -> tuple[str, str | None] | None:
        table = "searches" if job.type == "geolocation_search" else "imports"
        with self.queue._conn() as c:
            try:
                row = c.execute(
                    f"SELECT status, error FROM {table} WHERE id=?", (job.id,)
                ).fetchone()
            except Exception:
                return None
        if row and row["status"] in ("succeeded", "abstained", "failed", "cancelled"):
            return row["status"], row["error"]
        return None

    def _lost(self, job: Job, reason: str) -> None:
        jt = JOB_TYPES.get(job.type)
        if (
            jt
            and jt.resume_safe
            and job.attempt < job.max_attempts
            and self.queue.requeue_lost(
                job.id, reason + "; resuming automatically (safe for this job type)"
            )
        ):
            log.warning("job %s requeued: %s", job.id, reason)
            return
        self.queue.finish(
            job.id,
            None,
            "interrupted",
            error_code="worker_lost",
            error_message=reason
            + ". Partial outputs are kept and marked incomplete; use Retry to run it again.",
        )
        log.warning("job %s interrupted: %s", job.id, reason)

    def _kill(self, job: Job, sig: int) -> None:
        if not job.pid or not worker_alive(job):
            return
        try:
            if job.mode == "worker":
                os.killpg(os.getpgid(job.pid), sig)
            # inline jobs live in the user's CLI process: never signal it from here
        except OSError:
            pass

    def escalate_cancellations(self, now: float | None = None) -> None:
        now = now or time.time()
        for job in self.queue.list(statuses=list(LIVE), limit=1000):
            if not job.cancel_requested or job.mode != "worker":
                continue
            asked = self._cancel_asked_at(job) or now
            waited = now - asked
            if not worker_alive(job):
                self.queue.finish(
                    job.id,
                    None,
                    "cancelled",
                    error_code="cancelled",
                    error_message="cancelled by operator",
                )
            elif waited > 2 * self.cancel_grace_s:
                self._kill(job, signal.SIGKILL)
                self.queue.finish(
                    job.id,
                    None,
                    "cancelled",
                    error_code="cancelled",
                    error_message="cancelled by operator (worker did not stop; killed)",
                )
            elif waited > self.cancel_grace_s:
                self._kill(job, signal.SIGTERM)

    def fail_oversized(self, cap) -> list[str]:
        """Fail queued jobs whose reservation exceeds the whole budget (e.g. queued before the
        budget was lowered): they could never start and must not wait forever."""
        out = []
        for job in self.queue.list(statuses=["queued"], limit=1000):
            if job.mode == "worker" and job.memory_mb > cap.memory_budget_mb:
                if self.queue.finish(
                    job.id,
                    None,
                    "failed",
                    error_code="insufficient_memory",
                    error_message=(
                        f"needs about {job.memory_mb / 1e3:.1f} GB of memory but jobs may use at most "
                        f"{cap.memory_budget_mb / 1e3:.1f} GB on this machine "
                        "(AERODOPPLER_JOBS_MEMORY_BUDGET_MB); it can never start"
                    ),
                ):
                    out.append(job.id)
        return out

    def _cancel_asked_at(self, job: Job) -> float | None:
        with self.queue._conn() as c:
            row = c.execute("SELECT cancel_requested_at FROM jobs WHERE id=?", (job.id,)).fetchone()
        return row["cancel_requested_at"] if row else None

    # ---- loop --------------------------------------------------------------------------
    def tick(self) -> list[str]:
        """One scheduling round; returns the ids of jobs started."""
        if not self.queue.acquire_scheduler_lock(
            self.owner, os.getpid(), process_start_time(os.getpid())
        ):
            return []
        self._reap()
        self.reconcile()
        self.escalate_cancellations()
        started = []
        cap = capacity_for(self.settings)
        self.fail_oversized(cap)
        while (job := self.queue.claim_next(cap)) is not None:
            self.spawn(job)
            started.append(job.id)
        return started

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("scheduler tick failed")
            self._stop.wait(self.poll_s)
        self.queue.release_scheduler_lock(self.owner)

    def start(self) -> Scheduler:
        self._thread = threading.Thread(target=self._run, name="job-scheduler", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop dispatching. Workers keep running (they are separate processes) and are
        picked up again by the next scheduler."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)


def run_forever(settings: Settings) -> None:  # pragma: no cover - CLI entry
    s = Scheduler(settings)
    log.info("scheduler %s running (Ctrl-C to stop; workers keep running)", s.owner)
    try:
        s._run()
    except KeyboardInterrupt:
        s.queue.release_scheduler_lock(s.owner)


__all__ = ["Scheduler", "capacity_for", "worker_alive", "run_forever", "Path"]
