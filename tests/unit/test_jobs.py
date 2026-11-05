"""Persistent job queue: state machine, atomic admission, idempotency, retries, migration and
reconciliation (in-process; no worker subprocesses — see tests/integration/test_job_workers.py)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import pytest

from aerodoppler.config import Settings
from aerodoppler.jobs.queue import Capacity, SqliteJobQueue
from aerodoppler.jobs.scheduler import Scheduler, worker_alive
from aerodoppler.jobs.service import migrate_legacy_jobs, submit_job
from aerodoppler.jobs.store import Store

CAP = Capacity(class_limits={"import": 1, "analysis": 1, "light": 2}, memory_budget_mb=1000.0)


def _q(tmp_path) -> SqliteJobQueue:
    return SqliteJobQueue(tmp_path / "db.sqlite3")


def _submit(q, cls="light", mem=100.0, key=None, **kw):
    return q.submit(
        "diagnostic", {"seconds": 1, **kw}, resource_class=cls, memory_mb=mem, idempotency_key=key
    )


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_claiming_is_atomic_under_concurrent_claimers(tmp_path):
    q = _q(tmp_path)
    for _ in range(20):
        _submit(q)
    with ThreadPoolExecutor(max_workers=16) as pool:
        claimed = [j for j in pool.map(lambda _i: q.claim_next(CAP), range(40)) if j]
    assert len(claimed) == 2  # light class limit
    assert len({j.id for j in claimed}) == 2
    assert len(q.list(statuses=["claimed"])) == 2


def test_duplicate_submissions_resolve_to_one_job(tmp_path):
    q = _q(tmp_path)
    with ThreadPoolExecutor(max_workers=12) as pool:
        outs = list(pool.map(lambda _i: _submit(q, key="same-import"), range(24)))
    assert len({j.id for j, _ in outs}) == 1
    assert sum(1 for _, created in outs if created) == 1


def test_memory_budget_and_head_of_line_reservation(tmp_path):
    q = _q(tmp_path)
    a, _ = _submit(q, cls="import", mem=500)
    assert q.claim_next(CAP).id == a.id
    b, _ = _submit(q, cls="analysis", mem=800)  # does not fit next to a (500 + 800 > 1000)
    c, _ = _submit(q, cls="light", mem=300)  # would fit alone, but would starve b
    assert q.claim_next(CAP) is None
    q.finish(a.id, None, "succeeded")
    assert q.claim_next(CAP).id == b.id  # b first, not c
    assert q.claim_next(CAP) is None  # 800 + 300 > 1000
    q.finish(b.id, None, "succeeded")
    assert q.claim_next(CAP).id == c.id


def test_job_too_big_for_the_machine_is_refused_at_submission(tmp_path):
    s = Settings(data_dir=tmp_path, offline=True, jobs_memory_budget_mb=500)
    from aerodoppler.jobs.service import JobRejected

    with pytest.raises(JobRejected) as e:
        submit_job(s, "diagnostic", {"reserve_mb": 900})
    assert e.value.code == "insufficient_memory" and "0.5 GB" in str(e.value)


def test_cancel_queued_never_starts(tmp_path):
    q = _q(tmp_path)
    j, _ = _submit(q)
    assert q.request_cancel(j.id).status == "cancelled"
    assert q.claim_next(CAP) is None


def test_terminal_transition_requires_the_owner_and_is_idempotent(tmp_path):
    q = _q(tmp_path)
    _submit(q)
    j = q.claim_next(CAP)
    assert q.start(j.id, "wrong-token", 1, None, "h") is None
    assert q.start(j.id, j.worker_token, os.getpid(), None, "h") is not None
    assert not q.finish(j.id, "wrong-token", "succeeded")
    assert q.finish(j.id, j.worker_token, "succeeded", result={"ok": 1})
    assert not q.finish(j.id, j.worker_token, "failed")  # already terminal
    assert q.get(j.id).status == "succeeded"
    owner, _ = q.heartbeat(j.id, j.worker_token)
    assert not owner


def test_transient_failure_is_retried_with_backoff_then_stops(tmp_path):
    q = _q(tmp_path)
    q.submit("diagnostic", {}, resource_class="light", memory_mb=10, max_attempts=2)
    j = q.claim_next(CAP)
    q.start(j.id, j.worker_token, os.getpid(), None, "h")
    q.finish(
        j.id,
        j.worker_token,
        "failed",
        error_code="network",
        error_message="x",
        retry_at=time.time() + 60,
    )
    j2 = q.get(j.id)
    assert j2.status == "queued" and j2.stage == "waiting_to_retry"
    assert q.claim_next(CAP) is None  # not before next_attempt_at
    j3 = q.claim_next(CAP, now=time.time() + 61)
    assert j3.id == j.id and j3.attempt == 2
    # attempts exhausted: a lost worker is not requeued again
    assert not q.requeue_lost(j.id, "lost")


def test_events_have_monotonic_sequence_for_reconnection(tmp_path):
    q = _q(tmp_path)
    j, _ = _submit(q)
    c = q.claim_next(CAP)
    q.start(c.id, c.worker_token, os.getpid(), None, "h")
    for i in range(5):
        q.progress(c.id, c.worker_token, "sleeping", {"step": i})
    evs = q.events(j.id)
    seqs = [e["seq"] for e in evs]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    mid = seqs[3]
    assert [e["seq"] for e in q.events(j.id, after=mid)] == seqs[4:]


def test_legacy_rows_are_migrated_once_without_losing_data(tmp_path):
    """The reported failure: 17 'running' rows with dead workers blocked new searches."""
    s = Settings(data_dir=tmp_path, offline=True)
    s.ensure_dirs()
    store = Store(s.db_path)
    from aerodoppler.models import BBox, SearchMode, SearchRequest

    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id="rec_x",
        event_ids=["a", "b"],
        date_utc="2026-11-03",
        bbox=BBox(west=-74.1, south=40.7, east=-73.9, north=40.9),
    )
    dead = _dead_pid()
    for i in range(17):
        store.create_search(f"run_{i:02d}", req, tmp_path / f"run_{i}")
        store.update_search(f"run_{i:02d}", status="running", pid=dead, pid_started_at=1.0)
    store.create_search("run_done", req, tmp_path / "run_done")
    store.update_search("run_done", status="succeeded", stage="finished")
    q = SqliteJobQueue(s.db_path)
    assert migrate_legacy_jobs(s, q) == 18
    assert migrate_legacy_jobs(s, q) == 0  # once
    jobs = {j.id: j for j in q.list(limit=100)}
    assert {jobs[f"run_{i:02d}"].status for i in range(17)} == {"interrupted"}
    assert jobs["run_done"].status == "succeeded" and jobs["run_done"].legacy
    assert len(store.list_searches()) == 18  # the legacy table is untouched
    # nothing stale holds a slot: a new search is admitted at once
    submit_job(s, "diagnostic", {"resource_class": "analysis"}, queue=q)
    assert q.claim_next(Capacity({"analysis": 1}, 10_000.0)) is not None


def test_reconcile_dead_reused_and_hung_workers(tmp_path):
    s = Settings(data_dir=tmp_path, offline=True)
    s.ensure_dirs()
    q = SqliteJobQueue(s.db_path)
    sched = Scheduler(s, q, stall_kill_s=5.0)

    def running(pid, started, job_type="diagnostic", mode="worker"):
        j, _ = q.submit(
            job_type, {}, resource_class="light", memory_mb=1, mode=mode, max_attempts=3
        )
        with q._conn() as c:
            c.execute(
                "UPDATE jobs SET status='running', pid=?, pid_started_at=?, heartbeat_at=?, attempt=1, "
                "worker_token='t', lease_expires_at=? WHERE id=?",
                (pid, started, time.time(), time.time() + 30, j.id),
            )
        return j.id

    dead = running(_dead_pid(), 1.0)
    reused = running(os.getpid(), psutil.Process().create_time() - 500)  # PID alive, other process
    victim = subprocess.Popen(
        [sys.executable, "-c", "import time,sys; time.sleep(120)", "hung-job-placeholder"]
    )
    hung = running(victim.pid, psutil.Process(victim.pid).create_time(), mode="inline")
    with q._conn() as c:  # last heartbeat long ago
        c.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (time.time() - 60, hung))
    resumable = running(_dead_pid(), 1.0, job_type="flight_import")
    affected = sched.reconcile()
    assert {dead, reused, hung, resumable} <= set(affected)
    assert q.get(dead).status == "interrupted" and q.get(dead).error_code == "worker_lost"
    assert q.get(reused).status == "interrupted"
    assert psutil.pid_exists(os.getpid())  # the reused PID (this test) was not killed
    assert q.get(hung).status == "interrupted" and "heartbeats" in q.get(hung).error_message
    assert q.get(resumable).status == "queued"  # resume-safe type, attempts left
    victim.kill()
    victim.wait()


def test_worker_identity_checks_command_line(tmp_path):
    q = _q(tmp_path)
    j, _ = _submit(q)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with q._conn() as c:
            c.execute(
                "UPDATE jobs SET status='running', pid=?, pid_started_at=? WHERE id=?",
                (p.pid, psutil.Process(p.pid).create_time(), j.id),
            )
        assert not worker_alive(q.get(j.id))  # alive process, but not this job's worker
    finally:
        p.kill()
        p.wait()


def test_only_one_scheduler_dispatches(tmp_path):
    q = _q(tmp_path)
    assert q.acquire_scheduler_lock("a", os.getpid(), None)
    assert not q.acquire_scheduler_lock("b", os.getpid(), None)
    q.release_scheduler_lock("a")
    assert q.acquire_scheduler_lock("b", os.getpid(), None)


def test_child_job_of_a_lost_parent_is_finalised(tmp_path):
    s = Settings(data_dir=tmp_path, offline=True)
    s.ensure_dirs()
    q = SqliteJobQueue(s.db_path)
    parent, _ = q.submit("benchmark", {}, resource_class="analysis", memory_mb=1)
    child, _ = q.submit(
        "geolocation_search",
        {"_parent_job": parent.id},
        resource_class="child",
        memory_mb=0,
        mode="child",
    )
    with q._conn() as c:
        c.execute(
            "UPDATE jobs SET status='running', pid=?, pid_started_at=? WHERE id=?",
            (os.getpid(), psutil.Process().create_time(), child.id),
        )
    q.finish(parent.id, None, "failed", error_message="x")
    Scheduler(s, q).reconcile()
    assert q.get(child.id).status == "interrupted"


def test_job_view_hides_worker_token_and_traceback(tmp_path):
    from aerodoppler.api.jobs_routes import job_view

    q = _q(tmp_path)
    _submit(q)
    j = q.claim_next(CAP)
    q.finish(
        j.id,
        None,
        "failed",
        error_code="internal_error",
        error_message="boom",
        error_details={"traceback": "Traceback..."},
    )
    v = job_view(q.get(j.id))
    assert "worker_token" not in json.dumps(v) and "Traceback" not in json.dumps(v)
    assert "Traceback" in json.dumps(job_view(q.get(j.id), debug=True))


def test_inline_progress_never_reaches_stdout(tmp_path, capsys):
    """Regression (release smoke test): CLI stdout is machine-readable JSON (demo, search), so
    inline-job progress must go to stderr."""
    from aerodoppler.jobs.service import run_inline

    s = Settings(data_dir=tmp_path, offline=True)
    s.ensure_dirs()
    job, _ = run_inline(s, "diagnostic", {"seconds": 0.3})
    out = capsys.readouterr()
    assert job.status == "succeeded"
    assert out.out == "" and job.id in out.err


def test_spectrogram_reservation_covers_measured_peak():
    """Regression: a 2-hour recording's spectrogram job reserved 2.1 GB, used 4.3 GB and was
    killed by the memory guard. Measured peaks: 1.6 GB / 3.1 GB / 6.0 GB for 30 / 60 / 120 min."""
    from aerodoppler.jobs.handlers import JOB_TYPES

    s = Settings(offline=True)
    for dur, measured_mb in ((1800, 1614), (3600, 3076), (7200, 5960)):
        for t in ("spectrogram_generation", "ridge_detection"):
            assert JOB_TYPES[t].memory_mb(s, {"duration_s": dur}, 30_000) >= measured_mb


def test_retry_recomputes_the_memory_reservation(tmp_path):
    from aerodoppler.jobs.service import retry_job

    s = Settings(data_dir=tmp_path, offline=True)
    q = SqliteJobQueue(s.db_path)
    j, _ = q.submit(
        "spectrogram_generation",
        {"recording_id": "r", "duration_s": 7200},
        resource_class="light",
        memory_mb=2099,
    )
    q.finish(
        j.id, None, "failed", error_code="budget_exceeded", error_message="memory budget exceeded"
    )
    j2 = retry_job(s, q, j.id)
    assert j2.status == "queued" and j2.memory_mb > 6000


def test_retry_refuses_a_job_that_no_longer_fits_the_budget(tmp_path):
    """Regression: retry recomputed the reservation (6.7 GB for 2 h) without the budget check
    that submission applies, so a 4 GB machine could re-queue a job that can never start."""
    from aerodoppler.jobs.service import JobRejected, retry_job

    s = Settings(data_dir=tmp_path, offline=True, jobs_memory_budget_mb=4000)
    q = SqliteJobQueue(s.db_path)
    j, _ = q.submit(
        "spectrogram_generation",
        {"recording_id": "r", "duration_s": 7200},
        resource_class="light",
        memory_mb=2099,
    )
    q.finish(j.id, None, "failed", error_message="memory budget exceeded")
    with pytest.raises(JobRejected) as e:
        retry_job(s, q, j.id)
    assert e.value.code == "insufficient_memory"
    assert q.get(j.id).status == "failed"  # not re-queued


def test_oversized_queued_job_does_not_block_smaller_ones(tmp_path):
    """An oversized job already in the queue (e.g. queued before the budget was lowered) must
    not hold a head-of-line reservation, and the scheduler fails it with the reason."""
    s = Settings(data_dir=tmp_path, offline=True, jobs_memory_budget_mb=1000)
    q = SqliteJobQueue(s.db_path)
    big, _ = _submit(q, mem=6700)  # older than the small one
    small, _ = _submit(q, mem=100)
    claimed = q.claim_next(Capacity({"light": 2}, 1000.0))
    assert claimed is not None and claimed.id == small.id
    Scheduler(s, q).fail_oversized(Capacity({"light": 2}, 1000.0))
    j = q.get(big.id)
    assert j.status == "failed" and j.error_code == "insufficient_memory"
