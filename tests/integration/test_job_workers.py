"""Job system with REAL worker subprocesses: concurrency limits under load, crashes (SIGKILL),
cancellation (cooperative and forced), memory enforcement, automatic retry, server restarts,
progress reconnection and inline jobs. Small diagnostic jobs keep every test cheap."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import psutil
import pytest
from fastapi.testclient import TestClient

from aerodoppler.api.app import create_app
from aerodoppler.config import Settings
from aerodoppler.jobs.queue import TERMINAL, SqliteJobQueue
from aerodoppler.jobs.scheduler import Scheduler
from aerodoppler.jobs.service import run_inline, submit_job

pytestmark = pytest.mark.slow


def _settings(tmp: Path, **kw) -> Settings:
    s = Settings(data_dir=tmp, offline=True, jobs_scheduler_in_api=True, **kw)
    s.ensure_dirs()
    return s


def _wait(q: SqliteJobQueue, job_id: str, timeout: float = 60.0, until=TERMINAL):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = q.get(job_id)
        if j.status in until:
            return j
        time.sleep(0.1)
    raise AssertionError(f"{job_id} still {q.get(job_id).status} after {timeout} s")


def _worker_pids(job_ids: list[str]) -> list[int]:
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = " ".join(p.info["cmdline"] or [])
        if "aerodoppler.jobs.worker" in cl and any(j in cl for j in job_ids):
            out.append(p.info["pid"])
    return out


def test_concurrency_limit_holds_under_load(tmp_path):
    """A system configured for two light workers never runs more than two, however many
    jobs are submitted at once — checked both in the database and among live processes."""
    s = _settings(tmp_path, jobs_max_light=2)
    q = SqliteJobQueue(s.db_path)
    ids = [submit_job(s, "diagnostic", {"seconds": 1.5}, queue=q)[0].id for _ in range(7)]
    sched = Scheduler(s, q, poll_s=0.1).start()
    peak_db, peak_proc = 0, 0
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            live = q.list(statuses=["claimed", "running"])
            peak_db = max(peak_db, len(live))
            active = [j.id for j in q.list(statuses=["claimed", "running"])]
            # a worker counts while its job is active (one that has written its final state
            # and is exiting no longer runs work)
            peak_proc = max(peak_proc, len(_worker_pids(active)) if active else 0)
            if all(q.get(i).status in TERMINAL for i in ids):
                break
            time.sleep(0.05)
    finally:
        sched.stop()
    assert {q.get(i).status for i in ids} == {"succeeded"}
    assert peak_db == 2 and peak_proc <= 2


def test_sigkilled_worker_is_reconciled_and_frees_its_slot(tmp_path):
    s = _settings(tmp_path)
    q = SqliteJobQueue(s.db_path)
    j, _ = submit_job(s, "diagnostic", {"seconds": 60, "resource_class": "analysis"}, queue=q)
    sched = Scheduler(s, q, poll_s=0.1).start()
    try:
        _wait(q, j.id, until=("running",))
        time.sleep(0.5)
        os.kill(q.get(j.id).pid, signal.SIGKILL)  # an OOM kill looks exactly like this
        dead = _wait(q, j.id, timeout=30)
        assert dead.status == "interrupted" and dead.error_code == "worker_lost"
        # the slot is free again: the next analysis job runs
        j2, _ = submit_job(s, "diagnostic", {"seconds": 0.2, "resource_class": "analysis"}, queue=q)
        assert _wait(q, j2.id).status == "succeeded"
    finally:
        sched.stop()


def test_cancel_running_job_cooperatively_and_by_force(tmp_path):
    s = _settings(tmp_path)
    q = SqliteJobQueue(s.db_path)
    nice, _ = submit_job(s, "diagnostic", {"seconds": 60}, queue=q)
    stubborn, _ = submit_job(s, "diagnostic", {"seconds": 60, "ignore_cancel": True}, queue=q)
    sched = Scheduler(s, q, poll_s=0.2, cancel_grace_s=1.5).start()
    try:
        _wait(q, nice.id, until=("running",))
        _wait(q, stubborn.id, until=("running",))
        pid = q.get(stubborn.id).pid
        q.request_cancel(nice.id)
        q.request_cancel(stubborn.id)
        assert _wait(q, nice.id, timeout=20).status == "cancelled"
        j = _wait(q, stubborn.id, timeout=30)
        assert j.status == "cancelled" and "killed" in (j.error_message or "")
        time.sleep(0.5)
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        sched.stop()


def test_memory_overrun_is_stopped(tmp_path):
    """Controlled memory pressure: a job holding ~400 MB against a 50 MB reservation."""
    s = _settings(tmp_path)
    q = SqliteJobQueue(s.db_path)
    j, _ = submit_job(
        s, "diagnostic", {"seconds": 40, "allocate_mb": 400, "reserve_mb": 50}, queue=q
    )
    sched = Scheduler(s, q, poll_s=0.2).start()
    try:
        done = _wait(q, j.id, timeout=40)
        assert done.status == "failed" and "memory budget exceeded" in done.error_message
    finally:
        sched.stop()


def test_transient_failures_are_retried_with_a_limit(tmp_path):
    s = _settings(tmp_path, jobs_retry_base_s=0.3)
    q = SqliteJobQueue(s.db_path)
    ok, _ = submit_job(s, "diagnostic", {"seconds": 0.1, "fail_times": 2}, queue=q)  # 3 attempts
    bad, _ = submit_job(s, "diagnostic", {"seconds": 0.1, "fail_times": 5}, queue=q)
    sched = Scheduler(s, q, poll_s=0.1).start()
    try:
        j = _wait(q, ok.id, timeout=60)
        assert j.status == "succeeded" and j.attempt == 3
        b = _wait(q, bad.id, timeout=60)
        assert b.status == "failed" and b.attempt == 3  # never retried indefinitely
    finally:
        sched.stop()


def test_job_survives_api_restart_and_progress_reconnects(tmp_path):
    """Browser refresh and server restart: the job keeps running in its own process and the
    new server shows the same job; events resume after the last sequence number seen."""
    s = _settings(tmp_path)
    with TestClient(create_app(s, web_dir=tmp_path / "no-web")) as c:
        r = c.post("/api/jobs", json={"type": "diagnostic", "params": {"seconds": 6}})
        assert r.status_code == 422  # diagnostic is not offered through the generic endpoint
    # submit directly (as the CLI would) and let the API's scheduler run it
    q = SqliteJobQueue(s.db_path)
    j, _ = submit_job(s, "diagnostic", {"seconds": 6}, queue=q)
    with TestClient(create_app(s, web_dir=tmp_path / "no-web")) as c:
        _wait(q, j.id, until=("running",))
        first = c.get(f"/api/jobs/{j.id}/events?format=json").json()
        assert first["job"]["status"] == "running"
        seen = first["last_seq"]
    # the server is gone; the worker is a separate process and keeps going
    time.sleep(1.0)
    assert q.get(j.id).status == "running"
    with TestClient(create_app(s, web_dir=tmp_path / "no-web")) as c:
        later = c.get(f"/api/jobs/{j.id}/events?format=json&after={seen}").json()
        assert all(e["seq"] > seen for e in later["events"])
        done = _wait(q, j.id, timeout=60)
        assert done.status == "succeeded"
        listed = c.get("/api/jobs?status=succeeded&type=diagnostic").json()["jobs"]
        assert [x["job_id"] for x in listed] == [j.id]


def test_inline_job_waits_for_capacity_and_is_finalised(tmp_path):
    s = _settings(tmp_path)
    q = SqliteJobQueue(s.db_path)
    busy, _ = submit_job(s, "diagnostic", {"seconds": 3, "resource_class": "analysis"}, queue=q)
    sched = Scheduler(s, q, poll_s=0.1).start()
    try:
        _wait(q, busy.id, until=("running",))
        t0 = time.time()
        job, _ = run_inline(
            s, "diagnostic", {"seconds": 0.2, "resource_class": "analysis"}, log=lambda m: None
        )
        assert job.status == "succeeded"
        assert (
            q.get(busy.id).status == "succeeded"
            and q.get(busy.id).finished_at <= job.started_at + 0.5
        )
        assert time.time() - t0 >= 1.0  # it really waited for the slot
        with pytest.raises(ZeroDivisionError):
            run_inline(s, "diagnostic", {}, fn=lambda ctx: 1 / 0, log=lambda m: None)
        failed = q.list(statuses=["failed"])
        assert failed and failed[0].error_code == "internal_error"
    finally:
        sched.stop()


def test_sigkilled_inline_process_is_reconciled(tmp_path):
    s = _settings(tmp_path)
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from aerodoppler.config import Settings
        from aerodoppler.jobs.service import run_inline
        s = Settings(data_dir=Path({str(tmp_path)!r}), offline=True)
        run_inline(s, "diagnostic", {{"seconds": 120}}, log=lambda m: None)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script])
    q = SqliteJobQueue(s.db_path)
    deadline = time.time() + 60
    while time.time() < deadline and not q.list(statuses=["running"]):
        time.sleep(0.2)
    (j,) = q.list(statuses=["running"])
    assert j.mode == "inline"
    proc.kill()
    proc.wait()
    Scheduler(s, q).reconcile()
    assert q.get(j.id).status == "interrupted"


def test_worker_starvation_is_avoided(tmp_path):
    """A big queued job is not overtaken forever by a stream of small ones."""
    s = _settings(tmp_path, jobs_memory_budget_mb=1000, jobs_max_light=4)
    q = SqliteJobQueue(s.db_path)
    small = [
        submit_job(s, "diagnostic", {"seconds": 1.0, "reserve_mb": 300}, queue=q)[0]
        for _ in range(2)
    ]
    big, _ = submit_job(s, "diagnostic", {"seconds": 0.2, "reserve_mb": 900}, queue=q)
    stop = threading.Event()

    def feeder():  # keep submitting small jobs while the big one waits
        while not stop.is_set():
            submit_job(s, "diagnostic", {"seconds": 1.0, "reserve_mb": 300}, queue=q)
            time.sleep(0.3)

    th = threading.Thread(target=feeder, daemon=True)
    sched = Scheduler(s, q, poll_s=0.1).start()
    th.start()
    try:
        assert _wait(q, big.id, timeout=60).status == "succeeded"
    finally:
        stop.set()
        th.join()
        for j in q.list(statuses=["queued"]):
            q.request_cancel(j.id)
        sched.stop()
    assert all(_wait(q, x.id, timeout=60).status == "succeeded" for x in small)


def test_forked_pool_shutdown_does_not_cancel_the_job(tmp_path):
    """Regression: forked scoring-pool children inherited the worker's SIGTERM handler, so the
    pool's own terminate() turned into a cancellation of the whole search."""
    s = _settings(tmp_path)
    q = SqliteJobQueue(s.db_path)
    j, _ = submit_job(s, "diagnostic", {"seconds": 6, "fork_pool": True}, queue=q)
    sched = Scheduler(s, q, poll_s=0.1).start()
    try:
        done = _wait(q, j.id, timeout=60)
        assert done.status == "succeeded", done.error_message
        assert not done.cancel_requested
    finally:
        sched.stop()
