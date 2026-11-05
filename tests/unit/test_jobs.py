from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import psutil
import pytest

from aerodoppler.config import Settings
from aerodoppler.jobs import runner
from aerodoppler.jobs.store import Store
from aerodoppler.models import BBox, SearchMode, SearchRequest


def _request(recording_id: str = "rec_test") -> SearchRequest:
    return SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=recording_id,
        event_ids=["evt_1", "evt_2"],
        date_utc="2026-11-03",
        bbox=BBox(west=-75.1, south=39.9, east=-74.9, north=40.1),
    )


def test_job_slot_reservation_is_atomic(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    req = _request()

    def reserve(i: int) -> int | None:
        return store.create_search_if_capacity(f"run_{i}", req, tmp_path / f"run_{i}", max_active=2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(8)))

    assert results.count(None) == 2
    assert len(store.list_active_searches()) == 2
    assert all(result is None or result >= 2 for result in results)


def test_reconcile_fails_running_job_with_dead_pid(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    store.create_search("run_dead", _request(), tmp_path / "run_dead")
    store.update_search("run_dead", status="running", pid=2_000_000_000)

    assert runner.reconcile_stale_jobs(store) == ["run_dead"]
    job = store.get_search("run_dead")
    assert job is not None
    assert job["status"] == "failed"
    assert job["stage"] == "worker_lost"


def test_reconcile_preserves_live_job(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    store.create_search("run_live", _request(), tmp_path / "run_live")
    store.update_search(
        "run_live",
        status="running",
        pid=os.getpid(),
        pid_started_at=psutil.Process().create_time(),
    )

    assert runner.reconcile_stale_jobs(store) == []
    job = store.get_search("run_live")
    assert job is not None and job["status"] == "running"


def test_reconcile_rejects_reused_pid(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    store.create_search("run_reused", _request(), tmp_path / "run_reused")
    store.update_search(
        "run_reused",
        status="running",
        pid=os.getpid(),
        pid_started_at=psutil.Process().create_time() - 100,
    )

    assert runner.reconcile_stale_jobs(store) == ["run_reused"]
    job = store.get_search("run_reused")
    assert job is not None and job["status"] == "failed"


def test_cancel_finalizes_job_when_worker_is_already_dead(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    store.create_search("run_dead", _request(), tmp_path / "run_dead")
    store.update_search("run_dead", status="running", pid=2_000_000_000)

    assert runner.cancel_search(store, "run_dead")
    job = store.get_search("run_dead")
    assert job is not None
    assert job["status"] == "cancelled"
    assert job["cancel_requested"] is True


def test_inline_keyboard_interrupt_is_finalized(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path)
    settings.ensure_dirs()
    store = Store(settings.db_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_execute", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_search_inline(store, settings, _request())

    jobs = store.list_searches()
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["stage"] == "interrupted"
