"""Job lifecycle regressions: persistent imports, deduplication, atomic admission under
concurrent submissions, dead/killed/reused-PID workers, cancellation and API restarts."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest
from fastapi.testclient import TestClient

from aerodoppler.api.app import create_app
from aerodoppler.config import Settings
from aerodoppler.jobs import imports as imp
from aerodoppler.jobs import runner
from aerodoppler.jobs.store import Store
from aerodoppler.models import BBox, SearchMode, SearchRequest

TRACES = Path(__file__).parents[1] / "fixtures" / "adsblol" / "2026-11-02" / "traces"


def _req(date: str = "2026-11-02", west: float = -82.0) -> dict:
    return {
        "provider": "local",
        "local_path": str(TRACES),
        "date_utc": date,
        "bbox": {"west": west, "south": 28.0, "east": west + 1.0, "north": 29.0},
        "margin_km": 15.0,
        "confirm_download": True,
    }


def _settings(tmp: Path, **kw) -> Settings:
    s = Settings(data_dir=tmp, offline=True, **kw)
    s.ensure_dirs()
    return s


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_duplicate_import_request_reuses_the_running_job(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    out1 = store.create_import_if_capacity("imp_1", _req(), imp.dedup_key(_req()), max_active=1)
    out2 = store.create_import_if_capacity("imp_2", _req(), imp.dedup_key(_req()), max_active=1)
    out3 = store.create_import_if_capacity(
        "imp_3", _req(west=-90.0), imp.dedup_key(_req(west=-90.0)), max_active=1
    )
    assert out1 == ("created", "imp_1")
    assert out2 == ("reused", "imp_1")
    assert out3 == ("full", None)


def test_concurrent_import_submissions_never_oversubscribe(tmp_path):
    store = Store(tmp_path / "db.sqlite3")

    def go(i: int):
        r = _req(west=-100.0 + i)
        return store.create_import_if_capacity(f"imp_{i}", r, imp.dedup_key(r), max_active=2)

    with ThreadPoolExecutor(max_workers=12) as pool:
        outs = list(pool.map(go, range(24)))
    assert sum(1 for o, _ in outs if o == "created") == 2
    assert len(store.list_imports(active_only=True)) == 2


def test_seventeen_stale_search_rows_do_not_block_admission(tmp_path):
    """The reported failure: 17 'running' rows with dead workers -> '17 jobs already running'."""
    store = Store(tmp_path / "db.sqlite3")
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id="rec_x",
        event_ids=["a", "b"],
        date_utc="2026-11-03",
        bbox=BBox(west=-74.1, south=40.7, east=-73.9, north=40.9),
    )
    dead = _dead_pid()
    for i in range(17):
        store.create_search(f"run_{i}", req, tmp_path / f"run_{i}")
        store.update_search(f"run_{i}", status="running", pid=dead, pid_started_at=1.0)
    reconciled = runner.reconcile_stale_jobs(store)
    assert len(reconciled) == 17
    assert store.create_search_if_capacity("run_new", req, tmp_path / "new", max_active=2) is None
    statuses = {store.get_search(f"run_{i}")["status"] for i in range(17)}
    assert statuses == {"interrupted"}  # history kept, not deleted
    assert len(store.list_searches()) == 18


def test_sigkilled_import_worker_is_reconciled_to_interrupted(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    store.create_import_if_capacity("imp_k", _req(), imp.dedup_key(_req()), max_active=1)
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    store.update_import(
        "imp_k",
        status="running",
        stage="parse",
        pid=victim.pid,
        pid_started_at=psutil.Process(victim.pid).create_time(),
    )
    assert imp.reconcile_stale_imports(store) == []  # alive: untouched
    victim.send_signal(signal.SIGKILL)
    victim.wait()
    assert imp.reconcile_stale_imports(store) == ["imp_k"]
    job = store.get_import("imp_k")
    assert job["status"] == "interrupted" and job["error_kind"] == "worker_lost"
    assert "parse" in job["error"] and "retry" in job["error"].lower()


def test_reused_pid_is_not_mistaken_for_the_worker(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    store.create_import_if_capacity("imp_r", _req(), imp.dedup_key(_req()), max_active=1)
    # our own PID is alive but started at a different time than the recorded worker
    store.update_import(
        "imp_r",
        status="running",
        pid=os.getpid(),
        pid_started_at=psutil.Process().create_time() - 500.0,
    )
    assert imp.reconcile_stale_imports(store) == ["imp_r"]
    assert psutil.pid_exists(os.getpid())  # nothing was killed


def test_cancel_of_dead_import_finalises_immediately(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    store.create_import_if_capacity("imp_c", _req(), imp.dedup_key(_req()), max_active=1)
    store.update_import("imp_c", status="running", pid=_dead_pid(), pid_started_at=1.0)
    assert imp.cancel_import(store, "imp_c")
    assert store.get_import("imp_c")["status"] == "cancelled"
    assert not imp.cancel_import(store, "imp_c")  # terminal: idempotent, no second transition


def test_import_worker_runs_to_completion_inline(tmp_path):
    s = _settings(tmp_path)
    store = Store(s.db_path)
    req = _req()
    store.create_import_if_capacity("imp_ok", req, imp.dedup_key(req), max_active=1)
    store.update_import("imp_ok", status="running")
    assert imp.run_import_job(store, s, "imp_ok") == 0
    job = store.get_import("imp_ok")
    assert job["status"] == "succeeded" and job["dataset_id"]
    assert job["progress"]["stage"] == "done" and "parse" in job["progress"]["stages"]
    assert (s.normalized_dir / job["dataset_id"] / "manifest.json").exists()


def test_import_worker_honours_cancellation(tmp_path):
    s = _settings(tmp_path)
    store = Store(s.db_path)
    req = _req()
    store.create_import_if_capacity("imp_x", req, imp.dedup_key(req), max_active=1)
    store.update_import("imp_x", status="running")
    store.request_import_cancel("imp_x")
    assert imp.run_import_job(store, s, "imp_x") == 0
    assert store.get_import("imp_x")["status"] == "cancelled"


@pytest.mark.slow
def test_api_import_job_survives_api_restart(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_app(s, web_dir=tmp_path / "no-web")) as c:
        r = c.post("/api/flight-datasets/import", json=_req())
        assert r.status_code == 202, r.text
        job_id = r.json()["job"]["id"]
        again = c.post("/api/flight-datasets/import", json=_req()).json()
        assert again["outcome"] in ("reused", "exists")
    # a brand-new app instance (= API restart / page reload) sees the same job
    with TestClient(create_app(s, web_dir=tmp_path / "no-web")) as c:
        deadline = time.time() + 120
        while time.time() < deadline:
            j = c.get(f"/api/imports/{job_id}").json()
            if j["status"] not in ("queued", "running"):
                break
            time.sleep(0.5)
        assert j["status"] == "succeeded", j
        jobs = c.get("/api/jobs").json()
        assert any(x["id"] == job_id for x in jobs["imports"])
        assert jobs["counts"]["imports_active"] == 0
        assert jobs["resources"]["effective_limit_mb"] > 0
        assert c.get(f"/api/imports/{job_id}/log").json()["lines"]


def test_import_endpoint_rejects_impossible_days_before_any_job(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_app(s, web_dir=tmp_path / "no-web")) as c:
        body = {**_req(), "provider": "adsblol"}
        r = c.post("/api/flight-datasets/import", json={**body, "date_utc": "2999-01-01"})
        assert r.status_code == 422 and r.json()["detail"]["code"] == "future_date"
        r = c.post("/api/flight-datasets/import", json={**body, "date_utc": "1977-07-04"})
        assert r.json()["detail"]["code"] == "before_archive_start"
        r = c.post(
            "/api/flight-datasets/import",
            json={**body, "date_utc": "2026-10-01", "confirm_download": False},
        )
        assert r.status_code == 409 and r.json()["detail"]["code"] == "download_not_confirmed"
        assert c.get("/api/imports").json()["imports"] == []


def test_validation_errors_are_concise_and_url_free(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_app(s, web_dir=tmp_path / "no-web")) as c:
        r = c.post("/api/searches", json={"mode": "known-time", "recording_id": "rec_x"})
        assert r.status_code == 422
        d = r.json()["detail"]
        assert d["code"] == "invalid_request" and d["errors"]
        assert "http" not in r.text and "pydantic" not in r.text.lower()
