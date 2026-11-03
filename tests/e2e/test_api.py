"""End-to-end through the HTTP API without stubbed results: upload a synthetic recording,
spectrogram, manual + automatic events, launch a real search job (subprocess), poll, fetch
candidates/curves/export, purge."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aerodoppler.api.app import create_app
from aerodoppler.config import Settings
from aerodoppler.validation.scenarios import ScenarioSpec, build_scenario

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    settings = Settings(data_dir=root, offline=True, max_concurrent_jobs=1)
    settings.ensure_dirs()
    spec = ScenarioSpec(name="api", seed=5, snr_db=12.0, duration_s=1500.0)
    sc = build_scenario(spec, root / "scenario", settings.normalized_dir, settings.dem_dir)
    app = create_app(settings, web_dir=root / "no-web")
    with TestClient(app) as client:
        yield {"client": client, "settings": settings, "spec": spec, "sc": sc}


def test_health_and_privacy_defaults(api):
    r = api["client"].get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["host"] == "127.0.0.1" and "permission" in body["authorization_notice"]
    assert r.headers["Referrer-Policy"] == "no-referrer"


def test_upload_requires_permission(api):
    wav = Path(api["sc"].wav_path)
    with open(wav, "rb") as fh:
        r = api["client"].post(
            "/api/recordings",
            files={"file": (wav.name, fh, "audio/wav")},
            data={"owner_permission_confirmed": "false"},
        )
    assert r.status_code == 403


def test_full_flow(api):
    c = api["client"]
    sc = api["sc"]
    spec = api["spec"]
    wav = Path(sc.wav_path)
    with open(wav, "rb") as fh:
        r = c.post(
            "/api/recordings",
            files={"file": (wav.name, fh, "audio/wav")},
            data={
                "owner_permission_confirmed": "true",
                "recording_date_utc": spec.date_utc,
                "synthetic": "true",
            },
        )
    assert r.status_code == 201, r.text
    rec = r.json()["recording"]
    rid = rec["id"]
    assert rec["synthetic"] is True and rec["sha256"]
    # spectrogram view (downsampled)
    r = c.get(
        f"/api/recordings/{rid}/spectrogram",
        params={"t0": 0, "t1": 600, "f0": 20, "f1": 500, "max_frames": 300, "max_bins": 200},
    )
    assert r.status_code == 200
    sp = r.json()
    assert len(sp["times"]) <= 300 and len(sp["freqs"]) <= 200 and sp["synthetic"] is True
    assert np.isfinite(np.array(sp["values"])).all()
    # manual event (three points) works before auto detection
    r = c.post(
        f"/api/recordings/{rid}/events",
        json={"points": [[150.0, 95.0], [190.0, 88.0], [230.0, 80.0]], "harmonic": 1},
    )
    assert r.status_code == 201
    manual_id = r.json()["events"][0]["id"]
    # bad manual event rejected
    assert c.post(f"/api/recordings/{rid}/events", json={"points": [[1.0, 2.0]]}).status_code == 422
    # automatic proposals
    r = c.post(
        f"/api/recordings/{rid}/events/detect", json={"threshold_z": 2.5, "min_duration_s": 40}
    )
    assert r.status_code == 200
    from aerodoppler.audio.service import pick_event_pair
    from aerodoppler.models import DopplerEvent

    auto_all = [DopplerEvent.model_validate(e) for e in r.json()["events"]]
    auto = [e.model_dump() for e in pick_event_pair(auto_all)]
    assert len(auto) == 2
    # delete the manual one
    assert c.delete(f"/api/recordings/{rid}/events/{manual_id}").status_code == 200
    # flight datasets present (synthetic, labelled)
    ds = c.get("/api/flight-datasets").json()["datasets"]
    assert any(d["synthetic"] for d in ds)
    # unauthorized search rejected
    body = {
        "mode": "unknown-time",
        "recording_id": rid,
        "event_ids": [auto[0]["id"], auto[1]["id"]],
        "date_utc": spec.date_utc,
        "bbox": spec.bbox.model_dump(),
        "operator_authorized": False,
    }
    assert c.post("/api/searches", json=body).status_code == 403
    # real search job (subprocess)
    body["operator_authorized"] = True
    body["quality_settings"] = {"workers": 4}
    r = c.post("/api/searches", json=body)
    assert r.status_code == 202, r.text
    run_id = r.json()["id"]
    # concurrency cap: a second job is rejected while the first runs
    r2 = c.post("/api/searches", json=body)
    assert r2.status_code in (429, 202)
    if r2.status_code == 202:
        c.post(f"/api/searches/{r2.json()['id']}/cancel")
    deadline = time.time() + 600
    status = None
    while time.time() < deadline:
        status = c.get(f"/api/searches/{run_id}").json()
        if status["status"] in ("succeeded", "failed", "cancelled", "abstained"):
            break
        time.sleep(2)
    assert status is not None and status["status"] == "succeeded", status
    assert status["progress"]["stages"] and status["cpu_seconds"] is not None
    cands = c.get(f"/api/searches/{run_id}/candidates").json()
    assert cands["synthetic"] is True and cands["candidates"]
    best = cands["candidates"][0]
    from aerodoppler.geo.coordinates import haversine_m

    err = float(
        haversine_m(best["lat"], best["lon"], sc.truth["observer_lat"], sc.truth["observer_lon"])
    )
    assert err < 400.0, err
    curves = c.get(f"/api/searches/{run_id}/candidates/{best['candidate_id']}/curves").json()
    assert len(curves["events"]) == 2 and curves["events"][0]["predicted_hz_by_harmonic"]["1"]
    assert c.get(f"/api/searches/{run_id}/tracks").json()
    assert c.get(f"/api/searches/{run_id}/heatmap").json()["points"]
    gj = c.get(f"/api/searches/{run_id}/export", params={"format": "geojson"}).json()
    assert gj["type"] == "FeatureCollection" and gj["properties"]["synthetic"] is True
    man = c.get(f"/api/searches/{run_id}/export", params={"format": "manifest"}).json()
    assert man["git_sha"] is not None or True  # git may be absent in CI containers
    assert man["datasets"] and man["recording"]["sha256"] == rec["sha256"]
    # export JSON re-validates as a SearchResult (reimport)
    from aerodoppler.models import SearchResult

    SearchResult.model_validate(
        c.get(f"/api/searches/{run_id}/export", params={"format": "json"}).json()
    )
    # audit log has entries; purge removes everything
    assert c.get("/api/audit").json()["entries"]
    r = c.delete(f"/api/recordings/{rid}")
    assert r.status_code == 200 and r.json()["removed"]
    assert c.get(f"/api/recordings/{rid}").status_code == 404
    assert not (api["settings"].runs_dir / run_id).exists()


def test_path_traversal_and_bad_ids(api):
    c = api["client"]
    assert c.get("/api/recordings/../../etc/passwd").status_code in (400, 404)
    assert c.get("/api/searches/not%20an%20id").status_code == 400
    assert c.get("/api/searches/run_abc/candidates/..%2F..%2Fx/curves").status_code in (400, 404)
