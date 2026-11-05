"""Search prerequisites, event evidence and detection diagnostics through the HTTP API
(offline; recordings and datasets generated in the test)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from aerodoppler.adsb.ingest import import_samples_dataframe
from aerodoppler.api.app import create_app
from aerodoppler.audio.service import ingest_recording
from aerodoppler.config import Settings
from aerodoppler.jobs.store import Store
from aerodoppler.models import DopplerEvent, RidgePoint

SR = 2000
T0 = 1793718000.0  # 2026-11-03T15:00:00Z
BBOX = {"west": -74.1, "south": 40.7, "east": -73.9, "north": 40.9}


def _tone_wav(path: Path, seconds: float, f_of_t, on=(10.0, 60.0)) -> Path:
    t = np.arange(int(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f_of_t(t)) / SR
    gate = ((t >= on[0]) & (t <= on[1])).astype(float)
    x = 0.3 * gate * np.sin(phase) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    sf.write(path, x.astype(np.float32), SR)
    return path


def _event(rec_id: str, f_of_t, t0: float, t1: float) -> DopplerEvent:
    ts = np.arange(t0, t1, 0.25)
    return DopplerEvent(
        recording_id=rec_id,
        audio_start_s=t0,
        audio_end_s=t1,
        ridge_points=[RidgePoint(relative_t_s=float(t), frequency_hz=float(f_of_t(t))) for t in ts],
        detection_method="manual",
    )


def _doppler(t, f0=100.0, tc=60.0, w=15.0):
    # a descending sigmoid of ±12 %, like a light aircraft passing at ~40 m/s
    return f0 * (1.0 + 0.12 * np.tanh(-(np.asarray(t) - tc) / w))


def _flat(t):
    return 329.5 + 0.2 * np.sin(np.asarray(t) / 7.0)


@pytest.fixture()
def env(tmp_path):
    s = Settings(data_dir=tmp_path / "data", offline=True)
    s.ensure_dirs()
    store = Store(s.db_path)
    wav = _tone_wav(tmp_path / "r.wav", 120.0, _flat)
    rec = ingest_recording(store, s, wav, recording_date_utc="2026-11-03")
    rec_b = ingest_recording(
        store, s, wav, start_time_utc="2026-11-03T15:00:00Z", start_time_uncertainty_s=5.0
    )
    flat = store.add_event(_event(rec.id, _flat, 10.0, 60.0))
    dop1 = store.add_event(_event(rec.id, _doppler, 20.0, 100.0))
    dop2 = store.add_event(_event(rec.id, lambda t: _doppler(t, tc=70.0), 30.0, 110.0))
    dop_b = store.add_event(_event(rec_b.id, _doppler, 20.0, 100.0))
    # a tiny real-kind dataset for 2026-11-03 covering BBOX (two samples are enough here)
    df = pl.DataFrame(
        {
            "source": ["t", "t", "t"],
            "icao24": ["abc123"] * 3,
            "t": [T0, T0 + 10.0, T0 + 20.0],
            "lat": [40.8, 40.801, 40.802],
            "lon": [-74.0, -74.0, -74.0],
            "alt_baro_ft": [1500.0] * 3,
            "alt_geom_ft": [1500.0] * 3,
            "altitude_value_ft": [1500.0] * 3,
            "altitude_datum": ["geometric_wgs84_hae"] * 3,
            "on_ground": [False] * 3,
            "gs_mps": [50.0] * 3,
            "track_deg": [0.0] * 3,
            "vertical_rate_mps": [0.0] * 3,
            "flags": [None] * 3,
            "stale": [False] * 3,
            "new_leg": [False] * 3,
            "position_source": [None] * 3,
            "callsign": [None] * 3,
            "aircraft_type": ["C172"] * 3,
            "aircraft_desc": [None] * 3,
            "registration": [None] * 3,
            "db_flags": [None] * 3,
            "category": [None] * 3,
        },
        schema_overrides={
            "flags": pl.Utf8,
            "position_source": pl.Utf8,
            "callsign": pl.Utf8,
            "aircraft_desc": pl.Utf8,
            "registration": pl.Utf8,
            "db_flags": pl.Int64,
            "category": pl.Utf8,
        },
    )
    from aerodoppler.adsb.normalize import SAMPLE_SCHEMA

    df = df.cast({k: v for k, v in SAMPLE_SCHEMA.items() if k in df.columns})
    import_samples_dataframe(
        df,
        "real_2026-11-03",
        s.normalized_dir,
        notes=[],
        synthetic=False,
        extra_manifest={"request": {"date_utc": "2026-11-03", "bbox": [-74.2, 40.6, -73.8, 41.0]}},
    )
    import_samples_dataframe(
        df,
        "real_2026-11-02",
        s.normalized_dir,
        notes=[],
        synthetic=False,
        extra_manifest={"request": {"date_utc": "2026-11-02", "bbox": [-74.2, 40.6, -73.8, 41.0]}},
    )
    client = TestClient(create_app(s, web_dir=tmp_path / "no-web"))
    client.__enter__()  # runs the lifespan: the job scheduler executes detection jobs
    yield {
        "c": client,
        "rec": rec,
        "rec_b": rec_b,
        "flat": flat,
        "dop1": dop1,
        "dop2": dop2,
        "dop_b": dop_b,
    }


def _a(env, **kw):
    body = {
        "mode": "unknown-time",
        "recording_id": env["rec"].id,
        "date_utc": "2026-11-03",
        "bbox": BBOX,
    }
    body.update(kw)
    return body


def test_events_carry_evidence_labels(env):
    evs = env["c"].get(f"/api/recordings/{env['rec'].id}/events").json()["events"]
    labels = {e["id"]: e["assessment"]["label"] for e in evs}
    assert labels[env["flat"].id] == "tonal_only"
    assert labels[env["dop1"].id] == "plausible_doppler"


def test_flat_ridge_is_refused_until_reviewed(env):
    c = env["c"]
    r = c.post("/api/searches/estimate", json=_a(env, event_ids=[env["flat"].id, env["dop1"].id]))
    assert r.status_code == 422
    d = r.json()["detail"]
    assert d["code"] == "event_not_doppler" and d["event_id"] == env["flat"].id and d["action"]
    rv = c.post(
        f"/api/recordings/{env['rec'].id}/events/{env['flat'].id}/review", json={"accepted": True}
    )
    a = rv.json()["events"][0]["assessment"]
    assert a["label"] == "tonal_only" and a["review"]["accepted"] is True  # label unchanged
    r = c.post("/api/searches/estimate", json=_a(env, event_ids=[env["flat"].id, env["dop1"].id]))
    assert r.status_code == 200, r.text


def test_mode_prerequisites_are_explicit(env):
    c = env["c"]
    r = c.post(
        "/api/searches/estimate",
        json={
            "mode": "known-time",
            "recording_id": env["rec_b"].id,
            "event_ids": [env["dop_b"].id],
            "bbox": BBOX,
        },
    )
    assert r.json()["detail"]["code"] == "start_utc_required"
    assert r.json()["detail"]["field"] == "start_utc"
    r = c.post(
        "/api/searches/estimate",
        json={
            "mode": "known-time",
            "recording_id": env["rec_b"].id,
            "event_ids": [env["dop_b"].id],
            "bbox": BBOX,
            "start_utc": "2026-11-03T15:00:00",
        },
    )
    assert r.json()["detail"]["code"] == "start_utc_no_timezone"
    r = c.post(
        "/api/searches/estimate",
        json={
            "mode": "known-time",
            "recording_id": env["rec_b"].id,
            "event_ids": [env["dop_b"].id],
            "bbox": BBOX,
            "start_utc": "2026-11-03T15:00:00Z",
        },
    )
    assert r.json()["detail"]["code"] == "uncertainty_required"
    r = c.post("/api/searches/estimate", json=_a(env, event_ids=[env["dop1"].id]))
    assert r.json()["detail"]["code"] == "two_events_required"


def test_dataset_selection_is_explicit_and_checked(env):
    c = env["c"]
    ok = c.post("/api/searches/estimate", json=_a(env, event_ids=[env["dop1"].id, env["dop2"].id]))
    assert ok.status_code == 200, ok.text
    est = ok.json()
    assert est["datasets"] == ["real_2026-11-03"]  # not "all": the 11-02 archive is excluded
    assert est["grid_points"] > 0 and est["segments_in_window_and_area"] >= 1
    r = c.post(
        "/api/searches/estimate",
        json=_a(
            env, event_ids=[env["dop1"].id, env["dop2"].id], flight_dataset_ids=["real_2026-11-02"]
        ),
    )
    assert r.json()["detail"]["code"] == "date_mismatch"
    r = c.post(
        "/api/searches/estimate",
        json=_a(env, event_ids=[env["dop1"].id, env["dop2"].id], date_utc="2026-10-01"),
    )
    assert r.json()["detail"]["code"] == "no_dataset_for_date"
    far = {"west": 10.0, "south": 50.0, "east": 10.2, "north": 50.2}
    r = c.post(
        "/api/searches/estimate", json=_a(env, event_ids=[env["dop1"].id, env["dop2"].id], bbox=far)
    )
    assert r.json()["detail"]["code"] == "bbox_outside_data"


def test_detection_explains_a_stationary_tone(env):
    r = env["c"].post(
        f"/api/recordings/{env['rec'].id}/events/detect?wait_s=60",
        json={"threshold_z": 2.5, "min_duration_s": 40, "save": False},
    )
    assert r.status_code == 201, r.text  # finished within wait_s (a ridge_detection job)
    d = r.json()["diagnostics"]
    assert d["scanned"]["t_start_s"] == pytest.approx(0.0, abs=5) and d["scanned"]["t_end_s"] > 100
    assert "whole recording" in d["scanned"]["note"]
    # the 329.5 Hz tone is either rejected as stationary or proposed but labelled tonal-only;
    # it is never presented as aircraft evidence
    labels = d["evidence_labels"]
    assert d["rejected_counts"].get("stationary_tone", 0) > 0 or labels.get("tonal_only", 0) > 0
    assert not labels.get("plausible_doppler")
    assert d["suggestions"]
