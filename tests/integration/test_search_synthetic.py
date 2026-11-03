"""End-to-end synthetic recovery through the real pipeline (SPEC §13 integration tests):
scenario → ingest → spectrogram → auto events → Mode A (unknown time) and Mode B (known
time) → ranked candidates; plus negative cases (wrong date, no flights, no DEM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aerodoppler.audio.service import ensure_spectrogram, ingest_recording, propose_events
from aerodoppler.config import Settings
from aerodoppler.geo.coordinates import haversine_m
from aerodoppler.jobs.runner import run_search_inline
from aerodoppler.jobs.store import Store
from aerodoppler.models import BBox, QualitySettings, SearchMode, SearchRequest, unix_to_iso
from aerodoppler.search.engine import SearchInputError
from aerodoppler.validation.scenarios import ScenarioSpec, build_scenario

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    settings = Settings(data_dir=root, offline=True)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    spec = ScenarioSpec(name="itest", seed=11, snr_db=10.0, duration_s=1500.0)
    sc = build_scenario(spec, root / "scenario", settings.normalized_dir, settings.dem_dir)
    rec = ingest_recording(
        store,
        settings,
        sc.wav_path,
        owner_permission_confirmed=True,
        recording_date_utc=spec.date_utc,
        synthetic=True,
    )
    ensure_spectrogram(store, settings, rec.id)
    events = sorted(propose_events(store, settings, rec.id), key=lambda e: e.audio_start_s)
    return {
        "settings": settings,
        "store": store,
        "spec": spec,
        "sc": sc,
        "rec": rec,
        "events": events,
    }


def _err(c, truth):
    return float(haversine_m(c.lat, c.lon, truth["observer_lat"], truth["observer_lon"]))


def test_auto_events_found(world):
    ev = world["events"]
    assert len(ev) >= 2
    truth = world["sc"].truth
    # fundamentals within 10 % of the truth for the two audible aircraft
    f0s = sorted(e.f0_prior for e in ev[:2])
    assert abs(f0s[0] - truth["event_a"]["f0_hz"]) / truth["event_a"]["f0_hz"] < 0.1
    assert abs(f0s[1] - truth["event_b"]["f0_hz"]) / truth["event_b"]["f0_hz"] < 0.1


def test_mode_a_unknown_time_recovers_location_and_clock(world):
    w = world
    truth = w["sc"].truth
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[w["events"][0].id, w["events"][1].id],
        date_utc=w["spec"].date_utc,
        bbox=w["spec"].bbox,
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        operator_authorized=True,
        quality_settings=QualitySettings(workers=4),
    )
    res = run_search_inline(w["store"], w["settings"], req)
    assert not res.abstention.abstained, res.abstention
    best = res.candidates[0]
    assert _err(best, truth) < 300.0  # coarse grid 200 m, refined to 25 m
    assert {s["icao24"] for s in best.extra["segments"]} == {
        truth["event_a"]["icao24"],
        truth["event_b"]["icao24"],
    }
    from aerodoppler.models import iso_to_unix

    assert abs(iso_to_unix(best.inferred_start_utc) - truth["recording_start_unix"]) < 3.0
    assert abs(best.source_fundamental_hz_per_event[0] - truth["event_a"]["f0_hz"]) < 1.0
    assert "SYNTHETIC DEMO" in best.flags
    assert best.uncertainty_label.startswith("score-landscape")
    stages = {c.stage for c in res.stage_counters}
    assert {
        "segments_time_bbox_speed",
        "grid_points",
        "cpa_within_slant_bounds",
        "mode_a_pairs",
        "coarse_scoring",
        "fine_scoring",
        "refinement",
    } <= stages
    run_dir = w["settings"].runs_dir / res.run_id
    assert (
        (run_dir / "result.json").exists()
        and (run_dir / "manifest.json").exists()
        and (run_dir / "heatmap.csv").exists()
    )
    assert list((run_dir / "curves").glob("cand_*.json"))
    job = w["store"].get_search(res.run_id)
    assert job["status"] == "succeeded"


def test_mode_b_known_time_single_event(world):
    """Known time ±5 s. NOTE: for one straight constant-speed flyby, moving the observer along
    the track is exactly equivalent to shifting the clock, so the clock uncertainty maps to an
    along-track position uncertainty (±5 s × 62 m/s ≈ ±300 m) and the side of the track is
    ambiguous (mirror solution). The truth must therefore be among the leading distinct
    candidates, not necessarily rank 1."""
    w = world
    truth = w["sc"].truth
    req = SearchRequest(
        mode=SearchMode.KNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[w["events"][0].id],
        start_utc=unix_to_iso(truth["recording_start_unix"] + 3.0),
        start_uncertainty_s=5.0,
        bbox=w["spec"].bbox,
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        operator_authorized=True,
        quality_settings=QualitySettings(workers=4),
    )
    res = run_search_inline(w["store"], w["settings"], req)
    assert not res.abstention.abstained, res.abstention
    best = res.candidates[0]
    assert best.extra["segments"][0]["icao24"] == truth["event_a"]["icao24"]
    errs = [_err(c, truth) for c in res.candidates]
    assert min(errs[:3]) < 600.0, errs
    assert "mode_b_matches" in {c.stage for c in res.stage_counters}
    # the landscape radius must be reported (uncalibrated) and non-trivial for this geometry
    assert best.estimated_uncertainty_m is not None


def test_wrong_date_abstains_with_reason(world):
    w = world
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[w["events"][0].id, w["events"][1].id],
        date_utc="2026-09-09",
        bbox=w["spec"].bbox,
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        operator_authorized=True,
        quality_settings=QualitySettings(workers=2),
    )
    res = run_search_inline(w["store"], w["settings"], req)
    assert res.abstention.abstained and res.abstention.reason == "no_flight_coverage"
    assert res.candidates == []


def test_wrong_time_abstains(world):
    w = world
    truth = w["sc"].truth
    req = SearchRequest(
        mode=SearchMode.KNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[w["events"][0].id],
        start_utc=unix_to_iso(truth["recording_start_unix"] + 1800.0),
        start_uncertainty_s=30.0,
        bbox=w["spec"].bbox,
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        operator_authorized=True,
        quality_settings=QualitySettings(workers=2),
    )
    res = run_search_inline(w["store"], w["settings"], req)
    assert res.abstention.abstained
    assert res.abstention.reason in {
        "no_candidate_geometry",
        "no_flight_coverage",
        "weak_fit",
        "poor_coverage",
    }


def test_missing_dem_is_an_explicit_error(world, tmp_path: Path):
    w = world
    s2 = Settings(data_dir=w["settings"].data_dir, offline=True)
    s2 = s2.model_copy(update={"data_dir": w["settings"].data_dir})
    # a bbox with no DEM tile: elsewhere in the world, offline
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[w["events"][0].id, w["events"][1].id],
        date_utc=w["spec"].date_utc,
        bbox=BBox(west=10.0, south=50.0, east=10.3, north=50.3),
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        operator_authorized=True,
        quality_settings=QualitySettings(workers=2),
    )
    with pytest.raises(SearchInputError, match="DEM unavailable"):
        run_search_inline(w["store"], s2, req)
    # explicit flat-ground fallback is accepted (and flagged) — but has no flights there
    req2 = req.model_copy(
        update={
            "quality_settings": QualitySettings(
                workers=2, ground_elevation_m=200.0, ground_elevation_sigma_m=50.0
            )
        }
    )
    res = run_search_inline(w["store"], s2, req2)
    assert res.abstention.abstained and any("flat ground" in wmsg for wmsg in res.warnings)


def test_unauthorized_request_rejected(world):
    with pytest.raises(ValueError, match="operator_authorized"):
        SearchRequest(
            mode=SearchMode.UNKNOWN_TIME,
            recording_id="x",
            event_ids=["a", "b"],
            date_utc="2026-09-08",
            bbox=BBox(west=-75.2, south=39.9, east=-74.8, north=40.1),
            operator_authorized=False,
        )
