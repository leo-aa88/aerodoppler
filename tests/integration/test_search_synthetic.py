"""End-to-end synthetic recovery through the real pipeline (SPEC §13 integration tests):
scenario → ingest → spectrogram → auto events → Mode A (unknown time) and Mode B (known
time) → ranked candidates; plus negative cases (wrong date, no flights, no DEM)."""

from __future__ import annotations

import pytest

from aerodoppler.audio.service import (
    ensure_spectrogram,
    ingest_recording,
    pick_event_pair,
    propose_events,
)
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
    all_events = propose_events(store, settings, rec.id)
    events = pick_event_pair(all_events)
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
    from aerodoppler.jobs.queue import SqliteJobQueue

    job = SqliteJobQueue(w["settings"].db_path).get(res.run_id)
    assert job.status == "succeeded" and job.mode == "inline"
    assert not (run_dir / "INCOMPLETE").exists()


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
    # a dataset whose day cannot contain the recording is refused before searching ...
    with pytest.raises(
        SearchInputError, match="covers 2026-09-08, but the recording can only be on 2026-09-09"
    ):
        run_search_inline(w["store"], w["settings"], req)
    # ... and with the explicit development override the search abstains for lack of flights
    res = run_search_inline(
        w["store"], w["settings"], req.model_copy(update={"allow_date_mismatch": True})
    )
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


def test_area_outside_flight_data_is_refused(world):
    w = world
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[w["events"][0].id, w["events"][1].id],
        date_utc=w["spec"].date_utc,
        bbox=BBox(west=10.0, south=50.0, east=10.3, north=50.3),
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        quality_settings=QualitySettings(workers=2),
    )
    with pytest.raises(SearchInputError, match="outside every selected flight dataset"):
        run_search_inline(w["store"], w["settings"], req)


def test_missing_dem_is_an_explicit_error(world):
    import shutil

    w = world
    dem_dir = w["settings"].dem_dir
    hidden = dem_dir.with_name("dem_hidden_for_test")
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[w["events"][0].id, w["events"][1].id],
        date_utc=w["spec"].date_utc,
        bbox=w["spec"].bbox,
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        quality_settings=QualitySettings(workers=2),
    )
    shutil.move(dem_dir, hidden)
    dem_dir.mkdir()
    try:
        with pytest.raises(SearchInputError, match="DEM unavailable"):
            run_search_inline(w["store"], w["settings"], req)
        # an explicit flat-ground value is accepted, and flagged in the result
        req2 = req.model_copy(
            update={
                "quality_settings": QualitySettings(
                    workers=4, ground_elevation_m=200.0, ground_elevation_sigma_m=50.0
                )
            }
        )
        res = run_search_inline(w["store"], w["settings"], req2)
        assert any("flat ground" in wmsg for wmsg in res.warnings)
    finally:
        shutil.rmtree(dem_dir)
        shutil.move(hidden, dem_dir)


def test_search_over_memory_budget_stops_cleanly(world, monkeypatch):
    """A search that exceeds its memory budget fails visibly instead of exhausting RAM."""
    import aerodoppler.resources as res
    from aerodoppler.jobs.queue import JobError, SqliteJobQueue

    w = world
    monkeypatch.setattr(res, "tree_pss_mb", lambda pid=None: 1e9)
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[e.id for e in w["events"][:2]],
        date_utc=w["spec"].date_utc,
        bbox=w["spec"].bbox,
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        quality_settings=QualitySettings(workers=1),
    )
    with pytest.raises(JobError) as e:
        run_search_inline(w["store"], w["settings"], req)
    assert e.value.code == "memory_budget"
    job = SqliteJobQueue(w["settings"].db_path).list(types=["geolocation_search"])[0]
    assert job.status == "failed" and job.error_code == "memory_budget"
    assert "budget" in job.error_message


def test_search_without_datasets_uses_only_date_compatible_ones(world):
    """No dataset list: the engine resolves date-compatible data of the same kind instead of
    every imported archive; a wrong date is refused with the reason."""
    w = world
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[e.id for e in w["events"][:2]],
        date_utc="2020-01-01",
        bbox=w["spec"].bbox,
    )
    with pytest.raises(
        SearchInputError, match="no imported SYNTHETIC flight archive covers 2020-01-01"
    ):
        run_search_inline(w["store"], w["settings"], req)


def test_anytime_screen_reports_partial_coverage(world):
    """With a screen budget that runs out, the search scores candidates most-consistent first,
    says how many it scored, and is marked not exhaustive (never silently complete)."""
    w = world
    q = QualitySettings(workers=2, ultra_keep_top=200, screen_batch=500, screen_budget_s=0.0)
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[e.id for e in w["events"][:2]],
        date_utc=w["spec"].date_utc,
        bbox=w["spec"].bbox,
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        quality_settings=q,
    )
    res = run_search_inline(w["store"], w["settings"], req)
    gate = (
        next(g for g in res.gates if g["name"] == "search_complete")
        if isinstance(res.gates[0], dict)
        else next(g for g in res.gates if g.name == "search_complete")
    )
    detail = gate["detail"] if isinstance(gate, dict) else gate.detail
    passed = gate["passed"] if isinstance(gate, dict) else gate.passed
    assert not passed and "NOT exhaustive" in detail and "of " in detail
    screen = next(c for c in res.stage_counters if c.stage == "ultra_coarse_screen")
    assert screen.candidates_in > 500 and "stopped by the 0 s budget" in screen.note
    assert res.confidence_grade != "high"


@pytest.mark.parametrize("batched", [True, False])
def test_screen_releases_unkept_candidates_and_still_finds_the_answer(world, batched):
    """Many small screen batches with a small kept top, with the batched scorer and with the
    reference path (which releases candidates that fall out of the top as it goes): the kept
    top is still the right one."""
    w = world
    truth = w["sc"].truth
    q = QualitySettings(
        workers=2,
        ultra_keep_top=100,
        screen_batch=300,
        screen_budget_s=3600.0,
        batched_screen=batched,
    )
    req = SearchRequest(
        mode=SearchMode.UNKNOWN_TIME,
        recording_id=w["rec"].id,
        event_ids=[w["events"][0].id, w["events"][1].id],
        date_utc=w["spec"].date_utc,
        bbox=w["spec"].bbox,
        flight_dataset_ids=[w["sc"].dataset.dataset_id],
        quality_settings=q,
    )
    res = run_search_inline(w["store"], w["settings"], req)
    screen = next(c for c in res.stage_counters if c.stage == "ultra_coarse_screen")
    assert screen.candidates_in > 3 * 300 and "(complete)" in screen.note
    assert screen.candidates_out <= 100
    best = res.candidates[0]
    assert _err(best, truth) < 300.0
    assert {s["icao24"] for s in best.extra["segments"]} == {
        truth["event_a"]["icao24"],
        truth["event_b"]["icao24"],
    }


def test_batched_and_reference_screens_give_the_same_search(world):
    """End to end: the batched screen is an optimisation; with the same request the search keeps
    the same candidates and returns the same answer as the reference path."""
    w = world
    out = {}
    for batched in (True, False):
        q = QualitySettings(workers=2, ultra_keep_top=150, screen_batch=400, batched_screen=batched)
        req = SearchRequest(
            mode=SearchMode.UNKNOWN_TIME,
            recording_id=w["rec"].id,
            event_ids=[w["events"][0].id, w["events"][1].id],
            date_utc=w["spec"].date_utc,
            bbox=w["spec"].bbox,
            flight_dataset_ids=[w["sc"].dataset.dataset_id],
            quality_settings=q,
        )
        res = run_search_inline(w["store"], w["settings"], req)
        screen = next(c for c in res.stage_counters if c.stage == "ultra_coarse_screen")
        out[batched] = (
            [(round(c.lat, 7), round(c.lon, 7), round(c.score_total, 6)) for c in res.candidates],
            screen.candidates_in,
            screen.candidates_out,
            res.confidence_grade,
        )
    assert out[True] == out[False]
