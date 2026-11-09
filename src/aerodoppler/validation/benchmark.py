"""Benchmark runners (SPEC §13, M7).

* ``run_synthetic_benchmark``: a reproducible set of SYNTHETIC scenarios (seeds, SNRs, geometries,
  distractors, ADS-B gaps) evaluated with hidden truth. This measures the *inverse engine*, not
  real-world accuracy — labelled accordingly.
* ``run_manifest_benchmark``: owner-consented real recordings described by a JSON manifest
  (``cases[]`` with ``audio``, ``date_utc`` or ``start_utc``, ``bbox``, ``truth_lat``/``truth_lon``,
  optional ``events`` ridge points). Truth is read only after each result is committed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aerodoppler import __version__
from aerodoppler.audio.service import (
    add_manual_event,
    ensure_spectrogram,
    ingest_recording,
    pick_event_pair,
    pick_events,
    propose_events,
)
from aerodoppler.config import Settings
from aerodoppler.jobs.runner import run_search_inline
from aerodoppler.jobs.store import Store
from aerodoppler.models import BBox, QualitySettings, SearchMode, SearchRequest, unix_to_iso
from aerodoppler.util import git_sha
from aerodoppler.validation.metrics import CaseOutcome, evaluate_case, summarize
from aerodoppler.validation.scenarios import ScenarioSpec, build_scenario

SYNTHETIC_CASES: list[dict[str, Any]] = [
    {"name": "base_snr10", "seed": 101, "snr_db": 10.0},
    {"name": "lowsnr_4", "seed": 102, "snr_db": 4.0},
    {"name": "verylow_0", "seed": 103, "snr_db": 0.0},
    {"name": "far_cpa", "seed": 104, "snr_db": 10.0, "cpa_a_m": 3500.0, "cpa_b_m": 4200.0},
    {
        "name": "near_fast",
        "seed": 105,
        "snr_db": 10.0,
        "cpa_a_m": 600.0,
        "speed_a": 95.0,
        "speed_b": 110.0,
    },
    {"name": "short_sep_300", "seed": 106, "snr_db": 10.0, "event_separation_s": 300.0},
    {
        "name": "long_sep_1100",
        "seed": 107,
        "snr_db": 10.0,
        "event_separation_s": 1100.0,
        "duration_s": 1700.0,
    },
    {"name": "straight_b", "seed": 108, "snr_db": 10.0, "b_turns": False},
    {"name": "adsb_gap_b", "seed": 109, "snr_db": 10.0, "adsb_gap_in_b": True},
    {"name": "many_distractors", "seed": 110, "snr_db": 8.0, "n_distractors": 8},
    {"name": "heading_mix", "seed": 111, "snr_db": 10.0, "heading_a": 20.0, "heading_b": 250.0},
    {"name": "drifting_rpm", "seed": 112, "snr_db": 10.0, "drift_b": 4e-5},
]


def run_synthetic_benchmark(
    settings: Settings,
    out_dir: Path,
    cases: list[dict[str, Any]] | None = None,
    workers: int = 0,
    mode_b: bool = True,
    log=print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    cases = cases if cases is not None else SYNTHETIC_CASES
    outcomes_a: list[CaseOutcome] = []
    outcomes_b: list[CaseOutcome] = []
    rows: list[dict[str, Any]] = []
    for cs in cases:
        spec = ScenarioSpec(**{k: v for k, v in cs.items() if k != "name"}, name=cs["name"])
        log(f"[bench] {spec.name}: building scenario…")
        sc = build_scenario(spec, out_dir / spec.name, settings.normalized_dir, settings.dem_dir)
        truth = sc.truth
        rec = ingest_recording(
            store,
            settings,
            sc.wav_path,
            owner_permission_confirmed=True,
            recording_date_utc=spec.date_utc,
            synthetic=True,
        )
        ensure_spectrogram(store, settings, rec.id)
        events = pick_event_pair(propose_events(store, settings, rec.id))
        row: dict[str, Any] = {
            "case": spec.name,
            "seed": spec.seed,
            "snr_db": spec.snr_db,
            "n_events_detected": len(events),
            "recording_id": rec.id,
        }
        q = QualitySettings(workers=workers)
        if len(events) >= 2:
            t = time.time()
            req = SearchRequest(
                mode=SearchMode.UNKNOWN_TIME,
                recording_id=rec.id,
                event_ids=[events[0].id, events[1].id],
                date_utc=spec.date_utc,
                bbox=spec.bbox,
                flight_dataset_ids=[sc.dataset.dataset_id],
                operator_authorized=True,
                quality_settings=q,
                seed=spec.seed,
            )
            res = run_search_inline(store, settings, req)
            oc = evaluate_case(
                spec.name,
                res,
                truth["observer_lat"],
                truth["observer_lon"],
                {truth["event_a"]["icao24"], truth["event_b"]["icao24"]},
                time.time() - t,
            )
            row["mode_a"] = asdict(oc)
            row["mode_a"]["run_id"] = res.run_id
            outcomes_a.append(oc)
            log(
                f"[bench] {spec.name} Mode A: abstained={oc.abstained} top1 err={oc.top1_error_m} m, flights ok={oc.top1_flights_correct}, {oc.runtime_s:.0f}s"
            )
        else:
            oc = CaseOutcome(
                spec.name,
                True,
                "fewer_than_two_events_detected",
                None,
                None,
                None,
                None,
                None,
                0.0,
                0,
            )
            outcomes_a.append(oc)
            row["mode_a"] = asdict(oc)
            log(f"[bench] {spec.name}: only {len(events)} events detected → Mode A abstains")
        if mode_b and events:
            t = time.time()
            req_b = SearchRequest(
                mode=SearchMode.KNOWN_TIME,
                recording_id=rec.id,
                event_ids=[e.id for e in events[:2]],
                start_utc=unix_to_iso(truth["recording_start_unix"] + 3.0),
                start_uncertainty_s=10.0,
                bbox=spec.bbox,
                flight_dataset_ids=[sc.dataset.dataset_id],
                operator_authorized=True,
                quality_settings=q,
                seed=spec.seed,
            )
            res_b = run_search_inline(store, settings, req_b)
            icaos = {truth["event_a"]["icao24"]} | (
                {truth["event_b"]["icao24"]} if len(events) >= 2 else set()
            )
            ocb = evaluate_case(
                spec.name,
                res_b,
                truth["observer_lat"],
                truth["observer_lon"],
                icaos,
                time.time() - t,
            )
            row["mode_b"] = asdict(ocb)
            row["mode_b"]["run_id"] = res_b.run_id
            outcomes_b.append(ocb)
            log(
                f"[bench] {spec.name} Mode B (±10 s, {len(events[:2])} events): abstained={ocb.abstained} top1 err={ocb.top1_error_m} m, {ocb.runtime_s:.0f}s"
            )
        rows.append(row)
    report = {
        "label": "SYNTHETIC BENCHMARK — measures the inverse engine on simulated audio/flights; NOT real-world accuracy",
        "aerodoppler_version": __version__,
        "git_sha": git_sha(),
        "n_cases": len(cases),
        "mode_a": summarize(outcomes_a),
        "mode_b": summarize(outcomes_b) if mode_b else None,
        "cases": rows,
    }
    (out_dir / "synthetic_benchmark.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def run_manifest_benchmark(
    settings: Settings, manifest_path: Path, out_dir: Path, workers: int = 0, log=print
) -> dict[str, Any]:
    """Real-recording benchmark from a manifest of owner-consented recordings with surveyed
    coordinates. Truth is only read after the search result has been written."""
    man = json.loads(Path(manifest_path).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    outcomes: list[CaseOutcome] = []
    rows = []
    for case in man["cases"]:
        cid = case["id"]
        if not case.get("owner_consent", False):
            log(f"[bench] {cid}: skipped — owner_consent not asserted in manifest")
            continue
        rec = ingest_recording(
            store,
            settings,
            Path(case["audio"]),
            owner_permission_confirmed=True,
            recording_date_utc=case.get("date_utc"),
            start_time_utc=case.get("start_utc"),
            start_time_uncertainty_s=case.get("start_uncertainty_s"),
            notes=f"benchmark case {cid}",
        )
        ensure_spectrogram(store, settings, rec.id)
        if case.get("events"):
            events = [
                add_manual_event(
                    store,
                    rec.id,
                    [(p[0], p[1]) for p in ev["points"]],
                    harmonic=ev.get("harmonic", 1),
                    annotated_by="benchmark-manifest",
                )
                for ev in case["events"]
            ]
        else:
            events = pick_event_pair(propose_events(store, settings, rec.id))
        mode = SearchMode.KNOWN_TIME if case.get("start_utc") else SearchMode.UNKNOWN_TIME
        bbox = BBox.parse(case["bbox"]) if isinstance(case["bbox"], str) else BBox(**case["bbox"])
        t = time.time()
        try:
            req = SearchRequest(
                mode=mode,
                recording_id=rec.id,
                event_ids=[
                    e.id
                    for e in events[
                        : (1 if mode == SearchMode.KNOWN_TIME and len(events) == 1 else 2)
                    ]
                ],
                date_utc=case.get("date_utc"),
                start_utc=case.get("start_utc"),
                start_uncertainty_s=case.get("start_uncertainty_s"),
                bbox=bbox,
                flight_dataset_ids=case.get("datasets", []),
                operator_authorized=True,
                quality_settings=QualitySettings(workers=workers, **case.get("quality", {})),
            )
            res = run_search_inline(store, settings, req)
        except Exception as e:  # recorded as a failure, not hidden
            oc = CaseOutcome(
                cid,
                True,
                f"error: {type(e).__name__}: {e}",
                None,
                None,
                None,
                None,
                None,
                time.time() - t,
                0,
            )
            outcomes.append(oc)
            rows.append(asdict(oc))
            log(f"[bench] {cid}: {oc.abstention_reason}")
            continue
        # truth is read only now, after the result is committed to disk
        oc = evaluate_case(
            cid,
            res,
            float(case["truth_lat"]),
            float(case["truth_lon"]),
            set(case.get("truth_icao24", [])) or None,
            time.time() - t,
        )
        outcomes.append(oc)
        r = asdict(oc)
        r["run_id"] = res.run_id
        rows.append(r)
        log(f"[bench] {cid}: abstained={oc.abstained} top1 err={oc.top1_error_m} m")
    report = {
        "label": "REAL-RECORDING BENCHMARK (owner-consented, surveyed coordinates)",
        "manifest": str(manifest_path),
        "aerodoppler_version": __version__,
        "git_sha": git_sha(),
        "summary": summarize(outcomes),
        "cases": rows,
    }
    (out_dir / "real_benchmark.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def run_real_trajectory_benchmark(
    settings: Settings,
    dataset_id: str,
    observers: list[tuple[str, float, float]],
    out_dir: Path,
    workers: int = 0,
    seeds: tuple[int, ...] = (1,),
    snr_db: float = 10.0,
    log=print,
    max_assignments: int = 20_000_000,
    events_k: int = 2,
    oracle_events: bool = False,
) -> dict[str, Any]:
    """Semi-synthetic benchmark: simulated audio of REAL ADS-B trajectories at known observer
    positions, searched against all real traffic of the day with the real DEM."""
    from aerodoppler.adsb.index import FlightDataset
    from aerodoppler.geo.dem import open_dem_for_bbox
    from aerodoppler.validation.real_trajectory import (
        RealTrajectoryCase,
        build_real_trajectory_scenario,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    ds = FlightDataset(dataset_id, settings.normalized_dir / dataset_id)
    outcomes: list[CaseOutcome] = []
    oracle_outcomes: list[CaseOutcome] = []
    rows: list[dict[str, Any]] = []
    # checkpoint after every case so an hours-long run can be resumed with the same command:
    # finished cases are read back from <out_dir>/checkpoint.json and skipped
    ckpt = out_dir / "checkpoint.json"
    if ckpt.exists():
        rows = json.loads(ckpt.read_text())
        for row in rows:
            if "mode_a" in row:
                outcomes.append(CaseOutcome(**row["mode_a"]))
            if "mode_a_oracle_events" in row:
                oracle_outcomes.append(CaseOutcome(**row["mode_a_oracle_events"]))
        log(f"[realtraj] resuming: {len(rows)} case(s) already in {ckpt}")
    done = {row["case"] for row in rows}
    for name, lat, lon in observers:
        for seed in seeds:
            cname = f"{name}_s{seed}"
            if cname in done:
                continue
            local_bbox = BBox(west=lon - 0.18, south=lat - 0.15, east=lon + 0.18, north=lat + 0.15)
            try:
                dem = open_dem_for_bbox(settings.dem_dir, local_bbox, offline=settings.offline)
                truth = build_real_trajectory_scenario(
                    RealTrajectoryCase(cname, lat, lon, seed=seed, snr_db=snr_db),
                    ds,
                    dem,
                    out_dir / cname,
                )
            except Exception as e:
                log(f"[realtraj] {cname}: scenario not possible: {e}")
                rows.append({"case": cname, "error": str(e)})
                ckpt.write_text(json.dumps(rows, indent=2, default=str))
                continue
            rec = ingest_recording(
                store,
                settings,
                Path(truth["wav_path"]),
                owner_permission_confirmed=True,
                recording_date_utc=truth["date_utc"],
                synthetic=True,
                notes="SEMI-SYNTHETIC: real trajectories, simulated audio",
            )
            ensure_spectrogram(store, settings, rec.id)
            all_events = propose_events(store, settings, rec.id)
            events = (
                pick_events(all_events, k=events_k) if events_k > 2 else pick_event_pair(all_events)
            )
            row: dict[str, Any] = {
                "case": cname,
                "observer": [lat, lon],
                "n_events_detected": len(all_events),
                "chosen_pair": [
                    (p["icao24"], p["type"], round(p["slant"]), round(p["speed"], 1))
                    for p in truth["chosen_pair"]
                ],
                "audible_flights": len(truth["audible_flights"]),
                "event_separation_s": truth["event_separation_s"],
                "recording_id": rec.id,
            }
            if len(events) < 2:
                oc = CaseOutcome(
                    cname,
                    True,
                    "fewer_than_two_events_detected",
                    None,
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    0,
                )
                log(f"[realtraj] {cname}: only {len(all_events)} proposals → abstain")
            else:
                t = time.time()
                req = SearchRequest(
                    mode=SearchMode.UNKNOWN_TIME,
                    recording_id=rec.id,
                    event_ids=[e.id for e in events],
                    date_utc=truth["date_utc"],
                    bbox=local_bbox,
                    flight_dataset_ids=[dataset_id],
                    # semi-synthetic by design: simulated audio over REAL ADS-B tracks
                    allow_synthetic_mix=True,
                    operator_authorized=True,
                    quality_settings=QualitySettings(
                        workers=workers, max_assignments=max_assignments
                    ),
                    seed=seed,
                )
                try:
                    res = run_search_inline(store, settings, req)
                    truth_icaos = {p["icao24"] for p in truth["chosen_pair"]}
                    oc = evaluate_case(cname, res, lat, lon, None, time.time() - t)
                    oc.notes = {
                        "run_id": res.run_id,
                        "top1_flights": [s["icao24"] for s in res.candidates[0].extra["segments"]]
                        if res.candidates
                        else None,
                        "chosen_pair_icaos": sorted(truth_icaos),
                        "top1_uses_a_chosen_flight": bool(
                            res.candidates
                            and (
                                {s["icao24"] for s in res.candidates[0].extra["segments"]}
                                & truth_icaos
                            )
                        ),
                    }
                except Exception as e:
                    oc = CaseOutcome(
                        cname,
                        True,
                        f"error: {type(e).__name__}: {e}",
                        None,
                        None,
                        None,
                        None,
                        None,
                        time.time() - t,
                        0,
                    )
                log(
                    f"[realtraj] {cname}: abstained={oc.abstained} ({oc.abstention_reason}) top1 err={oc.top1_error_m} m, {oc.runtime_s:.0f}s, notes={oc.notes}"
                )
            outcomes.append(oc)
            row["mode_a"] = asdict(oc)
            if oracle_events and truth.get("oracle_ridges"):
                # operator-style manual events from the true ridges (fundamental); separates
                # detector failures from search failures
                man = []
                for rg in truth["oracle_ridges"].values():
                    pts = list(zip(rg["t_rel_s"], rg["f_hz"], strict=True))
                    if len(pts) >= 3:
                        man.append(
                            add_manual_event(
                                store,
                                rec.id,
                                pts,
                                harmonic=1,
                                uncertainty_hz=0.5,
                                annotated_by="oracle-truth-ridge",
                            )
                        )
                man.sort(key=lambda e: e.audio_start_s)
                if len(man) == 2:
                    t = time.time()
                    req = SearchRequest(
                        mode=SearchMode.UNKNOWN_TIME,
                        recording_id=rec.id,
                        event_ids=[e.id for e in man],
                        date_utc=truth["date_utc"],
                        bbox=local_bbox,
                        flight_dataset_ids=[dataset_id],
                        # semi-synthetic by design: simulated audio over REAL ADS-B tracks
                        allow_synthetic_mix=True,
                        operator_authorized=True,
                        quality_settings=QualitySettings(
                            workers=workers, max_assignments=max_assignments
                        ),
                        seed=seed,
                    )
                    try:
                        res = run_search_inline(store, settings, req)
                        oco = evaluate_case(
                            cname + "_oracle",
                            res,
                            lat,
                            lon,
                            {p["icao24"] for p in truth["chosen_pair"]},
                            time.time() - t,
                        )
                        oco.notes = {
                            "run_id": res.run_id,
                            "top1_flights": [
                                s["icao24"] for s in res.candidates[0].extra["segments"]
                            ]
                            if res.candidates
                            else None,
                            "events": "oracle (true ridges traced as manual events)",
                        }
                    except Exception as e:
                        oco = CaseOutcome(
                            cname + "_oracle",
                            True,
                            f"error: {type(e).__name__}: {e}",
                            None,
                            None,
                            None,
                            None,
                            None,
                            time.time() - t,
                            0,
                        )
                    log(
                        f"[realtraj] {cname} ORACLE events: abstained={oco.abstained} ({oco.abstention_reason}) top1 err={oco.top1_error_m} m, flights ok={oco.top1_flights_correct}, {oco.runtime_s:.0f}s"
                    )
                    oracle_outcomes.append(oco)
                    row["mode_a_oracle_events"] = asdict(oco)
            rows.append(row)
            ckpt.write_text(json.dumps(rows, indent=2, default=str))
    report = {
        "label": "SEMI-SYNTHETIC BENCHMARK — real ADS-B trajectories + real DEM, SIMULATED audio; not real-world accuracy",
        "dataset_id": dataset_id,
        "aerodoppler_version": __version__,
        "git_sha": git_sha(),
        "summary": summarize(outcomes),
        "summary_oracle_events": summarize(oracle_outcomes) if oracle_outcomes else None,
        "cases": rows,
    }
    (out_dir / "real_trajectory_benchmark.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return report
