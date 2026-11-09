"""Held-out evaluation of the blind search (semi-synthetic: real ADS-B tracks, simulated audio).

Three strictly separated phases:

1. **Scenarios** — for each case a random observer (seeded) is drawn inside the flight data's
   extent until two real propeller aircraft pass close to it; audio is simulated from every
   audible real aircraft. The observer position is written ONLY to ``truth/<case>.json``.
2. **Inference** — sees the audio file, the UTC day and the flight datasets, nothing else (no
   area, no observer, no truth path). Every prediction is written to ``predictions/<case>.json``
   and its SHA-256 appended to ``predictions/commit.json`` before any truth is opened.
3. **Scoring** — verifies every prediction against its committed hash, then reads the truth.

The flight data extent bounds where an answer can be (it is the search scope, reported in every
result); within it, nothing about the location is given to the search.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from aerodoppler.config import Settings
from aerodoppler.models import BBox


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_scenarios(
    settings: Settings,
    dataset_id: str,
    out_dir: Path,
    n_cases: int,
    seed: int,
    log: Callable[[str], None],
) -> list[dict[str, Any]]:
    from aerodoppler.adsb.index import FlightDataset
    from aerodoppler.geo.dem import open_dem_for_bbox
    from aerodoppler.search.blind import dataset_scope
    from aerodoppler.validation.real_trajectory import (
        RealTrajectoryCase,
        build_real_trajectory_scenario,
    )

    ds = FlightDataset(dataset_id, settings.normalized_dir / dataset_id)
    scope = dataset_scope([ds])
    assert scope is not None
    rng = np.random.default_rng(seed)
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    tries = 0
    # keep observers 0.3 deg inside the scope so their audible aircraft are inside the data
    while len(cases) < n_cases and tries < 400 * n_cases:
        tries += 1
        lat = float(rng.uniform(scope.south + 0.3, scope.north - 0.3))
        lon = float(rng.uniform(scope.west + 0.3, scope.east - 0.3))
        name = f"case{len(cases) + 1:02d}"
        local = BBox(west=lon - 0.2, south=lat - 0.17, east=lon + 0.2, north=lat + 0.17)
        try:
            dem = open_dem_for_bbox(settings.dem_dir, local, offline=settings.offline)
            truth = build_real_trajectory_scenario(
                RealTrajectoryCase(name, lat, lon, seed=seed * 100 + tries),
                ds,
                dem,
                out_dir / "work" / name,
                local_bbox=local,
            )
        except Exception as e:  # no two close passes here (or no terrain): draw again
            if tries % 25 == 0:
                log(f"[blind] {tries} draws, {len(cases)} cases so far (last: {str(e)[:80]})")
            continue
        wav = out_dir / "audio" / f"{name}.wav"
        Path(truth["wav_path"]).replace(wav)
        (out_dir / "truth" / f"{name}.json").write_text(
            json.dumps({k: v for k, v in truth.items() if k != "wav_path"}, indent=2, default=str)
        )
        cases.append({"case": name, "audio": str(wav), "date_utc": truth["date_utc"]})
        log(f"[blind] scenario {name} ready after {tries} draws")
    (out_dir / "cases.json").write_text(json.dumps(cases, indent=2))
    return cases


def _stage_logger(case: str, log: Callable[[str], None]) -> Callable[[str, dict[str, Any]], None]:
    last = {"key": None}

    def f(stage: str, info: dict[str, Any]) -> None:
        key = (stage, info.get("run"), info.get("stage"))
        if key != last["key"]:
            last["key"] = key
            log(f"[blind] {case} {stage}: {json.dumps(info, default=str)[:200]}")

    return f


def infer(
    settings: Settings,
    cases: list[dict[str, Any]],
    dataset_ids: list[str],
    out_dir: Path,
    budget_s: float,
    workers: int,
    log: Callable[[str], None],
) -> None:
    """Blind inference. Receives only audio paths, days and datasets."""
    from aerodoppler.audio.service import (
        ensure_spectrogram,
        ingest_recording,
        pick_events,
        propose_events,
    )
    from aerodoppler.jobs.store import Store
    from aerodoppler.search.blind import BlindConfig, run_blind_search
    from aerodoppler.search.events import assess_event, usable_as_evidence

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    commit_path = pred_dir / "commit.json"
    commit = json.loads(commit_path.read_text()) if commit_path.exists() else {}
    store = Store(settings.db_path)
    for c in cases:
        if c["case"] in commit:
            continue
        t0 = time.time()
        rec = ingest_recording(
            store, settings, Path(c["audio"]), recording_date_utc=c["date_utc"], synthetic=True
        )
        ensure_spectrogram(store, settings, rec.id)
        evs = [
            e
            for e in propose_events(store, settings, rec.id)
            if usable_as_evidence(assess_event(e))
        ]
        pair = pick_events(evs, k=4)
        if len(pair) < 2:
            pred: dict[str, Any] = {
                "case": c["case"],
                "verdict": {"answer": "abstain", "reason": f"{len(evs)} usable event(s)"},
            }
        else:
            res = run_blind_search(
                settings,
                store,
                recording_id=rec.id,
                event_ids=[e.id for e in pair],
                mode="unknown-time",
                date_utc=c["date_utc"],
                start_utc=None,
                start_uncertainty_s=None,
                flight_dataset_ids=dataset_ids,
                out_dir=out_dir / "runs" / c["case"],
                cfg=BlindConfig(budget_s=budget_s),
                workers=workers,
                allow_synthetic_mix=True,
                progress=_stage_logger(c["case"], log),
            )
            pred = {
                "case": c["case"],
                **{
                    k: res[k] for k in ("verdict", "coverage", "support", "scope_bbox", "elapsed_s")
                },
            }
        pred["inference_s"] = round(time.time() - t0, 1)
        p = pred_dir / f"{c['case']}.json"
        p.write_text(json.dumps(pred, indent=2, default=str))
        commit[c["case"]] = _sha(p)
        commit_path.write_text(json.dumps(commit, indent=2))  # committed before any truth is read
        log(f"[blind] {c['case']}: {pred['verdict']} ({pred['inference_s']} s) committed")


def score(out_dir: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    from aerodoppler.geo.coordinates import haversine_m

    commit = json.loads((out_dir / "predictions" / "commit.json").read_text())
    rows = []
    for c in cases:
        p = out_dir / "predictions" / f"{c['case']}.json"
        if _sha(p) != commit[c["case"]]:
            raise RuntimeError(f"prediction {c['case']} changed after it was committed")
        pred = json.loads(p.read_text())
        truth = json.loads((out_dir / "truth" / f"{c['case']}.json").read_text())
        v = pred["verdict"]
        row = {
            "case": c["case"],
            "answer": v["answer"],
            "grade": v.get("grade"),
            "coverage": pred.get("coverage"),
            "supported_area_km2": (pred.get("support") or {}).get("area_km2"),
            "inference_s": pred.get("inference_s"),
        }
        if v["answer"] == "location":
            row["error_m"] = float(
                haversine_m(v["lat"], v["lon"], truth["observer_lat"], truth["observer_lon"])
            )
            rb = v.get("region_bbox")
            row["truth_in_winning_region"] = bool(
                rb
                and rb[0] <= truth["observer_lon"] <= rb[2]
                and rb[1] <= truth["observer_lat"] <= rb[3]
            )
        rows.append(row)
    answered = [r for r in rows if r["answer"] == "location"]
    errs = sorted(r["error_m"] for r in answered)
    summary = {
        "label": "BLIND SEMI-SYNTHETIC BENCHMARK — real ADS-B tracks, simulated audio, held-out observers",
        "cases": len(rows),
        "answered": len(answered),
        "abstained": len(rows) - len(answered),
        "median_error_m": float(np.median(errs)) if errs else None,
        "within_300m": sum(e <= 300 for e in errs),
        "within_1km": sum(e <= 1000 for e in errs),
        "within_10km": sum(e <= 10000 for e in errs),
        "grades": {
            g: sum(1 for r in answered if r["grade"] == g) for g in ("high", "medium", "low")
        },
        "high_or_medium_beyond_1km": sum(
            1 for r in answered if r["grade"] in ("high", "medium") and r["error_m"] > 1000
        ),
        "commit_verified": True,
    }
    out = {"summary": summary, "cases": rows}
    (out_dir / "blind_benchmark.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run_blind_benchmark(
    settings: Settings,
    dataset_id: str,
    out_dir: Path,
    n_cases: int = 6,
    seed: int = 1,
    budget_s: float = 1800.0,
    workers: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    cases_path = out_dir / "cases.json"
    cases = (
        json.loads(cases_path.read_text())
        if cases_path.exists()
        else make_scenarios(settings, dataset_id, out_dir, n_cases, seed, log)
    )
    infer(settings, cases, [dataset_id], out_dir, budget_s, workers, log)
    return score(out_dir, cases)
