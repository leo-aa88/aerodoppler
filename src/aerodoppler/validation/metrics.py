"""Benchmark metrics (SPEC §13): top-1/top-K distance error, success/abstention rates, false
matches, runtime. Computed only from committed results and hidden truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aerodoppler.geo.coordinates import haversine_m


@dataclass
class CaseOutcome:
    case_id: str
    abstained: bool
    abstention_reason: str | None
    top1_error_m: float | None
    topk_min_error_m: float | None
    top1_flights_correct: bool | None
    top1_score: float | None
    runner_up_gap_z: float | None
    runtime_s: float
    n_candidates: int
    notes: dict[str, Any] = field(default_factory=dict)


def evaluate_case(
    case_id: str,
    result: Any,
    truth_lat: float,
    truth_lon: float,
    truth_icaos: set[str] | None,
    runtime_s: float,
    k: int = 3,
) -> CaseOutcome:
    cands = result.candidates
    if not cands:
        return CaseOutcome(
            case_id,
            result.abstention.abstained,
            result.abstention.reason,
            None,
            None,
            None,
            None,
            None,
            runtime_s,
            0,
        )
    errs = [float(haversine_m(c.lat, c.lon, truth_lat, truth_lon)) for c in cands]
    top = cands[0]
    flights_ok = None
    if truth_icaos is not None:
        flights_ok = {s["icao24"] for s in top.extra.get("segments", [])} == set(truth_icaos)
    gap = (cands[0].score_total - cands[1].score_total) if len(cands) > 1 else None
    return CaseOutcome(
        case_id,
        result.abstention.abstained,
        result.abstention.reason,
        errs[0],
        float(min(errs[:k])),
        flights_ok,
        top.score_total,
        gap,
        runtime_s,
        len(cands),
    )


def summarize(outcomes: list[CaseOutcome], k: int = 3, success_m: float = 300.0) -> dict[str, Any]:
    n = len(outcomes)
    answered = [o for o in outcomes if not o.abstained and o.top1_error_m is not None]
    e1 = np.array([o.top1_error_m for o in answered], dtype=float)
    ek = np.array([o.topk_min_error_m for o in answered], dtype=float)
    out: dict[str, Any] = {
        "n_cases": n,
        "n_answered": len(answered),
        "n_abstained": n - len(answered),
        "abstention_rate": (n - len(answered)) / n if n else None,
        "abstention_reasons": {},
        "top1_error_m": {
            "median": float(np.median(e1)) if e1.size else None,
            "p90": float(np.percentile(e1, 90)) if e1.size else None,
            "max": float(e1.max()) if e1.size else None,
            "values": [round(float(v), 1) for v in e1],
        },
        f"top{k}_min_error_m": {
            "median": float(np.median(ek)) if ek.size else None,
            "p90": float(np.percentile(ek, 90)) if ek.size else None,
        },
        f"success_rate_top1_within_{int(success_m)}m": float(np.mean(e1 <= success_m))
        if e1.size
        else None,
        f"success_rate_top{k}_within_{int(success_m)}m": float(np.mean(ek <= success_m))
        if ek.size
        else None,
        "false_match_rate_top1_over_1km": float(np.mean(e1 > 1000.0)) if e1.size else None,
        "flights_correct_rate": float(
            np.mean(
                [o.top1_flights_correct for o in answered if o.top1_flights_correct is not None]
            )
        )
        if any(o.top1_flights_correct is not None for o in answered)
        else None,
        "runtime_s": {
            "median": float(np.median([o.runtime_s for o in outcomes])) if n else None,
            "max": float(max(o.runtime_s for o in outcomes)) if n else None,
        },
        "calibration_note": "scores are relative, uncalibrated fit values; no confidence intervals are claimed",
    }
    for o in outcomes:
        if o.abstained:
            out["abstention_reasons"][o.abstention_reason or "unknown"] = (
                out["abstention_reasons"].get(o.abstention_reason or "unknown", 0) + 1
            )
    return out
