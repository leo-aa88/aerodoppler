"""Ranking, flags and abstention (SPEC §10.5): relative fit scores only, explicit abstention
reasons, ambiguity reporting when distinct locations score nearly as well."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerodoppler.models import Abstention


@dataclass
class RankConfig:
    min_score_z: float = 1.2  # mean whitened z along curves below this = no convincing ridge
    min_coverage: float = 0.25
    ambiguity_margin_z: float = 0.3
    min_candidates_for_ranking: int = 1


def decide_abstention(
    n_segments: int,
    n_assignments: int,
    best_score: float | None,
    best_coverage: float | None,
    cfg: RankConfig,
    event_flags: list[str],
) -> Abstention:
    if n_segments == 0:
        return Abstention(
            abstained=True,
            reason="no_flight_coverage",
            detail="no airborne flight segments with altitude overlap the date/time window and bbox (archive coverage gap, wrong date, or bbox)",
        )
    if n_assignments == 0:
        return Abstention(
            abstained=True,
            reason="no_candidate_geometry",
            detail="no grid point has flights whose delayed-CPA timing matches the event separation / known time within tolerance and the speed & slant bounds",
        )
    if best_score is None or not np.isfinite(best_score):
        return Abstention(
            abstained=True, reason="scoring_failed", detail="curve scoring produced no finite value"
        )
    if best_score < cfg.min_score_z:
        return Abstention(
            abstained=True,
            reason="weak_fit",
            detail=f"best relative fit score {best_score:.2f} < {cfg.min_score_z} (no convincing ridge energy along any predicted curve)",
        )
    if best_coverage is not None and best_coverage < cfg.min_coverage:
        return Abstention(
            abstained=True,
            reason="poor_coverage",
            detail=f"only {best_coverage:.0%} of the predicted curve shows ridge energy",
        )
    return Abstention(abstained=False)


def ambiguity_flags(scores_distinct: list[float], cfg: RankConfig) -> list[str]:
    if (
        len(scores_distinct) >= 2
        and scores_distinct[0] - scores_distinct[1] < cfg.ambiguity_margin_z
    ):
        return [
            f"ambiguous: runner-up distinct location within {cfg.ambiguity_margin_z} z of the best"
        ]
    return []


def gate_results(
    *,
    n_events: int,
    mode_known_time: bool,
    per_event_scores: list[float],
    per_event_coverage: list[float],
    null_mean: list[float],
    null_std: list[float],
    own_clock_offsets_s: list[float | None],
    margin_distinct_flights_z: float | None,
    unheard_close_pass_penalty: float | None,
    landscape_radius_m: float | None,
    min_coverage: float = 0.4,
    clock_tol_s: float = 4.0,
    margin_high_z: float = 0.5,
    margin_medium_z: float = 0.25,
    radius_max_m: float = 1000.0,
) -> tuple[str, list[dict]]:
    """Graded confidence after fdop's five gates, in this code's units (whitened z):

    core matches — every event's score beats the chance-match null (same curve at random clock
    offsets in the same recording) by >= 2 sigma and has coverage >= min_coverage, with at least
    two events in unknown-time mode (one in known-time mode); clock lock — every event's own best
    clock offset is within clock_tol_s of the shared clock; margin — the winner beats the best
    rival that uses different flights by margin_high_z (or no rival); no unheard close pass —
    no penalised unheard aircraft; radius — landscape radius <= radius_max_m.
    High = all gates; medium = core + clock + margin >= margin_medium_z; else low.
    """
    gates: list[dict] = []
    need = 1 if mode_known_time else 2
    ok_events = []
    details = []
    for i, (sc, cov) in enumerate(zip(per_event_scores, per_event_coverage, strict=True)):
        mu, sd = null_mean[i], max(null_std[i], 1e-6)
        sig = (sc - mu) / sd
        ok = sig >= 2.0 and cov >= min_coverage
        ok_events.append(ok)
        details.append(
            f"event {i + 1}: score {sc:.2f} vs chance {mu:.2f}±{sd:.2f} ({sig:.1f}σ), coverage {cov:.0%}"
        )
    core = sum(ok_events) >= need and len(ok_events) >= need
    gates.append(
        {
            "name": "core_matches",
            "passed": core,
            "detail": f"{sum(ok_events)}/{len(ok_events)} events beat chance by >= 2σ with coverage >= {min_coverage:.0%} (need {need}); "
            + "; ".join(details),
        }
    )
    offs = [o for o in own_clock_offsets_s if o is not None]
    clock = all(abs(o) <= clock_tol_s for o in offs) and bool(offs)
    gates.append(
        {
            "name": "clock_lock",
            "passed": clock,
            "detail": "each event's own best clock offset from the shared clock: "
            + ", ".join(f"{o:+.1f} s" for o in offs)
            + f" (tolerance ±{clock_tol_s:.0f} s)",
        }
    )
    if margin_distinct_flights_z is None:
        margin_ok, margin_med = True, True
        mdetail = "no rival hypothesis with different flights survived the funnel"
    else:
        margin_ok = margin_distinct_flights_z >= margin_high_z
        margin_med = margin_distinct_flights_z >= margin_medium_z
        mdetail = f"winner beats the best different-flight rival by {margin_distinct_flights_z:.2f} z (high needs {margin_high_z}, medium {margin_medium_z})"
    gates.append({"name": "margin", "passed": margin_ok, "detail": mdetail})
    if unheard_close_pass_penalty is None:
        gates.append(
            {
                "name": "no_unheard_close_pass",
                "passed": True,
                "detail": "not evaluated in this mode (silence term is Mode A only)",
            }
        )
        unheard_ok = True
    else:
        unheard_ok = unheard_close_pass_penalty <= 0.0
        gates.append(
            {
                "name": "no_unheard_close_pass",
                "passed": unheard_ok,
                "detail": f"silence penalty {unheard_close_pass_penalty:.2f} z (0 = no penalised unheard aircraft)",
            }
        )
    radius_ok = landscape_radius_m is not None and landscape_radius_m <= radius_max_m
    gates.append(
        {
            "name": "radius",
            "passed": radius_ok,
            "detail": f"score-landscape radius {landscape_radius_m if landscape_radius_m is None else round(landscape_radius_m)} m (<= {radius_max_m:.0f} m required; uncalibrated)",
        }
    )
    if core and clock and margin_ok and unheard_ok and radius_ok:
        grade = "high"
    elif core and clock and margin_med:
        grade = "medium"
    else:
        grade = "low"
    return grade, gates
