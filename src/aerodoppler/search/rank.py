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
    margin_distinct_location_z: float | None = None,
    null_n: list[int] | None = None,
    null_offset_range_s: tuple[float, float] | None = None,
    event_flags: list[list[str]] | None = None,
    null_diagnostics: dict | list[dict] | None = None,
    event_evidence: list[dict] | None = None,
    track_support: tuple[bool, str] | None = None,
    completeness: tuple[bool, str] | None = None,
    sensitivity: tuple[bool, str] | None = None,
    min_coverage: float = 0.4,
    clock_tol_s: float = 4.0,
    margin_high_z: float = 0.5,
    margin_medium_z: float = 0.25,
    radius_max_m: float = 1000.0,
    min_null_samples: int = 8,
) -> tuple[str, list[dict]]:
    """Graded confidence after fdop's five gates, in this code's units (whitened z):

    core matches — every event's score beats the chance-match null (same curve at random clock
    offsets in the same recording) by >= 2 sigma and has coverage >= min_coverage, with at least
    two events in unknown-time mode (one in known-time mode); clock lock — every event's own best
    clock offset is within clock_tol_s of the shared clock; margin — the winner beats the best
    rival that uses different flights by margin_high_z (or no rival); no unheard close pass —
    no penalised unheard aircraft; radius — landscape radius <= radius_max_m.
    event shape — every event reaches its plateaus with a clear descent (high only).
    High = all gates; medium = core + clock + margin >= margin_medium_z; else low.
    """
    gates: list[dict] = []
    need = 1 if mode_known_time else 2
    ok_events = []
    details = []
    for i, (sc, cov) in enumerate(zip(per_event_scores, per_event_coverage, strict=True)):
        n_null = null_n[i] if null_n is not None else None
        if n_null is not None and n_null < min_null_samples:
            # no usable null: fail visibly instead of comparing with a made-up null, and say
            # why (no admissible misalignment at all, or too few valid scores among them)
            ok_events.append(False)
            nd_all = null_diagnostics
            nd = (nd_all[i] if isinstance(nd_all, list) and i < len(nd_all) else nd_all) or {}
            if nd.get("band") is None and null_diagnostics is not None:
                why = (
                    "no clock shift satisfies "
                    + str(nd.get("rule", "the null's rules"))
                    + " — the event window is too long relative to the predicted curve"
                )
            elif null_diagnostics is not None:
                lo, hi = nd["band"]
                why = (
                    f"only {n_null} of {nd.get('drawn', '?')} shifts in the admissible band "
                    f"{lo:.0f}–{hi:.0f} s gave a valid score"
                )
            else:
                why = "too few valid shifted alignments"
            details.append(
                f"event {i + 1}: score {sc:.2f}, chance-match null NOT estimable ({n_null} "
                f"valid samples, need {min_null_samples}: {why}), coverage {cov:.0%}"
            )
            continue
        mu, sd = null_mean[i], max(null_std[i], 1e-6)
        sig = (sc - mu) / sd
        ok = sig >= 2.0 and cov >= min_coverage
        ok_events.append(ok)
        n_txt = f", n={n_null}" if n_null is not None else ""
        band_i = None
        if isinstance(null_diagnostics, list) and i < len(null_diagnostics):
            band_i = (null_diagnostics[i] or {}).get("band")
        if band_i:
            n_txt += f", |offsets| {band_i[0]:.0f}–{band_i[1]:.0f} s"
        elif null_offset_range_s is not None:
            # short clips only allow a narrow band of offsets, so the null spread (and this σ)
            # says little beyond "above chance"; show the band so the figure is not over-read
            lo, hi = null_offset_range_s
            n_txt += f", |offsets| {lo:.0f}–{hi:.0f} s"
        details.append(
            f"event {i + 1}: score {sc:.2f} vs chance {mu:.2f}±{sd:.2f} ({sig:.1f}σ{n_txt}), coverage {cov:.0%}"
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
    # one event always "agrees" with the clock fitted to it: that is not corroboration
    single = len(offs) < 2
    clock = (not single) and all(abs(o) <= clock_tol_s for o in offs)
    gates.append(
        {
            "name": "clock_lock",
            "passed": clock,
            "detail": (
                "not applicable with a single event: one event cannot corroborate the clock it "
                "was fitted to (high needs two or more events)"
            )
            if single
            else "each event's own best clock offset from the shared clock: "
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
    # a mirror solution of a single straight pass uses the SAME flight on the other side of
    # the track, so it escapes the different-flight margin; this gate compares the winner
    # with the best hypothesis at a distinct location (>= 1 km away), whatever its flights
    if margin_distinct_location_z is None:
        loc_ok, loc_med = True, True
        ldetail = "no rival at a distinct location (>= 1 km) survived"
    else:
        loc_ok = margin_distinct_location_z >= margin_high_z
        loc_med = margin_distinct_location_z >= margin_medium_z
        ldetail = f"winner beats the best rival >= 1 km away by {margin_distinct_location_z:.2f} z (high needs {margin_high_z}, medium {margin_medium_z})"
    gates.append({"name": "margin_distinct_location", "passed": loc_ok, "detail": ldetail})
    margin_ok = margin_ok and loc_ok
    margin_med = margin_med and loc_med
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
    # self-distrust gates (after DOPPLERGEIST's geometry validation, retention accounting and
    # sensitivity profiling): the matched tracks must physically support the fit, the funnel
    # must not have discarded a comparable competitor, and the answer must survive declared
    # perturbations of the nuisance physics
    extras_ok = True
    for name, item in (
        ("track_support", track_support),
        ("search_complete", completeness),
        ("sensitivity_stable", sensitivity),
    ):
        if item is None:
            gates.append({"name": name, "passed": True, "detail": "not evaluated"})
        else:
            gates.append({"name": name, "passed": bool(item[0]), "detail": item[1]})
            extras_ok = extras_ok and bool(item[0])
    # event shape: a ridge that never reaches its plateaus (or has no clear descent) leaves the
    # pass speed and slant range underdetermined, so a different aircraft at a different slant
    # can fit the same fragment; such evidence cannot support a high grade
    bad_shape = {"plateaus_not_reached", "no_descending_slope", "non_monotonic_ridge"}
    if event_flags is None:
        gates.append({"name": "event_shape", "passed": True, "detail": "not evaluated"})
    else:
        problems = [
            f"event {i + 1}: {', '.join(sorted(set(fl) & bad_shape))}"
            for i, fl in enumerate(event_flags)
            if set(fl) & bad_shape
        ]
        shape_ok = not problems
        gates.append(
            {
                "name": "event_shape",
                "passed": shape_ok,
                "detail": "every event reaches both Doppler plateaus with a clear descent"
                if shape_ok
                else "; ".join(problems) + " (speed/slant underdetermined; not high)",
            }
        )
        extras_ok = extras_ok and shape_ok
    # evidence: an event without a measurable Doppler transition carries no location
    # information, even when an operator accepted it; weak transitions cap the grade
    ev_cap = "high"
    if event_evidence is None:
        gates.append({"name": "event_evidence", "passed": True, "detail": "not evaluated"})
    else:
        tonal = [i + 1 for i, a in enumerate(event_evidence) if a.get("label") == "tonal_only"]
        weak = [i + 1 for i, a in enumerate(event_evidence) if a.get("label") == "weak_doppler"]
        if tonal:
            ev_cap = "low"
            reviewed = [
                i + 1
                for i, a in enumerate(event_evidence)
                if a.get("label") == "tonal_only" and (a.get("review") or {}).get("accepted")
            ]
            detail = (
                f"event(s) {tonal} show no measurable Doppler transition (tonal ridge only)"
                + (f"; event(s) {reviewed} were accepted by operator review" if reviewed else "")
                + ": the grade cannot exceed low"
            )
        elif weak:
            ev_cap = "medium"
            detail = f"event(s) {weak} show only a weak Doppler transition: the grade cannot exceed medium"
        else:
            detail = "every event shows a clear Doppler transition"
        gates.append({"name": "event_evidence", "passed": not tonal and not weak, "detail": detail})
    track_ok = track_support is None or bool(track_support[0])
    clock_med = clock or single  # a single known-time event may reach medium, never high
    if core and clock and margin_ok and unheard_ok and radius_ok and extras_ok:
        grade = "high"
    elif core and clock_med and margin_med and track_ok:
        grade = "medium"
    else:
        grade = "low"
    order = {"low": 0, "medium": 1, "high": 2}
    if order[grade] > order[ev_cap]:
        grade = ev_cap
    return grade, gates
