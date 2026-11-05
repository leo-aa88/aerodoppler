"""Measured event features and approximate kinematic bounds (SPEC §10.1-10.2).

From a Doppler ridge we extract the time of steepest descent (≈ delayed CPA), the received
frequency there (≈ h·f0, since the radial velocity is ~0 at CPA), the approach/recede
plateaus and the maximum slope. For a straight constant-speed flyby:

    f_app / f_rec = (c + v_r) / (c - v_r)      → v_r = c (R - 1) / (R + 1)
    |df/dt|_CPA  = f0 v² / (c d)               → d = f_CPA v² / (c |df/dt|)

``v_r`` is the *radial* speed reached on the plateaus, hence a lower bound on the aircraft
speed; the slant distance follows from the speed bounds. All bounds are deliberately generous
(SPEC: "do not over-prune").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerodoppler.models import DopplerEvent


@dataclass
class EventFeatures:
    event_id: str
    t_center_s: float
    f_center_hz: float
    ridge_harmonic: int
    f_max_hz: float
    f_min_hz: float
    plateau_ratio: float
    v_radial_mps: float
    slope_max_hz_per_s: float
    duration_s: float
    t_start_s: float
    t_end_s: float
    f0_lo_hz: float
    f0_hi_hz: float
    excursion_local: (
        float  # f(centre - W) / f(centre + W) on the smoothed ridge, W = EXCURSION_WINDOW_S
    )
    v_min_mps: float
    v_max_mps: float
    d_min_m: float
    d_max_m: float
    n_points: int
    flags: list[str]

    def f0_center(self) -> float:
        return self.f_center_hz / self.ridge_harmonic


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k | 1)
    if len(y) < k:
        return y.copy()
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


EXCURSION_WINDOW_S = 60.0


def robust_descent_centre(
    t: np.ndarray, f: np.ndarray, half_window_s: float = 25.0, smooth_s: float = 6.0
) -> tuple[int, float]:
    """Index and strength of the centre of the largest frequency *drop* over ±half_window_s.

    Using the drop over a physically sized window (a flyby transition lasts ~2d/v, tens of
    seconds) instead of the instantaneous slope makes the estimate immune to small tracking
    artefacts (a few-Hz jump between adjacent ridges) that otherwise dominate the slope.
    Returns (index, drop_hz); drop_hz <= 0 means no descending transition was found.
    """
    n = len(t)
    if n < 3:
        return int(np.argmax(-f)) if n else 0, 0.0
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    fs = _smooth(f, int(round(smooth_s / max(dt, 1e-3))))
    before = np.interp(t - half_window_s, t, fs)  # clamps at the ends
    after = np.interp(t + half_window_s, t, fs)
    drop = before - after
    i = int(np.argmax(drop))
    return i, float(drop[i])


def measure_event(
    ev: DopplerEvent,
    c_mps: float,
    speed_margin: float = 0.7,
    v_floor_mps: float = 25.0,
    v_ceiling_mps: float = 350.0,
    slant_margins: tuple[float, float] = (0.35, 3.0),
    d_floor_m: float = 150.0,
    d_ceiling_m: float = 15000.0,
) -> EventFeatures:
    pts = sorted(ev.ridge_points, key=lambda p: p.relative_t_s)
    t = np.array([p.relative_t_s for p in pts], dtype=float)
    f = np.array([p.frequency_hz for p in pts], dtype=float)
    flags: list[str] = []
    if len(t) < 3:
        raise ValueError("event needs at least three ridge points")
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    fs = _smooth(f, int(round(6.0 / max(dt, 1e-3))))
    slope = np.gradient(fs, t) if len(t) > 2 else np.zeros_like(t)
    i, drop = robust_descent_centre(t, f)
    if drop <= 0:
        flags.append("no_descending_slope")
        i = int(np.argmin(np.abs(fs - 0.5 * (fs.max() + fs.min()))))
    t_center = float(t[i])
    f_center = float(fs[i])
    # plateau estimates: robust extremes of the first/last 20 % of the ridge
    n = len(t)
    k = max(2, n // 5)
    f_max = float(np.percentile(fs[:k], 80)) if fs[:k].size else float(fs.max())
    f_min = float(np.percentile(fs[-k:], 20)) if fs[-k:].size else float(fs.min())
    if f_max <= f_min:
        f_max, f_min = float(fs.max()), float(fs.min())
        flags.append("non_monotonic_ridge")
    ratio = max(f_max / max(f_min, 1e-6), 1.0 + 1e-6)
    v_r = c_mps * (ratio - 1.0) / (ratio + 1.0)
    duration = float(t[-1] - t[0])
    if duration < 30.0:
        flags.append("short_event")
    # Did the ridge reach its plateaus? If the ends are still steep the ratio under-estimates v_r.
    end_slopes = np.abs(np.concatenate([slope[:k], slope[-k:]]))
    if end_slopes.size and np.median(end_slopes) > 0.3 * abs(slope[i]):
        flags.append("plateaus_not_reached")
    v_min = max(v_floor_mps, speed_margin * v_r)
    v_max = min(v_ceiling_mps, max(3.0 * v_r, 120.0))
    # transition slope: steepest smoothed slope within ±half window of the centre
    lo_i, hi_i = (
        max(0, i - int(25.0 / max(dt, 1e-3))),
        min(len(t), i + int(25.0 / max(dt, 1e-3)) + 1),
    )
    local = slope[lo_i:hi_i]
    slope_max = (
        float(-local.min()) if local.size and local.min() < 0 else float(np.max(np.abs(slope)))
    )
    if slope_max <= 1e-4:
        flags.append("flat_ridge")
        d_lo, d_hi = d_floor_m, d_ceiling_m
    else:
        d_at = lambda v: f_center * v**2 / (c_mps * slope_max)  # noqa: E731
        d_lo = max(d_floor_m, slant_margins[0] * d_at(v_min))
        d_hi = min(d_ceiling_m, slant_margins[1] * d_at(v_max))
        if d_hi <= d_lo:
            d_lo, d_hi = d_floor_m, d_ceiling_m
            flags.append("slant_bounds_degenerate")
    h = max(1, int(ev.ridge_harmonic))
    f0_lo = f_min / h / 1.03
    f0_hi = f_max / h * 1.03
    # local excursion around the centre (robust to composite ridges whose far ends belong to
    # another aircraft); compared with the predicted excursion over the same window in scoring
    f_before = float(np.interp(t_center - EXCURSION_WINDOW_S, t, fs))
    f_after = float(np.interp(t_center + EXCURSION_WINDOW_S, t, fs))
    excursion_local = max(f_before / max(f_after, 1e-6), 1.0)
    return EventFeatures(
        event_id=ev.id or "",
        t_center_s=t_center,
        f_center_hz=f_center,
        ridge_harmonic=h,
        f_max_hz=f_max,
        f_min_hz=f_min,
        plateau_ratio=ratio,
        v_radial_mps=v_r,
        slope_max_hz_per_s=slope_max,
        duration_s=duration,
        t_start_s=float(ev.audio_start_s),
        t_end_s=float(ev.audio_end_s),
        f0_lo_hz=f0_lo,
        f0_hi_hz=f0_hi,
        excursion_local=excursion_local,
        v_min_mps=v_min,
        v_max_mps=v_max,
        d_min_m=d_lo,
        d_max_m=d_hi,
        n_points=n,
        flags=flags,
    )


# ---- evidence assessment ----------------------------------------------------------------------

# A propeller aircraft at 40-70 m/s produces a full Doppler swing f_app/f_rec of 1.26-1.50.
# A ridge window that contains the transition shows several percent of change; a window on one
# plateau (or a stationary source: hum, machinery, a held musical note) shows almost none.
# Thresholds are evidence grades, not a classifier of the sound's origin.
DOPPLER_MIN_EXCURSION = 0.03  # relative change needed to count as a visible transition
WEAK_MIN_EXCURSION = 0.01
# whitened level above which a template-fit point is a measurement rather than the band maximum
# of noise (the maximum of ~30 noise bins is z ~ 2)
TEMPLATE_POINT_MIN_Z = 3.0


def assess_event(ev: DopplerEvent, c_mps: float = 343.0) -> dict:
    """Shape metrics and an evidence label for one event.

    Labels: ``plausible_doppler`` (a descending transition of several percent),
    ``weak_doppler`` (some descent but small or with unreached plateaus), ``tonal_only`` (a
    tonal ridge without a measurable Doppler transition — it may be a stationary source, or a
    real aircraft heard only on a plateau; either way it does not constrain a location).
    A user review (``ev.extra['review']``) is reported but never changes the measured label."""
    review = (ev.extra or {}).get("review")
    pz = (ev.extra or {}).get("point_z")
    if ev.detection_method == "template-fit" and pz and len(pz) == len(ev.ridge_points):
        # a template event's points are the strongest peak inside a band that follows the
        # fitted template; at noise level that peak traces the template itself. Judge the
        # shape only from points clearly above noise.
        strong = [p for p, z in zip(ev.ridge_points, pz, strict=True) if z >= TEMPLATE_POINT_MIN_Z]
        if len(strong) < max(10, int(0.4 * len(pz))):
            return {
                "label": "tonal_only",
                "reasons": [
                    f"only {len(strong)} of {len(pz)} template-fit points stand clearly above noise "
                    f"(z >= {TEMPLATE_POINT_MIN_Z:g}); the Doppler shape comes from the fitted "
                    "template, not from the audio"
                ],
                "metrics": {"n_points": len(pz), "strong_points": len(strong)},
                "review": review,
            }
        ev = ev.model_copy(update={"ridge_points": strong})
    pts = sorted(ev.ridge_points, key=lambda p: p.relative_t_s)
    reasons: list[str] = []
    if len(pts) < 3:
        return {"label": "tonal_only", "reasons": ["fewer than three ridge points"], "metrics": {}}
    ff = np.array([p.frequency_hz for p in pts])
    lo, hi = float(ff.min()), float(ff.max())
    if (hi - lo) / max(float(np.median(ff)), 1e-6) >= DOPPLER_MIN_EXCURSION:
        # a real pass sweeps continuously through every frequency between its plateaus; points
        # sitting on a few discrete levels are jumps between steady lines (e.g. the harmonics of
        # a constant engine), whatever the size of the "drop"
        occupied = int(np.unique(np.minimum((ff - lo) / (hi - lo) * 12, 11).astype(int)).size)
        if occupied < 6:
            return {
                "label": "tonal_only",
                "reasons": [
                    f"the frequency drop ({hi:.1f} → {lo:.1f} Hz) is not a continuous sweep: the "
                    f"ridge points occupy only {occupied} of 12 frequency bands between the two "
                    "ends — jumps between steady lines (e.g. harmonics of a constant sound)"
                ],
                "metrics": {"n_points": len(pts), "bands_occupied_of_12": occupied},
                "review": review,
            }
    f = np.array([p.frequency_hz for p in pts], dtype=float)
    try:
        feat = measure_event(ev, c_mps)
    except ValueError as e:
        return {"label": "tonal_only", "reasons": [str(e)], "metrics": {}}
    f_mean = float(np.mean(f))
    t = np.array([p.relative_t_s for p in pts], dtype=float)
    _, drop = robust_descent_centre(t, f)
    rel_exc = (feat.f_max_hz - feat.f_min_hz) / max(f_mean, 1e-6)
    rel_drop = max(drop, 0.0) / max(f_mean, 1e-6)
    metrics = {
        "duration_s": round(feat.duration_s, 1),
        "n_points": feat.n_points,
        "f_mean_hz": round(f_mean, 2),
        "f_range_hz": round(float(f.max() - f.min()), 2),
        "excursion_pct": round(100.0 * rel_exc, 2),
        "descent_pct": round(100.0 * rel_drop, 2),
        "max_slope_hz_per_s": round(feat.slope_max_hz_per_s, 3),
        "max_slope_pct_per_s": round(100.0 * feat.slope_max_hz_per_s / max(f_mean, 1e-6), 4),
        "steepest_point_s": round(feat.t_center_s, 1),
        "plateaus_reached": "plateaus_not_reached" not in feat.flags,
        "monotonic": "non_monotonic_ridge" not in feat.flags,
        "harmonics_seen": len(ev.candidate_harmonic_numbers or [1]),
        "implied_radial_speed_mps": round(feat.v_radial_mps, 1),
        "mean_z": (ev.extra or {}).get("mean_z"),
    }
    if "no_descending_slope" in feat.flags:
        reasons.append(
            "no descending frequency transition (an aircraft passing by always descends)"
        )
    if rel_drop < WEAK_MIN_EXCURSION:
        reasons.append(
            f"frequency changes by only {100 * rel_drop:.2f} % around the steepest point "
            f"(a passing propeller aircraft shows several %)"
        )
    if "plateaus_not_reached" in feat.flags:
        reasons.append(
            "the ridge ends before both plateaus are reached (speed and distance underdetermined)"
        )
    if not metrics["monotonic"]:
        reasons.append("the frequency does not fall monotonically from approach to recession")
    if feat.duration_s < 30.0:
        reasons.append(f"short ({feat.duration_s:.0f} s)")
    if rel_drop >= DOPPLER_MIN_EXCURSION and "no_descending_slope" not in feat.flags:
        label = "plausible_doppler" if metrics["monotonic"] else "weak_doppler"
    elif rel_drop >= WEAK_MIN_EXCURSION and "no_descending_slope" not in feat.flags:
        label = "weak_doppler"
    else:
        label = "tonal_only"
    return {"label": label, "reasons": reasons, "metrics": metrics, "review": review}


def usable_as_evidence(assessment: dict) -> bool:
    """True when an event may enter a search: measured Doppler evidence, or an explicit
    user review accepting a tonal-only event (recorded in the result manifest)."""
    if assessment.get("label") in ("plausible_doppler", "weak_doppler"):
        return True
    review = assessment.get("review") or {}
    return bool(review.get("accepted"))
