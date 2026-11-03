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
    i = int(np.argmin(slope))
    if slope[i] >= 0:
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
    slope_max = float(abs(slope[i])) if slope[i] < 0 else float(np.max(np.abs(slope)))
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
        v_min_mps=v_min,
        v_max_mps=v_max,
        d_min_m=d_lo,
        d_max_m=d_hi,
        n_points=n,
        flags=flags,
    )
