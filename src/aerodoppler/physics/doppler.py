"""Emission-time-correct Doppler forward model and delayed-CPA geometry (SPEC §9).

Given a trajectory ``p(t_e)``, ``v(t_e)`` and a stationary observer ``x``::

    r = |p(t_e) - x|,  u = (p - x)/r,  v_toward = -v . u
    t_receive = t_e + r/c
    f_receive = f_source(t_e) * c / (c - v_toward)

Every quantity is evaluated at the *emission* time and then mapped to reception time. The
Doppler factor ``D = c/(c - v_toward)`` is returned separately from ``f_source`` so a single
geometric evaluation can be reused for any candidate source fundamental and harmonic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from aerodoppler.physics.propagation import doppler_factor
from aerodoppler.physics.trajectory import Trajectory


@dataclass
class DopplerPrediction:
    t_emit: NDArray[np.float64]
    t_receive: NDArray[np.float64]
    factor: NDArray[np.float64]  # f_receive / f_source
    slant_range_m: NDArray[np.float64]
    v_toward_mps: NDArray[np.float64]
    observer_ecef: NDArray[np.float64]
    c_mps: float

    def received_frequency(self, f_source: ArrayLike, harmonic: int = 1) -> NDArray[np.float64]:
        """f_source may be scalar or an array aligned with t_emit (slow drift)."""
        return np.asarray(f_source, dtype=np.float64) * harmonic * self.factor

    def is_monotonic(self) -> bool:
        return bool(np.all(np.diff(self.t_receive) > 0))

    def emission_time_for_reception(self, t_receive_query: ArrayLike) -> NDArray[np.float64]:
        """Invert the monotonic t_receive(t_emit) relation by interpolation."""
        tq = np.asarray(t_receive_query, dtype=np.float64)
        out = np.interp(tq, self.t_receive, self.t_emit, left=np.nan, right=np.nan)
        return np.asarray(out)

    def factor_at_reception(self, t_receive_query: ArrayLike) -> NDArray[np.float64]:
        tq = np.asarray(t_receive_query, dtype=np.float64)
        return np.asarray(np.interp(tq, self.t_receive, self.factor, left=np.nan, right=np.nan))


def predict(
    traj: Trajectory,
    observer_ecef: ArrayLike,
    c_mps: float,
    t_emit: ArrayLike | None = None,
    dt_s: float = 0.5,
    t_min: float | None = None,
    t_max: float | None = None,
) -> DopplerPrediction:
    """Evaluate the forward model on an emission-time grid (default: resampled at dt_s)."""
    x = np.asarray(observer_ecef, dtype=np.float64).reshape(3)
    te = (
        traj.resample_times(dt_s, t_min, t_max)
        if t_emit is None
        else np.asarray(t_emit, dtype=np.float64)
    )
    p = traj.position(te)
    v = traj.velocity(te)
    ok = np.all(np.isfinite(p), axis=1)
    te, p, v = te[ok], p[ok], v[ok]
    d = p - x
    r = np.linalg.norm(d, axis=1)
    r_safe = np.where(r > 1e-9, r, 1e-9)
    u = d / r_safe[:, None]
    v_toward = -np.einsum("ij,ij->i", v, u)
    # guard against data glitches: the model is only valid for subsonic radial speeds
    v_toward = np.clip(v_toward, -0.9 * c_mps, 0.9 * c_mps)
    t_r = te + r / c_mps
    fac = doppler_factor(v_toward, c_mps)
    return DopplerPrediction(te, t_r, fac, r, v_toward, x, c_mps)


@dataclass
class ClosestApproach:
    t_emit_cpa: float
    slant_range_m: float
    t_receive_cpa: float  # "delayed CPA": reception time of the emission at minimum range
    position_ecef: NDArray[np.float64]
    speed_mps: float
    altitude_above_observer_m: float | None = None


def closest_approach(traj: Trajectory, observer_ecef: ArrayLike, c_mps: float) -> ClosestApproach:
    """Exact minimum of |p(t) - x| over the piecewise-linear trajectory (no sample snapping)."""
    x = np.asarray(observer_ecef, dtype=np.float64).reshape(3)
    a = traj.pos[:-1] - x
    seg = traj.pos[1:] - traj.pos[:-1]
    seg2 = np.einsum("ij,ij->i", seg, seg)
    s = np.where(seg2 > 0, -np.einsum("ij,ij->i", a, seg) / np.where(seg2 > 0, seg2, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    gap_ok = traj._gap <= traj.max_gap_s
    closest = a + seg * s[:, None]
    r = np.linalg.norm(closest, axis=1)
    r = np.where(gap_ok, r, np.inf)
    i = int(np.argmin(r))
    if not np.isfinite(r[i]):
        raise ValueError("trajectory has no gap-free segment")
    t_cpa = float(traj.t[i] + s[i] * traj._gap[i])
    v = traj.velocity(np.array([t_cpa]))[0]
    return ClosestApproach(
        t_emit_cpa=t_cpa,
        slant_range_m=float(r[i]),
        t_receive_cpa=t_cpa + float(r[i]) / c_mps,
        position_ecef=closest[i] + x,
        speed_mps=float(np.linalg.norm(v)),
    )


@njit(cache=True)
def _cpa_kernel(pos, t, gap_ok, X):
    """Exact closest approach of each observer in X to the piecewise-linear trajectory."""
    M = X.shape[0]
    N = pos.shape[0]
    t_out = np.empty(M)
    r_out = np.empty(M)
    for m in range(M):
        x0, x1, x2 = X[m, 0], X[m, 1], X[m, 2]
        best = 1e300
        best_t = t[0]
        for i in range(N - 1):
            if not gap_ok[i]:
                continue
            ax, ay, az = pos[i, 0] - x0, pos[i, 1] - x1, pos[i, 2] - x2
            sx, sy, sz = (
                pos[i + 1, 0] - pos[i, 0],
                pos[i + 1, 1] - pos[i, 1],
                pos[i + 1, 2] - pos[i, 2],
            )
            seg2 = sx * sx + sy * sy + sz * sz
            if seg2 > 0.0:
                s = -(ax * sx + ay * sy + az * sz) / seg2
                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
            else:
                s = 0.0
            cx, cy, cz = ax + sx * s, ay + sy * s, az + sz * s
            r2 = cx * cx + cy * cy + cz * cz
            if r2 < best:
                best = r2
                best_t = t[i] + s * (t[i + 1] - t[i])
        t_out[m] = best_t
        r_out[m] = np.sqrt(best)
    return t_out, r_out


def closest_approach_batch(
    traj: Trajectory, observers_ecef: NDArray[np.float64], c_mps: float, chunk: int = 4096
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised delayed CPA for many observers.

    Returns ``(t_emit_cpa, slant_range_m, t_receive_cpa)`` each of shape (M,).
    """
    X = np.asarray(observers_ecef, dtype=np.float64).reshape(-1, 3)
    gap_ok_all = traj._gap <= traj.max_gap_s
    if not gap_ok_all.any():
        raise ValueError("trajectory has no gap-free segment")
    if len(X) * len(traj.t) > 20000:  # numba path for large problems (no big temporaries)
        t_out, r_out = _cpa_kernel(traj.pos, traj.t, gap_ok_all, X)
        return t_out, r_out, t_out + r_out / c_mps
    seg = traj.pos[1:] - traj.pos[:-1]  # (N-1, 3)
    seg2 = np.einsum("ij,ij->i", seg, seg)
    seg2_safe = np.where(seg2 > 0, seg2, 1.0)
    gap_ok = traj._gap <= traj.max_gap_s
    t_out = np.empty(len(X))
    r_out = np.empty(len(X))
    for k in range(0, len(X), chunk):
        xs = X[k : k + chunk]  # (m, 3)
        a = traj.pos[None, :-1, :] - xs[:, None, :]  # (m, N-1, 3)
        s = -np.einsum("mij,ij->mi", a, seg) / seg2_safe[None, :]
        s = np.clip(np.where(seg2[None, :] > 0, s, 0.0), 0.0, 1.0)
        closest = a + seg[None, :, :] * s[..., None]
        r = np.sqrt(np.einsum("mij,mij->mi", closest, closest))
        r = np.where(gap_ok[None, :], r, np.inf)
        i = np.argmin(r, axis=1)
        rows = np.arange(len(xs))
        r_out[k : k + chunk] = r[rows, i]
        t_out[k : k + chunk] = traj.t[i] + s[rows, i] * traj._gap[i]
    return t_out, r_out, t_out + r_out / c_mps


def min_slant_range_to_polyline(
    traj: Trajectory, observers_ecef: NDArray[np.float64]
) -> NDArray[np.float64]:
    return closest_approach_batch(traj, observers_ecef, c_mps=343.0)[1]


def analytic_straight_flyby(
    t: ArrayLike,
    speed_mps: float,
    cpa_distance_m: float,
    f_source_hz: float,
    c_mps: float,
    t_cpa_emit: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Closed-form reference for a straight, constant-speed flyby.

    Emission time grid ``t``; aircraft along a line at perpendicular distance ``d`` from the
    observer, closest at ``t_cpa_emit``. Returns ``(t_receive, f_receive)``. Derived
    independently of :func:`predict` (no shared code) for testing.
    """
    te = np.asarray(t, dtype=np.float64)
    along = speed_mps * (te - t_cpa_emit)  # signed along-track position; negative = approaching
    r = np.sqrt(along**2 + cpa_distance_m**2)
    v_toward = -speed_mps * along / r
    t_r = te + r / c_mps
    f_r = f_source_hz * c_mps / (c_mps - v_toward)
    return t_r, f_r
