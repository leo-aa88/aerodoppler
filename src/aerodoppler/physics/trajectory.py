"""Interpolated 3D aircraft trajectories in ECEF.

Position: piecewise linear between samples. Velocity: smoothed central finite differences,
linearly interpolated. Queries outside the sampled interval, or inside a gap larger than
``max_gap_s``, return NaN rather than an extrapolated value (SPEC §6.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerodoppler.geo.coordinates import ecef_to_geodetic, geodetic_to_ecef


@dataclass
class Trajectory:
    t: NDArray[np.float64]  # (N,) unix seconds, strictly increasing
    pos: NDArray[np.float64]  # (N, 3) ECEF metres
    vel: NDArray[np.float64]  # (N, 3) ECEF m/s
    max_gap_s: float = 120.0
    segment_id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)
        if self.t.ndim != 1 or len(self.t) < 2:
            raise ValueError("trajectory needs at least two samples")
        if self.pos.shape != (len(self.t), 3) or self.vel.shape != (len(self.t), 3):
            raise ValueError("pos/vel must be (N, 3)")
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("trajectory times must be strictly increasing")
        if not (np.all(np.isfinite(self.pos)) and np.all(np.isfinite(self.vel))):
            raise ValueError("trajectory contains non-finite values")
        self._gap = np.diff(self.t)

    # ---- constructors ------------------------------------------------------------------
    @classmethod
    def from_ecef(
        cls, t: ArrayLike, pos: ArrayLike, max_gap_s: float = 120.0, smooth: int = 3, **kw
    ) -> Trajectory:
        t_a = np.asarray(t, dtype=np.float64)
        p = np.asarray(pos, dtype=np.float64)
        order = np.argsort(t_a, kind="stable")
        t_a, p = t_a[order], p[order]
        keep = np.concatenate([[True], np.diff(t_a) > 1e-6])  # drop duplicated timestamps
        t_a, p = t_a[keep], p[keep]
        vel = finite_difference_velocity(t_a, p, smooth=smooth)
        return cls(t=t_a, pos=p, vel=vel, max_gap_s=max_gap_s, **kw)

    @classmethod
    def from_geodetic(
        cls, t: ArrayLike, lat: ArrayLike, lon: ArrayLike, h_hae_m: ArrayLike, **kw
    ) -> Trajectory:
        return cls.from_ecef(t, geodetic_to_ecef(lat, lon, h_hae_m), **kw)

    # ---- properties --------------------------------------------------------------------
    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    @property
    def speed(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.vel, axis=1)

    def geodetic(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return ecef_to_geodetic(self.pos)

    # ---- interpolation -----------------------------------------------------------------
    def _locate(
        self, tq: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.bool_]]:
        idx = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, len(self.t) - 2)
        dt = self._gap[idx]
        frac = (tq - self.t[idx]) / dt
        valid = (tq >= self.t[0]) & (tq <= self.t[-1]) & (dt <= self.max_gap_s)
        return idx, frac, valid

    def position(self, t_query: ArrayLike) -> NDArray[np.float64]:
        tq = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        idx, frac, valid = self._locate(tq)
        out = self.pos[idx] + (self.pos[idx + 1] - self.pos[idx]) * frac[:, None]
        out[~valid] = np.nan
        return out

    def velocity(self, t_query: ArrayLike) -> NDArray[np.float64]:
        tq = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        idx, frac, valid = self._locate(tq)
        out = self.vel[idx] + (self.vel[idx + 1] - self.vel[idx]) * frac[:, None]
        out[~valid] = np.nan
        return out

    def resample_times(
        self, dt_s: float, t_min: float | None = None, t_max: float | None = None
    ) -> NDArray[np.float64]:
        a = self.t_start if t_min is None else max(self.t_start, t_min)
        b = self.t_end if t_max is None else min(self.t_end, t_max)
        if b <= a:
            return np.empty(0)
        n = int(np.floor((b - a) / dt_s)) + 1
        return a + dt_s * np.arange(n)

    def valid_mask(self, t_query: ArrayLike) -> NDArray[np.bool_]:
        tq = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        return self._locate(tq)[2]

    def subsample(self, min_dt_s: float) -> Trajectory:
        """Keep samples at least ``min_dt_s`` apart (always first and last); velocities are kept
        from the dense track. Used for coarse CPA screening; errors are a few metres for
        straight flight and tens of metres in turns, far below the coarse grid spacing."""
        keep = [0]
        last = self.t[0]
        for i in range(1, len(self.t) - 1):
            if self.t[i] - last >= min_dt_s:
                keep.append(i)
                last = self.t[i]
        keep.append(len(self.t) - 1)
        idx = np.array(sorted(set(keep)))
        return Trajectory(
            self.t[idx],
            self.pos[idx],
            self.vel[idx],
            self.max_gap_s + min_dt_s,
            self.segment_id,
            self.meta,
        )

    def slice(self, t_min: float, t_max: float, pad: int = 1) -> Trajectory:
        i0 = max(0, int(np.searchsorted(self.t, t_min)) - pad)
        i1 = min(len(self.t), int(np.searchsorted(self.t, t_max, side="right")) + pad)
        if i1 - i0 < 2:
            i0, i1 = max(0, i1 - 2), min(len(self.t), i0 + 2)
        return Trajectory(
            self.t[i0:i1],
            self.pos[i0:i1],
            self.vel[i0:i1],
            self.max_gap_s,
            self.segment_id,
            self.meta,
        )


MAX_PLAUSIBLE_SPEED_MPS = 400.0  # subsonic civil traffic; larger values are data glitches


def finite_difference_velocity(
    t: NDArray[np.float64], pos: NDArray[np.float64], smooth: int = 3
) -> NDArray[np.float64]:
    """Central differences (one-sided at the ends), then a short moving average.

    For a constant-velocity track this is exact regardless of smoothing. Smoothing is applied
    in *sample index* space and only across neighbours separated by less than 60 s so that a
    gap does not blur velocity across it.
    """
    n = len(t)
    v = np.empty_like(pos)
    if n == 2:
        v[:] = (pos[1] - pos[0]) / (t[1] - t[0])
        return v
    v[1:-1] = (pos[2:] - pos[:-2]) / (t[2:] - t[:-2])[:, None]
    v[0] = (pos[1] - pos[0]) / (t[1] - t[0])
    v[-1] = (pos[-1] - pos[-2]) / (t[-1] - t[-2])
    if smooth and smooth > 1 and n >= smooth:
        half = smooth // 2
        out = v.copy()
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            w = np.abs(t[lo:hi] - t[i]) < 60.0
            out[i] = v[lo:hi][w].mean(axis=0)
        v = out
    # near-duplicate timestamps with position noise can produce absurd velocities; clamp the
    # magnitude so downstream Doppler factors stay finite and subsonic (data glitch, flagged by
    # the segment splitter's implied-speed rule where it exceeds 350 m/s)
    speed = np.linalg.norm(v, axis=1)
    too_fast = speed > MAX_PLAUSIBLE_SPEED_MPS
    if too_fast.any():
        v[too_fast] *= (MAX_PLAUSIBLE_SPEED_MPS / speed[too_fast])[:, None]
    return v
