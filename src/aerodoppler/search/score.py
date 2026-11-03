"""Physically accurate curve scoring with locked two-event alignment (SPEC §10.3).

For a candidate observer ``x`` and an assigned flight segment, the forward model gives the
reception-time curve ``(t_receive_abs, D)`` with ``D = f_receive / f_source``. An observed
recording-relative time ``s`` corresponds to ``t0 + s`` where ``t0`` is the shared recording
clock origin. The score of one event is the harmonic-weighted mean of the whitened
spectrogram (robustly clipped) sampled bilinearly along ``f = f0 (1 + alpha (t - t_c)) h D(t)``
for every harmonic ``h``; ``f0`` (emission fundamental) and ``alpha`` (slow RPM drift,
penalised) are nuisance parameters swept per event. ``t0`` is swept once (small alignment
window) and **shared** by all events: event 2 is never slid independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit
from numpy.typing import NDArray

from aerodoppler.audio.spectrogram import Spectrogram
from aerodoppler.physics.doppler import predict
from aerodoppler.physics.trajectory import Trajectory

CLIP_LO = -2.0
CLIP_HI = 8.0
ON_RIDGE_Z = 1.5


@njit(cache=True)
def _score_kernel(
    W,
    t_spec0,
    hop,
    f_spec0,
    df,
    t_rel,
    D,
    t_c,
    win_lo,
    win_hi,
    f0_vals,
    shifts,
    alphas,
    harmonics,
    hweights,
):
    """f0_vals has shape (S, K): one f0 grid per time shift (centred on the ridge-derived
    estimate for that shift)."""
    T, F = W.shape
    S, A, H = len(shifts), len(alphas), len(harmonics)
    K = f0_vals.shape[1]
    scores = np.full((S, K, A), -9.0)
    coverage = np.zeros((S, K, A))
    valid = np.zeros((S, K, A))
    n = len(t_rel)
    wsum = 0.0
    for h in range(H):
        wsum += hweights[h]
    for si in range(S):
        sh = shifts[si]
        for ki in range(K):
            f0 = f0_vals[si, ki]
            for ai in range(A):
                al = alphas[ai]
                acc = 0.0
                cov = 0.0
                nvalid = 0
                nin = 0
                for i in range(n):
                    tr = t_rel[i] + sh
                    if tr < win_lo or tr > win_hi:
                        continue
                    nin += 1
                    ti = (tr - t_spec0) / hop
                    if ti < 0.0 or ti > T - 1.0:
                        continue
                    r0 = int(ti)
                    if r0 >= T - 1:
                        r0 = T - 2
                    fr = ti - r0
                    fsrc = f0 * (1.0 + al * (t_rel[i] - t_c))
                    point_ok = False
                    for hh in range(H):
                        f = fsrc * harmonics[hh] * D[i]
                        fi = (f - f_spec0) / df
                        if fi < 0.0 or fi > F - 1.0:
                            continue
                        c0 = int(fi)
                        if c0 >= F - 1:
                            c0 = F - 2
                        fc = fi - c0
                        v = (
                            W[r0, c0] * (1 - fr) * (1 - fc)
                            + W[r0, c0 + 1] * (1 - fr) * fc
                            + W[r0 + 1, c0] * fr * (1 - fc)
                            + W[r0 + 1, c0 + 1] * fr * fc
                        )
                        if v < CLIP_LO:
                            v = CLIP_LO
                        elif v > CLIP_HI:
                            v = CLIP_HI
                        acc += hweights[hh] * v
                        if v > ON_RIDGE_Z:
                            cov += hweights[hh]
                        point_ok = True
                    if point_ok:
                        nvalid += 1
                if nvalid > 0 and nin > 0:
                    scores[si, ki, ai] = acc / (nvalid * wsum)
                    coverage[si, ki, ai] = cov / (nvalid * wsum)
                    valid[si, ki, ai] = nvalid / nin
    return scores, coverage, valid


@dataclass
class PredictedCurve:
    t_receive_abs: NDArray[np.float64]
    factor: NDArray[np.float64]
    t_emit: NDArray[np.float64]
    slant_range_m: NDArray[np.float64]
    segment_id: str


def predicted_curve(
    traj: Trajectory,
    observer_ecef: NDArray[np.float64],
    c_mps: float,
    t_center_abs: float,
    half_window_s: float,
    dt_s: float,
) -> PredictedCurve:
    pred = predict(
        traj,
        observer_ecef,
        c_mps,
        dt_s=dt_s,
        t_min=t_center_abs - half_window_s - 60.0,
        t_max=t_center_abs + half_window_s + 60.0,
    )
    return PredictedCurve(
        pred.t_receive, pred.factor, pred.t_emit, pred.slant_range_m, traj.segment_id
    )


@dataclass
class EventScore:
    score: float
    f0_hz: float
    alpha_per_s: float
    coverage: float
    valid_fraction: float
    drift_penalty: float
    shift_index: int

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "f0_hz": self.f0_hz,
            "alpha_per_s": self.alpha_per_s,
            "coverage": self.coverage,
            "valid_fraction": self.valid_fraction,
            "drift_penalty": self.drift_penalty,
        }


@dataclass
class ScoreConfig:
    f0_step_fraction: float = 0.004
    f0_halfwidth_fraction: float = 0.03  # sweep ± this around the ridge-derived f0 estimate
    f0_polish_step_fraction: float = 0.0004  # second-pass fine f0 step (0 disables)
    f0_polish_top_shifts: int = 8
    alphas: tuple[float, ...] = (-2e-5, 0.0, 2e-5)
    alpha_penalty_z: float = 0.25  # penalty at |alpha| = max(alphas), quadratic
    harmonics: tuple[int, ...] = (1, 2, 3, 4)
    harmonic_weights: tuple[float, ...] = (1.0, 0.8, 0.6, 0.45)
    min_valid_fraction: float = 0.5
    t0_fit: str = "event1"  # "event1" (SPEC default) or "joint"


@dataclass
class EventInputs:
    t_center_s: float  # recording-relative measured centre
    t_start_s: float
    t_end_s: float
    f0_lo_hz: float
    f0_hi_hz: float
    weight: float = 1.0
    event_id: str = ""
    ridge_t: NDArray[np.float64] | None = None  # observed ridge points (recording-relative s)
    ridge_f: NDArray[np.float64] | None = None  # observed ridge frequencies (Hz)
    ridge_harmonic: int = 1


@dataclass
class LockedResult:
    t0_abs: float
    combined: float
    per_event: list[EventScore]
    shift_scores: NDArray[np.float64] = field(repr=False)  # combined score as function of t0 shift
    shifts: NDArray[np.float64] = field(repr=False)


def f0_grid(lo: float, hi: float, step_fraction: float) -> NDArray[np.float64]:
    lo, hi = max(lo, 5.0), max(hi, lo * 1.001)
    n = int(np.ceil(np.log(hi / lo) / np.log(1.0 + step_fraction))) + 1
    return lo * (1.0 + step_fraction) ** np.arange(n)


def f0_estimates_per_shift(
    curve_t_rel: NDArray[np.float64],
    curve_factor: NDArray[np.float64],
    shifts: NDArray[np.float64],
    ev: EventInputs,
) -> NDArray[np.float64]:
    """For each t0 shift, the median of f_ridge / (h * D(t_ridge)) over observed ridge points
    that fall inside the predicted curve — the emission fundamental implied by the data."""
    out = np.full(len(shifts), np.nan)
    if ev.ridge_t is None or ev.ridge_f is None or len(ev.ridge_t) == 0:
        return out
    for i, sh in enumerate(shifts):
        D = np.interp(ev.ridge_t - sh, curve_t_rel, curve_factor, left=np.nan, right=np.nan)
        ok = np.isfinite(D)
        if ok.sum() >= 3:
            out[i] = float(np.median(ev.ridge_f[ok] / (ev.ridge_harmonic * D[ok])))
    return out


def score_event_curve(
    spec: Spectrogram,
    curve: PredictedCurve,
    ev: EventInputs,
    t0_candidates: NDArray[np.float64],
    cfg: ScoreConfig,
) -> tuple[
    NDArray[np.float64],
    list[tuple[int, int]],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Returns (best score per t0 candidate, argmax (k, a) per t0, f0 grid (S, K), coverage, valid)."""
    alphas = np.asarray(cfg.alphas, dtype=np.float64)
    # recording-relative curve times for t0 = t0_candidates[0]; other t0s are shifts
    base_t0 = float(t0_candidates[0])
    t_rel = curve.t_receive_abs - base_t0
    shifts = base_t0 - np.asarray(
        t0_candidates, dtype=np.float64
    )  # t_rel' = t_abs - t0 = t_rel + (base_t0 - t0)
    # f0 grid per shift: narrow sweep around the ridge-derived estimate, falling back to the
    # broad event bounds when there are no ridge points
    est = f0_estimates_per_shift(t_rel, curve.factor, shifts, ev)
    rel = (1.0 + cfg.f0_step_fraction) ** np.arange(
        -int(np.ceil(cfg.f0_halfwidth_fraction / cfg.f0_step_fraction)),
        int(np.ceil(cfg.f0_halfwidth_fraction / cfg.f0_step_fraction)) + 1,
    )
    broad = f0_grid(ev.f0_lo_hz, ev.f0_hi_hz, cfg.f0_step_fraction)
    if np.all(np.isnan(est)):
        f0s = np.tile(broad, (len(shifts), 1))
    else:
        fill = float(np.nanmedian(est))
        est = np.where(np.isnan(est), fill, est)
        est = np.clip(est, ev.f0_lo_hz, ev.f0_hi_hz)
        f0s = est[:, None] * rel[None, :]
    harmonics = np.asarray(cfg.harmonics, dtype=np.float64)
    hw = np.asarray(cfg.harmonic_weights[: len(cfg.harmonics)], dtype=np.float64)
    args = (
        spec.whitened,
        float(spec.times[0]),
        spec.hop_s,
        float(spec.freqs[0]),
        spec.df_hz,
        t_rel,
        curve.factor,
        ev.t_center_s,
        ev.t_start_s,
        ev.t_end_s,
    )
    scores, coverage, valid = _score_kernel(*args, f0s, shifts, alphas, harmonics, hw)
    amax = np.max(np.abs(alphas)) or 1.0
    pen = cfg.alpha_penalty_z * (alphas / amax) ** 2
    adj = scores - pen[None, None, :]
    adj = np.where(valid >= cfg.min_valid_fraction, adj, -9.0)
    flat = adj.reshape(len(shifts), -1)
    best_idx = np.argmax(flat, axis=1)
    best = flat[np.arange(len(shifts)), best_idx]
    ka = [(int(b // len(alphas)), int(b % len(alphas))) for b in best_idx]
    # Second pass ("f0 polish"): the highest harmonic needs f0 to ~0.05 % for a ridge that is
    # only ~2 bins wide, so re-sweep a fine f0 grid around the best coarse value for the
    # leading time shifts only (cheap: a handful of shifts).
    if cfg.f0_polish_step_fraction > 0 and cfg.f0_polish_top_shifts > 0:
        top = np.argsort(-best)[: cfg.f0_polish_top_shifts]
        n_f = int(np.ceil(cfg.f0_step_fraction / cfg.f0_polish_step_fraction))
        rel_fine = (1.0 + cfg.f0_polish_step_fraction) ** np.arange(-n_f, n_f + 1)
        for si in top:
            if best[si] <= -9.0:
                continue
            k, _a = ka[si]
            f_fine = (f0s[si, k] * rel_fine)[None, :]
            sc2, cov2, val2 = _score_kernel(
                *args, f_fine, shifts[si : si + 1], alphas, harmonics, hw
            )
            adj2 = np.where(val2 >= cfg.min_valid_fraction, sc2 - pen[None, None, :], -9.0)[0]
            kk, aa = np.unravel_index(int(np.argmax(adj2)), adj2.shape)
            if adj2[kk, aa] > best[si]:
                best[si] = adj2[kk, aa]
                # splice the polished f0 into the grid slot so callers can read it back
                f0s[si, k] = f_fine[0, kk]
                coverage[si, k, aa] = cov2[0, kk, aa]
                valid[si, k, aa] = val2[0, kk, aa]
                ka[si] = (int(k), int(aa))
    return best, ka, f0s, coverage, valid


def locked_score(
    spec: Spectrogram,
    curves: list[PredictedCurve],
    events: list[EventInputs],
    t0_candidates: NDArray[np.float64],
    cfg: ScoreConfig,
) -> LockedResult:
    """Shared-t0 scoring of all events. ``t0`` is chosen from event 1 alone (SPEC default) or
    from the weighted combination (``cfg.t0_fit == 'joint'``); event 2..n are then evaluated at
    exactly that t0."""
    per_shift: list[NDArray[np.float64]] = []
    details = []
    for curve, ev in zip(curves, events, strict=True):
        best, ka, f0s, cov, val = score_event_curve(spec, curve, ev, t0_candidates, cfg)
        per_shift.append(best)
        details.append((ka, f0s, cov, val))
    weights = np.array([e.weight for e in events], dtype=np.float64)
    combined_by_shift = np.sum(np.stack(per_shift) * weights[:, None], axis=0) / weights.sum()
    if cfg.t0_fit == "joint":
        si = int(np.argmax(combined_by_shift))
    else:
        si = int(np.argmax(per_shift[0]))
    alphas = np.asarray(cfg.alphas)
    amax = np.max(np.abs(alphas)) or 1.0
    per_event = []
    for (ka, f0s, cov, val), best in zip(details, per_shift, strict=True):
        k, a = ka[si]
        pen = float(cfg.alpha_penalty_z * (alphas[a] / amax) ** 2)
        per_event.append(
            EventScore(
                float(best[si]),
                float(f0s[si, k]),
                float(alphas[a]),
                float(cov[si, k, a]),
                float(val[si, k, a]),
                pen,
                si,
            )
        )
    return LockedResult(
        float(t0_candidates[si]),
        float(combined_by_shift[si]),
        per_event,
        combined_by_shift,
        t0_candidates - t0_candidates[0],
    )
