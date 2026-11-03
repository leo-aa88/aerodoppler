"""Automatic tonal-ridge detection (SPEC §8.6).

Method: iterative Viterbi tracking on the whitened spectrogram. The hidden state per frame
is either OFF or a frequency bin; ON→ON transitions are limited to ±``max_jump_bins`` with a
quadratic penalty, switching ON/OFF costs a fixed penalty, and the emission reward is the
whitened value minus a threshold. The best path yields one or more ridge runs; the strongest
run is masked out and the procedure repeats. Runs are filtered for length and
frequency *variation* (stationary hum has none), then grouped into harmonic families.

Everything returned is an editable proposal with a confidence; manual annotation remains the
reference workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit
from numpy.typing import NDArray

from aerodoppler.audio.spectrogram import Spectrogram
from aerodoppler.models import DopplerEvent, RidgePoint

OFF = -1


@dataclass
class DetectorConfig:
    threshold_z: float = 2.5
    max_jump_bins: int = 6
    jump_penalty: float = 0.08
    switch_penalty: float = 14.0
    min_duration_s: float = 40.0
    min_mean_z: float = 3.0
    min_freq_variation_hz: float = 1.5
    min_frequency_hz: float = 25.0
    max_frequency_hz: float = 900.0
    max_ridges: int = 12
    smooth_bins: int = 3
    harmonic_tolerance: float = 0.03


@dataclass
class Ridge:
    frame_idx: NDArray[np.int64]
    times: NDArray[np.float64]
    freqs: NDArray[np.float64]
    values: NDArray[np.float64]
    mean_z: float
    duration_s: float
    freq_range_hz: float
    freq_std_hz: float
    harmonic: int = 1
    group: int = -1
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)

    def as_points(self, df_hz: float, harmonic: int = 1) -> list[RidgePoint]:
        unc = np.clip(2.0 * df_hz / np.sqrt(np.maximum(self.values, 1.0)), 0.25 * df_hz, 4 * df_hz)
        return [
            RidgePoint(relative_t_s=float(t), frequency_hz=float(f), uncertainty_hz=float(u))
            for t, f, u in zip(self.times, self.freqs, unc, strict=False)
        ]


@njit(cache=True)
def _viterbi(W: np.ndarray, thr: float, K: int, jump_pen: float, switch_pen: float) -> np.ndarray:
    T, F = W.shape
    NEG = -1e18
    score_on = np.full(F, NEG)
    score_off = 0.0
    back_on = np.zeros((T, F), dtype=np.int16)  # delta from previous bin, or -32768 for "from OFF"
    back_off = np.zeros(T, dtype=np.int32)  # -1 = from OFF, else bin index of ON
    FROM_OFF = -32768
    new_on = np.empty(F)
    for t in range(T):
        # OFF state update
        best_off = score_off
        best_src = -1
        for f in range(F):
            v = score_on[f] - switch_pen
            if v > best_off:
                best_off = v
                best_src = f
        back_off[t] = best_src
        # ON state update
        for f in range(F):
            best = score_off - switch_pen
            bsrc = FROM_OFF
            lo = f - K if f - K > 0 else 0
            hi = f + K if f + K < F - 1 else F - 1
            for g in range(lo, hi + 1):
                d = g - f
                v = score_on[g] - jump_pen * d * d
                if v > best:
                    best = v
                    bsrc = d
            new_on[f] = best + (W[t, f] - thr)
            back_on[t, f] = bsrc
        for f in range(F):
            score_on[f] = new_on[f]
        score_off = best_off
    # termination
    end_state = -1
    best = score_off
    for f in range(F):
        if score_on[f] > best:
            best = score_on[f]
            end_state = f
    path = np.empty(T, dtype=np.int32)
    s = end_state
    for t in range(T - 1, -1, -1):
        path[t] = s
        if s == -1:
            s = back_off[t]
        else:
            d = back_on[t, s]
            if d == FROM_OFF:
                s = -1
            else:
                s = s + d
    return path


def _refine_peak(W: np.ndarray, t: int, f: int) -> float:
    if 0 < f < W.shape[1] - 1:
        y0, y1, y2 = float(W[t, f - 1]), float(W[t, f]), float(W[t, f + 1])
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return 0.0


def detect_ridges(spec: Spectrogram, cfg: DetectorConfig | None = None) -> list[Ridge]:
    cfg = cfg or DetectorConfig()
    fmask = (spec.freqs >= cfg.min_frequency_hz) & (spec.freqs <= cfg.max_frequency_hz)
    f_off = int(np.argmax(fmask))
    W = spec.whitened[:, fmask].astype(np.float64)
    if cfg.smooth_bins > 1:
        k = np.ones(cfg.smooth_bins) / cfg.smooth_bins
        W = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, W)
    work = W.copy()
    ridges: list[Ridge] = []
    hop = spec.hop_s
    min_frames = int(cfg.min_duration_s / hop)
    for _ in range(cfg.max_ridges):
        path = _viterbi(
            work, cfg.threshold_z, cfg.max_jump_bins, cfg.jump_penalty, cfg.switch_penalty
        )
        on = path >= 0
        if not on.any():
            break
        # split into runs
        edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
        runs = list(zip(edges[::2], edges[1::2], strict=False))
        if not runs:
            break
        accepted_any = False
        for a, b in runs:
            idx = np.arange(a, b)
            bins = path[a:b]
            vals = W[idx, bins]
            # mask this run out for the next iteration regardless of acceptance
            for i, fb in zip(idx, bins, strict=False):
                lo, hi = (
                    max(0, fb - 2 * cfg.max_jump_bins),
                    min(work.shape[1], fb + 2 * cfg.max_jump_bins + 1),
                )
                work[i, lo:hi] = 0.0
            if len(idx) < min_frames:
                continue
            deltas = np.array(
                [_refine_peak(W, int(i), int(fb)) for i, fb in zip(idx, bins, strict=False)]
            )
            freqs = spec.freqs[f_off + bins] + deltas * spec.df_hz
            mean_z = float(vals.mean())
            fr = float(freqs.max() - freqs.min())
            fstd = float(freqs.std())
            flags = []
            if mean_z < cfg.min_mean_z:
                flags.append("weak")
            if fr < cfg.min_freq_variation_hz:
                flags.append("stationary_tone")  # hum / equipment, not an aircraft
            if flags:
                continue
            r = Ridge(idx, spec.times[idx], freqs, vals, mean_z, float(len(idx) * hop), fr, fstd)
            r.confidence = (
                float(1 - np.exp(-(mean_z - cfg.threshold_z) * r.duration_s / 200.0))
                if mean_z > cfg.threshold_z
                else 0.0
            )
            ridges.append(r)
            accepted_any = True
        if not accepted_any and all((b - a) < min_frames for a, b in runs):
            break
    ridges.sort(key=lambda r: r.times[0])
    _group_harmonics(ridges, cfg.harmonic_tolerance)
    return ridges


def _group_harmonics(ridges: list[Ridge], tol: float) -> None:
    """Assign group ids and harmonic numbers using frequency ratios at common times."""
    n = len(ridges)
    for r in ridges:
        r.group = -1
        r.harmonic = 1
    gid = 0
    for i in range(n):
        if ridges[i].group < 0:
            ridges[i].group = gid
            gid += 1
        for j in range(i + 1, n):
            a, b = ridges[i], ridges[j]
            t0 = max(a.times[0], b.times[0])
            t1 = min(a.times[-1], b.times[-1])
            if t1 - t0 < 20.0:
                continue
            tt = np.arange(t0, t1, 1.0)
            fa = np.interp(tt, a.times, a.freqs)
            fb = np.interp(tt, b.times, b.freqs)
            ratio = fb / fa
            med = float(np.median(ratio))
            spread = float(np.median(np.abs(ratio - med)))
            best = None
            for p in range(1, 9):
                for q in range(1, 9):
                    if np.gcd(p, q) != 1:
                        continue
                    if abs(med - p / q) <= tol * med and spread <= tol * med:
                        if best is None or p + q < best[0] + best[1]:
                            best = (p, q)
            if best is None:
                continue
            p, q = best  # fb/fa ≈ p/q  → fa ∝ q, fb ∝ p
            if b.group < 0:
                b.group = a.group
                a.harmonic, b.harmonic = _merge_harmonics(a.harmonic, q, p)
            elif a.group != b.group:
                # merge groups
                old = b.group
                for r in ridges:
                    if r.group == old:
                        r.group = a.group
    # Renumber harmonics per group. Hypothesis h_base (1..4) for the lowest ridge is chosen as
    # the smallest value for which members covering >= 80 % of the group's total duration have
    # near-integer implied harmonics; members that do not fit are split into their own group.
    next_gid = (max(r.group for r in ridges) + 1) if ridges else 0
    for g in sorted(set(r.group for r in ridges)):
        members = [r for r in ridges if r.group == g]
        if len(members) == 1:
            members[0].harmonic = 1
            continue
        base = min(members, key=lambda r: float(np.median(r.freqs)))
        base_f = float(np.median(base.freqs))
        ratios = np.array([_ratio_to(r, base) for r in members])
        durs = np.array([r.duration_s for r in members])
        chosen = None
        for hb in range(1, 5):
            cand = ratios * hb
            fit = np.abs(cand - np.round(cand)) <= 0.06 * cand
            if durs[fit].sum() >= 0.8 * durs.sum():
                chosen = (hb, cand, fit)
                break
        if chosen is None:
            hb = 1
            cand = ratios
            fit = np.abs(cand - np.round(cand)) <= 0.06 * cand
        else:
            hb, cand, fit = chosen
        for r, c, ok in zip(members, cand, fit, strict=False):
            if ok:
                r.harmonic = max(1, int(round(c)))
            else:
                r.flags.append("harmonic_mismatch")
                r.group = next_gid
                r.harmonic = 1
                next_gid += 1
        _ = base_f


def _ratio_to(r: Ridge, base: Ridge) -> float:
    """Median frequency ratio r/base over their common time span (falls back to medians)."""
    t0 = max(r.times[0], base.times[0])
    t1 = min(r.times[-1], base.times[-1])
    if t1 - t0 < 5.0:
        return float(np.median(r.freqs) / np.median(base.freqs))
    tt = np.arange(t0, t1, 0.5)
    return float(np.median(np.interp(tt, r.times, r.freqs) / np.interp(tt, base.times, base.freqs)))


def _merge_harmonics(ha: int, qa: int, pb: int) -> tuple[int, int]:
    return qa, pb


def estimate_f0_at_steepest(
    times: NDArray[np.float64], freqs: NDArray[np.float64], smooth_s: float = 8.0
) -> float:
    """Frequency at the point of steepest descent ≈ received frequency at the delayed CPA,
    where the radial velocity (and hence the Doppler shift) is ~zero."""
    if len(times) < 5:
        return float(np.median(freqs))
    dt = float(np.median(np.diff(times))) or 0.25
    k = max(3, int(round(smooth_s / dt)) | 1)
    kern = np.ones(k) / k
    fs = np.convolve(np.pad(freqs, (k // 2, k // 2), mode="edge"), kern, mode="valid")
    slope = np.gradient(fs, times)
    i = int(np.argmin(slope))
    if slope[i] >= 0:  # no descending part: fall back to the midpoint between extremes
        return float(0.5 * (fs.max() + fs.min()))
    return float(fs[i])


def ridges_to_events(
    ridges: list[Ridge], spec: Spectrogram, recording_id: str, method: str = "auto-viterbi"
) -> list[DopplerEvent]:
    """One DopplerEvent per harmonic group; ridge points from the strongest member, with its
    harmonic number so the fundamental can be inferred (f0 = f / h)."""
    events: list[DopplerEvent] = []
    for g in sorted(set(r.group for r in ridges)):
        members = [r for r in ridges if r.group == g]
        strongest = max(members, key=lambda r: r.mean_z * r.duration_s)
        t0 = float(min(r.times[0] for r in members))
        t1 = float(max(r.times[-1] for r in members))
        harmonics = sorted(set(r.harmonic for r in members))
        quality = float(np.clip(max(r.confidence for r in members), 0, 1))
        f0_est = estimate_f0_at_steepest(strongest.times, strongest.freqs) / strongest.harmonic
        ev = DopplerEvent(
            recording_id=recording_id,
            audio_start_s=max(0.0, t0 - 2.0),
            audio_end_s=t1 + 2.0,
            ridge_points=strongest.as_points(spec.df_hz, strongest.harmonic),
            candidate_harmonic_numbers=harmonics or [1],
            f0_prior=f0_est,
            detection_method=method,
            detection_quality=quality,
            annotated_by="auto",
            ridge_harmonic=strongest.harmonic,
            extra={
                "n_members": len(members),
                "mean_z": strongest.mean_z,
                "member_harmonics": [r.harmonic for r in members],
            },
        )
        events.append(ev)
    return events
