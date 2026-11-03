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
    merge_gap_s: float = 1.5
    merge_df_bins: float = 3.0
    max_bridge_s: float = 1.0  # longer sub-threshold stretches split a tracked path


@dataclass(eq=False)
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
        # split every ON run where the path bridges through sub-threshold values for more than
        # ``bridge_frames`` consecutive frames (the Viterbi path walking from one aircraft's ridge
        # to another's through noise)
        split_runs: list[tuple[int, int]] = []
        bridge_frames = max(2, int(round(cfg.max_bridge_s / hop)))
        for a, b in runs:
            vals_run = W[np.arange(a, b), path[a:b]]
            weak = vals_run < cfg.threshold_z
            start = a
            i = 0
            while i < b - a:
                if weak[i]:
                    j = i
                    while j < b - a and weak[j]:
                        j += 1
                    if j - i > bridge_frames:
                        if a + i > start:
                            split_runs.append((start, a + i))
                        start = a + j
                    i = j
                else:
                    i += 1
            if b > start:
                split_runs.append((start, b))
        runs = split_runs
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
    ridges = _merge_adjacent(
        ridges, hop, spec.df_hz, max_gap_s=cfg.merge_gap_s, max_df_bins=cfg.merge_df_bins
    )
    _group_harmonics(ridges, cfg.harmonic_tolerance)
    return ridges


def _merge_adjacent(
    ridges: list[Ridge], hop: float, df: float, max_gap_s: float, max_df_bins: float
) -> list[Ridge]:
    """Join pieces of one physical ridge that the iterative extraction returned separately:
    B starts within ``max_gap_s`` after A ends (or overlaps slightly) and the frequencies are
    continuous across the junction."""
    merged: list[Ridge] = []
    pool = sorted(ridges, key=lambda r: r.times[0])
    while pool:
        a = pool.pop(0)
        changed = True
        while changed:
            changed = False
            for b in list(pool):
                gap = b.times[0] - a.times[-1]
                if -2 * hop <= gap <= max_gap_s:
                    na = min(len(a.freqs), max(3, int(round(5.0 / hop))))
                    nb = min(len(b.freqs), max(3, int(round(5.0 / hop))))
                    slope_a = (a.freqs[-1] - a.freqs[-na]) / max(a.times[-1] - a.times[-na], hop)
                    slope_b = (b.freqs[nb - 1] - b.freqs[0]) / max(
                        b.times[nb - 1] - b.times[0], hop
                    )
                    fa = a.freqs[-1] + slope_a * max(gap, 0.0)  # linear continuation
                    slope_ok = abs(slope_b - slope_a) <= max(
                        0.08, 0.5 * max(abs(slope_a), abs(slope_b))
                    )
                    if abs(b.freqs[0] - fa) <= max_df_bins * df and slope_ok:
                        keep = b.times > a.times[-1]
                        if keep.sum() < 2:
                            pool.remove(b)
                            continue
                        a = Ridge(
                            np.concatenate([a.frame_idx, b.frame_idx[keep]]),
                            np.concatenate([a.times, b.times[keep]]),
                            np.concatenate([a.freqs, b.freqs[keep]]),
                            np.concatenate([a.values, b.values[keep]]),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        )
                        a.mean_z = float(a.values.mean())
                        a.duration_s = float(len(a.times) * hop)
                        a.freq_range_hz = float(a.freqs.max() - a.freqs.min())
                        a.freq_std_hz = float(a.freqs.std())
                        a.confidence = max(a.confidence, b.confidence)
                        pool.remove(b)
                        changed = True
        merged.append(a)
    return merged


def _group_harmonics(ridges: list[Ridge], tol: float) -> None:
    """Base-anchored harmonic grouping (no transitive unions).

    Ridges are visited by decreasing evidence (mean z × duration). Each ungrouped ridge becomes
    the base of a new group; another ungrouped ridge joins only if it overlaps the base in time
    by at least half of the shorter duration and its frequency ratio to the base is a small
    rational p/q (p, q ≤ 6) within ``tol`` with a stable ratio over the overlap. Harmonic numbers
    are then assigned with the smallest base-harmonic hypothesis that makes the members integer
    multiples; members that do not fit are split off into their own group.
    """
    for r in ridges:
        r.group = -1
        r.harmonic = 1
    order = sorted(
        range(len(ridges)), key=lambda i: ridges[i].mean_z * ridges[i].duration_s, reverse=True
    )
    gid = 0
    for bi in order:
        base = ridges[bi]
        if base.group >= 0:
            continue
        base.group = gid
        members = [base]
        for j in order:
            r = ridges[j]
            if r.group >= 0 or r is base:
                continue
            t0 = max(base.times[0], r.times[0])
            t1 = min(base.times[-1], r.times[-1])
            overlap = t1 - t0
            if overlap < 0.5 * min(base.duration_s, r.duration_s) or overlap < 10.0:
                continue
            tt = np.arange(t0, t1, 0.5)
            ratio = np.interp(tt, r.times, r.freqs) / np.interp(tt, base.times, base.freqs)
            med = float(np.median(ratio))
            spread = float(np.median(np.abs(ratio - med)))
            ok = False
            for p_ in range(1, 7):
                for q_ in range(1, 7):
                    if (
                        np.gcd(p_, q_) == 1
                        and abs(med - p_ / q_) <= tol * med
                        and spread <= tol * med
                    ):
                        ok = True
                        break
                if ok:
                    break
            if ok:
                r.group = gid
                members.append(r)
        gid += 1
    # harmonic numbers per group
    next_gid = gid
    for g in range(gid):
        members = [r for r in ridges if r.group == g]
        if len(members) == 1:
            members[0].harmonic = 1
            continue
        longest = max(r.duration_s for r in members)
        substantial = [r for r in members if r.duration_s >= 0.3 * longest] or members
        low = min(substantial, key=lambda r: float(np.median(r.freqs)))
        ratios = np.array([_ratio_to(r, low) for r in members])
        durs = np.array([r.duration_s for r in members])
        chosen = None
        for hb in range(1, 5):
            cand = ratios * hb
            fit = np.abs(cand - np.round(cand)) <= 0.06 * cand
            if durs[fit].sum() >= 0.8 * durs.sum():
                chosen = (cand, fit)
                break
        if chosen is None:
            cand = ratios
            fit = np.abs(cand - np.round(cand)) <= 0.06 * cand
        else:
            cand, fit = chosen
        for r, c, ok in zip(members, cand, fit, strict=False):
            if ok:
                r.harmonic = max(1, int(round(c)))
            else:
                r.flags.append("harmonic_mismatch")
                r.group = next_gid
                r.harmonic = 1
                next_gid += 1


def _ratio_to(r: Ridge, base: Ridge) -> float:
    """Median frequency ratio r/base evaluated at r's times. ``np.interp`` clamps to the
    base ridge's end values, so a piece that continues a plateau after the base ridge ended
    is compared against the plateau frequency rather than the base's global median."""
    fb = np.interp(r.times, base.times, base.freqs)
    return float(np.median(r.freqs / fb))


def estimate_f0_at_steepest(
    times: NDArray[np.float64], freqs: NDArray[np.float64], smooth_s: float = 8.0
) -> float:
    """Frequency at the centre of the largest descent (≈ received frequency at the delayed CPA,
    where the radial velocity and hence the Doppler shift are ~zero)."""
    from aerodoppler.search.events import robust_descent_centre

    if len(times) < 5:
        return float(np.median(freqs))
    dt = float(np.median(np.diff(times))) or 0.25
    k = max(3, int(round(smooth_s / dt)) | 1)
    fs = np.convolve(np.pad(freqs, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    i, drop = robust_descent_centre(times, freqs)
    if drop <= 0:  # no descending part: fall back to the midpoint between extremes
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
