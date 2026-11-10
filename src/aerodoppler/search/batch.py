"""Batched scoring of the ultra-coarse screen (Phase 2 of the scoring plan).

The reference path scores one candidate at a time in Python: per event a predicted curve from
the trajectory (``physics.doppler.predict``), ridge-implied f0 estimates per clock shift, a
slope weighting, the ``_score_kernel`` sweep and the shared-clock combination
(``search.score.locked_score``), then the unexplained-event rule (``engine._score_point``).
Profiling the Manhattan-style workload showed the numba kernel at ~15 % of a worker's time;
the rest was Python and small NumPy calls per candidate.

This module re-implements that exact computation for the ultra-coarse configuration (one drift
value, no f0 polish) as one numba function over a batch of candidates, run in threads with
``prange`` on shared arrays (no process pool, no copy-on-write growth). Every step mirrors the
reference, including its edge rules, and ``tests/unit/test_batch_scoring.py`` checks parity on
random candidates. Physics is unchanged: emission-time sampling of the trajectory, reception
time t_e + r/c, Doppler factor c / (c - v_toward) with v_toward clipped to ±0.9 c.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from aerodoppler.search.score import CLIP_HI, CLIP_LO, ON_RIDGE_Z

NEG = -9.0


@dataclass
class TrajectoryTable:
    """All segment trajectories as one structure-of-arrays block."""

    t: NDArray[np.float64]
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    off: NDArray[np.int64]  # segment k occupies [off[k], off[k+1])
    max_gap: NDArray[np.float64]

    @classmethod
    def from_segments(cls, segments) -> TrajectoryTable:
        ts, ps, vs, gaps = [], [], [], []
        off = [0]
        for s in segments:
            tr = s.traj_build.trajectory
            ts.append(tr.t)
            ps.append(tr.pos)
            vs.append(tr.vel)
            gaps.append(tr.max_gap_s)
            off.append(off[-1] + len(tr.t))
        return cls(
            np.concatenate(ts),
            np.concatenate(ps),
            np.concatenate(vs),
            np.asarray(off, dtype=np.int64),
            np.asarray(gaps, dtype=np.float64),
        )


@dataclass
class EventTable:
    """Per-event inputs of the reference ``EventInputs`` plus the curve sampling of the screen."""

    t_center: NDArray[np.float64]
    f0_lo: NDArray[np.float64]
    f0_hi: NDArray[np.float64]
    win_lo: NDArray[np.float64]  # t_start
    win_hi: NDArray[np.float64]  # t_end
    plateau: NDArray[np.float64]  # NaN = no observed ratio
    exc_window: NDArray[np.float64]
    half: NDArray[np.float64]  # curve half window (s) around the delayed CPA
    dt: NDArray[np.float64]  # curve emission-time step (s)
    ridge_t: NDArray[np.float64]
    ridge_f: NDArray[np.float64]
    ridge_off: NDArray[np.int64]
    ridge_h: NDArray[np.float64]


@njit(cache=True)
def _interp(x, xp, fp, lo, hi):  # pragma: no cover - numba
    """np.interp for increasing ``xp`` with explicit left/right values."""
    n = xp.shape[0]
    if x < xp[0]:
        return lo
    if x > xp[n - 1]:
        return hi
    if x == xp[n - 1]:
        return fp[n - 1]
    # rightmost j with xp[j] <= x
    a, b = 0, n - 1
    while b - a > 1:
        m = (a + b) // 2
        if xp[m] <= x:
            a = m
        else:
            b = m
    if xp[a + 1] == xp[a]:
        return fp[a]
    s = (fp[a + 1] - fp[a]) / (xp[a + 1] - xp[a])
    return s * (x - xp[a]) + fp[a]


@njit(cache=True)
def _predict(tt, tpos, tvel, max_gap, x0, x1, x2, c, tmin, tmax, dt):  # pragma: no cover
    """physics.doppler.predict on ``Trajectory.resample_times(dt, tmin, tmax)``: returns
    (t_receive, factor) of the samples where the trajectory is valid."""
    n_t = tt.shape[0]
    a = max(tt[0], tmin)
    b = min(tt[n_t - 1], tmax)
    if b <= a:
        return np.empty(0), np.empty(0)
    n = int(np.floor((b - a) / dt)) + 1
    tr = np.empty(n)
    fac = np.empty(n)
    k = 0
    for i in range(n):
        te = a + dt * i
        # Trajectory._locate: searchsorted(side="right") - 1, clipped to [0, N-2]
        lo_, hi_ = 0, n_t
        while lo_ < hi_:
            m = (lo_ + hi_) // 2
            if tt[m] <= te:
                lo_ = m + 1
            else:
                hi_ = m
        j = lo_ - 1
        if j < 0:
            j = 0
        if j > n_t - 2:
            j = n_t - 2
        gap = tt[j + 1] - tt[j]
        if te < tt[0] or te > tt[n_t - 1] or gap > max_gap:
            continue
        fr = (te - tt[j]) / gap
        px = tpos[j, 0] + (tpos[j + 1, 0] - tpos[j, 0]) * fr
        py = tpos[j, 1] + (tpos[j + 1, 1] - tpos[j, 1]) * fr
        pz = tpos[j, 2] + (tpos[j + 1, 2] - tpos[j, 2]) * fr
        vx = tvel[j, 0] + (tvel[j + 1, 0] - tvel[j, 0]) * fr
        vy = tvel[j, 1] + (tvel[j + 1, 1] - tvel[j, 1]) * fr
        vz = tvel[j, 2] + (tvel[j + 1, 2] - tvel[j, 2]) * fr
        dx, dy, dz = px - x0, py - x1, pz - x2
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        rs = r if r > 1e-9 else 1e-9
        vt = -(vx * dx + vy * dy + vz * dz) / rs
        lim = 0.9 * c
        if vt < -lim:
            vt = -lim
        elif vt > lim:
            vt = lim
        tr[k] = te + r / c
        fac[k] = c / (c - vt)
        k += 1
    return tr[:k], fac[:k]


@njit(cache=True)
def _gradient(y, x):  # pragma: no cover - numba
    """np.gradient(y, x) for non-uniform x (second-order interior, first-order edges)."""
    n = y.shape[0]
    g = np.empty(n)
    g[0] = (y[1] - y[0]) / (x[1] - x[0])
    g[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
    for i in range(1, n - 1):
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        g[i] = (
            (-h2 / (h1 * (h1 + h2))) * y[i - 1]
            + ((h2 - h1) / (h1 * h2)) * y[i]
            + (h1 / (h2 * (h1 + h2))) * y[i + 1]
        )
    return g


@njit(cache=True)
def _event_curve_score(  # pragma: no cover - numba
    W,
    t_spec0,
    hop,
    f_spec0,
    df,
    t_rel,
    D,
    shifts,
    tc,
    win_lo,
    win_hi,
    f0_lo,
    f0_hi,
    ridge_t,
    ridge_f,
    ridge_h,
    exc_window,
    harmonics,
    hweights,
    rel,
    broad,
    slope_floor,
    min_valid,
):
    """score_event_curve for one drift value (alpha = 0) and no f0 polish.
    Returns (best per shift, coverage of that best per shift, predicted excursion)."""
    S = shifts.shape[0]
    n = t_rel.shape[0]
    T, F = W.shape
    H = harmonics.shape[0]
    wsum = 0.0
    for h in range(H):
        wsum += hweights[h]
    # f0 estimates per shift: median of f / (h D(t)) over ridge points inside the curve
    est = np.full(S, np.nan)
    nr = ridge_t.shape[0]
    buf = np.empty(nr)
    for si in range(S):
        m = 0
        if n > 0:
            # ridge points are sorted by time (UltraBatchScorer), so the interpolation interval
            # only moves forward: a merge walk gives exactly np.interp's interval (the rightmost
            # j <= n-2 with t_rel[j] <= x) without a binary search per point
            j = 0
            last = t_rel[n - 1]
            for p in range(nr):
                x = ridge_t[p] - shifts[si]
                if x < t_rel[0] or x > last:
                    continue
                if x == last:
                    Dv = D[n - 1]
                else:
                    while j < n - 2 and t_rel[j + 1] <= x:
                        j += 1
                    if t_rel[j + 1] == t_rel[j]:
                        Dv = D[j]
                    else:
                        Dv = (D[j + 1] - D[j]) / (t_rel[j + 1] - t_rel[j]) * (x - t_rel[j]) + D[j]
                if np.isfinite(Dv):
                    buf[m] = ridge_f[p] / (ridge_h * Dv)
                    m += 1
        if m >= 3:
            est[si] = np.median(buf[:m])
    all_nan = True
    for si in range(S):
        if np.isfinite(est[si]):
            all_nan = False
    if all_nan:
        K = broad.shape[0]
        f0s = np.empty((S, K))
        for si in range(S):
            f0s[si, :] = broad
    else:
        fill = np.nanmedian(est)
        K = rel.shape[0]
        f0s = np.empty((S, K))
        for si in range(S):
            e = est[si] if np.isfinite(est[si]) else fill
            if e < f0_lo:
                e = f0_lo
            elif e > f0_hi:
                e = f0_hi
            for k in range(K):
                f0s[si, k] = e * rel[k]
    # information weighting
    pw = np.ones(n)
    if n >= 3:
        dD = np.abs(_gradient(D, t_rel))
        mx = max(dD.max(), 1e-12)
        for i in range(n):
            pw[i] = slope_floor + (1.0 - slope_floor) * dD[i] / mx
    best = np.full(S, NEG)
    best_cov = np.zeros(S)
    for si in range(S):
        sh = shifts[si]
        for ki in range(K):
            f0 = f0s[si, ki]
            acc = 0.0
            cov = 0.0
            wsum_pts = 0.0
            nvalid = 0
            nin = 0
            for i in range(n):
                tr = t_rel[i] + sh
                if tr < win_lo or tr > win_hi:
                    continue
                nin += 1
                w_i = pw[i]
                ti = (tr - t_spec0) / hop
                if ti < 0.0 or ti > T - 1.0:
                    continue
                r0 = int(ti)
                if r0 >= T - 1:
                    r0 = T - 2
                fr = ti - r0
                point_ok = False
                for hh in range(H):
                    f = f0 * harmonics[hh] * D[i]
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
                    acc += hweights[hh] * v * w_i
                    if v > ON_RIDGE_Z:
                        cov += hweights[hh] * w_i
                    point_ok = True
                if point_ok:
                    nvalid += 1
                    wsum_pts += w_i
            sc = NEG
            cv = 0.0
            if nvalid > 0 and nin > 0 and wsum_pts > 0.0:
                frac = nvalid / nin
                if frac >= min_valid:
                    sc = acc / (wsum_pts * wsum) * frac
                cv = cov / (wsum_pts * wsum) * frac
            # first maximum over k (np.argmax over the flattened (K, A=1) grid)
            if ki == 0 or sc > best[si]:
                best[si] = sc
                best_cov[si] = cv
    # predicted excursion at the central shift (np.interp with edge values outside)
    exc = np.nan
    mid = S // 2
    if n >= 2:
        a0 = t_rel[0] + shifts[mid]
        a1 = t_rel[n - 1] + shifts[mid]
        if a0 <= tc <= a1:
            tr_mid = t_rel + shifts[mid]
            db = _interp(tc - exc_window, tr_mid, D, D[0], D[n - 1])
            da = _interp(tc + exc_window, tr_mid, D, D[0], D[n - 1])
            exc = max(db / max(da, 1e-9), 1.0)
    return best, best_cov, exc


@njit(parallel=True, cache=True)
def ultra_scores(  # pragma: no cover - numba
    W,
    t_spec0,
    hop,
    f_spec0,
    df,
    tt,
    tpos,
    tvel,
    toff,
    tgap,
    ev_tc,
    ev_f0_lo,
    ev_f0_hi,
    ev_lo,
    ev_hi,
    ev_plateau,
    ev_excw,
    ev_half,
    ev_dt,
    ridge_t,
    ridge_f,
    ridge_off,
    ridge_h,
    broad_all,
    broad_off,
    obs,
    seg,
    tau,
    t0c,
    t0_offsets,
    c,
    harmonics,
    hweights,
    rel,
    slope_floor,
    min_valid,
    exc_z,
    min_cov,
    cov_z,
    imb_lambda,
    unexplained_score,
):
    """Combined ultra-coarse score of each candidate (``LockedResult.combined`` of the reference
    path with t0 chosen from the first explained event)."""
    B = obs.shape[0]
    E = seg.shape[1]
    out = np.full(B, NEG)
    for b in prange(B):
        t0s = t0c[b] + t0_offsets
        base = t0s[0]
        S = t0s.shape[0]
        shifts = base - t0s
        n_expl = 0
        for e in range(E):
            if seg[b, e] >= 0:
                n_expl += 1
        scores = np.empty(n_expl)
        covs = np.empty((n_expl, S))
        bests = np.empty((n_expl, S))
        j = 0
        for e in range(E):
            s = seg[b, e]
            if s < 0:
                continue
            a0, a1 = toff[s], toff[s + 1]
            tr_abs, D = _predict(
                tt[a0:a1],
                tpos[a0:a1],
                tvel[a0:a1],
                tgap[s],
                obs[b, 0],
                obs[b, 1],
                obs[b, 2],
                c,
                tau[b, e] - ev_half[e] - 60.0,
                tau[b, e] + ev_half[e] + 60.0,
                ev_dt[e],
            )
            t_rel = tr_abs - base
            r0, r1 = ridge_off[e], ridge_off[e + 1]
            best, bcov, exc = _event_curve_score(
                W,
                t_spec0,
                hop,
                f_spec0,
                df,
                t_rel,
                D,
                shifts,
                ev_tc[e],
                ev_lo[e],
                ev_hi[e],
                ev_f0_lo[e],
                ev_f0_hi[e],
                ridge_t[r0:r1],
                ridge_f[r0:r1],
                ridge_h[e],
                ev_excw[e],
                harmonics,
                hweights,
                rel,
                broad_all[broad_off[e] : broad_off[e + 1]],
                slope_floor,
                min_valid,
            )
            pen_exc = 0.0
            obs_r = ev_plateau[e]
            if np.isfinite(obs_r) and np.isfinite(exc) and obs_r > 1.0:
                pen_exc = exc_z * abs(np.log(max(exc, 1.0) / obs_r))
            for si in range(S):
                bests[j, si] = best[si] - pen_exc
                covs[j, si] = bcov[si]
            j += 1
        # t0 from the first explained event (cfg.t0_fit == "event1"): first maximum
        si_best = 0
        for si in range(1, S):
            if bests[0, si] > bests[0, si_best]:
                si_best = si
        tot = 0.0
        mx = -1e300
        mn = 1e300
        for j in range(n_expl):
            cv = covs[j, si_best]
            pen_cov = cov_z * max(0.0, min_cov - cv)
            sc = bests[j, si_best] - pen_cov
            scores[j] = sc
            tot += sc
            mx = max(mx, sc)
            mn = min(mn, sc)
        if n_expl == E:
            comb = tot / n_expl
        else:
            comb = (tot + unexplained_score * (E - n_expl)) / E
        if n_expl > 1:
            comb -= imb_lambda * 0.5 * (mx - mn)
        out[b] = comb
    return out


class UltraBatchScorer:
    """Prepared inputs for :func:`ultra_scores` from an engine context (built once per search)."""

    def __init__(self, ctx, cfg) -> None:
        from aerodoppler.search.score import f0_grid

        spec = ctx.spec
        self.W = np.ascontiguousarray(spec.whitened)
        self.t_spec0 = float(spec.times[0])
        self.hop = float(spec.hop_s)
        self.f_spec0 = float(spec.freqs[0])
        self.df = float(spec.df_hz)
        self.traj = TrajectoryTable.from_segments(ctx.segments)
        feats, events = ctx.feats, ctx.events
        half = np.array([0.5 * (f.t_end_s - f.t_start_s) + 30.0 for f in feats])
        rt, rf, roff, rh, broad, boff = [], [], [0], [], [], [0]
        for f, ev in zip(feats, events, strict=True):
            t_ = np.array([p.relative_t_s for p in ev.ridge_points], dtype=np.float64)
            f_ = np.array([p.frequency_hz for p in ev.ridge_points], dtype=np.float64)
            o_ = np.argsort(
                t_, kind="stable"
            )  # sorted for the merge walk; the median ignores order
            rt.append(t_[o_])
            rf.append(f_[o_])
            roff.append(roff[-1] + len(rt[-1]))
            rh.append(float(ev.ridge_harmonic))
            g = f0_grid(f.f0_lo_hz, f.f0_hi_hz, cfg.f0_step_fraction)
            broad.append(g)
            boff.append(boff[-1] + len(g))
        self.ev = EventTable(
            t_center=np.array([f.t_center_s for f in feats], dtype=np.float64),
            f0_lo=np.array([f.f0_lo_hz for f in feats], dtype=np.float64),
            f0_hi=np.array([f.f0_hi_hz for f in feats], dtype=np.float64),
            win_lo=np.array([f.t_start_s for f in feats], dtype=np.float64),
            win_hi=np.array([f.t_end_s for f in feats], dtype=np.float64),
            plateau=np.array(
                [np.nan if f.excursion_local is None else f.excursion_local for f in feats],
                dtype=np.float64,
            ),
            exc_window=np.full(len(feats), 60.0),
            half=half,
            dt=np.maximum(2.0, 2.0 * half / 150.0),
            ridge_t=np.concatenate(rt) if rt else np.empty(0),
            ridge_f=np.concatenate(rf) if rf else np.empty(0),
            ridge_off=np.asarray(roff, dtype=np.int64),
            ridge_h=np.asarray(rh, dtype=np.float64),
        )
        self.broad = np.concatenate(broad)
        self.broad_off = np.asarray(boff, dtype=np.int64)
        n_rel = int(np.ceil(cfg.f0_halfwidth_fraction / cfg.f0_step_fraction))
        self.rel = (1.0 + cfg.f0_step_fraction) ** np.arange(-n_rel, n_rel + 1)
        self.harmonics = np.asarray(cfg.harmonics, dtype=np.float64)
        self.hweights = np.asarray(cfg.harmonic_weights[: len(cfg.harmonics)], dtype=np.float64)
        self.cfg = cfg
        self.c = float(ctx.c.c_mps)
        if tuple(cfg.alphas) != (0.0,) or cfg.f0_polish_step_fraction > 0 or cfg.t0_fit != "event1":
            raise ValueError(
                "the batched screen implements alphas=(0,), no polish, t0 from event 1"
            )

    def score(
        self,
        obs: NDArray[np.float64],
        seg: NDArray[np.int64],
        tau: NDArray[np.float64],
        t0c: NDArray[np.float64],
        t0_offsets: NDArray[np.float64],
        unexplained_score: float,
    ) -> NDArray[np.float64]:
        cfg = self.cfg
        return ultra_scores(
            self.W,
            self.t_spec0,
            self.hop,
            self.f_spec0,
            self.df,
            self.traj.t,
            self.traj.pos,
            self.traj.vel,
            self.traj.off,
            self.traj.max_gap,
            self.ev.t_center,
            self.ev.f0_lo,
            self.ev.f0_hi,
            self.ev.win_lo,
            self.ev.win_hi,
            self.ev.plateau,
            self.ev.exc_window,
            self.ev.half,
            self.ev.dt,
            self.ev.ridge_t,
            self.ev.ridge_f,
            self.ev.ridge_off,
            self.ev.ridge_h,
            self.broad,
            self.broad_off,
            np.ascontiguousarray(obs, dtype=np.float64),
            np.ascontiguousarray(seg, dtype=np.int64),
            np.ascontiguousarray(tau, dtype=np.float64),
            np.ascontiguousarray(t0c, dtype=np.float64),
            np.ascontiguousarray(t0_offsets, dtype=np.float64),
            self.c,
            self.harmonics,
            self.hweights,
            self.rel,
            float(cfg.slope_weight_floor),
            float(cfg.min_valid_fraction),
            float(cfg.excursion_penalty_z),
            float(cfg.min_coverage),
            float(cfg.coverage_penalty_z),
            float(cfg.imbalance_lambda),
            float(unexplained_score),
        )
