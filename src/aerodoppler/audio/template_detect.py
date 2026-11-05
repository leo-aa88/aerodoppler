"""Parametric flyover detector: grid search of an ideal straight constant-speed pass.

For weak recordings (phone audio in wind) a ridge is often only 1–3 z above the whitened
background per frame, below what a frame-by-frame tracker can follow, yet it extends over a
minute. Integrating the whitened map along a *physically shaped* curve recovers it: the
received frequency of a straight pass is a 4-parameter family (frequency at closest approach
``f_c``, speed ``v``, slant distance ``d``, reception time of CPA ``t_c``), evaluated with the
closed-form flyby of :func:`aerodoppler.physics.doppler.analytic_straight_flyby`. The grid
search keeps local maxima in ``t_c``, refines them continuously (Nelder–Mead) and re-measures
the ridge frame by frame around the fitted curve, so the returned event carries *observed*
ridge points, not the template. (Idea after mjshiggins/dopplertrack's ``detect``; this is an
independent implementation.) The fitted parameters only seed the search's generous bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.optimize import minimize

from aerodoppler.audio.spectrogram import Spectrogram
from aerodoppler.models import DopplerEvent, RidgePoint
from aerodoppler.physics.doppler import analytic_straight_flyby


@dataclass
class TemplateConfig:
    f_min_hz: float = 50.0
    f_max_hz: float = 450.0
    f_step_fraction: float = 0.015
    speeds_mps: tuple[float, ...] = (40.0, 55.0, 70.0, 90.0, 120.0, 160.0)
    distances_m: tuple[float, ...] = (300.0, 600.0, 1000.0, 1600.0, 2500.0, 4000.0)
    t_step_s: float = 2.0
    half_window_s: float = 45.0
    n_samples: int = 120
    min_score_z: float = 1.0
    min_separation_s: float = 60.0
    max_events: int = 8
    c_mps: float = 343.0
    harmonic2_weight: float = 0.5
    min_half_score_z: float = 0.3  # each half of the pass must show evidence


@njit(cache=True)
def _template_scores(W, t0_spec, hop, f0_spec, df, t_rel_tpl, ratio_tpl, f_vals, t_centres, h2w):
    """Mean whitened z along f = f_c * ratio(t - t_c) (+ optional second harmonic) for every
    (template, f_c, t_c). ratio_tpl: (P, n) curve shapes relative to f_c, t_rel_tpl: (P, n)."""
    T, F = W.shape
    P, n = ratio_tpl.shape
    K = len(f_vals)
    C = len(t_centres)
    out = np.full((P, K, C), -9.0)
    for p in range(P):
        for k in range(K):
            fc = f_vals[k]
            for ci in range(C):
                acc = 0.0
                cnt = 0
                for i in range(n):
                    tr = t_centres[ci] + t_rel_tpl[p, i]
                    ti = (tr - t0_spec) / hop
                    if ti < 0.0 or ti > T - 1.0:
                        continue
                    r0 = int(ti)
                    if r0 >= T - 1:
                        r0 = T - 2
                    fr = ti - r0
                    f = fc * ratio_tpl[p, i]
                    fi = (f - f0_spec) / df
                    if fi < 0.0 or fi > F - 1.0:
                        continue
                    c0 = int(fi)
                    if c0 >= F - 1:
                        c0 = F - 2
                    fcf = fi - c0
                    v = (
                        W[r0, c0] * (1 - fr) * (1 - fcf)
                        + W[r0, c0 + 1] * (1 - fr) * fcf
                        + W[r0 + 1, c0] * fr * (1 - fcf)
                        + W[r0 + 1, c0 + 1] * fr * fcf
                    )
                    if v < -2.0:
                        v = -2.0
                    elif v > 8.0:
                        v = 8.0
                    acc += v
                    cnt += 1
                    if h2w > 0.0:
                        f2 = 2.0 * f
                        fi2 = (f2 - f0_spec) / df
                        if fi2 >= 0.0 and fi2 <= F - 1.0:
                            c2 = int(fi2)
                            if c2 >= F - 1:
                                c2 = F - 2
                            fc2 = fi2 - c2
                            v2 = (
                                W[r0, c2] * (1 - fr) * (1 - fc2)
                                + W[r0, c2 + 1] * (1 - fr) * fc2
                                + W[r0 + 1, c2] * fr * (1 - fc2)
                                + W[r0 + 1, c2 + 1] * fr * fc2
                            )
                            if v2 < -2.0:
                                v2 = -2.0
                            elif v2 > 8.0:
                                v2 = 8.0
                            acc += h2w * v2
                if cnt >= n // 2:
                    out[p, k, ci] = acc / (cnt * (1.0 + h2w))
    return out


def _templates(
    cfg: TemplateConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[tuple[float, float]]]:
    """Curve shapes (ratio to f_c vs reception time relative to delayed CPA) for all (v, d)."""
    params = [(v, d) for v in cfg.speeds_mps for d in cfg.distances_m]
    t_rel = np.empty((len(params), cfg.n_samples))
    ratio = np.empty((len(params), cfg.n_samples))
    te = np.linspace(-cfg.half_window_s * 1.2, cfg.half_window_s * 1.2, cfg.n_samples)
    for i, (v, d) in enumerate(params):
        tr, fr = analytic_straight_flyby(te, v, d, 1.0, cfg.c_mps, t_cpa_emit=0.0)
        tr = tr - d / cfg.c_mps  # relative to the delayed CPA (reception time of the CPA emission)
        t_rel[i] = tr
        ratio[i] = fr
    return t_rel, ratio, params


@dataclass
class TemplateFit:
    f_c_hz: float
    speed_mps: float
    distance_m: float
    t_c_s: float
    score_z: float
    t_start_s: float
    t_end_s: float


def _curve_score(
    spec: Spectrogram, f_c: float, v: float, d: float, t_c: float, cfg: TemplateConfig
) -> float:
    te = np.linspace(-cfg.half_window_s * 1.2, cfg.half_window_s * 1.2, cfg.n_samples)
    tr, fr = analytic_straight_flyby(te, v, d, f_c, cfg.c_mps, t_cpa_emit=0.0)
    tr = tr - d / cfg.c_mps + t_c
    z = spec.sample_whitened(tr, fr, fill=np.nan)
    z2 = spec.sample_whitened(tr, 2 * fr, fill=np.nan) if cfg.harmonic2_weight > 0 else None
    ok = np.isfinite(z)
    if ok.sum() < cfg.n_samples // 2:
        return -9.0
    # both halves of the pass (approach, recede) must carry evidence: steady hum, clicks and
    # one-sided plateau pieces then cannot pass as a flyover (after fdop's matched filter)
    half = len(te) // 2
    a_ok, b_ok = ok[:half], ok[half:]
    if a_ok.sum() >= 10 and b_ok.sum() >= 10:
        za = np.clip(z[:half][a_ok], -2, 8).mean()
        zb = np.clip(z[half:][b_ok], -2, 8).mean()
        if min(za, zb) < cfg.min_half_score_z:
            return -9.0
    s = np.clip(z[ok], -2, 8).mean()
    if z2 is not None:
        ok2 = np.isfinite(z2)
        if ok2.any():
            s = (s + cfg.harmonic2_weight * np.clip(z2[ok2], -2, 8).mean()) / (
                1 + cfg.harmonic2_weight
            )
    return float(s)


def detect_flyovers(spec: Spectrogram, cfg: TemplateConfig | None = None) -> list[TemplateFit]:
    cfg = cfg or TemplateConfig()
    t_rel, ratio, params = _templates(cfg)
    n_f = int(np.ceil(np.log(cfg.f_max_hz / cfg.f_min_hz) / np.log(1 + cfg.f_step_fraction))) + 1
    f_vals = cfg.f_min_hz * (1 + cfg.f_step_fraction) ** np.arange(n_f)
    t_centres = np.arange(spec.times[0], spec.times[-1] + 1e-9, cfg.t_step_s)
    scores = _template_scores(
        spec.whitened,
        float(spec.times[0]),
        spec.hop_s,
        float(spec.freqs[0]),
        spec.df_hz,
        t_rel,
        ratio,
        f_vals,
        t_centres,
        cfg.harmonic2_weight,
    )
    best_c = scores.max(axis=(0, 1))  # (C,)
    fits: list[TemplateFit] = []
    order = np.argsort(-best_c)
    sep = max(1, int(round(cfg.min_separation_s / cfg.t_step_s)))
    taken = np.zeros(len(t_centres), dtype=bool)
    for ci in order:
        if best_c[ci] < cfg.min_score_z or len(fits) >= cfg.max_events:
            break
        if taken[max(0, ci - sep) : ci + sep + 1].any():
            continue
        p, k = np.unravel_index(int(np.argmax(scores[:, :, ci])), scores.shape[:2])
        v, d = params[p]
        x0 = np.array([np.log(f_vals[k]), np.log(v), np.log(d), t_centres[ci]])

        def obj(x: np.ndarray, spec: Spectrogram = spec, cfg: TemplateConfig = cfg) -> float:
            return -_curve_score(
                spec,
                float(np.exp(x[0])),
                float(np.clip(np.exp(x[1]), 20, 300)),
                float(np.clip(np.exp(x[2]), 100, 20000)),
                float(x[3]),
                cfg,
            )

        r = minimize(
            obj, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 300}
        )
        fc, vv, dd, tc = (
            float(np.exp(r.x[0])),
            float(np.clip(np.exp(r.x[1]), 20, 300)),
            float(np.clip(np.exp(r.x[2]), 100, 20000)),
            float(r.x[3]),
        )
        sc = -float(r.fun)
        if sc < cfg.min_score_z:
            continue
        taken[max(0, ci - sep) : ci + sep + 1] = True
        fits.append(
            TemplateFit(
                fc,
                vv,
                dd,
                tc,
                sc,
                max(spec.times[0], tc - cfg.half_window_s),
                min(spec.times[-1], tc + cfg.half_window_s),
            )
        )
    fits.sort(key=lambda f: f.t_c_s)
    return fits


def remeasure_ridge(
    spec: Spectrogram,
    fit: TemplateFit,
    cfg: TemplateConfig,
    band_hz: float = 4.0,
    min_z: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Observed ridge: per-frame peak of the whitened map within ±band of the fitted curve."""
    ti = np.where((spec.times >= fit.t_start_s) & (spec.times <= fit.t_end_s))[0]
    te = np.linspace(-cfg.half_window_s * 1.5, cfg.half_window_s * 1.5, 600)
    tr, fr = analytic_straight_flyby(
        te, fit.speed_mps, fit.distance_m, fit.f_c_hz, cfg.c_mps, t_cpa_emit=0.0
    )
    tr = tr - fit.distance_m / cfg.c_mps + fit.t_c_s
    f_pred = np.interp(spec.times[ti], tr, fr)
    out_t, out_f, out_z = [], [], []
    nb = max(1, int(round(band_hz / spec.df_hz)))
    for i, fp in zip(ti, f_pred, strict=True):
        k = int(round((fp - spec.freqs[0]) / spec.df_hz))
        lo, hi = max(0, k - nb), min(len(spec.freqs) - 1, k + nb)
        if hi <= lo:
            continue
        seg = spec.whitened[i, lo : hi + 1]
        j = int(np.argmax(seg))
        if seg[j] >= min_z:
            kk = lo + j
            delta = 0.0
            if 0 < kk < len(spec.freqs) - 1:
                y0, y1, y2 = (
                    spec.whitened[i, kk - 1],
                    spec.whitened[i, kk],
                    spec.whitened[i, kk + 1],
                )
                den = y0 - 2 * y1 + y2
                if den < 0:
                    delta = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            out_t.append(float(spec.times[i]))
            out_f.append(float(spec.freqs[kk] + delta * spec.df_hz))
            out_z.append(float(seg[j]))
    return np.array(out_t), np.array(out_f), np.array(out_z)


def flyovers_to_events(
    spec: Spectrogram, fits: list[TemplateFit], recording_id: str, cfg: TemplateConfig | None = None
) -> list[DopplerEvent]:
    cfg = cfg or TemplateConfig()
    events = []
    for fit in fits:
        t, f, z = remeasure_ridge(spec, fit, cfg)
        if len(t) < 10:
            continue
        unc = np.clip(
            2.0 * spec.df_hz / np.sqrt(np.maximum(z, 1.0)), 0.25 * spec.df_hz, 4 * spec.df_hz
        )
        quality = float(
            np.clip(1 - np.exp(-(fit.score_z - 0.5) * (fit.t_end_s - fit.t_start_s) / 60.0), 0, 1)
        )
        events.append(
            DopplerEvent(
                recording_id=recording_id,
                audio_start_s=float(max(0.0, t[0] - 2.0)),
                audio_end_s=float(t[-1] + 2.0),
                ridge_points=[
                    RidgePoint(
                        relative_t_s=float(a), frequency_hz=float(b), uncertainty_hz=float(u)
                    )
                    for a, b, u in zip(t, f, unc, strict=True)
                ],
                candidate_harmonic_numbers=[1, 2],
                f0_prior=fit.f_c_hz,
                detection_method="template-fit",
                detection_quality=quality,
                annotated_by="auto",
                ridge_harmonic=1,
                extra={
                    "fit": {
                        "f_c_hz": fit.f_c_hz,
                        "speed_mps": fit.speed_mps,
                        "distance_m": fit.distance_m,
                        "t_c_s": fit.t_c_s,
                        "score_z": fit.score_z,
                    },
                    "note": "straight-pass template fit seeds only; ridge points are re-measured per frame",
                    # per-point whitened level: points near noise level only echo the template's
                    # shape (peak-picking inside a band that follows the template), so the
                    # evidence assessment uses only the points clearly above noise
                    "point_z": [round(float(v), 2) for v in z],
                },
            )
        )
    return events
