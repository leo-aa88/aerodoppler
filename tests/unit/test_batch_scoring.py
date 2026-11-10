"""The batched ultra-coarse scorer must reproduce the reference per-candidate path
(``engine._score_point(..., ultra=True)``) on the same inputs: it is an optimisation, not a new
model. Random spectrogram, straight-ish tracks, ridge points and candidates, including an
unexplained third event and curves that leave the trajectory."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from aerodoppler.geo.coordinates import geodetic_to_ecef
from aerodoppler.models import SearchMode
from aerodoppler.physics.doppler import closest_approach
from aerodoppler.physics.trajectory import Trajectory
from aerodoppler.search.batch import BatchScorer
from aerodoppler.search.engine import UNEXPLAINED_EVENT_SCORE, _score_point, _t0_candidates
from aerodoppler.search.score import ScoreConfig

C = 343.0
T0 = 1.79e9


def _track(rng, lat0, lon0, t_mid):
    n, dt = 400, 1.0 + rng.uniform(0, 1.5)
    t = t_mid + (np.arange(n) - n / 2) * dt
    hd = np.radians(rng.uniform(0, 360) + np.cumsum(rng.normal(0, 0.2, n)))
    v = rng.uniform(35, 80)
    north = np.cumsum(v * dt * np.cos(hd))
    east = np.cumsum(v * dt * np.sin(hd))
    north -= north[n // 2]
    east -= east[n // 2]
    lat = lat0 + north / 111_320.0
    lon = lon0 + east / (111_320.0 * np.cos(np.radians(lat0)))
    alt = rng.uniform(200, 1500) + np.cumsum(rng.normal(0, 0.3, n))
    if rng.random() < 0.3:  # a data gap longer than max_gap_s: part of the curve is invalid
        t[n // 3 :] += 200.0
    return Trajectory.from_ecef(t, geodetic_to_ecef(lat, lon, alt).reshape(-1, 3))


def _world(seed=7):
    rng = np.random.default_rng(seed)
    hop, df = 0.25, 0.5
    times = np.arange(0, 1200, hop)
    freqs = np.arange(20, 900, df)
    W = rng.normal(0, 1, (len(times), len(freqs))).astype(np.float32)
    W[rng.random(W.shape) < 0.02] = 7.0  # bright speckle, so clipping and coverage matter
    spec = SimpleNamespace(whitened=W, times=times, hop_s=hop, freqs=freqs, df_hz=df)
    centres = [200.0, 520.0, 900.0]
    feats, events = [], []
    for k, tc in enumerate(centres):
        f0 = rng.uniform(70, 130)
        rt = tc + np.sort(rng.uniform(-60, 60, 120))
        rf = f0 * (1 + 0.08 * np.tanh(-(rt - tc) / 15.0)) * (1 + rng.normal(0, 0.003, len(rt)))
        if k == 1:
            rf *= 2.0  # this event's ridge is the 2nd harmonic
        feats.append(
            SimpleNamespace(
                t_center_s=tc,
                t_start_s=tc - 70.0,
                t_end_s=tc + 70.0,
                f0_lo_hz=f0 * 0.8,
                f0_hi_hz=f0 * 1.2,
                event_id=f"e{k}",
                excursion_local=float(rng.uniform(1.0, 1.2)) if k != 2 else None,
            )
        )
        events.append(
            SimpleNamespace(
                ridge_points=[
                    SimpleNamespace(relative_t_s=float(a), frequency_hz=float(b))
                    for a, b in zip(rt, rf, strict=True)
                ],
                ridge_harmonic=2 if k == 1 else 1,
            )
        )
    segs = []
    for _ in range(10):
        tr = _track(
            rng,
            29.0 + rng.uniform(-0.03, 0.03),
            -81.2 + rng.uniform(-0.03, 0.03),
            T0 + rng.uniform(0, 1000),
        )
        segs.append(SimpleNamespace(traj_build=SimpleNamespace(trajectory=tr)))
    cfg = ScoreConfig(
        f0_step_fraction=0.03,
        f0_halfwidth_fraction=0.03,
        harmonics=(1, 2),
        harmonic_weights=(1.0, 0.8),
        alphas=(0.0,),
        f0_polish_step_fraction=0.0,
    )
    # the engine's coarse-stage settings: 3 harmonics, 1.5 % f0 step, default f0 polish
    coarse = ScoreConfig(
        f0_step_fraction=0.015,
        harmonics=(1, 2, 3),
        harmonic_weights=(1.0, 0.8, 0.6),
        alphas=(0.0,),
    )
    ctx = SimpleNamespace(
        spec=spec,
        segments=segs,
        feats=feats,
        events=events,
        c=SimpleNamespace(c_mps=C),
        ultra_cfg=cfg,
        coarse_cfg=coarse,
        coarse_step_s=2.0,
        curve_dt_s=0.5,
        mode=SearchMode.UNKNOWN_TIME,
        sweep_s=24.0,
        sigma_t0=24.0,
        ultra_step_s=4.0,
        t0_known=None,
    )
    return rng, ctx, cfg


@pytest.mark.parametrize(
    ("ultra", "polish_span"), [(True, None), (False, 1.0), (False, 0.5)]
)  # coarse stage at full and half f0 polish width
@pytest.mark.parametrize("seed", [7, 11])
def test_batched_scores_match_the_reference_path(seed, ultra, polish_span):
    rng, ctx, cfg = _world(seed)
    if polish_span is not None:
        ctx.coarse_cfg = replace(ctx.coarse_cfg, f0_polish_span=polish_span)
    obs, seg, tau, t0c = [], [], [], []
    for _ in range(60):
        lat, lon = 29.0 + rng.uniform(-0.04, 0.04), -81.2 + rng.uniform(-0.04, 0.04)
        x = geodetic_to_ecef(lat, lon, 5.0).reshape(3)
        s = rng.choice(len(ctx.segments), 3, replace=False)
        ta = [
            closest_approach(ctx.segments[k].traj_build.trajectory, x, C).t_receive_cpa for k in s
        ]
        ta = [t + rng.normal(0, 5.0) for t in ta]
        if rng.random() < 0.3:
            s[2] = -1  # third event left unexplained by this hypothesis
        obs.append(x)
        seg.append(s)
        tau.append(ta)
        t0c.append(ta[0] - ctx.feats[0].t_center_s + rng.normal(0, 6.0))
    obs, seg, tau, t0c = map(np.asarray, (obs, seg, tau, t0c))
    offsets = _t0_candidates(ctx, 0.0, True, ultra)
    got = BatchScorer(ctx, cfg if ultra else ctx.coarse_cfg, ultra=ultra).score(
        obs, seg, tau, t0c, offsets, UNEXPLAINED_EVENT_SCORE
    )
    want = []
    for b in range(len(obs)):
        a = SimpleNamespace(
            seg_idx=tuple(int(v) for v in seg[b]), tau=tuple(tau[b]), t0_center=float(t0c[b])
        )
        want.append(_score_point(ctx, a, obs[b], coarse=True, ultra=ultra).combined)
    want = np.asarray(want)
    assert np.isfinite(want).all() and (want > -9.0).sum() >= 20  # meaningful, not all invalid
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)
