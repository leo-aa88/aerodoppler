"""The compact candidate pool (sparse-CPA entry indices, pairs built block by block) must give
exactly the candidates, order and values of the one-shot construction it replaced."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aerodoppler.search import preselect
from aerodoppler.search.preselect import SparseCPA, mode_a_pairs, mode_b_matches


def _sp(rng, n=4000, n_seg=40, n_pts=60):
    return SparseCPA(
        rng.integers(0, n_seg, n).astype(np.int32),
        rng.integers(0, n_pts, n).astype(np.int32),
        rng.uniform(200, 8000, n),
        1.79e9 + rng.uniform(0, 3000, n),
        rng.uniform(30, 130, n),
    )


def _feat(tc, v=(30.0, 120.0), d=(200.0, 7000.0)):
    return SimpleNamespace(
        t_center_s=tc,
        v_min_mps=v[0],
        v_max_mps=v[1],
        d_min_m=d[0],
        d_max_m=d[1],
        f_center_hz=110.0,
        slope_max_hz_per_s=0.8,
    )


def _segments(n_seg, rng):
    return [SimpleNamespace(class_prior=float(rng.choice([0.0, -0.1, -0.4]))) for _ in range(n_seg)]


def _reference_mode_a(sp, segs, feats, dt, tol, cap, c):
    """The previous construction: all pairs at once, copies of every field."""
    f1, f2 = feats
    ok1 = (sp.speed >= f1.v_min_mps) & (sp.speed <= f1.v_max_mps)
    ok1 &= (sp.slant >= f1.d_min_m) & (sp.slant <= f1.d_max_m)
    ok2 = (sp.speed >= f2.v_min_mps) & (sp.speed <= f2.v_max_mps)
    ok2 &= (sp.slant >= f2.d_min_m) & (sp.slant <= f2.d_max_m)
    a_idx, b_idx = np.flatnonzero(ok1), np.flatnonzero(ok2)
    tbase = float(np.min(sp.tau))
    key_b = sp.point_idx[b_idx].astype(np.float64) * 1e6 + (sp.tau[b_idx] - tbase)
    order = np.argsort(key_b)
    key_b_sorted, b_sorted = key_b[order], b_idx[order]
    key_a = sp.point_idx[a_idx].astype(np.float64) * 1e6 + (sp.tau[a_idx] - tbase)
    lo = np.searchsorted(key_b_sorted, key_a + dt - tol, side="left")
    hi = np.searchsorted(key_b_sorted, key_a + dt + tol, side="right")
    counts = hi - lo
    n = int(counts.sum())
    rep_a = np.repeat(np.arange(len(a_idx)), counts)
    offsets = np.arange(n) - np.repeat(np.cumsum(counts) - counts, counts)
    ia, ib = a_idx[rep_a], b_sorted[np.repeat(lo, counts) + offsets]
    same = sp.seg_idx[ia] == sp.seg_idx[ib]
    ia, ib = ia[~same], ib[~same]
    cost = preselect._slope_prior(sp, ia, f1, c) + preselect._slope_prior(sp, ib, f2, c)
    order = np.argsort(cost, kind="stable")[:cap]
    ia, ib, cost = ia[order], ib[order], cost[order]
    prior = np.array([s.class_prior for s in segs])
    return dict(
        point_idx=sp.point_idx[ia].astype(np.int64),
        seg=np.stack([sp.seg_idx[ia], sp.seg_idx[ib]], axis=1).astype(np.int64),
        tau=np.stack([sp.tau[ia], sp.tau[ib]], axis=1),
        slant=np.stack([sp.slant[ia], sp.slant[ib]], axis=1),
        t0_center=sp.tau[ia] - f1.t_center_s,
        class_prior=0.5 * (prior[sp.seg_idx[ia]] + prior[sp.seg_idx[ib]]),
        cost=cost,
    )


@pytest.mark.parametrize("block", [7, 1000, 4_000_000])
@pytest.mark.parametrize("cap", [10**9, 500])
def test_mode_a_pool_equals_the_one_shot_construction(monkeypatch, block, cap):
    rng = np.random.default_rng(5)
    sp = _sp(rng)
    segs = _segments(40, rng)
    feats = [_feat(100.0), _feat(400.0, v=(40.0, 130.0), d=(300.0, 8000.0))]
    monkeypatch.setattr(preselect, "PAIR_BLOCK", block)
    pool = mode_a_pairs(sp, segs, feats, 300.0, 20.0, [], cap, [], c_mps=343.0)
    want = _reference_mode_a(sp, segs, feats, 300.0, 20.0, cap, 343.0)
    assert len(pool) == len(want["cost"]) > 100
    point_idx, seg, tau, slant, t0c, prior = pool.rows(slice(0, len(pool)))
    np.testing.assert_array_equal(point_idx, want["point_idx"])
    np.testing.assert_array_equal(seg, want["seg"])
    np.testing.assert_array_equal(tau, want["tau"])
    np.testing.assert_array_equal(slant, want["slant"])
    np.testing.assert_array_equal(t0c, want["t0_center"])
    np.testing.assert_array_equal(prior, want["class_prior"])
    np.testing.assert_array_equal(pool.cost, want["cost"])
    # objects for a slice and for chosen indices carry the same values
    objs = pool.take(3, 9) + pool.take_indices(np.array([0, len(pool) - 1]))
    for a, i in zip(objs, [3, 4, 5, 6, 7, 8, 0, len(pool) - 1], strict=True):
        assert a.point_idx == want["point_idx"][i]
        assert a.seg_idx == tuple(want["seg"][i]) and a.tau == tuple(want["tau"][i])
        assert a.t0_center == want["t0_center"][i]
        assert a.class_prior == want["class_prior"][i]


def test_mode_b_rows_share_one_grid_point_and_the_known_start():
    rng = np.random.default_rng(9)
    sp = _sp(rng, n=3000)
    segs = _segments(40, rng)
    feats = [_feat(100.0), _feat(400.0)]
    t0 = 1.79e9 + 1000.0
    pool = mode_b_matches(sp, segs, feats, t0, 30.0, 20.0, [], 10**9, [], c_mps=343.0)
    assert len(pool) > 10
    point_idx, seg, tau, _slant, t0c, _prior = pool.rows(slice(0, len(pool)))
    e = pool.ent
    assert (sp.point_idx[e[:, 0]] == sp.point_idx[e[:, 1]]).all()
    np.testing.assert_array_equal(point_idx, sp.point_idx[e[:, 0]])
    assert (t0c == t0).all()
    assert (seg[:, 0] != seg[:, 1]).all()
    for k, f in enumerate(feats):
        assert (np.abs(tau[:, k] - (t0 + f.t_center_s)) <= 50.0).all()
