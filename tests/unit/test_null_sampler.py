"""Chance-match null sampler: the reported failure (6,380 s recording, one 48 s event,
'0 valid random-clock samples') and the admissibility rules."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from aerodoppler.search.engine import SearchEngine


def _ctx(duration_s: float, windows: list[tuple[float, float]]):
    return SimpleNamespace(
        spec=SimpleNamespace(times=np.arange(0.0, duration_s, 0.25)),
        feats=[SimpleNamespace(t_start_s=a, t_end_s=b) for a, b in windows],
    )


def test_long_recording_short_event_gets_a_null():
    # the Central Park case: event window 13.3-61.5 s, predicted curve = window + 30 s around CPA
    ctx = _ctx(6380.0, [(13.3, 61.5)])
    centre = 20.5
    half = 0.5 * (61.5 - 13.3) + 30.0
    offs, diag = SearchEngine._null_offsets(ctx, seed=0, extents=[(centre - half, centre + half)])
    assert offs.size == 48, diag
    assert diag["band"] is not None
    lo, hi = diag["band"]
    assert lo >= 24.0 and hi < 80.0  # shifts beyond the curve's reach are never drawn
    assert np.all(np.abs(offs) >= 24.0)
    assert np.all(np.diff(np.sort(offs)) >= 1.0 - 1e-9)  # spaced, not 48 copies of one shift


def test_window_longer_than_curve_reach_is_not_estimable():
    # a window whose half-length exceeds what the curve can still cover after shifting
    ctx = _ctx(400.0, [(0.0, 390.0)])
    offs, diag = SearchEngine._null_offsets(ctx, seed=0, extents=[(-30.0, 60.0)])
    assert offs.size == 0 and diag["band"] is None and "rule" in diag


def test_offsets_respect_every_event():
    ctx = _ctx(1500.0, [(100.0, 200.0), (900.0, 960.0)])
    ext = [(80.0, 220.0), (870.0, 990.0)]
    offs, _ = SearchEngine._null_offsets(ctx, seed=3, extents=ext)
    assert offs.size > 0
    for (a, b), (c0, c1) in zip([(100.0, 200.0), (900.0, 960.0)], ext, strict=True):
        dur = b - a
        assert np.all(np.abs(offs) >= max(20.0, 0.5 * dur))
        cov = np.minimum(c1 - offs, b) - np.maximum(c0 - offs, a)
        assert np.all(cov >= 0.4 * dur - 1e-9)
