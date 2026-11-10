"""Coarse-stage f0 polish width at its boundary: a pitch halfway between two coarse f0 grid
points (the farthest a pitch can be from its nearest grid point), strong, weak and very weak
ridges, two equal ridges, and a competing-ridge case where half width is known to fall short.

The coarse f0 grid steps by 1.5 % around the ridge-derived estimate; the polish re-sweeps
0.04 % steps around the best grid point. Half width (+-0.75 %) reaches any pitch within half a
grid step of the chosen grid point, so it equals full width (+-1.5 %) whenever the coarse sweep
picks a grid point adjacent to the peak. It does not when another ridge makes the coarse sweep
pick a grid point farther than half a step from the strongest peak: then half width scores the
weaker ridge. The coarse score only ranks candidates for the fine stage, which re-sweeps +-3 %
in 0.4 % steps with its own polish."""

from __future__ import annotations

import numpy as np
import pytest

from aerodoppler.audio.spectrogram import Spectrogram
from aerodoppler.models import QualitySettings
from aerodoppler.search.score import EventInputs, PredictedCurve, ScoreConfig, score_event_curve

T0 = 1.79e9
STEP = 0.015  # QualitySettings.coarse_f0_step_fraction
E = 100.0  # ridge-derived f0 estimate: the centre of the coarse f0 grid


def _D(t):
    return 1 + 0.06 * np.tanh(-(t - 150.0) / 12.0)


def _scene(ridges, seed=3):
    """Whitened noise plus harmonics 1-3 of every (f0, amplitude) ridge along one Doppler
    curve; the observed ridge points imply f0 = E exactly."""
    rng = np.random.default_rng(seed)
    times, freqs = np.arange(0, 300, 0.25), np.arange(20, 600, 0.5)
    W = rng.normal(0, 1, (len(times), len(freqs)))
    for f0, amp in ridges:
        for h in (1, 2, 3):
            fh = f0 * h * _D(times)
            W += amp * np.exp(-0.5 * ((freqs[None, :] - fh[:, None]) / 0.3) ** 2)
    spec = Spectrogram(times, freqs, np.zeros_like(W, np.float32), W.astype(np.float32))
    tc = np.arange(40, 260, 0.5)
    curve = PredictedCurve(T0 + tc, _D(tc), T0 + tc, np.full(len(tc), 1000.0), "s")
    rt = np.arange(80, 220, 1.0)
    ev = EventInputs(150.0, 80.0, 220.0, E * 0.8, E * 1.2, ridge_t=rt, ridge_f=E * _D(rt))
    return spec, curve, ev


def _coarse(scene, span, step=STEP):
    cfg = ScoreConfig(
        f0_step_fraction=step,
        harmonics=(1, 2, 3),
        harmonic_weights=(1.0, 0.8, 0.6),
        alphas=(0.0,),
        f0_polish_span=span,
        f0_polish_top_shifts=8 if span > 0 else 0,
    )
    best, ka, f0s, *_ = score_event_curve(*scene, np.array([T0]), cfg)
    return float(best[0]), float(f0s[0, ka[0][0]])


def _rel_err(f, f_true):
    return abs(f / f_true - 1.0)


@pytest.mark.parametrize("amp", [4.0, 1.0, 0.5])
@pytest.mark.parametrize("offset_steps", [0.0, 0.25, 0.5])
def test_half_width_equals_full_width_up_to_half_a_coarse_step(offset_steps, amp):
    f_true = E * (1 + STEP) ** offset_steps
    scene = _scene([(f_true, amp)])
    half, full = _coarse(scene, 0.5), _coarse(scene, 1.0)
    assert half == pytest.approx(full, abs=1e-12)
    if amp >= 1.0:  # the polished pitch is the true one (0.04 % polish steps)
        assert _rel_err(half[1], f_true) < 5e-4
    # (amp 0.5 halfway between grid points: both widths lose the ridge alike)


def test_a_narrower_polish_misses_the_halfway_pitch():
    """The test above is sensitive: a quarter-width or no polish does not reach the midpoint."""
    f_true = E * (1 + STEP) ** 0.5
    scene = _scene([(f_true, 4.0)])
    half = _coarse(scene, 0.5)
    for span in (0.25, 0.0):
        s, f = _coarse(scene, span)
        assert _rel_err(f, f_true) > 3e-3
        assert s < 0.5 * half[0]


def test_two_equal_ridges_halfway_between_grid_points():
    ridges = [(E * (1 + STEP) ** 0.5, 2.0), (E * (1 + STEP) ** -1.5, 2.0)]
    scene = _scene(ridges)
    half, full = _coarse(scene, 0.5), _coarse(scene, 1.0)
    assert half == pytest.approx(full, abs=1e-12)
    assert min(_rel_err(half[1], f) for f, _ in ridges) < 5e-4


@pytest.mark.parametrize("weak_amp", [2.0, 3.0])
def test_competing_ridge_known_limitation_of_half_width(weak_amp):
    """Known limitation: the ridge points sit on a weaker ridge at a grid point, and a stronger
    ridge lies 0.7 coarse steps away. Full width reaches the stronger ridge; half width stays on
    the weaker one (lower coarse score for this candidate, never a higher one)."""
    weak, strong = E, E * (1 + STEP) ** 0.7
    scene = _scene([(weak, weak_amp), (strong, 4.0)])
    half, full, off = _coarse(scene, 0.5), _coarse(scene, 1.0), _coarse(scene, 0.0)
    assert _rel_err(full[1], strong) < 5e-4
    assert _rel_err(half[1], weak) < 5e-4
    assert off[0] <= half[0] < full[0]
    if weak_amp == 2.0:
        # the fine stage (0.4 % steps, +-3 %, its own polish) recovers the stronger ridge here
        q = QualitySettings()
        s_fine, f_fine = _coarse(scene, 1.0, step=q.f0_step_fraction)
        assert _rel_err(f_fine, strong) < 5e-4
        assert s_fine > half[0]
