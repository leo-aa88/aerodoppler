"""Locked two-event alignment regression (SPEC §13): independently sliding event 2 in time
would let a wrong flight pair / location win; the shared-t0 rule must prevent that."""

from __future__ import annotations

import numpy as np

from aerodoppler.audio.spectrogram import Spectrogram
from aerodoppler.search.score import (
    EventInputs,
    PredictedCurve,
    ScoreConfig,
    locked_score,
    score_event_curve,
)


def _synthetic_spec(duration_s=1200.0, hop=0.25, df=0.25, fmax=400.0):
    times = np.arange(0.0, duration_s, hop)
    freqs = np.arange(0.0, fmax, df)
    W = np.random.default_rng(0).normal(0.0, 1.0, (len(times), len(freqs))).astype(np.float32)
    return Spectrogram(times, freqs, W.copy(), W, {"hop_s": hop})


def _paint_ridge(spec: Spectrogram, t_rel, f_hz, z=8.0):
    ti = np.round((t_rel - spec.times[0]) / spec.hop_s).astype(int)
    fi = np.round((f_hz - spec.freqs[0]) / spec.df_hz).astype(int)
    ok = (ti >= 0) & (ti < len(spec.times)) & (fi >= 1) & (fi < len(spec.freqs) - 1)
    for a, b in zip(ti[ok], fi[ok], strict=False):
        spec.whitened[a, b - 1 : b + 2] = z


def _flyby_curve(t_abs_cpa, duration=300.0, v=70.0, d=1500.0, c=343.0, seg="s"):
    te = np.arange(t_abs_cpa - duration / 2, t_abs_cpa + duration / 2, 0.25)
    along = v * (te - t_abs_cpa)
    r = np.sqrt(along**2 + d**2)
    vt = -v * along / r
    return PredictedCurve(te + r / c, c / (c - vt), te, r, seg)


def test_independent_event2_alignment_would_pick_wrong_hypothesis():
    spec = _synthetic_spec()
    t0_true = 1_000_000.0  # absolute time of recording start
    f0a, f0b = 100.0, 120.0
    # truth: event 1 at rel 200 s, event 2 at rel 900 s (Δt = 700 s)
    ca_true = _flyby_curve(t0_true + 200.0)
    cb_true = _flyby_curve(t0_true + 900.0)
    for curve, f0 in ((ca_true, f0a), (cb_true, f0b)):
        tr = curve.t_receive_abs - t0_true
        for h in (1, 2):
            _paint_ridge(spec, tr, f0 * h * curve.factor)
    # observed ridge points (as a real DopplerEvent carries): fundamental of each flyby
    ra = ca_true.t_receive_abs - t0_true
    rb = cb_true.t_receive_abs - t0_true
    ev1 = EventInputs(
        200.0, 50.0, 350.0, 90.0, 110.0, ridge_t=ra[::8], ridge_f=f0a * ca_true.factor[::8]
    )
    ev2 = EventInputs(
        900.0, 750.0, 1050.0, 108.0, 132.0, ridge_t=rb[::8], ridge_f=f0b * cb_true.factor[::8]
    )
    cfg = ScoreConfig(
        f0_step_fraction=0.005, alphas=(0.0,), harmonics=(1, 2), harmonic_weights=(1.0, 0.8)
    )
    # Hypothesis H_true: pair whose predicted delayed CPAs are 700 s apart
    t0_grid = t0_true + np.arange(-20.0, 20.5, 1.0)
    true_res = locked_score(spec, [ca_true, cb_true], [ev1, ev2], t0_grid, cfg)
    # Hypothesis H_wrong: a flight pair whose second CPA is 60 s too early (Δt = 640 s):
    # geometrically inconsistent with the observed separation.
    cb_wrong = _flyby_curve(t0_true + 840.0, seg="wrong")
    wrong_res = locked_score(spec, [ca_true, cb_wrong], [ev1, ev2], t0_grid, cfg)
    assert true_res.combined > wrong_res.combined + 1.0
    assert abs(true_res.t0_abs - t0_true) <= 1.0
    # If event 2 were allowed its own independent time shift (the forbidden procedure), the
    # wrong hypothesis would score as well as the truth — the evidence in Δt is destroyed.
    best2_indep, *_ = score_event_curve(
        spec, cb_wrong, ev2, t0_true + np.arange(-80.0, 80.5, 1.0), cfg
    )
    assert best2_indep.max() > true_res.per_event[1].score - 0.3
    # ...while at the locked t0 the wrong pair's event 2 shows no ridge energy
    assert wrong_res.per_event[1].score < 1.0


def test_joint_fit_option_and_mode_b_window():
    spec = _synthetic_spec(duration_s=600.0)
    t0_true = 5_000_000.0
    ca = _flyby_curve(t0_true + 200.0)
    _paint_ridge(spec, ca.t_receive_abs - t0_true, 100.0 * ca.factor)
    ev1 = EventInputs(
        200.0,
        50.0,
        350.0,
        90.0,
        110.0,
        ridge_t=(ca.t_receive_abs - t0_true)[::8],
        ridge_f=100.0 * ca.factor[::8],
    )
    cfg = ScoreConfig(
        f0_step_fraction=0.005,
        alphas=(0.0,),
        harmonics=(1,),
        harmonic_weights=(1.0,),
        t0_fit="joint",
    )
    res = locked_score(spec, [ca], [ev1], t0_true + np.arange(-30.0, 30.5, 1.0), cfg)
    assert abs(res.t0_abs - t0_true) <= 1.0 and res.per_event[0].f0_hz == pytest_approx(100.0, 0.01)


def pytest_approx(v, rel):
    import pytest

    return pytest.approx(v, rel=rel)


def test_confidence_gates_grading():
    from aerodoppler.search.rank import gate_results

    good = dict(
        n_events=2,
        mode_known_time=False,
        per_event_scores=[4.0, 3.5],
        per_event_coverage=[0.9, 0.8],
        null_mean=[0.5, 0.4],
        null_std=[0.4, 0.4],
        own_clock_offsets_s=[0.0, 1.0],
        margin_distinct_flights_z=0.8,
        unheard_close_pass_penalty=0.0,
        landscape_radius_m=300.0,
    )
    grade, gates = gate_results(**good)
    assert grade == "high" and all(g["passed"] for g in gates)
    # one event does not beat chance -> low
    grade, _ = gate_results(**{**good, "per_event_scores": [4.0, 0.9]})
    assert grade == "low"
    # clock disagreement -> low
    grade, _ = gate_results(**{**good, "own_clock_offsets_s": [0.0, 7.0]})
    assert grade == "low"
    # thin margin -> medium; unheard close pass -> medium (all else fine)
    assert gate_results(**{**good, "margin_distinct_flights_z": 0.3})[0] == "medium"
    assert gate_results(**{**good, "unheard_close_pass_penalty": 0.6})[0] == "medium"
    # a mirror solution (same flight, other side of the track) within 0.1 z -> not high
    assert gate_results(**{**good, "margin_distinct_location_z": 0.1})[0] == "low"
    assert gate_results(**{**good, "margin_distinct_location_z": 0.3})[0] == "medium"
    # single event is enough in known-time mode
    assert (
        gate_results(
            **{
                **good,
                "mode_known_time": True,
                "per_event_scores": [4.0],
                "per_event_coverage": [0.9],
                "null_mean": [0.5],
                "null_std": [0.4],
                "own_clock_offsets_s": [0.0],
            }
        )[0]
        == "high"
    )
