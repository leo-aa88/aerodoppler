"""Forward-model unit tests (SPEC §9 and §13)."""

import numpy as np
import pytest

from aerodoppler.geo.coordinates import LocalFrame
from aerodoppler.physics.doppler import (
    analytic_straight_flyby,
    closest_approach,
    closest_approach_batch,
    predict,
)
from aerodoppler.physics.propagation import SoundSpeed, doppler_factor, speed_of_sound_mps
from aerodoppler.physics.trajectory import Trajectory

C = 343.0
FRAME = LocalFrame(40.0, -75.0, 0.0)


def straight_traj(
    speed=60.0,
    d=1500.0,
    alt=1000.0,
    t_cpa=1000.0,
    t_span=600.0,
    dt=5.0,
    heading_deg=90.0,
    t_offset=0.0,
):
    """Constant-velocity level flight; CPA (in the horizontal plane) at t_cpa, east-bound by default."""
    t = np.arange(t_cpa - t_span, t_cpa + t_span + dt, dt)
    h = np.radians(heading_deg)
    along = speed * (t - t_cpa)
    e = along * np.sin(h) + d * np.cos(h)
    n = along * np.cos(h) - d * np.sin(h)
    enu = np.stack([e, n, np.full_like(t, alt)], axis=1)
    return Trajectory.from_ecef(t + t_offset, FRAME.enu_to_ecef(enu), max_gap_s=60.0)


def test_speed_of_sound_default_and_temperature():
    assert speed_of_sound_mps(20.0) == pytest.approx(343.42, abs=0.01)
    assert speed_of_sound_mps(0.0) == pytest.approx(331.3)
    assert SoundSpeed.from_temperature(None).sigma_mps > SoundSpeed.from_temperature(20.0).sigma_mps
    with pytest.raises(ValueError):
        speed_of_sound_mps(500.0)


def test_doppler_sign_convention():
    c = 340.0
    assert doppler_factor(50.0, c) > 1.0  # approaching -> higher
    assert doppler_factor(-50.0, c) < 1.0  # receding -> lower
    assert doppler_factor(0.0, c) == 1.0
    with pytest.raises(ValueError):
        doppler_factor(400.0, c)


def test_analytic_straight_flyby_matches_forward_model():
    speed, d, alt = 60.0, 1500.0, 1000.0
    traj = straight_traj(speed=speed, d=d, alt=alt)
    obs = FRAME.enu_to_ecef(np.array([0.0, 0.0, 0.0]))
    pred = predict(traj, obs, C, dt_s=1.0)
    slant = np.hypot(d, alt)
    t_r_ref, f_r_ref = analytic_straight_flyby(
        pred.t_emit, speed, slant, 100.0, C, t_cpa_emit=1000.0
    )
    assert np.allclose(pred.t_receive, t_r_ref, atol=2e-3)
    assert np.allclose(pred.received_frequency(100.0), f_r_ref, rtol=2e-4)
    # approach then recede
    f = pred.received_frequency(100.0)
    assert f[0] > 100.0 > f[-1]
    assert f[0] == pytest.approx(
        100.0 * C / (C - speed * (60000.0 / np.hypot(60000.0, slant))), rel=1e-3
    )


def test_zero_radial_velocity_at_cpa_and_delayed_cpa():
    traj = straight_traj(speed=80.0, d=2000.0, alt=800.0, t_cpa=1000.0)
    obs = FRAME.enu_to_ecef(np.array([0.0, 0.0, 0.0]))
    cpa = closest_approach(traj, obs, C)
    assert cpa.t_emit_cpa == pytest.approx(1000.0, abs=0.05)
    assert cpa.slant_range_m == pytest.approx(np.hypot(2000.0, 800.0), rel=1e-4)
    assert cpa.t_receive_cpa == pytest.approx(1000.0 + cpa.slant_range_m / C, abs=0.05)
    pred = predict(traj, obs, C, t_emit=np.array([cpa.t_emit_cpa]))
    assert abs(pred.v_toward_mps[0]) < 0.5
    assert pred.factor[0] == pytest.approx(1.0, abs=2e-3)


def test_propagation_delay_and_monotonic_reception():
    traj = straight_traj(speed=120.0, d=500.0, alt=300.0)
    obs = FRAME.enu_to_ecef(np.array([0.0, 0.0, 0.0]))
    pred = predict(traj, obs, C, dt_s=0.5)
    assert np.allclose(pred.t_receive - pred.t_emit, pred.slant_range_m / C)
    assert pred.is_monotonic()
    # reception-time inversion round trip
    te = pred.emission_time_for_reception(pred.t_receive[10:-10])
    assert np.allclose(te, pred.t_emit[10:-10], atol=1e-6)


def test_clock_shift_invariance():
    a = straight_traj()
    b = straight_traj(t_offset=86400.0 * 3)
    obs = FRAME.enu_to_ecef(np.array([100.0, -50.0, 0.0]))
    pa = predict(a, obs, C, dt_s=1.0)
    pb = predict(b, obs, C, dt_s=1.0)
    assert np.allclose(pb.t_receive - pa.t_receive, 86400.0 * 3)
    assert np.allclose(pa.factor, pb.factor)


def test_emission_vs_reception_geometry_differs_on_curved_track():
    """Using aircraft state at reception time plus a constant shift is wrong (SPEC §9)."""
    # Circular arc: aircraft turning with radius 3 km at 70 m/s, observer near the arc.
    t = np.arange(0.0, 400.0, 2.0)
    r_turn, speed = 3000.0, 70.0
    ang = speed * (t - 200.0) / r_turn
    enu = np.stack(
        [r_turn * np.sin(ang), r_turn * np.cos(ang) - 2500.0, np.full_like(t, 600.0)], axis=1
    )
    traj = Trajectory.from_ecef(t, FRAME.enu_to_ecef(enu), max_gap_s=60.0)
    obs = FRAME.enu_to_ecef(np.zeros(3))
    correct = predict(traj, obs, C, dt_s=1.0)
    f_correct = correct.received_frequency(100.0)
    # naive: evaluate geometry at reception time, then shift by a constant delay
    naive_shift = float(np.median(correct.t_receive - correct.t_emit))
    t_naive_emit = correct.t_receive - naive_shift
    naive = predict(traj, obs, C, t_emit=t_naive_emit)
    f_naive = np.interp(
        correct.t_receive, naive.t_emit + naive_shift, naive.received_frequency(100.0)
    )
    assert np.max(np.abs(f_correct - f_naive)) > 0.5  # clearly distinguishable at 0.24 Hz bins


def test_trajectory_interpolation_continuity_and_gaps():
    traj = straight_traj(dt=10.0)
    tq = np.linspace(traj.t_start, traj.t_end, 5000)
    p = traj.position(tq)
    assert np.all(np.isfinite(p))
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert np.all(step < 15.0)  # 0.24 s * 60 m/s = 14.4 m
    # velocity is constant for a straight track regardless of smoothing
    assert np.allclose(np.linalg.norm(traj.vel, axis=1), 60.0, atol=1e-3)
    # no extrapolation
    assert np.all(np.isnan(traj.position([traj.t_start - 1.0, traj.t_end + 1.0])))
    # gap larger than max_gap -> NaN inside the gap, finite outside
    t = traj.t.copy()
    keep = (t < 900) | (t > 1100)
    g = Trajectory.from_ecef(t[keep], traj.pos[keep], max_gap_s=60.0)
    assert np.isnan(g.position([1000.0])).all()
    assert np.isfinite(g.position([800.0])).all()


def test_corrupted_points_and_duplicates_handled():
    traj = straight_traj(dt=5.0)
    t = np.concatenate([traj.t, [traj.t[5]]])  # duplicate timestamp
    pos = np.concatenate([traj.pos, [traj.pos[5]]])
    g = Trajectory.from_ecef(t, pos)
    assert len(g.t) == len(traj.t)
    with pytest.raises(ValueError):
        bad = traj.pos.copy()
        bad[3] = np.nan
        Trajectory.from_ecef(traj.t, bad)


def test_noisy_data_remains_finite():
    rng = np.random.default_rng(0)
    traj = straight_traj(dt=3.0)
    noisy = Trajectory.from_ecef(
        traj.t + rng.uniform(-0.4, 0.4, len(traj.t)), traj.pos + rng.normal(0, 15.0, traj.pos.shape)
    )
    obs = FRAME.enu_to_ecef(np.array([0.0, 0.0, 0.0]))
    pred = predict(noisy, obs, C, dt_s=0.5)
    assert np.all(np.isfinite(pred.factor)) and np.all(np.isfinite(pred.t_receive))
    assert np.all(np.abs(pred.v_toward_mps) < 120.0)
    assert pred.is_monotonic()


def test_zero_speed_degenerate():
    t = np.arange(0.0, 100.0, 5.0)
    pos = np.repeat(FRAME.enu_to_ecef(np.array([[1000.0, 0.0, 500.0]])), len(t), axis=0)
    traj = Trajectory.from_ecef(t, pos)
    pred = predict(traj, FRAME.enu_to_ecef(np.zeros(3)), C)
    assert np.allclose(pred.factor, 1.0)
    cpa = closest_approach(traj, FRAME.enu_to_ecef(np.zeros(3)), C)
    assert cpa.slant_range_m == pytest.approx(np.hypot(1000.0, 500.0), rel=1e-6)


def test_batch_cpa_matches_scalar():
    traj = straight_traj(speed=90.0, d=1200.0, alt=700.0)
    rng = np.random.default_rng(1)
    enu = np.stack(
        [rng.uniform(-5000, 5000, 50), rng.uniform(-5000, 5000, 50), np.zeros(50)], axis=1
    )
    obs = FRAME.enu_to_ecef(enu)
    te, r, tr = closest_approach_batch(traj, obs, C, chunk=7)
    for i in range(len(obs)):
        s = closest_approach(traj, obs[i], C)
        assert te[i] == pytest.approx(s.t_emit_cpa, abs=1e-6)
        assert r[i] == pytest.approx(s.slant_range_m, rel=1e-9)
        assert tr[i] == pytest.approx(s.t_receive_cpa, abs=1e-6)


def test_numba_cpa_kernel_matches_numpy_path_and_subsampling_is_close():
    from aerodoppler.physics.doppler import _cpa_kernel

    traj = straight_traj(speed=90.0, d=1200.0, alt=700.0, dt=3.0)
    rng = np.random.default_rng(2)
    n = 600  # large enough to take the numba path in closest_approach_batch
    enu = np.stack([rng.uniform(-8000, 8000, n), rng.uniform(-8000, 8000, n), np.zeros(n)], axis=1)
    obs = FRAME.enu_to_ecef(enu)
    te, r, tr = closest_approach_batch(traj, obs, C)
    te_k, r_k = _cpa_kernel(traj.pos, traj.t, traj._gap <= traj.max_gap_s, obs)
    assert np.allclose(te, te_k) and np.allclose(r, r_k)
    for i in range(0, n, 97):
        s = closest_approach(traj, obs[i], C)
        assert te[i] == pytest.approx(s.t_emit_cpa, abs=1e-6) and r[i] == pytest.approx(
            s.slant_range_m, rel=1e-9
        )
    coarse = traj.subsample(10.0)
    assert len(coarse.t) < len(traj.t)
    _, r_c, tr_c = closest_approach_batch(coarse, obs, C)
    assert np.max(np.abs(r_c - r)) < 5.0 and np.max(np.abs(tr_c - tr)) < 0.1


def test_glitchy_samples_do_not_produce_supersonic_velocities():
    """Near-duplicate timestamps with position noise (seen in real ADSB.lol traces)."""
    traj = straight_traj(dt=5.0)
    t = traj.t.copy()
    pos = traj.pos.copy()
    t = np.insert(t, 10, t[10] + 0.02)  # a sample 20 ms after another, 30 m off
    pos = np.insert(pos, 10, pos[10] + np.array([30.0, 0.0, 0.0]), axis=0)
    g = Trajectory.from_ecef(t, pos)
    assert np.all(np.linalg.norm(g.vel, axis=1) <= 400.0 + 1e-6)
    pred = predict(g, FRAME.enu_to_ecef(np.zeros(3)), C, dt_s=0.5)
    assert np.all(np.isfinite(pred.factor)) and np.all(np.abs(pred.v_toward_mps) < C)


def test_attach_extra_events_explains_matching_flight_only():
    from aerodoppler.search.engine import attach_extra_events
    from aerodoppler.search.events import EventFeatures
    from aerodoppler.search.preselect import Assignment, SparseCPA

    def feat(tc):
        return EventFeatures(
            "e",
            tc,
            100.0,
            1,
            110.0,
            90.0,
            1.2,
            30.0,
            0.5,
            100.0,
            tc - 50,
            tc + 50,
            80.0,
            120.0,
            1.2,
            25.0,
            200.0,
            100.0,
            15000.0,
            10,
            [],
        )

    feats = [feat(100.0), feat(700.0), feat(400.0)]
    # sparse CPA entries at grid point 5: segment 7 passes at tau = t0 + 400 (match), segment 8 at t0 + 480 (no match)
    t0 = 1_000_000.0
    sp = SparseCPA(
        np.array([7, 8, 7]),
        np.array([5, 5, 6]),
        np.zeros(3),
        np.array([1500.0, 1500.0, 1500.0]),
        np.array([t0 + 400.0, t0 + 480.0, t0 + 400.0]),
        np.array([60.0, 60.0, 60.0]),
    )
    a = Assignment(5, (1, 2), (t0 + 100.0, t0 + 700.0), (1000.0, 1200.0), t0, 0.0)
    b = Assignment(
        6, (1, 2), (t0 + 100.0, t0 + 700.0), (1000.0, 1200.0), t0 + 300.0, 0.0
    )  # at point 6 the only entry is 300 s off
    n = attach_extra_events([a, b], sp, feats, tol_s=20.0)
    assert a.seg_idx == (1, 2, 7) and abs(a.tau[2] - (t0 + 400.0)) < 1e-9
    assert b.seg_idx == (1, 2, -1) and n == 1
