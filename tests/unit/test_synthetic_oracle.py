"""Independent simulator vs forward model (the synthetic oracle, SPEC §9/§13)."""

import numpy as np

from aerodoppler.audio.spectrogram import compute_spectrogram, ridge_from_peaks
from aerodoppler.audio.synthetic import (
    SyntheticFlight,
    SyntheticScene,
    flight_to_adsb_samples,
    straight_path,
    synthesize_scene,
    turning_path,
)
from aerodoppler.geo.altitude import to_ellipsoidal_height
from aerodoppler.geo.coordinates import LocalFrame, geodetic_to_ecef
from aerodoppler.physics.doppler import predict
from aerodoppler.physics.trajectory import Trajectory

C = 343.0


def _ridge_error(scene, flight, truth_f0, t_win, f_lo, f_hi, fs=4000, snr=25.0, seed=0):
    rng = np.random.default_rng(seed)
    audio, truth = synthesize_scene(scene, rng, snr_db_at_cpa=snr, hum_hz=None, gust_modulation=0.0)
    from scipy.signal import resample_poly

    a2 = resample_poly(audio.astype(np.float64), 1, fs // 2000)
    spec = compute_spectrogram(a2, 2000, whiten=True)
    tt, ff, vals = ridge_from_peaks(spec, *t_win, f_lo, f_hi)
    # forward model from ADS-B-like samples of the same flight (different interpolation!)
    samples = flight_to_adsb_samples(
        flight, rng, cadence_s=4.0, jitter_s=1.0, dropout_fraction=0.0, position_noise_m=0.0
    )
    lat = np.array([s.latitude for s in samples])
    lon = np.array([s.longitude for s in samples])
    h = to_ellipsoidal_height(
        np.array([s.alt_geom_ft for s in samples]) * 0.3048, "geometric_wgs84_hae", lat, lon
    ).h_m
    traj = Trajectory.from_geodetic(np.array([s.utc_unix_s for s in samples]), lat, lon, h)
    pred = predict(traj, scene.observer_ecef, C, dt_s=0.25)
    f_pred = np.interp(tt + scene.t_start_abs, pred.t_receive, pred.received_frequency(truth_f0))
    good = vals > 4.0
    assert good.mean() > 0.8, f"ridge not detected in enough frames ({good.mean():.2f})"
    err = ff[good] - f_pred[good]
    return err, spec


def test_straight_flyby_ridge_matches_forward_model():
    frame = LocalFrame(40.0, -75.0, 100.0)
    t0 = 1_700_000_000.0
    fl = SyntheticFlight(
        "A",
        "a00001",
        t0,
        t0 + 300,
        straight_path(frame, t0 + 150, np.array([0.0, 1200.0, 900.0]), 90.0, 65.0),
        95.0,
    )
    scene = SyntheticScene(40.0, -75.0, 100.0, frame, [fl], t0, 300.0, 4000, C)
    err, _ = _ridge_error(scene, fl, 95.0, (40, 260), 60.0, 130.0)
    assert np.percentile(np.abs(err), 90) < 0.6  # bin is 0.244 Hz
    assert abs(np.median(err)) < 0.25


def test_turning_flyby_with_drift_matches_forward_model():
    frame = LocalFrame(40.0, -75.0, 100.0)
    t0 = 1_700_000_000.0
    path = turning_path(
        frame,
        t0,
        np.array([-9000.0, 1500.0, 700.0]),
        90.0,
        70.0,
        300.0,
        turn_rate_deg_s=1.5,
        turn_start_s=120.0,
        turn_duration_s=60.0,
    )
    fl = SyntheticFlight("B", "a00002", t0, t0 + 300, path, 110.0, drift_per_s=2e-5)
    scene = SyntheticScene(40.0, -75.0, 100.0, frame, [fl], t0, 300.0, 4000, C)
    rng = np.random.default_rng(0)
    audio, _ = synthesize_scene(scene, rng, snr_db_at_cpa=25.0, hum_hz=None, gust_modulation=0.0)
    from scipy.signal import resample_poly

    spec = compute_spectrogram(resample_poly(audio.astype(np.float64), 1, 2), 2000)
    tt, ff, vals = ridge_from_peaks(spec, 40, 260, 80.0, 150.0)
    samples = flight_to_adsb_samples(
        fl, rng, cadence_s=4.0, jitter_s=1.0, dropout_fraction=0.0, position_noise_m=0.0
    )
    lat = np.array([s.latitude for s in samples])
    lon = np.array([s.longitude for s in samples])
    h = np.array([s.alt_geom_ft for s in samples]) * 0.3048
    traj = Trajectory.from_geodetic(np.array([s.utc_unix_s for s in samples]), lat, lon, h)
    pred = predict(traj, scene.observer_ecef, C, dt_s=0.25)
    f_src = fl.f0(pred.t_emit)
    f_pred = np.interp(tt + t0, pred.t_receive, pred.received_frequency(f_src))
    good = vals > 4.0
    assert good.mean() > 0.7
    err = ff[good] - f_pred[good]
    assert np.percentile(np.abs(err), 90) < 0.8
    # the Doppler curve of a turning aircraft is visibly non-symmetric: check model & data agree
    # better than a straight-line surrogate would
    straight = straight_path(frame, t0 + 150, np.array([0.0, 1500.0, 700.0]), 90.0, 70.0)
    fl_s = SyntheticFlight("S", "a00003", t0, t0 + 300, straight, 110.0)
    ss = flight_to_adsb_samples(
        fl_s, rng, cadence_s=4.0, jitter_s=0.0, dropout_fraction=0.0, position_noise_m=0.0
    )
    ts = Trajectory.from_geodetic(
        np.array([s.utc_unix_s for s in ss]),
        np.array([s.latitude for s in ss]),
        np.array([s.longitude for s in ss]),
        np.array([s.alt_geom_ft for s in ss]) * 0.3048,
    )
    ps = predict(ts, scene.observer_ecef, C, dt_s=0.25)
    f_str = np.interp(tt + t0, ps.t_receive, ps.received_frequency(110.0))
    assert np.median(np.abs(ff[good] - f_str[good])) > np.median(np.abs(err))


def test_adsb_like_samples_carry_both_altitudes():
    frame = LocalFrame(40.0, -75.0, 100.0)
    t0 = 1_700_000_000.0
    fl = SyntheticFlight(
        "A",
        "a00001",
        t0,
        t0 + 100,
        straight_path(frame, t0 + 50, np.array([0.0, 1200.0, 900.0]), 90.0, 65.0),
        95.0,
    )
    s = flight_to_adsb_samples(fl, np.random.default_rng(0), baro_offset_m=25.0)
    assert len(s) > 10
    geoid = to_ellipsoidal_height(0.0, "orthometric_geoid", s[0].latitude, s[0].longitude).h_m[()]
    diff_m = (s[0].alt_geom_ft - s[0].alt_baro_ft) * 0.3048
    assert abs(diff_m - (geoid - 25.0)) < 20.0  # feet rounding to 25 ft
    assert geodetic_to_ecef(s[0].latitude, s[0].longitude, 0.0).shape == (3,)
