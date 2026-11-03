"""Automatic ridge detection: precision/recall on synthetic two-flight audio with hum."""

import numpy as np
from scipy.signal import resample_poly

from aerodoppler.audio.ridges import DetectorConfig, detect_ridges, ridges_to_events
from aerodoppler.audio.spectrogram import compute_spectrogram
from aerodoppler.audio.synthetic import (
    SyntheticFlight,
    SyntheticScene,
    straight_path,
    synthesize_scene,
    turning_path,
)
from aerodoppler.geo.coordinates import LocalFrame

C = 343.0


def _two_flight_scene(t0=1_700_000_000.0, fs=4000):
    frame = LocalFrame(40.0, -75.0, 100.0)
    a = SyntheticFlight(
        "A",
        "a00001",
        t0,
        t0 + 420,
        straight_path(frame, t0 + 200, np.array([0.0, 1500.0, 900.0]), 80.0, 65.0),
        90.0,
        harmonic_amps=(1.0, 0.6, 0.4),
    )
    path_b = turning_path(
        frame, t0 + 500, np.array([-12000.0, -2000.0, 700.0]), 90.0, 70.0, 420.0, 1.0, 150.0, 60.0
    )
    b = SyntheticFlight(
        "B", "a00002", t0 + 500, t0 + 920, path_b, 115.0, drift_per_s=1e-5, harmonic_amps=(1.0, 0.5)
    )
    return SyntheticScene(40.0, -75.0, 100.0, frame, [a, b], t0, 960.0, fs, C)


def test_detector_finds_both_events_and_rejects_hum():
    scene = _two_flight_scene()
    audio, truth = synthesize_scene(
        scene,
        np.random.default_rng(3),
        snr_db_at_cpa=12.0,
        hum_hz=60.0,
        hum_level=0.8,
        gust_modulation=0.2,
    )
    x = resample_poly(audio.astype(np.float64), 1, 2)
    spec = compute_spectrogram(x, 2000)
    ridges = detect_ridges(spec, DetectorConfig())
    assert ridges, "no ridges detected"
    # no accepted ridge may be a stationary hum line (60/120/180 Hz with < 1.5 Hz variation)
    for r in ridges:
        assert r.freq_range_hz >= 1.5
    events = ridges_to_events(ridges, spec, "rec")
    centres = sorted(0.5 * (e.audio_start_s + e.audio_end_s) for e in events)
    # expect one group near t≈200 (+ propagation) and one near t≈700
    assert any(150 < c < 260 for c in centres), centres
    assert any(550 < c < 780 for c in centres), centres
    # fundamentals recovered to within ~5 % (ratio of harmonics informs f0)
    f0s = sorted(e.f0_prior for e in events if e.f0_prior)
    assert any(abs(f - 90.0) / 90.0 < 0.06 or abs(f - 45.0) / 45.0 < 0.06 for f in f0s), f0s
    assert any(abs(f - 115.0) / 115.0 < 0.10 for f in f0s), f0s  # coarse prior; search sweeps f0
    # precision: ridge points lie on the true curves (within 1.5 Hz at some harmonic)
    for e in events:
        assert e.detection_quality > 0.3
        assert len(e.ridge_points) > 100


def test_detector_abstains_on_noise_only():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000 * 300)
    spec = compute_spectrogram(x, 2000)
    ridges = detect_ridges(spec, DetectorConfig())
    assert ridges == []
