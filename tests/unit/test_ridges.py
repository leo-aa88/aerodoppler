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


def test_pick_event_pair_prefers_time_separated_credible_events():
    from aerodoppler.audio.service import pick_event_pair
    from aerodoppler.models import DopplerEvent, RidgePoint

    def ev(t0, f, q, nh):
        pts = [
            RidgePoint(relative_t_s=t0 + i, frequency_hz=f - 0.1 * i, uncertainty_hz=1.0)
            for i in range(60)
        ]
        return DopplerEvent(
            id=f"e{t0}",
            recording_id="r",
            audio_start_s=t0,
            audio_end_s=t0 + 60,
            ridge_points=pts,
            candidate_harmonic_numbers=list(range(1, nh + 1)),
            detection_quality=q,
        )

    a = ev(0, 100.0, 0.99, 5)  # strongest
    frag = ev(5, 200.0, 0.4, 1)  # overlapping fragment of the same flyby
    b = ev(900, 120.0, 0.9, 3)
    pair = pick_event_pair([frag, b, a])
    assert [e.id for e in pair] == ["e0", "e900"]
    assert len(pick_event_pair([a, frag])) == 1  # no time-separated partner
    assert pick_event_pair([]) == []


def test_merge_adjacent_ridge_pieces():
    import numpy as np

    from aerodoppler.audio.ridges import Ridge, _merge_adjacent

    t1 = np.arange(0.0, 50.0, 0.25)
    t2 = np.arange(51.0, 100.0, 0.25)
    r1 = Ridge(np.arange(len(t1)), t1, 100.0 - 0.1 * t1, np.full(len(t1), 5.0), 5.0, 50.0, 5.0, 1.0)
    r2 = Ridge(np.arange(len(t2)), t2, 100.0 - 0.1 * t2, np.full(len(t2), 4.0), 4.0, 49.0, 5.0, 1.0)
    far = Ridge(
        np.arange(len(t2)), t2, 300.0 - 0.1 * t2, np.full(len(t2), 4.0), 4.0, 49.0, 5.0, 1.0
    )
    merged = _merge_adjacent([r1, r2, far], hop=0.25, df=0.244, max_gap_s=6.0, max_df_bins=6.0)
    assert len(merged) == 2
    assert merged[0].times[-1] == t2[-1] and len(merged[0].times) == len(t1) + len(t2)


def test_pick_events_adds_time_separated_extras():
    from aerodoppler.audio.service import pick_events
    from aerodoppler.models import DopplerEvent, RidgePoint

    def ev(t0, f, q, nh, eid):
        pts = [
            RidgePoint(relative_t_s=t0 + i, frequency_hz=f - 0.1 * i, uncertainty_hz=1.0)
            for i in range(60)
        ]
        return DopplerEvent(
            id=eid,
            recording_id="r",
            audio_start_s=t0,
            audio_end_s=t0 + 60,
            ridge_points=pts,
            candidate_harmonic_numbers=list(range(1, nh + 1)),
            detection_quality=q,
        )

    evs = [
        ev(0, 100, 0.99, 5, "a"),
        ev(900, 120, 0.9, 3, "b"),
        ev(400, 90, 0.8, 2, "c"),
        ev(410, 180, 0.7, 1, "c2"),
        ev(1500, 110, 0.2, 1, "weak"),
    ]
    got = pick_events(evs, k=4)
    assert [e.id for e in got[:2]] == ["a", "b"]  # pairing events first, time ordered
    assert (
        "c" in [e.id for e in got]
        and "c2" not in [e.id for e in got]
        and "weak" not in [e.id for e in got]
    )
