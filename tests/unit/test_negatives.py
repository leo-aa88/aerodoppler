"""Hard negatives: sounds that are not passing aircraft must not become Doppler evidence."""

from __future__ import annotations

import numpy as np
import pytest

from aerodoppler.audio.spectrogram import compute_spectrogram
from aerodoppler.audio.template_detect import detect_flyovers, flyovers_to_events
from aerodoppler.config import Settings
from aerodoppler.search.events import assess_event
from aerodoppler.validation import negatives as N


def _spec(x: np.ndarray):
    from scipy.signal import resample_poly

    y = resample_poly(x, 1, 4)  # 8 kHz -> 2 kHz analysis rate, as the ingest does
    return compute_spectrogram(y, 2000)


def test_harmonic_switch_of_a_steady_tone_is_not_a_sweep():
    """Regression: the template re-measurement stepped from a held tone's 2nd harmonic
    (360 Hz) to its fundamental (180 Hz) and reported an 89 % 'Doppler' drop."""
    x = N.held_tones(np.random.default_rng(3003))
    spec = _spec(x)
    rejected: list[dict] = []
    events = flyovers_to_events(spec, detect_flyovers(spec), "rec_x", rejected=rejected)
    assert not [e for e in events if assess_event(e)["label"] != "tonal_only"]
    assert any("steady_tones_joined" in r["reasons"] for r in rejected)


@pytest.mark.parametrize("name", ["rising_chirps", "engine_idle", "spin_down", "mains_hum"])
def test_negative_classes_yield_no_doppler_evidence_from_templates(name):
    spec = _spec(N.NEGATIVES[name](np.random.default_rng(7)))
    events = flyovers_to_events(spec, detect_flyovers(spec), "rec_x")
    assert all(assess_event(e)["label"] == "tonal_only" for e in events)


@pytest.mark.slow
def test_negative_benchmark_detection_level(tmp_path):
    s = Settings(data_dir=tmp_path / "data", offline=True)
    rep = N.run_negative_benchmark(s, tmp_path, search=False, log=lambda m: None)
    sm = rep["summary"]
    assert sm["negative_doppler_events"] == 0, rep["clips"]
    assert sm["positive_true_passes_found"] == [2, 2, 2]
