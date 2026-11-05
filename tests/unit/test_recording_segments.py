"""Long recordings are analysed in disclosed, offset-correct segments; time metadata cannot
pretend to more precision than it has (offline; files generated in the test)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aerodoppler.audio.decode import AudioDecodeError, decode_audio
from aerodoppler.audio.service import (
    TimeMetadataError,
    check_time_metadata,
    create_segment,
    ingest_recording,
    segment_notice,
)
from aerodoppler.config import Settings
from aerodoppler.jobs.store import Store

SR = 200  # low rate keeps 2-hour fixtures small (7484 s x 200 Hz = 1.5 M samples)


def _wav(tmp: Path, seconds: float, name: str = "long.wav") -> Path:
    p = tmp / name
    t = np.arange(int(seconds * SR)) / SR
    sf.write(p, (0.1 * np.sin(2 * np.pi * 37.0 * t)).astype(np.float32), SR)
    return p


def _env(tmp: Path) -> tuple[Settings, Store]:
    s = Settings(data_dir=tmp / "data", offline=True)
    s.ensure_dirs()
    return s, Store(s.db_path)


@pytest.mark.parametrize(
    "seconds,truncated", [(7199.0, False), (7200.0, False), (7201.0, True), (7484.0, True)]
)
def test_limit_boundaries(tmp_path, seconds, truncated):
    d = decode_audio(_wav(tmp_path, seconds), max_duration_s=7200.0)
    assert d.truncated is truncated
    assert d.duration_s == pytest.approx(min(seconds, 7200.0))
    assert d.source_duration_s == pytest.approx(seconds)
    assert d.segment_start_s == 0.0


def test_long_file_is_accepted_and_rest_is_reachable(tmp_path):
    """The reported failure: a 7,484 s file was rejected for a 284 s overrun."""
    s, store = _env(tmp_path)
    src = _wav(tmp_path, 7484.0)
    first = ingest_recording(
        store, s, src, start_time_utc="2026-11-03T15:00:00Z", start_time_uncertainty_s=5.0
    )
    assert first.truncated and first.duration_s == pytest.approx(7200.0)
    assert "0–7200 s of 7484 s" in segment_notice(first)
    rest = create_segment(store, s, first.id, start_s=7200.0, duration_s=None)
    assert rest.segment_start_s == pytest.approx(7200.0)
    assert rest.duration_s == pytest.approx(284.0)
    assert not rest.truncated
    assert rest.parent_recording_id == first.id
    # UTC of the analysed interval is shifted by the offset; the source start is preserved
    assert rest.source_start_time_utc == "2026-11-03T15:00:00Z"
    assert rest.start_time_utc == "2026-11-03T17:00:00.000Z"
    assert rest.sha256 == first.sha256  # same source, verified by hash
    # the original is preserved (hard-linked, not re-copied) for both recordings
    o1 = next(store.recording_dir(first.id).glob("original_*"))
    o2 = next(store.recording_dir(rest.id).glob("original_*"))
    assert o1.stat().st_ino == o2.stat().st_ino or o1.read_bytes() == o2.read_bytes()


def test_segment_beyond_end_is_rejected(tmp_path):
    with pytest.raises(AudioDecodeError, match="beyond the end"):
        decode_audio(_wav(tmp_path, 10.0), start_s=11.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_segment_seek_is_time_correct(tmp_path):
    """A chirp f(t) = 300 + 10 t Hz: the dominant frequency of the decoded 20–25 s segment must
    be ~ 300 + 10 * 22.5 = 525 Hz, proving the seek lands at the right time."""
    sr = 8000
    t = np.arange(40 * sr) / sr
    phase = 2 * np.pi * (300 * t + 5 * t**2)
    wav = tmp_path / "chirp.wav"
    sf.write(wav, (0.5 * np.sin(phase)).astype(np.float32), sr)
    mp3 = tmp_path / "chirp.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(mp3)], check=True)
    d = decode_audio(mp3, start_s=20.0, duration_s=5.0)
    assert d.decoder == "ffmpeg" and d.truncated
    assert d.duration_s == pytest.approx(5.0, abs=0.05)
    x = d.samples[:, 0]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f_peak = np.fft.rfftfreq(len(x), 1 / d.sample_rate_hz)[np.argmax(spec)]
    assert f_peak == pytest.approx(525.0, abs=30.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_corrupt_mp3_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"ID3" + b"\x00" * 64 + b"definitely not mpeg audio" * 100)
    with pytest.raises(AudioDecodeError):
        decode_audio(bad)


@pytest.mark.parametrize(
    "args,field",
    [
        ((None, None, 30.0, None, None), "start_time_uncertainty_s"),  # uncertainty, no start
        ((None, "2026-11-03T15:30:00", 5.0, None, None), "start_time_utc"),  # no timezone
        ((None, "2026-11-03T15:30:00Z", None, None, None), "start_time_uncertainty_s"),
        (("2026-11-04", "2026-11-03T15:30:00Z", 5.0, None, None), "recording_date_utc"),
        ((None, None, None, "2026-11-05", "2026-11-01"), "date_earliest_utc"),
        (("2026-12-01", None, None, None, "2026-11-04"), "recording_date_utc"),
    ],
)
def test_time_metadata_rejects_pretend_precision(args, field):
    with pytest.raises(TimeMetadataError) as e:
        check_time_metadata(*args)
    assert e.value.field == field


def test_time_basis_is_recorded(tmp_path):
    s, store = _env(tmp_path)
    src = _wav(tmp_path, 5.0)
    assert ingest_recording(store, s, src).time_basis == "unknown"
    assert (
        ingest_recording(store, s, src, recording_date_utc="2026-11-03").time_basis == "date_only"
    )
    pub = ingest_recording(store, s, src, date_latest_utc="2026-11-04")
    assert pub.time_basis == "upper_bound_only" and pub.recording_date_utc is None
    rng = ingest_recording(
        store, s, src, date_earliest_utc="2026-11-01", date_latest_utc="2026-11-04"
    )
    assert rng.time_basis == "date_range"
    v = ingest_recording(
        store, s, src, start_time_utc="2026-11-03T15:30:00+01:00", start_time_uncertainty_s=2
    )
    assert v.time_basis == "verified_start" and v.start_time_utc == "2026-11-03T14:30:00.000Z"
