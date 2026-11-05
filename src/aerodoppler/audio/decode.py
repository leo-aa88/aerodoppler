"""Audio ingestion (SPEC §8.1-8.2): decode WAV/FLAC/OGG via libsndfile, anything else
(MP3, MP4/MOV/MKV video …) via FFmpeg to float PCM. Original files are preserved; the
analysis copy is downmixed and resampled with an anti-alias polyphase filter. No speech
enhancement, gating or destructive high-pass filtering is applied in the detection path."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from aerodoppler.util import sha256_file

SNDFILE_EXT = {".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".w64", ".caf"}


class AudioDecodeError(RuntimeError):
    pass


@dataclass
class DecodedAudio:
    samples: NDArray[np.float32]  # (n, channels)
    sample_rate_hz: int
    channels: int
    duration_s: float
    sha256: str
    decoder: str
    original_filename: str
    source_duration_s: float | None = None
    truncated: bool = False


def ffmpeg_binary() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffprobe_duration(path: Path) -> float | None:
    probe = shutil.which("ffprobe")
    if not probe:
        return None
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def decode_audio(path: Path, max_duration_s: float | None = None) -> DecodedAudio:
    path = Path(path)
    if not path.exists():
        raise AudioDecodeError(f"file not found: {path}")
    digest = sha256_file(path)
    ext = path.suffix.lower()
    if ext in SNDFILE_EXT:
        try:
            info = sf.info(str(path))
            truncated = max_duration_s is not None and info.duration > max_duration_s
            frames = int(max_duration_s * info.samplerate) if truncated else -1
            data, sr = sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
        except AudioDecodeError:
            raise
        except Exception as e:
            raise AudioDecodeError(f"libsndfile failed on {path.name}: {e}") from e
        return DecodedAudio(
            data,
            int(sr),
            data.shape[1],
            data.shape[0] / sr,
            digest,
            "libsndfile",
            path.name,
            source_duration_s=info.duration,
            truncated=truncated,
        )
    ff = ffmpeg_binary()
    if ff is None:
        raise AudioDecodeError(
            "ffmpeg not available: install ffmpeg or `pip install imageio-ffmpeg`"
        )
    dur = ffprobe_duration(path)
    truncated = max_duration_s is not None and dur is not None and dur > max_duration_s
    # Probe stream parameters first so the native sample rate/channel count are preserved.
    sr, ch = _probe_stream(path)
    cmd = [
        ff,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
    ]
    if max_duration_s is not None:
        # This is an output duration cap, so it must follow the input path.  Putting it
        # between ``-i`` and the path makes FFmpeg interpret ``-t`` as the input filename.
        cmd.extend(["-t", str(max_duration_s)])
    cmd.extend(
        [
            "-vn",
            "-map",
            "0:a:0",
            "-ac",
            str(ch),
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired as e:
        raise AudioDecodeError("ffmpeg timed out") from e
    if proc.returncode != 0 or not proc.stdout:
        raise AudioDecodeError(
            f"ffmpeg failed on {path.name}: {proc.stderr.decode(errors='replace')[:400]}"
        )
    arr = np.frombuffer(proc.stdout, dtype=np.float32)
    arr = arr[: (len(arr) // ch) * ch].reshape(-1, ch)
    if max_duration_s is not None:
        max_frames = int(max_duration_s * sr)
        if dur is None and arr.shape[0] >= max_frames:
            truncated = True
        arr = arr[:max_frames]
    return DecodedAudio(
        arr.copy(),
        sr,
        ch,
        arr.shape[0] / sr,
        digest,
        "ffmpeg",
        path.name,
        source_duration_s=dur,
        truncated=truncated,
    )


def _probe_stream(path: Path) -> tuple[int, int]:
    probe = shutil.which("ffprobe")
    if probe:
        try:
            out = subprocess.run(
                [
                    probe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate,channels",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            st = json.loads(out.stdout)["streams"][0]
            return int(st["sample_rate"]), int(st["channels"])
        except Exception:
            pass
    return 48000, 2


def to_analysis_mono(
    audio: DecodedAudio, target_rate_hz: int = 2000, remove_dc: bool = True
) -> NDArray[np.float64]:
    """Downmix to mono and resample with an anti-aliasing polyphase filter."""
    x = audio.samples.astype(np.float64).mean(axis=1)
    if remove_dc:
        x = x - np.mean(x)
    if audio.sample_rate_hz != target_rate_hz:
        from math import gcd

        g = gcd(audio.sample_rate_hz, target_rate_hz)
        x = resample_poly(x, target_rate_hz // g, audio.sample_rate_hz // g)
    return x


def write_wav(
    path: Path, samples: NDArray[np.floating], sample_rate_hz: int, subtype: str = "PCM_16"
) -> None:
    sf.write(str(path), np.asarray(samples), sample_rate_hz, subtype=subtype)


def encode_with_ffmpeg(src_wav: Path, dst: Path, codec_args: list[str] | None = None) -> Path:
    """Re-encode (e.g. to MP3) to create codec-artifact test inputs."""
    ff = ffmpeg_binary()
    if ff is None:
        raise AudioDecodeError("ffmpeg not available")
    args = codec_args or ["-codec:a", "libmp3lame", "-b:a", "64k"]
    proc = subprocess.run(
        [ff, "-v", "error", "-nostdin", "-y", "-i", str(src_wav), *args, str(dst)],
        capture_output=True,
        timeout=600,
    )
    if proc.returncode != 0:
        raise AudioDecodeError(proc.stderr.decode(errors="replace")[:400])
    return dst
