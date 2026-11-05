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
    truncated: bool = False  # audio exists after the decoded interval
    segment_start_s: float = 0.0  # offset of the decoded interval in the source file


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


def decode_audio(
    path: Path,
    max_duration_s: float | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> DecodedAudio:
    """Decode ``[start_s, start_s + duration)`` of a file (whole file by default), never more
    than ``max_duration_s``. Only the requested interval is decoded (libsndfile seeks; FFmpeg
    uses input seeking), so analysing the last minutes of a long file is cheap.

    ``truncated`` is true when audio exists after the decoded interval; together with
    ``segment_start_s`` and ``source_duration_s`` it lets callers disclose exactly what was
    analysed and map segment times back to source times."""
    path = Path(path)
    if not path.exists():
        raise AudioDecodeError(f"file not found: {path}")
    if start_s < 0:
        raise AudioDecodeError("segment start must be >= 0 s")
    want = min(
        duration_s if duration_s is not None else float("inf"),
        max_duration_s if max_duration_s is not None else float("inf"),
    )
    if want <= 0:
        raise AudioDecodeError("segment duration must be > 0 s")
    digest = sha256_file(path)
    ext = path.suffix.lower()
    if ext in SNDFILE_EXT:
        try:
            info = sf.info(str(path))
            if start_s >= info.duration:
                raise AudioDecodeError(
                    f"segment start {start_s:.0f} s is beyond the end of the file ({info.duration:.0f} s)"
                )
            first = int(round(start_s * info.samplerate))
            frames = -1 if want == float("inf") else int(want * info.samplerate)
            data, sr = sf.read(
                str(path), start=first, frames=frames, dtype="float32", always_2d=True
            )
        except AudioDecodeError:
            raise
        except Exception as e:
            raise AudioDecodeError(f"libsndfile failed on {path.name}: {e}") from e
        seg_dur = data.shape[0] / sr
        return DecodedAudio(
            data,
            int(sr),
            data.shape[1],
            seg_dur,
            digest,
            "libsndfile",
            path.name,
            source_duration_s=info.duration,
            truncated=start_s + seg_dur < info.duration - 1.0 / sr,
            segment_start_s=first / info.samplerate,
        )
    ff = ffmpeg_binary()
    if ff is None:
        raise AudioDecodeError(
            "ffmpeg not available: install ffmpeg or `pip install imageio-ffmpeg`"
        )
    dur = ffprobe_duration(path)
    if dur is not None and start_s >= dur:
        raise AudioDecodeError(
            f"segment start {start_s:.0f} s is beyond the end of the file ({dur:.0f} s)"
        )
    # Probe stream parameters first so the native sample rate/channel count are preserved.
    sr, ch = _probe_stream(path)
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.6f}"]  # input seeking: frames before start are not decoded
    cmd += ["-i", str(path)]
    if want != float("inf"):
        # an output duration cap must follow the input path (between -i and the path FFmpeg
        # would read "-t" as the input filename)
        cmd += ["-t", f"{want:g}"]
    cmd += ["-vn", "-map", "0:a:0", "-ac", str(ch), "-ar", str(sr)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired as e:
        raise AudioDecodeError("ffmpeg timed out") from e
    if proc.returncode != 0:
        raise AudioDecodeError(
            f"ffmpeg could not decode {path.name}: {proc.stderr.decode(errors='replace')[-400:]}"
        )
    arr = np.frombuffer(proc.stdout, dtype=np.float32)
    arr = arr[: (len(arr) // ch) * ch].reshape(-1, ch)
    if want != float("inf"):
        arr = arr[: int(want * sr)]
    seg_dur = arr.shape[0] / sr if sr else 0.0
    if arr.shape[0] == 0:
        raise AudioDecodeError(f"{path.name}: no audio samples in the requested interval")
    if dur is not None:
        truncated = start_s + seg_dur < dur - 0.5
    else:
        truncated = want != float("inf") and seg_dur >= want - 1.0 / sr
    return DecodedAudio(
        arr.copy(),
        sr,
        ch,
        seg_dur,
        digest,
        "ffmpeg",
        path.name,
        source_duration_s=dur,
        truncated=truncated,
        segment_start_s=start_s,
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
