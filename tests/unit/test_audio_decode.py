from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from aerodoppler.audio import decode


def test_ffmpeg_duration_guard_follows_input_path(tmp_path: Path, monkeypatch) -> None:
    media = tmp_path / "recording.mp3"
    media.write_bytes(b"test media")
    pcm = np.zeros((4, 2), dtype=np.float32).tobytes()
    commands: list[list[str]] = []

    monkeypatch.setattr(decode, "ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(decode, "ffprobe_duration", lambda _path: 7_484.0)
    monkeypatch.setattr(decode, "_probe_stream", lambda _path: (48_000, 2))

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=pcm, stderr=b"")

    monkeypatch.setattr(decode.subprocess, "run", run)

    result = decode.decode_audio(media, max_duration_s=60)

    command = commands[0]
    input_index = command.index("-i")
    duration_index = command.index("-t")
    assert command[input_index + 1] == str(media)
    assert duration_index > input_index + 1
    assert command[duration_index + 1] == "60"
    assert result.duration_s == 4 / 48_000
    assert result.source_duration_s == 7_484.0
    assert result.truncated is True


def test_libsndfile_caps_audio_at_duration_limit(tmp_path: Path) -> None:
    media = tmp_path / "recording.wav"
    sf.write(media, np.zeros(1_000, dtype=np.float32), 100)

    result = decode.decode_audio(media, max_duration_s=2)

    assert result.duration_s == 2
    assert result.source_duration_s == 10
    assert result.truncated is True
