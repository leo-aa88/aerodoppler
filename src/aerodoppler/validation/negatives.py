"""Hard-negative benchmark: sounds that are NOT passing aircraft (SYNTHETIC, deterministic).

Each clip goes through the real path — ingest (decode, 2 kHz analysis copy), spectrogram,
automatic detection (ridge tracker + template fit), evidence labels — and, when it yields two
events that may enter a search, through a full unknown-time search against SYNTHETIC flights.
The question is never "is it detected" (many of these ARE tonal) but "does anything present it
as aircraft Doppler evidence, and does a search then produce anything but an abstention or a low
grade". Positive controls (synthetic aircraft scenes from the independent simulator, 10/4/0 dB)
run through the same path so a fix cannot simply switch detection off.

All audio here is generated: it stands in for the real-world classes (hum, machinery, music,
speech, wind, sirens, codec and noise-suppression artefacts) but is not a recording of them.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from aerodoppler.config import Settings

FS = 8000
DUR = 600.0


def _t(dur: float = DUR) -> np.ndarray:
    return np.arange(int(dur * FS)) / FS


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    from aerodoppler.audio.synthetic import coloured_noise

    return coloured_noise(n, FS, rng, alpha=1.0)


def _tone(f: np.ndarray, harmonics=(1.0, 0.5, 0.3)) -> np.ndarray:
    """Harmonic tone with instantaneous fundamental ``f`` (array, Hz)."""
    phase = 2 * np.pi * np.cumsum(f) / FS
    return sum(a * np.sin(k * phase) for k, a in enumerate(harmonics, start=1))


def _gate(t: np.ndarray, a: float, b: float, ramp: float = 2.0) -> np.ndarray:
    return np.clip((t - a) / ramp, 0, 1) * np.clip((b - t) / ramp, 0, 1)


# ---- generators: each returns float audio at FS ------------------------------------------


def mains_hum(rng):
    t = _t()
    f = np.full_like(t, 50.0 * (1 + 0.001 * np.sin(2 * np.pi * t / 300)))
    return 0.6 * _tone(f, (1.0, 0.6, 0.4, 0.3, 0.2)) + 0.3 * _pink(t.size, rng)


def held_tones(rng):
    """The Central Park case: a 329.5 Hz line for ~50 s, and a 180 Hz one later."""
    t = _t()
    f1 = 329.5 + 0.3 * np.sin(2 * np.pi * t / 7.0)
    f2 = 180.0 + 0.2 * np.sin(2 * np.pi * t / 5.0)
    x = _gate(t, 13, 62) * _tone(f1) + 0.8 * _gate(t, 200, 320) * _tone(f2)
    return x + 0.2 * _pink(t.size, rng)


def melody(rng):
    t = _t()
    notes = 220.0 * 2 ** (rng.integers(-5, 8, size=600) / 12.0)
    dur = rng.uniform(0.25, 1.5, size=600)
    starts = np.concatenate([[0.0], np.cumsum(dur)[:-1]])
    f = notes[np.clip(np.searchsorted(starts, t, side="right") - 1, 0, 599)]
    return 0.7 * _gate(t, 30, 400) * _tone(f, (1.0, 0.6, 0.4, 0.2)) + 0.2 * _pink(t.size, rng)


def speech_like(rng):
    t = _t()
    x = np.zeros_like(t)
    pos = 5.0
    while pos < DUR - 5:
        d = rng.uniform(0.12, 0.45)
        i0, i1 = int(pos * FS), int((pos + d) * FS)
        tt = t[i0:i1] - pos
        f0 = rng.uniform(100, 220) * (1 + rng.uniform(-0.15, 0.15) * tt / d)
        x[i0:i1] += np.hanning(i1 - i0) * _tone(f0, (1.0, 0.8, 0.6, 0.5, 0.4, 0.3))
        pos += d + rng.uniform(0.05, 0.6) + (rng.uniform(1, 4) if rng.random() < 0.15 else 0)
    return 0.6 * x + 0.15 * _pink(t.size, rng)


def wind(rng):
    from aerodoppler.audio.synthetic import coloured_noise

    t = _t()
    base = coloured_noise(t.size, FS, rng, alpha=2.0)
    gust = coloured_noise(t.size, FS, rng, alpha=2.5)
    gust = 1.0 + 1.5 * np.clip(gust / (np.abs(gust).max() + 1e-9), -0.6, 1.0)
    return base * gust


def rising_chirps(rng):
    """Doppler-sized pitch changes in the WRONG direction (pitch rises)."""
    t = _t()
    x = np.zeros_like(t)
    for c, f0 in ((120.0, 95.0), (330.0, 140.0), (500.0, 75.0)):
        f = f0 * (1.0 + 0.12 * np.tanh((t - c) / 15.0))
        x += _gate(t, c - 60, c + 60) * _tone(f)
    return x + 0.3 * _pink(t.size, rng)


def siren(rng):
    t = _t()
    f = 380.0 + 70.0 * np.sin(2 * np.pi * t / 4.0)
    return 0.6 * _gate(t, 100, 250) * _tone(f, (1.0, 0.4)) + 0.3 * _pink(t.size, rng)


def engine_idle(rng):
    t = _t()
    f = 31.0 * (1 + 0.004 * np.cumsum(rng.standard_normal(t.size)) / np.sqrt(FS * 50))
    return 0.6 * _tone(f, (1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)) + 0.3 * _pink(t.size, rng)


def spin_down(rng):
    """A machine slowing down: a descending tone, but only ~1 % over two minutes."""
    t = _t()
    f = 240.0 * (1.0 - 0.01 * np.clip((t - 150) / 120.0, 0, 1))
    return 0.6 * _gate(t, 100, 330) * _tone(f) + 0.3 * _pink(t.size, rng)


def ground_vehicle(rng):
    """A car passing 15 m away at 25 m/s: real Doppler, but over a few seconds."""
    t = _t()
    x = np.zeros_like(t)
    for c in (90.0, 260.0, 410.0):
        v, d, f0 = 25.0, 15.0, 110.0
        s = v * (t - c)
        vr = v * s / np.sqrt(s**2 + d**2)
        f = f0 * 343.0 / (343.0 + vr)
        amp = 1.0 / np.sqrt(1 + (s / 40.0) ** 2)
        x += amp * _gate(t, c - 15, c + 15, 1.0) * _tone(f, (1.0, 0.6, 0.4))
    return x + 0.2 * _pink(t.size, rng)


def noise_suppressed(rng):
    """Spectral-gated noise: isolated short tonal blips ('musical noise')."""
    t = _t()
    x = _pink(t.size, rng)
    n = 512
    frames = x[: (x.size // n) * n].reshape(-1, n)
    spec = np.fft.rfft(frames * np.hanning(n), axis=1)
    mag = np.abs(spec)
    spec[mag < np.percentile(mag, 97)] = 0.0
    return np.fft.irfft(spec, n=n, axis=1).reshape(-1)


def mp3_artifacts(rng):
    """Pink noise plus faint steady tones, through a 24 kbit/s MP3 round trip."""
    t = _t()
    x = (
        0.5 * _pink(t.size, rng)
        + 0.05 * _tone(np.full_like(t, 147.0))
        + 0.04 * _tone(np.full_like(t, 263.0))
    )
    return x  # encoded in run_negative_benchmark (needs ffmpeg)


NEGATIVES: dict[str, Callable[[np.random.Generator], np.ndarray]] = {
    "mains_hum": mains_hum,
    "held_tones": held_tones,
    "melody": melody,
    "speech_like": speech_like,
    "wind": wind,
    "rising_chirps": rising_chirps,
    "siren": siren,
    "engine_idle": engine_idle,
    "spin_down": spin_down,
    "ground_vehicle": ground_vehicle,
    "noise_suppressed": noise_suppressed,
    "mp3_artifacts": mp3_artifacts,
}


@dataclass
class ClipOutcome:
    clip: str
    kind: str  # negative | positive
    proposals: int
    labels: dict[str, int]
    doppler_events: int  # plausible + weak
    rejected: dict[str, int]
    search: dict[str, Any] | None
    seconds: float
    notes: dict[str, Any]


def _write(path: Path, x: np.ndarray) -> Path:
    x = x / (np.abs(x).max() * 1.05 + 1e-12)
    sf.write(path, x.astype(np.float32), FS)
    return path


def _mp3_roundtrip(wav: Path) -> Path:
    mp3 = wav.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "24k", str(mp3)], check=True
    )
    return mp3


def _analyse(
    settings: Settings,
    path: Path,
    date_utc: str,
    kind: str,
    name: str,
    scenario: Any | None,
    search: bool,
    workers: int,
) -> ClipOutcome:
    from aerodoppler.audio.service import ingest_recording, propose_events
    from aerodoppler.jobs.store import Store
    from aerodoppler.search.events import assess_event, usable_as_evidence

    t0 = time.time()
    store = Store(settings.db_path)
    rec = ingest_recording(store, settings, path, recording_date_utc=date_utc, synthetic=True)
    diag: dict[str, Any] = {}
    events = propose_events(store, settings, rec.id, diagnostics=diag)
    labels: dict[str, int] = {}
    usable = []
    for e in events:
        a = assess_event(e)
        labels[a["label"]] = labels.get(a["label"], 0) + 1
        if usable_as_evidence(a):
            usable.append(e)
    outcome: dict[str, Any] | None = None
    notes: dict[str, Any] = {"recording_id": rec.id}
    if kind == "positive" and scenario is not None:
        cpas = [scenario.truth[k]["t_cpa_emit_rel_s"] for k in ("event_a", "event_b")]
        hit = [
            any(
                e.audio_start_s - 20 <= c <= e.audio_end_s + 20
                and assess_event(e)["label"] == "plausible_doppler"
                for e in events
            )
            for c in cpas
        ]
        notes["true_passes_with_plausible_event"] = int(sum(hit))
    if search and scenario is not None:
        from aerodoppler.audio.service import pick_event_pair
        from aerodoppler.jobs.runner import run_search_inline
        from aerodoppler.models import QualitySettings, SearchMode, SearchRequest

        pair = pick_event_pair(usable)
        if len(pair) < 2:
            outcome = {
                "result": "refused before search",
                "reason": f"{len(usable)} usable event(s), 2 needed",
            }
        else:
            req = SearchRequest(
                mode=SearchMode.UNKNOWN_TIME,
                recording_id=rec.id,
                event_ids=[e.id for e in pair],
                date_utc=date_utc,
                bbox=scenario.spec.bbox,
                flight_dataset_ids=[scenario.dataset.dataset_id],
                quality_settings=QualitySettings(workers=workers),
            )
            try:
                res = run_search_inline(store, settings, req)
                grade = None if res.abstention.abstained else res.confidence_grade
                outcome = {
                    "result": "abstained" if res.abstention.abstained else f"answered ({grade})",
                    "grade": grade,
                    "abstention_reason": res.abstention.reason,
                }
                if kind == "positive" and res.candidates:
                    from aerodoppler.geo.coordinates import haversine_m

                    c = res.candidates[0]
                    outcome["top1_error_m"] = float(
                        haversine_m(
                            c.lat,
                            c.lon,
                            scenario.truth["observer_lat"],
                            scenario.truth["observer_lon"],
                        )
                    )
            except Exception as e:  # input refusals are a valid outcome for a negative
                outcome = {"result": "refused", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
    return ClipOutcome(
        clip=name,
        kind=kind,
        proposals=len(events),
        labels=labels,
        doppler_events=labels.get("plausible_doppler", 0) + labels.get("weak_doppler", 0),
        rejected=diag.get("rejected_counts", {}),
        search=outcome,
        seconds=round(time.time() - t0, 1),
        notes=notes,
    )


def run_negative_benchmark(
    settings: Settings,
    out_dir: Path,
    clips: list[str] | None = None,
    positives: tuple[float, ...] = (10.0, 4.0, 0.0),
    search: bool = True,
    workers: int = 4,
    seed: int = 3,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    from aerodoppler.validation.scenarios import ScenarioSpec, build_scenario

    out_dir.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    # one SYNTHETIC flight world serves as the search target for every clip (negatives are
    # searched against real-looking passes they did not record)
    world_spec = ScenarioSpec(name="neg_world", seed=seed, snr_db=10.0, duration_s=1500.0)
    world = build_scenario(world_spec, out_dir / "world", settings.normalized_dir, settings.dem_dir)
    rows: list[ClipOutcome] = []
    for i, name in enumerate(clips or list(NEGATIVES)):
        rng = np.random.default_rng(seed * 1000 + i)
        wav = _write(audio_dir / f"{name}.wav", NEGATIVES[name](rng))
        path = _mp3_roundtrip(wav) if name == "mp3_artifacts" else wav
        o = _analyse(settings, path, world_spec.date_utc, "negative", name, world, search, workers)
        rows.append(o)
        log(f"[negatives] {name}: {o.proposals} proposals, labels {o.labels}, search {o.search}")
    for snr in positives:
        spec = ScenarioSpec(
            name=f"pos_{int(snr)}db", seed=seed + int(snr), snr_db=snr, duration_s=1500.0
        )
        sc = build_scenario(spec, out_dir / spec.name, settings.normalized_dir, settings.dem_dir)
        o = _analyse(
            settings, Path(sc.wav_path), spec.date_utc, "positive", spec.name, sc, search, workers
        )
        rows.append(o)
        log(
            f"[negatives] positive {spec.name}: {o.proposals} proposals, labels {o.labels}, {o.notes}, search {o.search}"
        )
    neg = [r for r in rows if r.kind == "negative"]
    pos = [r for r in rows if r.kind == "positive"]
    summary = {
        "label": "SYNTHETIC HARD-NEGATIVE BENCHMARK — generated stand-ins for non-aircraft sounds",
        "negatives": len(neg),
        "negative_clips_with_doppler_evidence": sum(1 for r in neg if r.doppler_events),
        "negative_doppler_events": sum(r.doppler_events for r in neg),
        "negative_searches_not_refused_or_abstained": [
            r.clip for r in neg if r.search and r.search["result"].startswith("answered")
        ],
        "negative_searches_graded_above_low": [
            r.clip for r in neg if r.search and r.search.get("grade") in ("medium", "high")
        ],
        "positives": len(pos),
        "positive_true_passes_found": [
            r.notes.get("true_passes_with_plausible_event") for r in pos
        ],
        "positive_search": [r.search for r in pos],
    }
    report = {"summary": summary, "clips": [r.__dict__ for r in rows]}
    (out_dir / "negative_benchmark.json").write_text(json.dumps(report, indent=2, default=str))
    return report
