"""Recording-level operations shared by the CLI and the API: ingest, spectrogram, automatic
event proposals, manual events, purge (SPEC §8, §11)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from aerodoppler.audio.decode import AudioDecodeError, decode_audio, to_analysis_mono, write_wav
from aerodoppler.audio.ridges import DetectorConfig, detect_ridges, ridges_to_events
from aerodoppler.audio.spectrogram import Spectrogram, compute_spectrogram
from aerodoppler.config import Settings
from aerodoppler.jobs.store import Store
from aerodoppler.models import DopplerEvent, Recording, RidgePoint
from aerodoppler.util import new_id, safe_filename


class AuthorizationRequired(PermissionError):
    pass


class NotFound(KeyError):
    pass


def ingest_recording(
    store: Store,
    settings: Settings,
    path: Path,
    *,
    owner_permission_confirmed: bool,
    recording_date_utc: str | None = None,
    start_time_utc: str | None = None,
    start_time_uncertainty_s: float | None = None,
    synthetic: bool = False,
    notes: str | None = None,
    keep_original: bool = True,
) -> Recording:
    if not owner_permission_confirmed:
        raise AuthorizationRequired(
            "owner permission must be confirmed before a recording is ingested"
        )
    size_mb = Path(path).stat().st_size / 1e6
    if size_mb > settings.max_upload_mb:
        raise AudioDecodeError(f"file is {size_mb:.0f} MB; limit is {settings.max_upload_mb} MB")
    dec = decode_audio(Path(path), max_duration_s=settings.max_audio_duration_s)
    rec_id = new_id("rec")
    rec_dir = settings.recordings_dir / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    if keep_original:
        shutil.copy2(path, rec_dir / ("original_" + safe_filename(dec.original_filename)))
    mono = to_analysis_mono(dec, settings.analysis_sample_rate_hz)
    write_wav(
        rec_dir / "analysis_2khz.wav",
        mono.astype(np.float32),
        settings.analysis_sample_rate_hz,
        subtype="FLOAT",
    )
    rec = Recording(
        id=rec_id,
        sha256=dec.sha256,
        original_filename=dec.original_filename,
        duration_s=dec.duration_s,
        sample_rate_hz=dec.sample_rate_hz,
        channels=dec.channels,
        start_time_utc=start_time_utc,
        start_time_uncertainty_s=start_time_uncertainty_s,
        recording_date_utc=recording_date_utc or (start_time_utc[:10] if start_time_utc else None),
        owner_permission_confirmed=True,
        synthetic=synthetic,
        notes=notes,
    )
    (rec_dir / "recording.json").write_text(rec.model_dump_json(indent=2))
    (rec_dir / "decode.json").write_text(
        json.dumps(
            {
                "decoder": dec.decoder,
                "analysis_rate_hz": settings.analysis_sample_rate_hz,
                "original_kept": keep_original,
            },
            indent=2,
        )
    )
    store.add_recording(rec, rec_dir)
    return rec


def analysis_audio(store: Store, rec_id: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    d = store.recording_dir(rec_id)
    if d is None:
        raise NotFound(rec_id)
    x, sr = sf.read(str(d / "analysis_2khz.wav"), dtype="float64")
    return x, int(sr)


def ensure_spectrogram(
    store: Store, settings: Settings, rec_id: str, force: bool = False, **params: Any
) -> Spectrogram:
    p = store.spectrogram_path(rec_id)
    if p and p.exists() and not force and not params:
        return Spectrogram.load(p)
    d = store.recording_dir(rec_id)
    if d is None:
        raise NotFound(rec_id)
    x, sr = analysis_audio(store, rec_id)
    spec = compute_spectrogram(
        x,
        sr,
        window_s=params.get("window_s", settings.stft_window_s),
        hop_s=params.get("hop_s", settings.stft_hop_s),
        fmax_hz=params.get("fmax_hz", settings.max_analysis_freq_hz),
    )
    out = d / "spectrogram.npz"
    spec.save(out)
    store.set_spectrogram_path(rec_id, out)
    store.audit("recording.spectrogram", rec_id, spec.params)
    return spec


def propose_events(
    store: Store,
    settings: Settings,
    rec_id: str,
    cfg: DetectorConfig | None = None,
    save: bool = True,
) -> list[DopplerEvent]:
    spec = ensure_spectrogram(store, settings, rec_id)
    ridges = detect_ridges(spec, cfg)
    events = ridges_to_events(ridges, spec, rec_id)
    if save:
        events = [store.add_event(e) for e in events]
    return events


def add_manual_event(
    store: Store,
    rec_id: str,
    points: list[tuple[float, float]],
    harmonic: int = 1,
    uncertainty_hz: float = 1.0,
    annotated_by: str = "operator",
    pad_s: float = 2.0,
) -> DopplerEvent:
    if store.get_recording(rec_id) is None:
        raise NotFound(rec_id)
    if len(points) < 3:
        raise ValueError("a manual event needs at least three ridge points")
    pts = [
        RidgePoint(relative_t_s=t, frequency_hz=f, uncertainty_hz=uncertainty_hz) for t, f in points
    ]
    ts = [p.relative_t_s for p in pts]
    ev = DopplerEvent(
        recording_id=rec_id,
        audio_start_s=max(0.0, min(ts) - pad_s),
        audio_end_s=max(ts) + pad_s,
        ridge_points=pts,
        candidate_harmonic_numbers=[harmonic],
        detection_method="manual",
        detection_quality=1.0,
        annotated_by=annotated_by,
        ridge_harmonic=harmonic,
    )
    return store.add_event(ev)


def purge_recording(store: Store, settings: Settings, rec_id: str) -> dict[str, Any]:
    """Delete the original, analysis copy, spectrogram, events and all run outputs."""
    d = store.recording_dir(rec_id)
    if d is None:
        raise NotFound(rec_id)
    removed = []
    for run_dir in store.delete_searches_for_recording(rec_id):
        if run_dir.exists() and run_dir.resolve().is_relative_to(settings.runs_dir.resolve()):
            shutil.rmtree(run_dir, ignore_errors=True)
            removed.append(str(run_dir))
    if d.exists() and d.resolve().is_relative_to(settings.recordings_dir.resolve()):
        shutil.rmtree(d, ignore_errors=True)
        removed.append(str(d))
    store.mark_recording_deleted(rec_id)
    return {"recording_id": rec_id, "removed": removed}


def spectrogram_payload(
    spec: Spectrogram,
    t0: float | None,
    t1: float | None,
    f0: float | None,
    f1: float | None,
    max_frames: int = 1500,
    max_bins: int = 800,
    layer: str = "whitened",
) -> dict[str, Any]:
    """Downsampled view for the UI (block-max pooling preserves thin ridges)."""
    sub = spec.crop(
        t0 if t0 is not None else spec.times[0],
        t1 if t1 is not None else spec.times[-1],
        f0 if f0 is not None else spec.freqs[0],
        f1 if f1 is not None else spec.freqs[-1],
    )
    arr = sub.whitened if layer == "whitened" else sub.power_db
    T, F = arr.shape
    st = max(1, int(np.ceil(T / max_frames)))
    sf_ = max(1, int(np.ceil(F / max_bins)))
    if st > 1 or sf_ > 1:
        Tp, Fp = (T // st) * st, (F // sf_) * sf_
        a = arr[:Tp, :Fp].reshape(Tp // st, st, Fp // sf_, sf_).max(axis=(1, 3))
        times = sub.times[:Tp].reshape(-1, st).mean(axis=1)
        freqs = sub.freqs[:Fp].reshape(-1, sf_).mean(axis=1)
    else:
        a, times, freqs = arr, sub.times, sub.freqs
    return {
        "times": np.round(times, 3).tolist(),
        "freqs": np.round(freqs, 3).tolist(),
        "values": np.round(a.astype(np.float32), 2).tolist(),
        "layer": layer,
        "params": spec.params,
        "pooled": [st, sf_],
    }


def pick_event_pair(
    events: list[DopplerEvent], min_separation_s: float = 60.0
) -> list[DopplerEvent]:
    """Choose the two most credible events with clearly distinct time windows (for Mode A).

    Ranking: detection quality × number of harmonics seen × ridge length. The second event must
    have its steepest point at least ``min_separation_s`` away from the first. Returns the
    chosen events sorted by time (fewer than two if no such pair exists).
    """
    from aerodoppler.audio.ridges import estimate_f0_at_steepest  # noqa: F401  (same module family)

    def centre(e: DopplerEvent) -> float:
        from aerodoppler.search.events import robust_descent_centre

        t = np.array([p.relative_t_s for p in e.ridge_points])
        f = np.array([p.frequency_hz for p in e.ridge_points])
        if len(t) < 5:
            return 0.5 * (e.audio_start_s + e.audio_end_s)
        i, drop = robust_descent_centre(t, f)
        return float(t[i]) if drop > 0 else 0.5 * (e.audio_start_s + e.audio_end_s)

    def descents(e: DopplerEvent) -> int:
        """Number of distinct steep-descent episodes (a clean flyby has exactly one)."""
        t = np.array([p.relative_t_s for p in e.ridge_points])
        f = np.array([p.frequency_hz for p in e.ridge_points])
        if len(t) < 20:
            return 1
        k = max(3, int(round(10.0 / max(float(np.median(np.diff(t))), 0.05))) | 1)
        fs = np.convolve(np.pad(f, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        slope = np.gradient(fs, t)
        if np.min(slope) >= 0:
            return 1
        steep = slope < 0.3 * float(np.min(slope))
        edges = np.flatnonzero(np.diff(np.concatenate([[0], steep.astype(int), [0]])))
        return max(1, len(edges) // 2)

    def weight(e: DopplerEvent) -> float:
        shape = 1.0 / descents(e)  # merged multi-aircraft ridges are penalised
        return (
            (0.2 + e.detection_quality)
            * len(e.candidate_harmonic_numbers)
            * np.sqrt(max(len(e.ridge_points), 1))
            * shape
        )

    ranked = sorted(events, key=weight, reverse=True)
    if not ranked:
        return []
    first = ranked[0]
    c1 = centre(first)
    for e in ranked[1:]:
        if abs(centre(e) - c1) >= min_separation_s:
            return sorted([first, e], key=lambda x: x.audio_start_s)
    return [first]


def pick_events(
    events: list[DopplerEvent], k: int = 4, min_separation_s: float = 60.0
) -> list[DopplerEvent]:
    """Up to ``k`` credible, mutually time-separated events for a multi-event Mode A search.

    The first two are the pair chosen by :func:`pick_event_pair` (they drive the Δt pairing);
    further events are added by decreasing credibility when their centre is at least
    ``min_separation_s`` away from every event already chosen. Returned in that priority order
    (the search keeps the first two as the pairing events).
    """
    pair = pick_event_pair(events, min_separation_s)
    if len(pair) < 2:
        return pair

    from aerodoppler.search.events import robust_descent_centre

    def centre(e: DopplerEvent) -> float:
        t = np.array([p.relative_t_s for p in e.ridge_points])
        f = np.array([p.frequency_hz for p in e.ridge_points])
        if len(t) < 5:
            return 0.5 * (e.audio_start_s + e.audio_end_s)
        i, drop = robust_descent_centre(t, f)
        return float(t[i]) if drop > 0 else 0.5 * (e.audio_start_s + e.audio_end_s)

    def weight(e: DopplerEvent) -> float:
        return (
            (0.2 + e.detection_quality)
            * len(e.candidate_harmonic_numbers)
            * np.sqrt(max(len(e.ridge_points), 1))
        )

    chosen = list(pair)
    centres = [centre(e) for e in chosen]
    for e in sorted(events, key=weight, reverse=True):
        if len(chosen) >= k:
            break
        if any(e.id == c.id for c in chosen) or e.detection_quality < 0.3:
            continue
        c = centre(e)
        if all(abs(c - cc) >= min_separation_s for cc in centres):
            chosen.append(e)
            centres.append(c)
    return chosen
