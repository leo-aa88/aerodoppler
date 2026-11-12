"""Recording-level operations shared by the CLI and the API: ingest, spectrogram, automatic
event proposals, manual events, purge (SPEC §8, §11)."""

from __future__ import annotations

import datetime as dt
import json
import os
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


class RecordingBusy(RuntimeError):
    pass


def _shift_iso(iso: str, seconds: float) -> str:
    t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00")) + dt.timedelta(seconds=seconds)
    return t.astimezone(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ingest_recording(
    store: Store,
    settings: Settings,
    path: Path,
    *,
    owner_permission_confirmed: bool = True,
    recording_date_utc: str | None = None,
    start_time_utc: str | None = None,
    start_time_uncertainty_s: float | None = None,
    synthetic: bool = False,
    notes: str | None = None,
    keep_original: bool = True,
    segment_start_s: float = 0.0,
    segment_duration_s: float | None = None,
    parent_recording_id: str | None = None,
    original_filename: str | None = None,
    date_earliest_utc: str | None = None,
    date_latest_utc: str | None = None,
) -> Recording:
    """Ingest one interval of an audio/video file (the whole file when it fits).

    ``start_time_utc`` is the UTC time of the first sample of the SOURCE file; the
    recording's own ``start_time_utc`` is shifted by ``segment_start_s`` so every event time
    maps to the right UTC instant. Files longer than ``max_audio_duration_s`` are not
    rejected: the first ``max_audio_duration_s`` (or the requested segment) are analysed and
    the result says so (``truncated``); the original is kept so other segments can be made."""
    size_mb = Path(path).stat().st_size / 1e6
    if size_mb > settings.max_upload_mb:
        raise AudioDecodeError(f"file is {size_mb:.0f} MB; limit is {settings.max_upload_mb} MB")
    dec = decode_audio(
        Path(path),
        max_duration_s=settings.max_audio_duration_s,
        start_s=segment_start_s,
        duration_s=segment_duration_s,
    )
    if original_filename:
        dec.original_filename = original_filename
    rec_id = new_id("rec")
    rec_dir = settings.recordings_dir / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    original_path = None
    if keep_original:
        original_path = rec_dir / ("original_" + safe_filename(dec.original_filename))
        try:
            os.link(path, original_path)  # another segment of the same source: no copy
        except OSError:
            shutil.copy2(path, original_path)
    mono = to_analysis_mono(dec, settings.analysis_sample_rate_hz)
    write_wav(
        rec_dir / "analysis_2khz.wav",
        mono.astype(np.float32),
        settings.analysis_sample_rate_hz,
        subtype="FLOAT",
    )
    seg0 = float(dec.segment_start_s)
    rec_start = _shift_iso(start_time_utc, seg0) if start_time_utc else None
    rec_date = recording_date_utc or (rec_start[:10] if rec_start else None)
    if rec_start:
        basis = "verified_start"
    elif rec_date:
        basis = "date_only"
    elif date_earliest_utc and date_latest_utc:
        basis = "date_range"
    elif date_latest_utc:
        basis = "upper_bound_only"
    else:
        basis = "unknown"
    rec = Recording(
        id=rec_id,
        sha256=dec.sha256,
        original_filename=dec.original_filename,
        duration_s=dec.duration_s,
        sample_rate_hz=dec.sample_rate_hz,
        channels=dec.channels,
        source_duration_s=dec.source_duration_s,
        truncated=dec.truncated,
        segment_start_s=seg0,
        parent_recording_id=parent_recording_id,
        source_start_time_utc=start_time_utc,
        start_time_utc=rec_start,
        start_time_uncertainty_s=start_time_uncertainty_s,
        recording_date_utc=rec_date,
        time_basis=basis,
        date_earliest_utc=date_earliest_utc,
        date_latest_utc=date_latest_utc,
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
                "original_path": str(original_path) if original_path else None,
                "source_sha256": dec.sha256,
                "source_duration_s": dec.source_duration_s,
                "segment_start_s": seg0,
                "segment_end_s": seg0 + dec.duration_s,
                "truncated": dec.truncated,
                "parent_recording_id": parent_recording_id,
            },
            indent=2,
        )
    )
    store.add_recording(rec, rec_dir)
    return rec


def segment_notice(rec: Recording) -> str | None:
    """Human description of what part of the source was analysed, or None for whole files."""
    if not rec.truncated and rec.segment_start_s == 0:
        return None
    a, b = rec.segment_start_s, rec.segment_start_s + rec.duration_s
    total = f" of {rec.source_duration_s:.0f} s" if rec.source_duration_s else ""
    return (
        f"only {a:.0f}–{b:.0f} s{total} of the source file is analysed "
        f"(limit {rec.duration_s / 60:.0f} min per analysis copy). Event times are relative to "
        f"{a:.0f} s into the file; create another segment to analyse the rest."
    )


def recording_warnings(rec: Recording) -> list[str]:
    warnings = []
    note = segment_notice(rec)
    if note:
        warnings.append(note)
    if rec.time_basis == "upper_bound_only":
        warnings.append(
            f"only an upper bound on the date is known (recorded on or before "
            f"{rec.date_latest_utc}); flight matching needs the UTC day, so a search is not "
            "possible until the day is established"
        )
    elif rec.time_basis == "date_range":
        warnings.append(
            f"the day is only known to lie between {rec.date_earliest_utc} and "
            f"{rec.date_latest_utc}; each candidate day needs its own ~3.5 GB flight archive"
        )
    elif rec.time_basis == "unknown":
        warnings.append(
            "no recording date or time: flight matching needs at least the UTC day "
            "(unknown-time mode) or the UTC start time (known-time mode)"
        )
    return warnings


def create_segment(
    store: Store, settings: Settings, rec_id: str, start_s: float, duration_s: float | None
) -> Recording:
    """New recording for another interval of the same preserved source file."""
    parent = store.get_recording(rec_id)
    d = store.recording_dir(rec_id)
    if parent is None or d is None:
        raise NotFound(rec_id)
    originals = sorted(d.glob("original_*"))
    if not originals:
        raise AudioDecodeError("the original file of this recording was not kept")
    root_id = parent.parent_recording_id or parent.id
    return ingest_recording(
        store,
        settings,
        originals[0],
        recording_date_utc=parent.recording_date_utc if not parent.source_start_time_utc else None,
        start_time_utc=parent.source_start_time_utc,
        start_time_uncertainty_s=parent.start_time_uncertainty_s,
        synthetic=parent.synthetic,
        notes=parent.notes,
        segment_start_s=start_s,
        segment_duration_s=duration_s,
        parent_recording_id=root_id,
        original_filename=parent.original_filename,
        date_earliest_utc=parent.date_earliest_utc,
        date_latest_utc=parent.date_latest_utc,
    )


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
    template: bool = True,
    diagnostics: dict | None = None,
) -> list[DopplerEvent]:
    """Automatic proposals from two detectors: the Viterbi ridge tracker (precise ridges,
    harmonic families) and the straight-pass template fit (weak but long sweeps). Template
    events are added only where no tracked event overlaps them substantially.

    When ``diagnostics`` is a dict it is filled with what was scanned, the parameters, every
    rejected candidate with its reason, and suggested next steps."""
    from aerodoppler.audio.template_detect import detect_flyovers, flyovers_to_events
    from aerodoppler.search.events import assess_event

    cfg = cfg or DetectorConfig()
    spec = ensure_spectrogram(store, settings, rec_id)
    rejected: list[dict] = []
    ridges = detect_ridges(spec, cfg, rejected=rejected)
    events = ridges_to_events(ridges, spec, rec_id)
    n_tracked = len(events)
    if template:
        for te in flyovers_to_events(spec, detect_flyovers(spec), rec_id, rejected=rejected):
            overlap = any(
                min(e.audio_end_s, te.audio_end_s) - max(e.audio_start_s, te.audio_start_s)
                > 0.5 * (te.audio_end_s - te.audio_start_s)
                for e in events
            )
            if not overlap:
                events.append(te)
    if save:
        events = [store.add_event(e) for e in events]
    if diagnostics is not None:
        diagnostics.update(
            detection_diagnostics(spec, cfg, events, n_tracked, rejected, assess_event)
        )
    return events


_REJECT_TEXT = {
    "too_short": "shorter than the minimum duration",
    "weak": "too faint (mean whitened level below the minimum)",
    "stationary_tone": "frequency almost constant (hum, machinery, a held note — or a plateau)",
    "steady_tones_joined": "two steady lines joined by a jump while one keeps sounding (a constant source's harmonics), not a sweep",
}


def detection_diagnostics(
    spec: Spectrogram,
    cfg: DetectorConfig,
    events: list[DopplerEvent],
    n_tracked: int,
    rejected: list[dict],
    assess: Any,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in rejected:
        for reason in r["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    labels: dict[str, int] = {}
    for e in events:
        lab = assess(e)["label"]
        labels[lab] = labels.get(lab, 0) + 1
    suggestions: list[str] = []
    if not events:
        suggestions.append(
            "No tonal ridge passed the detector. That does not prove no aircraft is audible: "
            "check the rejected candidates below, look at the spectrogram yourself, and draw "
            "a ridge manually if you see a slanted line."
        )
    if counts.get("too_short"):
        suggestions.append(
            f"{counts['too_short']} candidate(s) were shorter than {cfg.min_duration_s:.0f} s; "
            "lower 'min duration' if passes in this recording are short."
        )
    if counts.get("weak"):
        suggestions.append(
            f"{counts['weak']} candidate(s) were too faint; lower the threshold (z) to see them, "
            "at the cost of more noise proposals."
        )
    if counts.get("stationary_tone"):
        suggestions.append(
            f"{counts['stationary_tone']} candidate(s) were rejected as nearly constant tones "
            "(no Doppler change); these are usually hum or machinery."
        )
    if labels.get("tonal_only"):
        suggestions.append(
            f"{labels['tonal_only']} proposed event(s) show no measurable Doppler transition "
            "(labelled 'tonal ridge only'); they cannot locate the recording unless you review "
            "them and are sure they are aircraft."
        )
    rejected_sorted = sorted(rejected, key=lambda r: -r["mean_z"])[:60]
    for r in rejected_sorted:
        r["reason_text"] = "; ".join(_REJECT_TEXT.get(x, x) for x in r["reasons"])
    return {
        "scanned": {
            "t_start_s": float(spec.times[0]),
            "t_end_s": float(spec.times[-1]),
            "f_min_hz": cfg.min_frequency_hz,
            "f_max_hz": min(cfg.max_frequency_hz, float(spec.freqs[-1])),
            "note": "the detector always scans the whole recording; the view range does not limit it",
        },
        "parameters": {
            "threshold_z": cfg.threshold_z,
            "min_duration_s": cfg.min_duration_s,
            "min_mean_z": cfg.min_mean_z,
            "min_freq_variation_hz": cfg.min_freq_variation_hz,
        },
        "proposals": len(events),
        "from_ridge_tracker": n_tracked,
        "from_template_fit": len(events) - n_tracked,
        "evidence_labels": labels,
        "rejected_counts": counts,
        "rejected": rejected_sorted,
        "suggestions": suggestions,
    }


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
    from aerodoppler.jobs.queue import ACTIVE, SqliteJobQueue

    q = SqliteJobQueue(settings.db_path)
    active = q.list(recording_id=rec_id, statuses=list(ACTIVE))
    if active:
        raise RecordingBusy(
            f"{len(active)} job(s) of this recording are still queued or running "
            f"({', '.join(j.id for j in active)}); cancel them first"
        )
    removed = []
    jobs_root = (settings.data_dir / "jobs").resolve()
    for j in q.list(recording_id=rec_id, limit=10000):
        wd = Path(j.work_dir) if j.work_dir else None
        if (
            wd
            and wd.exists()
            and (
                wd.resolve().is_relative_to(settings.runs_dir.resolve())
                or wd.resolve().is_relative_to(jobs_root)
            )
        ):
            shutil.rmtree(wd, ignore_errors=True)
            removed.append(str(wd))
    with q._conn() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM jobs WHERE recording_id=?", (rec_id,))]
        c.execute(
            "DELETE FROM job_events WHERE job_id IN (SELECT id FROM jobs WHERE recording_id=?)",
            (rec_id,),
        )
        c.execute("DELETE FROM jobs WHERE recording_id=?", (rec_id,))
    if ids:
        store.audit("jobs.purge", rec_id, {"jobs": ids})
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


# An automatically chosen event should be one aircraft passing by: the search places its delayed
# closest approach at the event's steepest point and reads speed and distance bounds from its
# plateaus. In dense traffic the longest, most harmonic-rich ridges are often something else:
# the approach of one aircraft only (no closest approach inside the window; a reached-plateau
# ratio implying a radial speed far below any flight speed) or two aircraft merged into one
# ridge (plateaus not reached, several descents).
PASS_MIN_RADIAL_SPEED_MPS = 20.0  # the search's 25 m/s speed floor less 20 % plateau error


def looks_like_a_pass(e: DopplerEvent) -> bool:
    """One complete pass: measured Doppler evidence (``plausible_doppler``) with both plateaus
    reached, a single steep descent, and a plateau ratio implying at least
    ``PASS_MIN_RADIAL_SPEED_MPS`` of radial speed."""
    from aerodoppler.search.events import assess_event

    a = assess_event(e)
    m = a.get("metrics") or {}
    return (
        a.get("label") == "plausible_doppler"
        and bool(m.get("plateaus_reached"))
        and (m.get("implied_radial_speed_mps") or 0.0) >= PASS_MIN_RADIAL_SPEED_MPS
        and _descents(e) == 1
    )


def _descents(e: DopplerEvent) -> int:
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


def pick_event_pair(
    events: list[DopplerEvent], min_separation_s: float = 60.0
) -> list[DopplerEvent]:
    """Choose the two most credible events with clearly distinct time windows (for Mode A).

    Events that look like one complete aircraft pass (:func:`looks_like_a_pass`) are ranked
    before all others; within each group the ranking is detection quality × number of
    harmonics seen × ridge length. The second event must have its steepest point at least
    ``min_separation_s`` away from the first. Returns the chosen events sorted by time (fewer
    than two if no such pair exists).
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

    def weight(e: DopplerEvent) -> float:
        shape = 1.0 / _descents(e)  # merged multi-aircraft ridges are penalised
        return (
            (0.2 + e.detection_quality)
            * len(e.candidate_harmonic_numbers)
            * np.sqrt(max(len(e.ridge_points), 1))
            * shape
        )

    ranked = sorted(events, key=lambda e: (not looks_like_a_pass(e), -weight(e)))
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
    for e in sorted(events, key=lambda e: (not looks_like_a_pass(e), -weight(e))):
        if len(chosen) >= k:
            break
        if any(e.id == c.id for c in chosen) or e.detection_quality < 0.3:
            continue
        c = centre(e)
        if all(abs(c - cc) >= min_separation_s for cc in centres):
            chosen.append(e)
            centres.append(c)
    return chosen


class TimeMetadataError(ValueError):
    def __init__(self, message: str, field: str, action: str = ""):
        super().__init__(message)
        self.field = field
        self.action = action


def check_time_metadata(
    recording_date_utc: str | None,
    start_time_utc: str | None,
    start_time_uncertainty_s: float | None,
    date_earliest_utc: str | None,
    date_latest_utc: str | None,
) -> None:
    """Reject time metadata that pretends to more than it says (raises TimeMetadataError)."""

    def day(v: str, field: str) -> dt.date:
        try:
            return dt.date.fromisoformat(v)
        except ValueError as e:
            raise TimeMetadataError(f"'{v}' is not a date (YYYY-MM-DD)", field) from e

    if start_time_uncertainty_s is not None and not start_time_utc:
        raise TimeMetadataError(
            "an uncertainty only makes sense around a known start time; none was given",
            "start_time_uncertainty_s",
            "enter the UTC start time, or leave the uncertainty empty",
        )
    if start_time_utc:
        try:
            t = dt.datetime.fromisoformat(start_time_utc.replace("Z", "+00:00"))
        except ValueError as e:
            raise TimeMetadataError(
                f"'{start_time_utc}' is not an ISO-8601 time",
                "start_time_utc",
                "use the form 2026-11-03T15:30:00Z",
            ) from e
        if t.tzinfo is None:
            raise TimeMetadataError(
                "the start time has no timezone; a local clock time is ambiguous",
                "start_time_utc",
                "add Z for UTC (e.g. 2026-11-03T15:30:00Z) or an explicit offset (+01:00)",
            )
        if start_time_uncertainty_s is None:
            raise TimeMetadataError(
                "a start time needs its uncertainty: how many seconds could it be off?",
                "start_time_uncertainty_s",
                "e.g. 2 for a network-synced phone, 60 for a clock read by eye",
            )
        if recording_date_utc and t.astimezone(dt.UTC).date() != day(
            recording_date_utc, "recording_date_utc"
        ):
            raise TimeMetadataError(
                "the UTC date and the UTC start time disagree",
                "recording_date_utc",
                "leave the date empty when the start time is known",
            )
    lo = day(date_earliest_utc, "date_earliest_utc") if date_earliest_utc else None
    hi = day(date_latest_utc, "date_latest_utc") if date_latest_utc else None
    if lo and hi and lo > hi:
        raise TimeMetadataError("the earliest date is after the latest date", "date_earliest_utc")
    if recording_date_utc:
        d = day(recording_date_utc, "recording_date_utc")
        if (lo and d < lo) or (hi and d > hi):
            raise TimeMetadataError(
                "the recording date lies outside the given date range", "recording_date_utc"
            )
