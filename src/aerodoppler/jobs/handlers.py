"""Job types: what each one does, what it needs, and how it may be retried.

A job type declares its resource class (``import``, ``analysis`` or ``light`` — per-class
concurrency limits), its memory reservation (counted against the shared budget at admission and
handed to the work as its own budget), whether it is safe to resume automatically after its
worker was lost, how duplicates are detected, and which errors are transient.

Handlers run inside a worker process (or inline, for the CLI) and receive a :class:`JobContext`
for progress, cancellation and logging; they never touch the queue directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerodoppler.config import Settings
from aerodoppler.jobs.queue import Job, JobCancelled, JobError
from aerodoppler.jobs.store import Store

log = logging.getLogger(__name__)


class JobContext:
    """What a handler sees: progress reporting, cancellation, logging, budgets, paths."""

    def __init__(
        self,
        job: Job,
        settings: Settings,
        store: Store,
        report: Callable[[str, dict[str, Any]], None],
        cancel_event: threading.Event,
        work_dir: Path,
        logger: Callable[[str], None] | None = None,
    ):
        self.job = job
        self.settings = settings
        self.store = store
        self._report = report
        self.cancel_event = cancel_event
        self.work_dir = work_dir
        self.memory_budget_mb = job.memory_mb
        self._log = logger or (lambda m: log.info("%s: %s", job.id, m))
        self._last: dict[str, float] = {}

    def progress(self, stage: str, info: dict[str, Any], min_interval_s: float = 1.0) -> None:
        now = time.time()
        if now - self._last.get(stage, 0.0) < min_interval_s and stage == self.job.stage:
            return
        self._last[stage] = now
        self.job.stage = stage
        self._report(stage, info)

    def should_cancel(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled("cancelled by operator")

    def log(self, msg: str) -> None:
        self._log(msg)


@dataclass
class JobType:
    name: str
    resource_class: str
    run: Callable[[JobContext], dict[str, Any]]
    memory_mb: Callable[[Settings, dict[str, Any], float], float]
    resume_safe: bool = False
    max_attempts: int = 1
    idempotency: Callable[[dict[str, Any]], str | None] = lambda p: None
    links: Callable[[dict[str, Any]], tuple[str | None, str | None]] = lambda p: (None, None)


def _key(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def shared_budget_mb(settings: Settings) -> float:
    """Memory all jobs together may reserve: a fraction of what the machine (cgroup/WSL
    aware) allows, or the configured value."""
    from aerodoppler.resources import snapshot

    if settings.jobs_memory_budget_mb:
        return float(settings.jobs_memory_budget_mb)
    return max(1024.0, 0.7 * snapshot().effective_limit_mb)


# ---- flight_import ------------------------------------------------------------------------

_TRANSIENT_PROVIDER = {"network", "rate_limited", "provider_error", "not_yet_published"}


def _run_flight_import(ctx: JobContext) -> dict[str, Any]:
    from aerodoppler.adsb.adsblol import AdsbLolProvider, ProviderUnavailable
    from aerodoppler.adsb.ingest import (
        ImportCancelled,
        ImportRequest,
        MemoryBudgetExceeded,
        StorageBudgetExceeded,
        import_flights,
    )
    from aerodoppler.adsb.providers import LocalTraceProvider
    from aerodoppler.models import BBox

    p = ctx.job.params
    s = ctx.settings
    if p.get("provider") == "local":
        prov: Any = LocalTraceProvider(Path(p["local_path"]))
    else:
        prov = AdsbLolProvider(offline=s.offline, github_token=s.github_token)
    req = ImportRequest(
        date_utc=p["date_utc"],
        bbox=BBox.model_validate(p["bbox"]),
        margin_km=float(p.get("margin_km", s.adsb_bbox_margin_km)),
        storage_budget_gb=p.get("storage_budget_gb"),
        confirm_download=bool(p.get("confirm_download")),
        workers=int(p.get("workers") or 0),
        memory_budget_mb=ctx.memory_budget_mb,
        exclude_jets=bool(p.get("exclude_jets")),
        gap_split_s=float(p.get("gap_split_s") or s.adsb_gap_split_s),
    )
    rate = {"t": 0.0, "b": 0}

    def on_progress(stage: str, info: dict[str, Any]) -> None:
        if stage == "download":
            now, done = time.time(), int(info.get("bytes_done") or 0)
            if rate["t"]:
                dt = now - rate["t"]
                if dt > 0:
                    info = {**info, "throughput_mb_s": round((done - rate["b"]) / dt / 1e6, 1)}
            rate["t"], rate["b"] = now, done
        ctx.progress(stage, info)

    try:
        ds = import_flights(
            prov,
            req,
            s.raw_dir,
            s.normalized_dir,
            progress=False,
            on_progress=on_progress,
            should_cancel=ctx.should_cancel,
        )
    except ImportCancelled as e:
        raise JobCancelled(str(e)) from e
    except ProviderUnavailable as e:
        if e.kind == "cancelled":
            raise JobCancelled(str(e)) from e
        raise JobError(
            str(e), code=e.kind, transient=e.kind in _TRANSIENT_PROVIDER, **e.details
        ) from e
    except StorageBudgetExceeded as e:
        raise JobError(str(e), code="download_not_confirmed") from e
    except MemoryBudgetExceeded as e:
        raise JobError(str(e), code="memory_budget") from e
    m = ds.manifest()
    ctx.progress("complete", {"segments": (m.get("summary") or {}).get("segments")})
    return {
        "dataset_id": ds.dataset_id,
        "summary": m.get("summary"),
        "storage_bytes": m.get("storage_bytes"),
        "resources": m.get("resources"),
    }


def _import_memory(s: Settings, p: dict[str, Any], budget: float) -> float:
    if p.get("memory_budget_mb"):
        return float(p["memory_budget_mb"])
    if s.import_memory_budget_mb:
        return float(s.import_memory_budget_mb)
    return max(1024.0, min(4096.0, 0.4 * budget))


def _import_key(p: dict[str, Any]) -> str:
    b = p.get("bbox") or {}
    bt = (
        [round(float(b[k]), 4) for k in ("west", "south", "east", "north")]
        if isinstance(b, dict)
        else b
    )
    return "flight_import:" + _key(
        p.get("provider", "adsblol"),
        p["date_utc"],
        bt,
        round(float(p.get("margin_km", 15.0)), 3),
        bool(p.get("exclude_jets")),
        p.get("local_path") or "",
    )


# ---- audio_ingest ------------------------------------------------------------------------


def _run_audio_ingest(ctx: JobContext) -> dict[str, Any]:
    from aerodoppler.audio.decode import AudioDecodeError
    from aerodoppler.audio.service import create_segment, ingest_recording

    p = ctx.job.params
    ctx.progress("decode", {"file": p.get("original_filename") or p.get("parent_recording_id")})
    try:
        if p.get("parent_recording_id"):
            rec = create_segment(
                ctx.store,
                ctx.settings,
                p["parent_recording_id"],
                float(p["start_s"]),
                p.get("duration_s"),
            )
        else:
            src = Path(p["upload_path"])
            if not src.exists():
                raise JobError(f"the uploaded file is gone ({src.name})", code="upload_missing")
            rec = ingest_recording(
                ctx.store,
                ctx.settings,
                src,
                recording_date_utc=p.get("recording_date_utc"),
                start_time_utc=p.get("start_time_utc"),
                start_time_uncertainty_s=p.get("start_time_uncertainty_s"),
                synthetic=bool(p.get("synthetic")),
                notes=p.get("notes"),
                segment_start_s=float(p.get("segment_start_s") or 0.0),
                segment_duration_s=p.get("segment_duration_s"),
                date_earliest_utc=p.get("date_earliest_utc"),
                date_latest_utc=p.get("date_latest_utc"),
                original_filename=p.get("original_filename"),
            )
    except AudioDecodeError as e:
        raise JobError(f"audio could not be decoded: {e}", code="decode_failed") from e
    staged = Path(p["upload_path"]).resolve() if p.get("upload_path") else None
    uploads = (ctx.settings.data_dir / "uploads").resolve()
    if staged is not None and staged.is_relative_to(uploads):
        # only an upload staged by the API belongs to this job (never a CLI user's own file);
        # the recording keeps its own copy of the original
        shutil.rmtree(staged.parent, ignore_errors=True)
    from aerodoppler.audio.service import recording_warnings

    return {"recording_id": rec.id, "warnings": recording_warnings(rec)}


def _ingest_memory(s: Settings, p: dict[str, Any], budget: float) -> float:
    # decoding holds the native-rate samples (float32) plus the resampled analysis copy
    dur = float(p.get("probe_duration_s") or s.max_audio_duration_s)
    dur = min(
        dur, float(p.get("segment_duration_s") or s.max_audio_duration_s), s.max_audio_duration_s
    )
    sr, ch = float(p.get("probe_sample_rate") or 48000), float(p.get("probe_channels") or 2)
    return max(512.0, 400.0 + 2.5 * dur * sr * ch * 4 / 1e6)


# ---- spectrogram_generation / ridge_detection --------------------------------------------


def _rec_memory(s: Settings, p: dict[str, Any], budget: float) -> float:
    """Spectrogram / detection peak memory. Measured 2026-11-05 on 2 kHz audio with the default
    STFT (0.25 s hop, 900 Hz, 4.096 s window): 1.6 GB for 30 min, 3.1 GB for 60 min, 6.0 GB for
    120 min — about 56 bytes per spectrogram cell (several float64 working arrays). 60 bytes per
    cell plus 300 MB of runtime keeps a margin."""
    dur = float(p.get("duration_s") or s.max_audio_duration_s)
    frames = dur / s.stft_hop_s
    bins = s.max_analysis_freq_hz * s.stft_window_s
    return max(512.0, 300.0 + 60.0 * frames * bins / 1e6)


def _run_spectrogram(ctx: JobContext) -> dict[str, Any]:
    from aerodoppler.audio.service import NotFound, ensure_spectrogram

    rid = ctx.job.params["recording_id"]
    ctx.progress("stft", {"recording_id": rid})
    try:
        spec = ensure_spectrogram(
            ctx.store, ctx.settings, rid, force=bool(ctx.job.params.get("force"))
        )
    except NotFound as e:
        raise JobError("recording not found", code="recording_not_found") from e
    return {"recording_id": rid, "frames": int(len(spec.times)), "bins": int(len(spec.freqs))}


def _run_detection(ctx: JobContext) -> dict[str, Any]:
    from aerodoppler.audio.ridges import DetectorConfig
    from aerodoppler.audio.service import NotFound, propose_events

    p = ctx.job.params
    ctx.progress("detect", {"recording_id": p["recording_id"]})
    diag: dict[str, Any] = {}
    try:
        events = propose_events(
            ctx.store,
            ctx.settings,
            p["recording_id"],
            DetectorConfig(
                threshold_z=float(p.get("threshold_z", 2.5)),
                min_duration_s=float(p.get("min_duration_s", 40.0)),
            ),
            bool(p.get("save", True)),
            True,
            diag,
        )
    except NotFound as e:
        raise JobError("recording not found", code="recording_not_found") from e
    return {"event_ids": [e.id for e in events], "diagnostics": diag}


# ---- geolocation_search -------------------------------------------------------------------


def _search_memory(s: Settings, p: dict[str, Any], budget: float) -> float:
    if s.search_memory_budget_mb:
        return float(s.search_memory_budget_mb)
    return max(1024.0, min(8192.0, 0.55 * budget))


def _run_search(ctx: JobContext) -> dict[str, Any]:
    from aerodoppler.jobs.runner import execute_search
    from aerodoppler.models import SearchRequest

    req = SearchRequest.model_validate(ctx.job.params)
    _res, summary = execute_search(ctx, req, ctx.work_dir)
    return summary


def _search_key(p: dict[str, Any]) -> str:
    return "geolocation_search:" + _key(
        {k: v for k, v in p.items() if k not in ("operator_authorized",)}
    )


# ---- benchmark ----------------------------------------------------------------------------


def _run_benchmark(ctx: JobContext) -> dict[str, Any]:
    from aerodoppler.validation import benchmark as B

    p = ctx.job.params
    out = Path(p.get("out") or (ctx.work_dir / "benchmark"))
    workers = int(p.get("workers") or 0)
    kind = p.get("kind", "synthetic")
    ctx.progress("running", {"kind": kind, "out": str(out)})

    def logf(msg: str) -> None:
        ctx.log(msg)
        ctx.check_cancel()

    if kind == "synthetic":
        res = B.run_synthetic_benchmark(
            ctx.settings, out, workers=workers, cases=p.get("cases"), log=logf
        )
    elif kind == "real_trajectories":
        res = B.run_real_trajectory_benchmark(ctx.settings, out, **p["args"], log=logf)
    elif kind == "livermore":
        from aerodoppler.validation.livermore import run_livermore_benchmark

        res = run_livermore_benchmark(
            ctx.settings, Path(p["zdir"]), out, p["datasets"], workers=workers, log=logf
        )
    else:
        raise JobError(f"unknown benchmark kind {kind}", code="invalid_params")
    return {"out": str(out), "summary_keys": sorted(res.keys()) if isinstance(res, dict) else None}


# ---- diagnostic -------------------------------------------------------------------------


def _busy_child(_i: int) -> int:
    t = time.time()
    while time.time() - t < 5.0:  # busy in Python bytecode, where signal handlers run
        sum(range(1000))
    return 0


def _run_diagnostic(ctx: JobContext) -> dict[str, Any]:
    """Self-test job: sleeps ``seconds`` in small steps reporting progress, optionally holding
    ``allocate_mb`` of memory, failing transiently ``fail_times`` times (counted in a file in its
    work dir), or ignoring cancellation and SIGTERM (``ignore_cancel``) to exercise the
    scheduler's escalation. Used by the reliability tests and ``aerodoppler jobs selftest``."""
    import signal

    p = ctx.job.params
    if p.get("ignore_cancel"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    counter = ctx.work_dir / "failures"
    if int(p.get("fail_times") or 0) > 0:
        n = int(counter.read_text()) if counter.exists() else 0
        if n < int(p["fail_times"]):
            counter.write_text(str(n + 1))
            raise JobError(f"simulated transient failure {n + 1}", code="network", transient=True)
    if p.get("fork_pool"):
        # like the search engine: a forked pool whose children are busy in Python code when
        # the pool is terminated (SIGTERM to each child)
        import multiprocessing as mp

        pool = mp.get_context("fork").Pool(2)
        pool.map_async(_busy_child, range(4))
        time.sleep(0.5)
        pool.terminate()
        pool.join()
        time.sleep(1.0)
    hold = bytearray(int(float(p.get("allocate_mb") or 0) * 1e6))
    for i in range(len(hold) // 4096):
        hold[i * 4096] = 1  # touch every page so it is resident
    steps = max(1, int(float(p.get("seconds", 1.0)) / 0.1))
    for i in range(steps):
        if not p.get("ignore_cancel"):
            ctx.check_cancel()
        ctx.progress("sleeping", {"step": i + 1, "steps": steps}, min_interval_s=0.5)
        time.sleep(0.1)
    return {"slept_s": steps * 0.1, "held_mb": len(hold) / 1e6}


# ---- registry -----------------------------------------------------------------------------

JOB_TYPES: dict[str, JobType] = {
    "flight_import": JobType(
        "flight_import",
        "import",
        _run_flight_import,
        _import_memory,
        resume_safe=True,  # atomic dataset write, resumable/verified download
        max_attempts=3,
        idempotency=_import_key,
    ),
    "audio_ingest": JobType(
        "audio_ingest",
        "light",
        _run_audio_ingest,
        _ingest_memory,
        resume_safe=False,  # would create a second recording
        links=lambda p: (p.get("parent_recording_id"), None),
    ),
    "spectrogram_generation": JobType(
        "spectrogram_generation",
        "light",
        _run_spectrogram,
        _rec_memory,
        resume_safe=True,
        max_attempts=2,
        idempotency=lambda p: (
            "spectrogram:" + p["recording_id"] + (":force" if p.get("force") else "")
        ),
        links=lambda p: (p["recording_id"], None),
    ),
    "ridge_detection": JobType(
        "ridge_detection",
        "light",
        _run_detection,
        _rec_memory,
        resume_safe=False,  # saving events twice would duplicate them
        idempotency=lambda p: "detect:" + _key(p),
        links=lambda p: (p["recording_id"], None),
    ),
    "geolocation_search": JobType(
        "geolocation_search",
        "analysis",
        _run_search,
        _search_memory,
        resume_safe=False,  # partial outputs are marked incomplete; retry is explicit
        idempotency=_search_key,
        links=lambda p: (p.get("recording_id"), None),
    ),
    "benchmark": JobType(
        "benchmark",
        "analysis",
        _run_benchmark,
        _search_memory,
        resume_safe=False,
    ),
    "diagnostic": JobType(
        "diagnostic",
        "light",
        _run_diagnostic,
        lambda s, p, b: float(p.get("reserve_mb") or 64.0),
        resume_safe=False,
        max_attempts=3,
    ),
}


def resource_class_for(job_type: str, params: dict[str, Any]) -> str:
    if job_type == "diagnostic" and params.get("resource_class"):
        return str(params["resource_class"])
    return JOB_TYPES[job_type].resource_class
