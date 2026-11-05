"""HTTP routes (SPEC §11). Localhost-only by default; no data leaves the machine. Errors are
structured (see api/errors.py); long work (imports, searches) runs as persistent jobs."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from aerodoppler import __version__
from aerodoppler.api import schemas as S
from aerodoppler.api.errors import api_error, scrub
from aerodoppler.audio.decode import AudioDecodeError
from aerodoppler.audio.ridges import DetectorConfig
from aerodoppler.audio.service import (
    NotFound,
    add_manual_event,
    ensure_spectrogram,
    ingest_recording,
    propose_events,
    purge_recording,
    spectrogram_payload,
)
from aerodoppler.config import Settings
from aerodoppler.jobs.runner import JobRejected, cancel_search, submit_search
from aerodoppler.jobs.store import Store
from aerodoppler.models import SearchMode, SearchRequest, SearchResult
from aerodoppler.util import safe_filename

router = APIRouter()

AUTH_NOTICE = (
    "Analyse only recordings you own or have explicit permission to analyse. Results are ranked "
    "hypotheses with uncalibrated relative scores, not verified locations."
)


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_store_dep(request: Request) -> Store:
    return request.app.state.store


def _safe_id(value: str, kind: str) -> str:
    if not value or len(value) > 64 or not all(c.isalnum() or c in "_-" for c in value):
        raise HTTPException(status_code=400, detail=f"invalid {kind} id")
    return value


# ---- health -----------------------------------------------------------------------------


@router.get("/health", response_model=S.HealthResponse)
def health(settings: Settings = Depends(get_settings_dep)) -> S.HealthResponse:
    return S.HealthResponse(
        status="ok",
        version=__version__,
        offline=settings.offline,
        data_dir=str(settings.data_dir),
        host=settings.host,
        authorization_notice=AUTH_NOTICE,
    )


# ---- recordings -------------------------------------------------------------------------


@router.post("/api/recordings", response_model=S.RecordingCreateResponse, status_code=201)
async def create_recording(
    file: UploadFile = File(...),
    owner_permission_confirmed: bool = Form(True),
    recording_date_utc: str | None = Form(None),
    start_time_utc: str | None = Form(None),
    start_time_uncertainty_s: float | None = Form(None),
    synthetic: bool = Form(False),
    notes: str | None = Form(None),
    date_earliest_utc: str | None = Form(None),
    date_latest_utc: str | None = Form(None),
    segment_start_s: float = Form(0.0, ge=0),
    segment_duration_s: float | None = Form(None, gt=0),
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> S.RecordingCreateResponse:
    """Upload a file. ``start_time_utc`` is the UTC time of the file's first sample (with
    its uncertainty); files longer than the analysis limit are analysed in segments."""
    from aerodoppler.audio.service import TimeMetadataError, check_time_metadata

    empty = lambda v: v if v not in ("", None) else None  # noqa: E731 (form fields)
    recording_date_utc, start_time_utc = empty(recording_date_utc), empty(start_time_utc)
    date_earliest_utc, date_latest_utc = empty(date_earliest_utc), empty(date_latest_utc)
    try:
        check_time_metadata(
            recording_date_utc,
            start_time_utc,
            start_time_uncertainty_s,
            date_earliest_utc,
            date_latest_utc,
        )
    except TimeMetadataError as e:
        raise api_error(422, "time_metadata", str(e), field=e.field, action=e.action) from e
    name = safe_filename(file.filename or "upload.wav")
    limit = settings.max_upload_mb * 1024 * 1024
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / name
        size = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise api_error(
                        413,
                        "upload_too_large",
                        f"the file exceeds the {settings.max_upload_mb} MB upload limit",
                        field="file",
                        action="extract the audio track (e.g. ffmpeg -i video.mp4 -vn -c:a flac out.flac) or raise AERODOPPLER_MAX_UPLOAD_MB",
                    )
                fh.write(chunk)
        try:
            rec = await asyncio.to_thread(
                ingest_recording,
                store,
                settings,
                tmp,
                owner_permission_confirmed=True,
                recording_date_utc=recording_date_utc,
                start_time_utc=start_time_utc,
                start_time_uncertainty_s=start_time_uncertainty_s,
                synthetic=synthetic,
                notes=notes,
                segment_start_s=segment_start_s,
                segment_duration_s=segment_duration_s,
                date_earliest_utc=date_earliest_utc,
                date_latest_utc=date_latest_utc,
            )
        except AudioDecodeError as e:
            raise api_error(
                422,
                "decode_failed",
                f"audio could not be decoded: {e}",
                field="file",
                action="check that the file plays; for videos the first audio track is used",
            ) from e
    return S.RecordingCreateResponse(recording=rec, warnings=_recording_warnings(rec))


def _recording_warnings(rec: Any) -> list[str]:
    from aerodoppler.audio.service import segment_notice

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


@router.post("/api/recordings/{rec_id}/segments", status_code=201)
def create_recording_segment(
    rec_id: str,
    body: S.SegmentRequest,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> S.RecordingCreateResponse:
    """Analyse another interval of the same preserved source file (e.g. the rest of a file
    longer than the analysis limit). Creates a new recording; times map back to the source."""
    from aerodoppler.audio.service import create_segment

    try:
        rec = create_segment(
            store, settings, _safe_id(rec_id, "recording"), body.start_s, body.duration_s
        )
    except NotFound as e:
        raise api_error(404, "recording_not_found", "recording not found") from e
    except AudioDecodeError as e:
        raise api_error(422, "segment_invalid", str(e), field="start_s") from e
    return S.RecordingCreateResponse(recording=rec, warnings=_recording_warnings(rec))


@router.get("/api/recordings", response_model=S.RecordingList)
def list_recordings(store: Store = Depends(get_store_dep)) -> S.RecordingList:
    return S.RecordingList(recordings=store.list_recordings())


@router.get("/api/recordings/{rec_id}")
def get_recording(rec_id: str, store: Store = Depends(get_store_dep)) -> dict[str, Any]:
    rec = store.get_recording(_safe_id(rec_id, "recording"))
    if rec is None:
        raise HTTPException(status_code=404, detail="recording not found")
    return {
        "recording": rec.model_dump(),
        "events": [e.model_dump() for e in store.list_events(rec.id)],
        "searches": store.list_searches(rec.id),
    }


@router.get("/api/recordings/{rec_id}/spectrogram", response_model=S.SpectrogramResponse)
def get_spectrogram(
    rec_id: str,
    t0: float | None = Query(None, ge=0),
    t1: float | None = Query(None, ge=0),
    f0: float | None = Query(None, ge=0),
    f1: float | None = Query(None, ge=0),
    layer: str = Query("whitened", pattern="^(whitened|power)$"),
    max_frames: int = Query(1200, ge=50, le=4000),
    max_bins: int = Query(700, ge=50, le=2000),
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> S.SpectrogramResponse:
    rec = store.get_recording(_safe_id(rec_id, "recording"))
    if rec is None:
        raise HTTPException(status_code=404, detail="recording not found")
    try:
        spec = ensure_spectrogram(store, settings, rec.id)
    except NotFound as e:
        raise HTTPException(status_code=404, detail="recording not found") from e
    p = spectrogram_payload(
        spec, t0, t1, f0, f1, max_frames=max_frames, max_bins=max_bins, layer=layer
    )
    return S.SpectrogramResponse(recording_id=rec.id, synthetic=rec.synthetic, **p)


@router.get("/api/recordings/{rec_id}/spectrogram.png")
def get_spectrogram_png(
    rec_id: str,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> Response:
    rec = store.get_recording(_safe_id(rec_id, "recording"))
    if rec is None:
        raise HTTPException(status_code=404, detail="recording not found")
    spec = ensure_spectrogram(store, settings, rec.id)
    d = store.recording_dir(rec.id)
    assert d is not None
    from aerodoppler.validation.plots import plot_spectrogram

    out = d / "spectrogram.png"
    if not out.exists():
        plot_spectrogram(spec, out, title=f"{rec.original_filename}", synthetic=rec.synthetic)
    return FileResponse(out, media_type="image/png")


@router.post("/api/recordings/{rec_id}/events", response_model=S.EventList, status_code=201)
def add_event(
    rec_id: str, body: S.ManualEventRequest, store: Store = Depends(get_store_dep)
) -> S.EventList:
    try:
        ev = add_manual_event(
            store,
            _safe_id(rec_id, "recording"),
            body.points,
            harmonic=body.harmonic,
            uncertainty_hz=body.uncertainty_hz,
            annotated_by=body.annotated_by,
        )
    except NotFound as e:
        raise HTTPException(status_code=404, detail="recording not found") from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return S.EventList(events=[ev])


def _with_assessment(events: list[Any]) -> list[Any]:
    from aerodoppler.search.events import assess_event

    return [e.model_copy(update={"assessment": assess_event(e)}) for e in events]


@router.get("/api/recordings/{rec_id}/events", response_model=S.EventList)
def list_events(rec_id: str, store: Store = Depends(get_store_dep)) -> S.EventList:
    if store.get_recording(_safe_id(rec_id, "recording")) is None:
        raise api_error(404, "recording_not_found", "recording not found")
    return S.EventList(events=_with_assessment(store.list_events(rec_id)))


@router.post("/api/recordings/{rec_id}/events/{event_id}/review", response_model=S.EventList)
def review_event(
    rec_id: str, event_id: str, body: S.EventReview, store: Store = Depends(get_store_dep)
) -> S.EventList:
    """Record the operator's judgement of an event. The measured label never changes; an
    accepted 'tonal ridge only' event may then be used in a search, and the override is
    recorded in that search's manifest."""
    from aerodoppler.models import utc_now_iso

    ev = store.get_event(_safe_id(event_id, "event"))
    if ev is None or ev.recording_id != rec_id:
        raise api_error(404, "event_not_found", "event not found")
    extra = dict(ev.extra or {})
    extra["review"] = {"accepted": body.accepted, "note": body.note, "at": utc_now_iso()}
    ev = store.add_event(ev.model_copy(update={"extra": extra, "assessment": None}))
    store.audit("event.review", ev.id, extra["review"])
    return S.EventList(events=_with_assessment([ev]))


@router.post("/api/recordings/{rec_id}/events/detect", response_model=S.EventList)
async def detect_events(
    rec_id: str,
    body: S.DetectRequest | None = None,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> S.EventList:
    body = body or S.DetectRequest()
    if store.get_recording(_safe_id(rec_id, "recording")) is None:
        raise HTTPException(status_code=404, detail="recording not found")
    diag: dict[str, Any] = {}
    try:
        events = await asyncio.to_thread(
            propose_events,
            store,
            settings,
            rec_id,
            DetectorConfig(threshold_z=body.threshold_z, min_duration_s=body.min_duration_s),
            body.save,
            True,
            diag,
        )
    except NotFound as e:
        raise api_error(404, "recording_not_found", "recording not found") from e
    return S.EventList(events=_with_assessment(events), diagnostics=diag)


@router.delete("/api/recordings/{rec_id}/events/{event_id}")
def delete_event(
    rec_id: str, event_id: str, store: Store = Depends(get_store_dep)
) -> dict[str, Any]:
    ev = store.get_event(_safe_id(event_id, "event"))
    if ev is None or ev.recording_id != rec_id:
        raise HTTPException(status_code=404, detail="event not found")
    store.delete_event(event_id)
    return {"deleted": event_id}


@router.delete("/api/recordings/{rec_id}", response_model=S.PurgeResponse)
def delete_recording(
    rec_id: str,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> S.PurgeResponse:
    try:
        out = purge_recording(store, settings, _safe_id(rec_id, "recording"))
    except NotFound as e:
        raise HTTPException(status_code=404, detail="recording not found") from e
    return S.PurgeResponse(**out)


# ---- flight datasets --------------------------------------------------------------------


@router.get("/api/flight-datasets", response_model=S.FlightDatasetList)
def flight_datasets(settings: Settings = Depends(get_settings_dep)) -> S.FlightDatasetList:
    from aerodoppler.adsb.index import list_datasets

    out = []
    for ds in list_datasets(settings.normalized_dir):
        m = ds.manifest()
        out.append(
            S.FlightDatasetInfo(
                dataset_id=ds.dataset_id,
                provider=(m.get("provider") or {}).get("provider"),
                date_utc=(m.get("request") or {}).get("date_utc"),
                bbox=list((m.get("request") or {}).get("bbox") or []) or None,
                segments=(m.get("summary") or {}).get("segments"),
                aircraft=(m.get("summary") or {}).get("aircraft"),
                synthetic=bool(m.get("synthetic", False)),
                license=m.get("license"),
                storage_bytes=m.get("storage_bytes"),
            )
        )
    return S.FlightDatasetList(datasets=out)


@router.get("/api/flight-datasets/{dataset_id}")
def flight_dataset_detail(
    dataset_id: str, settings: Settings = Depends(get_settings_dep)
) -> dict[str, Any]:
    from aerodoppler.adsb.index import FlightDataset, FlightIndex

    ds = FlightDataset(_safe_id(dataset_id, "dataset"), settings.normalized_dir / dataset_id)
    if not ds.exists():
        raise HTTPException(status_code=404, detail="dataset not found")
    return {"manifest": ds.manifest(), "coverage": FlightIndex([ds]).coverage_summary()}


@router.post("/api/flight-datasets/import", status_code=202)
def flight_import(
    body: S.FlightImportRequest,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> dict[str, Any]:
    """Start (or reuse) a persistent import job; poll ``/api/imports/{id}`` for progress.

    Cheap checks happen here, before any job or download: date format, a day that has not
    ended or predates the archive, a missing local path, an unconfirmed multi-GB download."""
    import datetime as dt

    from aerodoppler.adsb.adsblol import ARCHIVE_FIRST_DAY, availability_kind_for_missing
    from aerodoppler.adsb.ingest import dataset_id_for
    from aerodoppler.jobs.imports import ImportRejected, submit_import

    try:
        day = dt.date.fromisoformat(body.date_utc)
    except ValueError as e:
        raise api_error(
            422, "invalid_date", f"{body.date_utc} is not a valid date", field="date_utc"
        ) from e
    if body.provider == "local":
        if not body.local_path or not Path(body.local_path).exists():
            raise api_error(
                422,
                "local_path_missing",
                "a local provider needs an existing local_path",
                field="local_path",
            )
    else:
        kind = availability_kind_for_missing(body.date_utc)
        if kind == "future_date":
            raise api_error(
                422,
                "future_date",
                f"{body.date_utc} (UTC) has not ended yet, so ADSB.lol cannot have its archive. "
                "Days are published a few hours after they end.",
                field="date_utc",
                action="pick a past UTC day",
            )
        if kind == "before_archive_start":
            raise api_error(
                422,
                "before_archive_start",
                f"ADSB.lol archives start on {ARCHIVE_FIRST_DAY.isoformat()}; there is no flight "
                f"data for {day.isoformat()} from this provider, so flight matching is impossible "
                "for this recording.",
                field="date_utc",
            )
        if not body.confirm_download and body.storage_budget_gb is None:
            raise api_error(
                409,
                "download_not_confirmed",
                "ADSB.lol publishes one whole-sky archive per UTC day (about 3.5 GB). It is "
                "downloaded once and cached, even for a small area, because the upstream format "
                "cannot be subset remotely.",
                field="confirm_download",
                action="tick the download confirmation and retry",
            )
    ds_id = dataset_id_for(
        "local" if body.provider == "local" else "adsblol",
        body.date_utc,
        body.bbox,
        body.margin_km,
        body.exclude_jets,
    )
    if not body.force and (settings.normalized_dir / ds_id / "manifest.json").exists():
        return {"outcome": "exists", "dataset_id": ds_id, "job": None}
    request = body.model_dump()
    request["bbox"] = body.bbox.model_dump()
    try:
        job, outcome = submit_import(store, settings, request)
    except ImportRejected as e:
        raise api_error(
            429, "import_slot_busy", str(e), action="wait for it or cancel it on the Jobs page"
        ) from e
    return {"outcome": outcome, "dataset_id": ds_id, "job": _import_view(job)}


def _import_view(job: dict[str, Any]) -> dict[str, Any]:
    from aerodoppler.jobs.proc import is_stalled, job_process_alive

    out = {k: v for k, v in job.items() if k not in ("pid_started_at",)}
    out["kind"] = "import"
    out["stalled"] = is_stalled(job)
    out["worker_alive"] = job_process_alive(job) if job["status"] == "running" else False
    if out.get("error"):
        out["error"] = scrub(out["error"])
    return out


@router.get("/api/imports")
def list_imports(store: Store = Depends(get_store_dep)) -> dict[str, Any]:
    from aerodoppler.jobs.imports import reconcile_stale_imports

    reconcile_stale_imports(store)
    return {"imports": [_import_view(j) for j in store.list_imports()]}


@router.get("/api/imports/{import_id}")
def get_import(import_id: str, store: Store = Depends(get_store_dep)) -> dict[str, Any]:
    from aerodoppler.jobs.imports import reconcile_stale_imports

    reconcile_stale_imports(store)
    job = store.get_import(_safe_id(import_id, "import"))
    if job is None:
        raise api_error(404, "not_found", "import not found")
    return _import_view(job)


@router.get("/api/imports/{import_id}/log")
def import_log(
    import_id: str, lines: int = Query(200, ge=1, le=5000), store: Store = Depends(get_store_dep)
) -> dict[str, Any]:
    job = store.get_import(_safe_id(import_id, "import"))
    if job is None:
        raise api_error(404, "not_found", "import not found")
    p = Path(job["log_path"]) if job.get("log_path") else None
    text = p.read_text(errors="replace").splitlines()[-lines:] if p and p.exists() else []
    return {"lines": [scrub(t) for t in text]}


@router.post("/api/imports/{import_id}/cancel")
def cancel_import_route(import_id: str, store: Store = Depends(get_store_dep)) -> dict[str, Any]:
    from aerodoppler.jobs.imports import cancel_import

    if not cancel_import(store, _safe_id(import_id, "import")):
        raise api_error(409, "not_running", "this import is not running")
    job = store.get_import(import_id)
    assert job is not None
    return _import_view(job)


@router.post("/api/imports/{import_id}/retry", status_code=202)
def retry_import(
    import_id: str,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> dict[str, Any]:
    """Start a new job with the same request (cached and partial downloads are reused)."""
    from aerodoppler.jobs.imports import ImportRejected, submit_import

    job = store.get_import(_safe_id(import_id, "import"))
    if job is None:
        raise api_error(404, "not_found", "import not found")
    if job["status"] in ("queued", "running"):
        raise api_error(409, "still_running", "this import is still running")
    try:
        new, outcome = submit_import(store, settings, job["request"])
    except ImportRejected as e:
        raise api_error(429, "import_slot_busy", str(e)) from e
    return {"outcome": outcome, "job": _import_view(new)}


@router.get("/api/jobs")
def all_jobs(
    settings: Settings = Depends(get_settings_dep), store: Store = Depends(get_store_dep)
) -> dict[str, Any]:
    """Every job on this machine (all recordings), with live counts and resource limits."""
    from aerodoppler.jobs.imports import reconcile_stale_imports
    from aerodoppler.jobs.proc import is_stalled
    from aerodoppler.jobs.runner import reconcile_stale_jobs
    from aerodoppler.resources import disk_free_bytes, snapshot

    reconcile_stale_jobs(store)
    reconcile_stale_imports(store)
    searches = store.list_searches()[:50]
    imports = store.list_imports(limit=50)
    recs = {r.id: r.original_filename for r in store.list_recordings()}
    sv = []
    for j in searches:
        sv.append(
            {
                "kind": "search",
                "id": j["id"],
                "recording_id": j["recording_id"],
                "recording": recs.get(j["recording_id"]),
                "status": j["status"],
                "stage": j["stage"],
                "created_at": j["created_at"],
                "updated_at": j["updated_at"],
                "error": scrub(j["error"]) if j.get("error") else None,
                "progress": j["progress"],
                "stalled": is_stalled(j),
                "mode": (j["request"] or {}).get("mode"),
            }
        )
    return {
        "searches": sv,
        "imports": [_import_view(j) for j in imports],
        "counts": {
            "searches_active": sum(1 for j in searches if j["status"] in ("queued", "running")),
            "searches_limit": settings.max_concurrent_jobs,
            "imports_active": sum(1 for j in imports if j["status"] in ("queued", "running")),
            "imports_limit": settings.max_concurrent_imports,
        },
        "resources": {
            **snapshot().as_dict(),
            "data_dir": str(settings.data_dir),
            "disk_free_gb": round(disk_free_bytes(settings.data_dir) / 1e9, 1),
        },
    }


# ---- searches ---------------------------------------------------------------------------


def _validated_search(
    body: S.SearchCreateRequest, settings: Settings, store: Store
) -> tuple[Any, SearchRequest]:
    """Check every prerequisite of a search and return ``(recording, request)`` with the
    flight datasets resolved explicitly; raises structured errors naming field and fix."""
    import datetime as dt

    from aerodoppler.search.datasets import DatasetSelectionError, resolve_datasets, search_dates

    rec = store.get_recording(_safe_id(body.recording_id, "recording"))
    if rec is None:
        raise api_error(404, "recording_not_found", "recording not found", field="recording_id")
    from aerodoppler.search.events import assess_event, usable_as_evidence

    for eid in body.event_ids:
        ev = store.get_event(_safe_id(eid, "event"))
        if ev is None or ev.recording_id != rec.id:
            raise api_error(
                404,
                "event_not_found",
                f"event {eid} is not an event of this recording",
                field="event_ids",
            )
        a = assess_event(ev)
        if not usable_as_evidence(a):
            raise api_error(
                422,
                "event_not_doppler",
                f"event {eid} ({ev.audio_start_s:.0f}–{ev.audio_end_s:.0f} s) is a tonal ridge "
                f"without a measurable Doppler transition: {'; '.join(a['reasons'])}. It cannot "
                "locate the recording.",
                field="event_ids",
                action="deselect it, or review it in the Spectrogram lab and accept it if you are sure it is an aircraft",
                event_id=eid,
            )
    if body.mode == SearchMode.KNOWN_TIME:
        if not body.start_utc:
            raise api_error(
                422,
                "start_utc_required",
                "Known-time mode needs the recording's verified start time in UTC.",
                field="start_utc",
                action="enter it (e.g. 2026-11-03T15:30:00Z), or use unknown-time mode with the UTC date",
            )
        try:
            t = dt.datetime.fromisoformat(body.start_utc.replace("Z", "+00:00"))
        except ValueError as e:
            raise api_error(
                422,
                "start_utc_invalid",
                f"'{body.start_utc}' is not an ISO-8601 UTC time",
                field="start_utc",
                action="use the form 2026-11-03T15:30:00Z",
            ) from e
        if t.tzinfo is None:
            raise api_error(
                422,
                "start_utc_no_timezone",
                "the start time has no timezone",
                field="start_utc",
                action="add Z for UTC or an explicit offset",
            )
        if body.start_uncertainty_s is None:
            raise api_error(
                422,
                "uncertainty_required",
                "Known-time mode needs the uncertainty of that start time in seconds.",
                field="start_uncertainty_s",
                action="enter how far off the start time could be (e.g. 10 for a synced phone clock)",
            )
    else:
        if not body.date_utc:
            raise api_error(
                422,
                "date_required",
                "Unknown-time mode needs the recording's UTC date.",
                field="date_utc",
            )
        if len(body.event_ids) < 2:
            raise api_error(
                422,
                "two_events_required",
                "Unknown-time mode needs two events (two aircraft passes): their time "
                "separation is what replaces the unknown clock.",
                field="event_ids",
                action="detect or draw a second event, or use known-time mode if the start time is known",
            )
    try:
        req = SearchRequest(**body.model_dump())
    except ValueError as e:
        raise api_error(422, "invalid_search", str(e)) from e
    try:
        ds_ids, _warn = resolve_datasets(
            settings.normalized_dir,
            dates=search_dates(
                req.date_utc, req.start_utc, rec.duration_s, float(req.start_uncertainty_s or 0.0)
            ),
            recording_synthetic=rec.synthetic,
            requested=req.flight_dataset_ids,
            bbox=req.bbox,
            allow_date_mismatch=req.allow_date_mismatch,
            allow_synthetic_mix=req.allow_synthetic_mix,
        )
    except DatasetSelectionError as e:
        raise api_error(422, e.code, str(e), field=e.field, action=e.action) from e
    return rec, req.model_copy(update={"flight_dataset_ids": ds_ids})


@router.post("/api/searches/estimate")
def estimate_search(
    body: S.SearchCreateRequest,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> dict[str, Any]:
    """Dry run: validate the request and count the work, without starting a job."""
    from aerodoppler.search.estimate import estimate_workload

    rec, req = _validated_search(body, settings, store)
    return estimate_workload(settings, rec, req)


@router.post("/api/searches", response_model=S.SearchStatus, status_code=202)
def create_search(
    body: S.SearchCreateRequest,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> S.SearchStatus:
    """Validate every prerequisite before a job exists; errors name the field and the fix."""
    _rec, req = _validated_search(body, settings, store)
    area = req.bbox.approx_area_km2() if req.bbox else 0
    if area > 40_000 and not req.national_scale_confirmed:
        raise api_error(
            422,
            "large_area_unconfirmed",
            f"the search area is {area:.0f} km²; areas above 40,000 km² are slow and memory heavy",
            field="national_scale_confirmed",
            action="confirm the large-scale search or shrink the area",
        )
    try:
        run_id = submit_search(store, settings, req)
    except JobRejected as e:
        raise api_error(
            429,
            "search_slots_full",
            str(e),
            action="wait for a search to finish or cancel one on the Jobs page",
        ) from e
    return _status(store, run_id)


def _status(store: Store, run_id: str) -> S.SearchStatus:
    job = store.get_search(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="search not found")
    return S.SearchStatus(
        id=job["id"],
        recording_id=job["recording_id"],
        status=job["status"],
        stage=job["stage"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        progress=job["progress"],
        error=job["error"],
        cancel_requested=job["cancel_requested"],
        cpu_seconds=job["cpu_seconds"],
        peak_rss_mb=job["peak_rss_mb"],
        request=job["request"],
    )


@router.get("/api/searches")
def list_searches(
    recording_id: str | None = None, store: Store = Depends(get_store_dep)
) -> dict[str, Any]:
    return {"searches": store.list_searches(recording_id)}


@router.get("/api/searches/{run_id}", response_model=S.SearchStatus)
def search_status(run_id: str, store: Store = Depends(get_store_dep)) -> S.SearchStatus:
    return _status(store, _safe_id(run_id, "search"))


def _load_result(store: Store, run_id: str) -> tuple[dict[str, Any], SearchResult | None]:
    job = store.get_search(_safe_id(run_id, "search"))
    if job is None:
        raise HTTPException(status_code=404, detail="search not found")
    p = Path(job["result_dir"]) / "result.json"
    if not p.exists():
        return job, None
    return job, SearchResult.model_validate_json(p.read_text())


@router.get("/api/searches/{run_id}/candidates", response_model=S.CandidatesResponse)
def search_candidates(run_id: str, store: Store = Depends(get_store_dep)) -> S.CandidatesResponse:
    job, res = _load_result(store, run_id)
    if res is None:
        return S.CandidatesResponse(
            run_id=run_id,
            status=job["status"],
            candidates=[],
            stage_counters=[],
            abstention=None,
            warnings=[job["error"]] if job.get("error") else [],
            score_label="relative fit score (uncalibrated)",
            synthetic=False,
        )
    return S.CandidatesResponse(
        run_id=run_id,
        status=job["status"],
        candidates=res.candidates,
        stage_counters=res.stage_counters,
        abstention=res.abstention,
        warnings=res.warnings,
        score_label=res.score_label,
        synthetic=bool(res.manifest.get("synthetic_data")),
        confidence_grade=res.confidence_grade,
        gates=[g.model_dump() for g in res.gates],
    )


@router.get("/api/searches/{run_id}/candidates/{candidate_id}/curves")
def candidate_curves(run_id: str, candidate_id: str, store: Store = Depends(get_store_dep)) -> Any:
    job = store.get_search(_safe_id(run_id, "search"))
    if job is None:
        raise HTTPException(status_code=404, detail="search not found")
    p = Path(job["result_dir"]) / "curves" / f"{_safe_id(candidate_id, 'candidate')}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="candidate curves not found")
    return JSONResponse(content=json.loads(p.read_text()))


@router.get("/api/searches/{run_id}/tracks")
def search_tracks(run_id: str, store: Store = Depends(get_store_dep)) -> Any:
    job = store.get_search(_safe_id(run_id, "search"))
    if job is None:
        raise HTTPException(status_code=404, detail="search not found")
    p = Path(job["result_dir"]) / "tracks.json"
    if not p.exists():
        return JSONResponse(content=[])
    return JSONResponse(content=json.loads(p.read_text()))


@router.get("/api/searches/{run_id}/heatmap")
def search_heatmap(run_id: str, store: Store = Depends(get_store_dep)) -> Any:
    job = store.get_search(_safe_id(run_id, "search"))
    if job is None:
        raise HTTPException(status_code=404, detail="search not found")
    p = Path(job["result_dir"]) / "heatmap.csv"
    if not p.exists():
        return JSONResponse(content={"points": []})
    rows = p.read_text().strip().split("\n")[1:]
    pts = []
    for r in rows:
        lat, lon, _e, _n, score, stage = r.split(",")
        pts.append([float(lat), float(lon), float(score), stage])
    return JSONResponse(content={"points": pts})


@router.get("/api/searches/{run_id}/events")
async def search_events(
    run_id: str, request: Request, store: Store = Depends(get_store_dep)
) -> EventSourceResponse:
    """Server-sent progress events until the job reaches a terminal state."""
    _safe_id(run_id, "search")

    async def gen():
        last = None
        while True:
            if await request.is_disconnected():
                break
            job = store.get_search(run_id)
            if job is None:
                yield {"event": "error", "data": json.dumps({"detail": "not found"})}
                break
            snap = json.dumps(
                {
                    "status": job["status"],
                    "stage": job["stage"],
                    "progress": job["progress"],
                    "error": job["error"],
                },
                default=str,
            )
            if snap != last:
                yield {"event": "progress", "data": snap}
                last = snap
            if job["status"] in ("succeeded", "failed", "cancelled", "abstained"):
                yield {"event": "done", "data": snap}
                break
            await asyncio.sleep(1.0)

    return EventSourceResponse(gen())


@router.post("/api/searches/{run_id}/cancel", response_model=S.SearchStatus)
def search_cancel(run_id: str, store: Store = Depends(get_store_dep)) -> S.SearchStatus:
    if not cancel_search(store, _safe_id(run_id, "search")):
        raise HTTPException(status_code=409, detail="search is not running")
    return _status(store, run_id)


@router.get("/api/searches/{run_id}/export")
def search_export(
    run_id: str,
    format: str = Query("json", pattern="^(json|geojson|manifest|zip)$"),
    store: Store = Depends(get_store_dep),
) -> Any:
    job, res = _load_result(store, run_id)
    out_dir = Path(job["result_dir"])
    if format == "manifest":
        p = out_dir / "manifest.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="manifest not available yet")
        return FileResponse(p, media_type="application/json", filename=f"{run_id}_manifest.json")
    if format == "zip":
        tmp = Path(tempfile.mkdtemp()) / f"{run_id}"
        shutil.make_archive(str(tmp), "zip", out_dir)
        return FileResponse(
            str(tmp) + ".zip", media_type="application/zip", filename=f"{run_id}.zip"
        )
    if res is None:
        raise HTTPException(status_code=404, detail="result not available yet")
    if format == "json":
        return JSONResponse(content=json.loads(res.model_dump_json()))
    return JSONResponse(content=result_to_geojson(res))


def result_to_geojson(res: SearchResult) -> dict[str, Any]:
    feats = []
    for c in res.candidates:
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
                "properties": {
                    "rank": c.rank,
                    "score_total": c.score_total,
                    "score_label": res.score_label,
                    "flight_segment_ids": c.flight_segment_ids,
                    "inferred_start_utc": c.inferred_start_utc,
                    "ground_elevation_m": c.ground_elevation_m,
                    "estimated_uncertainty_m": c.estimated_uncertainty_m,
                    "uncertainty_label": c.uncertainty_label,
                    "flags": c.flags,
                    "nominal_grid_spacing_m": c.nominal_grid_spacing_m,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": feats,
        "properties": {
            "run_id": res.run_id,
            "abstention": res.abstention.model_dump(),
            "warnings": res.warnings,
            "synthetic": bool(res.manifest.get("synthetic_data")),
            "attribution": [
                d["manifest"].get("attribution") for d in res.manifest.get("datasets", [])
            ],
        },
    }


@router.get("/api/audit")
def audit_log(
    limit: int = Query(200, ge=1, le=2000), store: Store = Depends(get_store_dep)
) -> dict[str, Any]:
    return {"entries": store.audit_log(limit)}
