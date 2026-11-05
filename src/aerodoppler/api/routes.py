"""HTTP routes (SPEC §11). Localhost-only by default; no data leaves the machine. Errors are
structured (see api/errors.py); long work (imports, searches) runs as persistent jobs."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from aerodoppler import __version__
from aerodoppler.api import schemas as S
from aerodoppler.api.errors import api_error, scrub
from aerodoppler.audio.service import (
    NotFound,
    RecordingBusy,
    add_manual_event,
    purge_recording,
    spectrogram_payload,
)
from aerodoppler.config import Settings
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


@router.post("/api/recordings", status_code=202)
async def create_recording(
    request: Request,
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
    wait_s: float = Query(0.0, ge=0, le=600, description="wait this long for the ingest job"),
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> Any:
    """Upload a file and queue its decoding as an ``audio_ingest`` job (202 + job). The upload
    itself is the only work done in the request. ``start_time_utc`` is the UTC time of the
    file's first sample; files longer than the analysis limit are analysed in segments. With
    ``wait_s`` the request waits for the job and answers 201 with the recording when done."""
    from aerodoppler.audio.decode import ffprobe_duration
    from aerodoppler.audio.service import TimeMetadataError, check_time_metadata
    from aerodoppler.jobs.service import JobRejected, submit_job
    from aerodoppler.util import new_id

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
    stage_dir = settings.data_dir / "uploads" / new_id("upl")
    stage_dir.mkdir(parents=True, exist_ok=True)
    tmp = stage_dir / name
    size = 0
    with open(tmp, "wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                fh.close()
                shutil.rmtree(stage_dir, ignore_errors=True)
                raise api_error(
                    413,
                    "upload_too_large",
                    f"the file exceeds the {settings.max_upload_mb} MB upload limit",
                    field="file",
                    action="extract the audio track (e.g. ffmpeg -i video.mp4 -vn -c:a flac out.flac) or raise AERODOPPLER_MAX_UPLOAD_MB",
                )
            fh.write(chunk)
    probe: dict[str, Any] = {}
    try:
        import soundfile as sf

        info = sf.info(str(tmp))
        probe = {
            "probe_duration_s": info.duration,
            "probe_sample_rate": info.samplerate,
            "probe_channels": info.channels,
        }
    except Exception:
        dur = await asyncio.to_thread(ffprobe_duration, tmp)
        if dur:
            probe = {"probe_duration_s": dur}
    params = {
        "upload_path": str(tmp),
        "original_filename": file.filename or name,
        "recording_date_utc": recording_date_utc,
        "start_time_utc": start_time_utc,
        "start_time_uncertainty_s": start_time_uncertainty_s,
        "synthetic": synthetic,
        "notes": notes,
        "segment_start_s": segment_start_s,
        "segment_duration_s": segment_duration_s,
        "date_earliest_utc": date_earliest_utc,
        "date_latest_utc": date_latest_utc,
        **probe,
    }
    try:
        job, _ = submit_job(settings, "audio_ingest", params, queue=request.app.state.queue)
    except JobRejected as e:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise api_error(422, e.code, str(e), field="file") from e
    return await _maybe_wait(request, job.id, wait_s)


async def _maybe_wait(request: Request, job_id: str, wait_s: float) -> Any:
    """Return 202 + the job, or — when ``wait_s`` > 0 and the job finishes in time — 201 with
    the job and its result (compatibility for scripts that expect the object directly)."""
    from aerodoppler.api.jobs_routes import job_view
    from aerodoppler.jobs.queue import TERMINAL

    q = request.app.state.queue
    deadline = time.time() + wait_s
    job = q.get(job_id)
    while wait_s > 0 and job.status not in TERMINAL and time.time() < deadline:
        await asyncio.sleep(0.25)
        job = q.get(job_id)
    view = job_view(job)
    if job.status in ("succeeded", "abstained"):
        out = {**view, "job": view}
        rid = (job.result or {}).get("recording_id")
        if rid:
            rec = request.app.state.store.get_recording(rid)
            out["recording"] = rec.model_dump() if rec else None
            out["warnings"] = (job.result or {}).get("warnings", [])
        return JSONResponse(status_code=201, content=json.loads(json.dumps(out, default=str)))
    if job.status in TERMINAL:
        err = view.get("error") or {}
        raise api_error(
            422,
            err.get("code") or job.status,
            err.get("message") or f"job {job.status}",
            job_id=job.id,
        )
    return JSONResponse(status_code=202, content=json.loads(json.dumps(view, default=str)))


@router.post("/api/recordings/{rec_id}/segments", status_code=202)
async def create_recording_segment(
    rec_id: str,
    body: S.SegmentRequest,
    request: Request,
    wait_s: float = Query(0.0, ge=0, le=600),
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> Any:
    """Analyse another interval of the same preserved source file (an ``audio_ingest`` job)."""
    from aerodoppler.jobs.service import JobRejected, submit_job

    parent = store.get_recording(_safe_id(rec_id, "recording"))
    if parent is None:
        raise api_error(404, "recording_not_found", "recording not found")
    if parent.source_duration_s and body.start_s >= parent.source_duration_s:
        raise api_error(
            422,
            "segment_invalid",
            f"segment start {body.start_s:.0f} s is beyond the end of the file ({parent.source_duration_s:.0f} s)",
            field="start_s",
        )
    params = {
        "parent_recording_id": rec_id,
        "start_s": body.start_s,
        "duration_s": body.duration_s,
        "probe_sample_rate": parent.sample_rate_hz,
        "probe_channels": parent.channels,
        "probe_duration_s": body.duration_s,
    }
    try:
        job, _ = submit_job(settings, "audio_ingest", params, queue=request.app.state.queue)
    except JobRejected as e:
        raise api_error(422, e.code, str(e)) from e
    return await _maybe_wait(request, job.id, wait_s)


@router.get("/api/recordings", response_model=S.RecordingList)
def list_recordings(store: Store = Depends(get_store_dep)) -> S.RecordingList:
    return S.RecordingList(recordings=store.list_recordings())


@router.get("/api/recordings/{rec_id}")
def get_recording(
    rec_id: str, request: Request, store: Store = Depends(get_store_dep)
) -> dict[str, Any]:
    rec = store.get_recording(_safe_id(rec_id, "recording"))
    if rec is None:
        raise HTTPException(status_code=404, detail="recording not found")
    return {
        "recording": rec.model_dump(),
        "events": [e.model_dump() for e in store.list_events(rec.id)],
        "searches": [
            _search_view(j).model_dump()
            for j in request.app.state.queue.list(
                types=["geolocation_search"], recording_id=rec.id, limit=200
            )
        ],
    }


def _spectrogram_or_job(request: Request, rec: Any) -> Any:
    """The cached spectrogram, or a 202 with the (deduplicated) job computing it."""
    from aerodoppler.api.jobs_routes import job_view
    from aerodoppler.audio.spectrogram import Spectrogram
    from aerodoppler.jobs.service import JobRejected, submit_job

    store: Store = request.app.state.store
    p = store.spectrogram_path(rec.id)
    if p and p.exists():
        return Spectrogram.load(p)
    try:
        job, _ = submit_job(
            request.app.state.settings,
            "spectrogram_generation",
            {"recording_id": rec.id, "duration_s": rec.duration_s},
            queue=request.app.state.queue,
        )
    except JobRejected as e:
        raise api_error(422, e.code, str(e)) from e
    return JSONResponse(status_code=202, content=json.loads(json.dumps(job_view(job), default=str)))


@router.get("/api/recordings/{rec_id}/spectrogram", response_model=S.SpectrogramResponse)
def get_spectrogram(
    rec_id: str,
    request: Request,
    t0: float | None = Query(None, ge=0),
    t1: float | None = Query(None, ge=0),
    f0: float | None = Query(None, ge=0),
    f1: float | None = Query(None, ge=0),
    layer: str = Query("whitened", pattern="^(whitened|power)$"),
    max_frames: int = Query(1200, ge=50, le=4000),
    max_bins: int = Query(700, ge=50, le=2000),
    store: Store = Depends(get_store_dep),
) -> Any:
    """A view of the spectrogram. While it is being computed (first access of a recording)
    the answer is 202 with the spectrogram job; poll it and ask again."""
    rec = store.get_recording(_safe_id(rec_id, "recording"))
    if rec is None:
        raise api_error(404, "recording_not_found", "recording not found")
    spec = _spectrogram_or_job(request, rec)
    if isinstance(spec, JSONResponse):
        return spec
    p = spectrogram_payload(
        spec, t0, t1, f0, f1, max_frames=max_frames, max_bins=max_bins, layer=layer
    )
    return S.SpectrogramResponse(recording_id=rec.id, synthetic=rec.synthetic, **p)


@router.get("/api/recordings/{rec_id}/spectrogram.png")
def get_spectrogram_png(
    rec_id: str, request: Request, store: Store = Depends(get_store_dep)
) -> Response:
    rec = store.get_recording(_safe_id(rec_id, "recording"))
    if rec is None:
        raise api_error(404, "recording_not_found", "recording not found")
    spec = _spectrogram_or_job(request, rec)
    if isinstance(spec, JSONResponse):
        return spec
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


@router.post("/api/recordings/{rec_id}/events/detect", status_code=202)
async def detect_events(
    rec_id: str,
    request: Request,
    body: S.DetectRequest | None = None,
    wait_s: float = Query(0.0, ge=0, le=600),
    store: Store = Depends(get_store_dep),
) -> Any:
    """Queue automatic ridge detection (a ``ridge_detection`` job). Its result holds the new
    event ids and the diagnostics; the events themselves appear in the events list."""
    from aerodoppler.jobs.service import JobRejected, submit_job

    body = body or S.DetectRequest()
    rec = store.get_recording(_safe_id(rec_id, "recording"))
    if rec is None:
        raise api_error(404, "recording_not_found", "recording not found")
    try:
        job, _ = submit_job(
            request.app.state.settings,
            "ridge_detection",
            {
                "recording_id": rec_id,
                "threshold_z": body.threshold_z,
                "min_duration_s": body.min_duration_s,
                "save": body.save,
                "duration_s": rec.duration_s,
            },
            queue=request.app.state.queue,
        )
    except JobRejected as e:
        raise api_error(422, e.code, str(e)) from e
    resp = await _maybe_wait(request, job.id, wait_s)
    if resp.status_code == 201:
        j = request.app.state.queue.get(job.id)
        ids = set((j.result or {}).get("event_ids") or [])
        evs = [e for e in store.list_events(rec_id) if e.id in ids]
        payload = json.loads(resp.body)
        payload["events"] = [e.model_dump(mode="json") for e in _with_assessment(evs)]
        payload["diagnostics"] = (j.result or {}).get("diagnostics")
        return JSONResponse(status_code=201, content=json.loads(json.dumps(payload, default=str)))
    return resp


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
        raise api_error(404, "recording_not_found", "recording not found") from e
    except RecordingBusy as e:
        raise api_error(
            409, "recording_busy", str(e), action="cancel its jobs on the Jobs page, then delete"
        ) from e
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


def validated_import(
    raw: dict[str, Any], settings: Settings
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Check an import request before any job or download exists. Returns ``(params, None)``,
    or ``(params, response)`` when nothing needs to run (the dataset is already imported)."""
    import datetime as dt

    from aerodoppler.adsb.adsblol import ARCHIVE_FIRST_DAY, availability_kind_for_missing
    from aerodoppler.adsb.ingest import dataset_id_for

    try:
        body = S.FlightImportRequest.model_validate(raw)
    except ValueError as e:
        raise api_error(422, "invalid_request", str(e)) from e
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
    params = body.model_dump()
    params["bbox"] = body.bbox.model_dump()
    if not body.force and (settings.normalized_dir / ds_id / "manifest.json").exists():
        return params, {"outcome": "exists", "dataset_id": ds_id, "job": None, "job_id": None}
    return params, None


@router.post("/api/flight-datasets/import", status_code=202)
def flight_import(body: S.FlightImportRequest, request: Request) -> Any:
    """Start (or reuse) a persistent ``flight_import`` job; progress at ``/api/jobs/{id}``."""
    from aerodoppler.adsb.ingest import dataset_id_for
    from aerodoppler.api.jobs_routes import accepted
    from aerodoppler.jobs.service import JobRejected, submit_job

    settings: Settings = request.app.state.settings
    params, early = validated_import(body.model_dump(), settings)
    if early is not None:
        return JSONResponse(status_code=200, content=early)
    try:
        job, created = submit_job(settings, "flight_import", params, queue=request.app.state.queue)
    except JobRejected as e:
        raise api_error(422, e.code, str(e)) from e
    ds_id = dataset_id_for(
        "local" if body.provider == "local" else "adsblol",
        body.date_utc,
        body.bbox,
        body.margin_km,
        body.exclude_jets,
    )
    view = accepted(job, created)
    return {**view, "dataset_id": ds_id, "job": view}


# compatibility aliases for the pre-queue import endpoints (documented in docs/jobs.md)
@router.get("/api/imports")
def list_imports(request: Request) -> dict[str, Any]:
    from aerodoppler.api.jobs_routes import job_view

    return {
        "imports": [
            job_view(j) for j in request.app.state.queue.list(types=["flight_import"], limit=100)
        ]
    }


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


def validated_search_params(
    raw: dict[str, Any], settings: Settings, store: Store
) -> dict[str, Any]:
    try:
        body = S.SearchCreateRequest.model_validate(raw)
    except ValueError as e:
        raise api_error(422, "invalid_request", str(e)) from e
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
    return req.model_dump(mode="json")


@router.post("/api/searches", response_model=S.SearchStatus, status_code=202)
def create_search(body: S.SearchCreateRequest, request: Request) -> S.SearchStatus:
    """Validate every prerequisite, then queue a ``geolocation_search`` job (202). The same
    request while an identical search is queued or running returns that search."""
    from aerodoppler.jobs.service import JobRejected, submit_job

    settings: Settings = request.app.state.settings
    params = validated_search_params(
        body.model_dump(mode="json"), settings, request.app.state.store
    )
    try:
        job, _ = submit_job(settings, "geolocation_search", params, queue=request.app.state.queue)
    except JobRejected as e:
        raise api_error(422, e.code, str(e)) from e
    return _search_view(job)


def _search_job(request: Request, run_id: str) -> Any:
    j = request.app.state.queue.get(_safe_id(run_id, "search"))
    if j is None or j.type != "geolocation_search":
        raise api_error(404, "not_found", "search not found")
    return j


def _search_view(job: Any) -> S.SearchStatus:
    from aerodoppler.api.jobs_routes import _iso

    prog = dict(job.progress or {})
    prog.setdefault(
        "elapsed_s",
        round(((job.finished_at or time.time()) - job.started_at), 1) if job.started_at else 0,
    )
    res = job.resources or {}
    prog.setdefault("cpu_seconds", res.get("cpu_s"))
    prog.setdefault("rss_mb", res.get("pss_mb"))
    return S.SearchStatus(
        id=job.id,
        recording_id=job.recording_id or (job.params or {}).get("recording_id", ""),
        status=job.status,
        stage=job.stage,
        created_at=_iso(job.created_at) or "",
        updated_at=_iso(job.updated_at) or "",
        progress=prog,
        error=scrub(job.error_message) if job.error_message else None,
        cancel_requested=job.cancel_requested,
        cpu_seconds=res.get("cpu_s") or res.get("cpu_seconds"),
        peak_rss_mb=res.get("peak_pss_mb") or res.get("peak_rss_mb"),
        request=job.params,
    )


@router.get("/api/searches")
def list_searches(request: Request, recording_id: str | None = None) -> dict[str, Any]:
    jobs = request.app.state.queue.list(
        types=["geolocation_search"], recording_id=recording_id, limit=500
    )
    return {"searches": [_search_view(j).model_dump() for j in jobs]}


@router.get("/api/searches/{run_id}", response_model=S.SearchStatus)
def search_status(run_id: str, request: Request) -> S.SearchStatus:
    return _search_view(_search_job(request, run_id))


def _result_dir(job: Any) -> Path:
    return Path(job.work_dir) if job.work_dir else Path(job.params.get("_result_dir", "."))


def _load_result(request: Request, run_id: str) -> tuple[Any, SearchResult | None]:
    """The job and its result; a directory still marked INCOMPLETE is never read as a result."""
    from aerodoppler.jobs.runner import INCOMPLETE

    job = _search_job(request, run_id)
    d = _result_dir(job)
    p = d / "result.json"
    if not p.exists() or (d / INCOMPLETE).exists():
        return job, None
    return job, SearchResult.model_validate_json(p.read_text())


@router.get("/api/searches/{run_id}/candidates", response_model=S.CandidatesResponse)
def search_candidates(run_id: str, request: Request) -> S.CandidatesResponse:
    job, res = _load_result(request, run_id)
    if res is None:
        return S.CandidatesResponse(
            run_id=run_id,
            status=job.status,
            candidates=[],
            stage_counters=[],
            abstention=None,
            warnings=[scrub(job.error_message)] if job.error_message else [],
            score_label="relative fit score (uncalibrated)",
            synthetic=False,
        )
    return S.CandidatesResponse(
        run_id=run_id,
        status=job.status,
        candidates=res.candidates,
        stage_counters=res.stage_counters,
        abstention=res.abstention,
        warnings=res.warnings,
        score_label=res.score_label,
        synthetic=bool(res.manifest.get("synthetic_data")),
        confidence_grade=res.confidence_grade,
        gates=[g.model_dump() for g in res.gates],
        alternatives=res.alternatives,
    )


@router.get("/api/searches/{run_id}/candidates/{candidate_id}/curves")
def candidate_curves(run_id: str, candidate_id: str, request: Request) -> Any:
    job = _search_job(request, run_id)
    p = _result_dir(job) / "curves" / f"{_safe_id(candidate_id, 'candidate')}.json"
    if not p.exists():
        raise api_error(404, "not_found", "candidate curves not found")
    return JSONResponse(content=json.loads(p.read_text()))


@router.get("/api/searches/{run_id}/tracks")
def search_tracks(run_id: str, request: Request) -> Any:
    p = _result_dir(_search_job(request, run_id)) / "tracks.json"
    return JSONResponse(content=json.loads(p.read_text()) if p.exists() else [])


@router.get("/api/searches/{run_id}/heatmap")
def search_heatmap(run_id: str, request: Request) -> Any:
    p = _result_dir(_search_job(request, run_id)) / "heatmap.csv"
    if not p.exists():
        return JSONResponse(content={"points": []})
    pts = []
    for r in p.read_text().strip().split("\n")[1:]:
        lat, lon, _e, _n, score, stage = r.split(",")
        pts.append([float(lat), float(lon), float(score), stage])
    return JSONResponse(content={"points": pts})


@router.get("/api/searches/{run_id}/events")
async def search_events(run_id: str, request: Request) -> EventSourceResponse:
    """Server-sent progress of a search (compatibility; ``/api/jobs/{id}/events`` is the
    general form with resumable event ids)."""
    from aerodoppler.jobs.queue import TERMINAL

    _search_job(request, run_id)
    q = request.app.state.queue

    async def gen():
        last = None
        while True:
            if await request.is_disconnected():
                break
            job = q.get(run_id)
            snap = json.dumps(
                {
                    "status": job.status,
                    "stage": job.stage,
                    "progress": job.progress,
                    "error": job.error_message,
                },
                default=str,
            )
            if snap != last:
                yield {"event": "progress", "data": snap}
                last = snap
            if job.status in TERMINAL:
                yield {"event": "done", "data": snap}
                break
            await asyncio.sleep(1.0)

    return EventSourceResponse(gen())


@router.post("/api/searches/{run_id}/cancel", response_model=S.SearchStatus)
def search_cancel(run_id: str, request: Request) -> S.SearchStatus:
    from aerodoppler.jobs.queue import TERMINAL

    job = _search_job(request, run_id)
    if job.status in TERMINAL:
        raise api_error(409, "not_active", "search is not running")
    return _search_view(request.app.state.queue.request_cancel(run_id))


@router.get("/api/searches/{run_id}/export")
def search_export(
    run_id: str,
    request: Request,
    format: str = Query("json", pattern="^(json|geojson|manifest|zip)$"),
) -> Any:
    job, res = _load_result(request, run_id)
    out_dir = _result_dir(job)
    if format == "manifest":
        p = out_dir / "manifest.json"
        if not p.exists():
            raise api_error(404, "not_found", "manifest not available yet")
        return FileResponse(p, media_type="application/json", filename=f"{run_id}_manifest.json")
    if format == "zip":
        tmp = Path(tempfile.mkdtemp()) / f"{run_id}"
        shutil.make_archive(str(tmp), "zip", out_dir)
        return FileResponse(
            str(tmp) + ".zip", media_type="application/zip", filename=f"{run_id}.zip"
        )
    if res is None:
        raise api_error(
            404, "not_found", "result not available (search unfinished, failed or incomplete)"
        )
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
