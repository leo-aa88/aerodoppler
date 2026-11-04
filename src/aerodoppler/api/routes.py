"""HTTP routes (SPEC §11). Localhost-only by default; every recording ingest and search needs
an explicit owner-permission acknowledgement; no data leaves the machine."""

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
from aerodoppler.models import SearchRequest, SearchResult
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
    owner_permission_confirmed: bool = Form(False),
    recording_date_utc: str | None = Form(None),
    start_time_utc: str | None = Form(None),
    start_time_uncertainty_s: float | None = Form(None),
    synthetic: bool = Form(False),
    notes: str | None = Form(None),
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> S.RecordingCreateResponse:
    if not owner_permission_confirmed:
        raise HTTPException(
            status_code=403, detail="owner_permission_confirmed must be true: " + AUTH_NOTICE
        )
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
                    raise HTTPException(
                        status_code=413, detail=f"upload exceeds {settings.max_upload_mb} MB"
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
            )
        except AudioDecodeError as e:
            raise HTTPException(status_code=422, detail=f"audio could not be decoded: {e}") from e
    warnings = []
    if not rec.recording_date_utc and not rec.start_time_utc:
        warnings.append(
            "no recording date/time supplied: a search will need --date (unknown-time) or --start-utc (known-time)"
        )
    return S.RecordingCreateResponse(recording=rec, warnings=warnings)


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


@router.get("/api/recordings/{rec_id}/events", response_model=S.EventList)
def list_events(rec_id: str, store: Store = Depends(get_store_dep)) -> S.EventList:
    if store.get_recording(_safe_id(rec_id, "recording")) is None:
        raise HTTPException(status_code=404, detail="recording not found")
    return S.EventList(events=store.list_events(rec_id))


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
    try:
        events = await asyncio.to_thread(
            propose_events,
            store,
            settings,
            rec_id,
            DetectorConfig(threshold_z=body.threshold_z, min_duration_s=body.min_duration_s),
            body.save,
        )
    except NotFound as e:
        raise HTTPException(status_code=404, detail="recording not found") from e
    return S.EventList(events=events)


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


@router.post("/api/flight-datasets/import", response_model=S.FlightImportResponse)
async def flight_import(
    body: S.FlightImportRequest, settings: Settings = Depends(get_settings_dep)
) -> S.FlightImportResponse:
    from aerodoppler.adsb.adsblol import AdsbLolProvider, ProviderUnavailable
    from aerodoppler.adsb.ingest import ImportRequest, StorageBudgetExceeded, import_flights
    from aerodoppler.adsb.providers import LocalTraceProvider

    if body.provider == "local":
        if not body.local_path:
            raise HTTPException(status_code=422, detail="local_path required for provider=local")
        p = Path(body.local_path).resolve()
        if not p.exists():
            raise HTTPException(status_code=422, detail="local_path does not exist")
        prov: Any = LocalTraceProvider(p)
    else:
        prov = AdsbLolProvider(offline=settings.offline)
    req = ImportRequest(
        date_utc=body.date_utc,
        bbox=body.bbox,
        margin_km=body.margin_km,
        storage_budget_gb=body.storage_budget_gb,
        confirm_download=body.confirm_download,
    )
    try:
        ds = await asyncio.to_thread(
            import_flights, prov, req, settings.raw_dir, settings.normalized_dir, False
        )
    except StorageBudgetExceeded as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ProviderUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    m = ds.manifest()
    return S.FlightImportResponse(
        dataset_id=ds.dataset_id,
        summary=m["summary"],
        filter_counts={k: v for k, v in m["filter_counts"].items() if k != "errors"},
        storage_bytes=m["storage_bytes"],
        license=m["license"],
    )


# ---- searches ---------------------------------------------------------------------------


@router.post("/api/searches", response_model=S.SearchStatus, status_code=202)
def create_search(
    body: S.SearchCreateRequest,
    settings: Settings = Depends(get_settings_dep),
    store: Store = Depends(get_store_dep),
) -> S.SearchStatus:
    if not body.operator_authorized:
        raise HTTPException(
            status_code=403, detail="operator_authorized must be true: " + AUTH_NOTICE
        )
    rec = store.get_recording(_safe_id(body.recording_id, "recording"))
    if rec is None:
        raise HTTPException(status_code=404, detail="recording not found")
    for eid in body.event_ids:
        ev = store.get_event(_safe_id(eid, "event"))
        if ev is None or ev.recording_id != rec.id:
            raise HTTPException(status_code=404, detail=f"event {eid} not found for this recording")
    try:
        req = SearchRequest(**body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    area = req.bbox.approx_area_km2() if req.bbox else 0
    if area > 40_000 and not req.national_scale_confirmed:
        raise HTTPException(
            status_code=422,
            detail=f"bbox area {area:.0f} km² is a large-scale search; set national_scale_confirmed=true to run it (slow, resource intensive)",
        )
    try:
        run_id = submit_search(store, settings, req)
    except JobRejected as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
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
