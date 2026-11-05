"""Job-management API: submit, list/filter, inspect, cancel, retry, progress events, logs.

``POST`` endpoints that start work answer ``202 Accepted`` with ``{job_id, type, status,
created_at, status_url}``. Progress is available as Server-Sent Events with monotonically
increasing ids (``Last-Event-ID`` resumes after a reconnect) and, for clients without SSE, as
JSON from the same endpoint with ``?format=json&after=<seq>``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from aerodoppler.api.errors import api_error, scrub
from aerodoppler.config import Settings
from aerodoppler.jobs.queue import ACTIVE, TERMINAL, Job, SqliteJobQueue
from aerodoppler.jobs.store import Store

router = APIRouter()
STALL_S = 120.0


def get_queue(request: Request) -> SqliteJobQueue:
    return request.app.state.queue


def _iso(t: float | None) -> str | None:
    return dt.datetime.fromtimestamp(t, dt.UTC).isoformat().replace("+00:00", "Z") if t else None


def job_view(job: Job, debug: bool = False) -> dict[str, Any]:
    """The public shape of a job (no worker token, scrubbed errors, timings)."""
    from aerodoppler.jobs.scheduler import worker_alive

    now = time.time()
    running = job.status in ("claimed", "running")
    queue_wait = (
        ((job.claimed_at or now) - job.created_at)
        if job.status != "cancelled" or job.claimed_at
        else None
    )
    details = (
        job.error_details
        if debug
        else (
            {k: v for k, v in (job.error_details or {}).items() if k != "traceback"} or None
            if isinstance(job.error_details, dict)
            else job.error_details
        )
    )
    return {
        "job_id": job.id,
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "stage": job.stage,
        "status_url": f"/api/jobs/{job.id}",
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "updated_at": _iso(job.updated_at),
        "finished_at": _iso(job.finished_at),
        "queue_wait_s": round(queue_wait, 1) if queue_wait is not None else None,
        "duration_s": round((job.finished_at or now) - job.started_at, 1)
        if job.started_at
        else None,
        "params": job.params,
        "recording_id": job.recording_id,
        "dataset_id": job.dataset_id,
        "mode": job.mode,
        "attempt": job.attempt,
        "max_attempts": job.max_attempts,
        "next_attempt_at": _iso(job.next_attempt_at),
        "cancel_requested": job.cancel_requested,
        "progress": job.progress,
        "resources": job.resources,
        "memory_reserved_mb": round(job.memory_mb),
        "resource_class": job.resource_class,
        "error": {
            "code": job.error_code,
            "message": scrub(job.error_message) if job.error_message else None,
            "details": details,
        }
        if job.error_message or job.error_code
        else None,
        "result": job.result,
        "has_log": bool(job.log_path and Path(job.log_path).exists()),
        "stalled": bool(running and job.heartbeat_at and now - job.heartbeat_at > STALL_S),
        "worker_alive": worker_alive(job) if running else False,
        "legacy": job.legacy,
    }


def accepted(job: Job, created: bool) -> dict[str, Any]:
    return {**job_view(job), "outcome": "created" if created else "reused"}


class JobSubmit(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=200)


@router.post("/api/jobs", status_code=202)
def submit(body: JobSubmit, request: Request) -> dict[str, Any]:
    """Generic submission. Validation is type-specific and happens before the job exists."""
    from aerodoppler.api.routes import validated_import, validated_search_params
    from aerodoppler.jobs.service import JobRejected, submit_job

    settings: Settings = request.app.state.settings
    store: Store = request.app.state.store
    if body.type == "flight_import":
        params, early = validated_import(body.params, settings)
        if early is not None:
            return early
    elif body.type == "geolocation_search":
        params = validated_search_params(body.params, settings, store)
    elif body.type in ("spectrogram_generation", "ridge_detection"):
        rid = str(body.params.get("recording_id", ""))
        rec = store.get_recording(rid)
        if rec is None:
            raise api_error(
                404, "recording_not_found", "recording not found", field="params.recording_id"
            )
        params = {**body.params, "duration_s": rec.duration_s}
    elif body.type == "benchmark":
        if body.params.get("kind", "synthetic") not in (
            "synthetic",
            "real_trajectories",
            "livermore",
        ):
            raise api_error(
                422,
                "invalid_params",
                "benchmark kind must be synthetic, real_trajectories or livermore",
                field="params.kind",
            )
        params = body.params
    elif body.type == "audio_ingest":
        raise api_error(
            422, "use_upload", "audio ingest jobs are created by POST /api/recordings (file upload)"
        )
    else:
        raise api_error(422, "unknown_type", f"unknown job type '{body.type}'", field="type")
    try:
        job, created = submit_job(
            settings,
            body.type,
            params,
            queue=get_queue(request),
            idempotency_key=body.idempotency_key,
        )
    except JobRejected as e:
        raise api_error(422, e.code, str(e)) from e
    return accepted(job, created)


def _ts(v: str | None) -> float | None:
    if not v:
        return None
    try:
        return dt.datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except ValueError as e:
        raise api_error(422, "invalid_date", f"'{v}' is not an ISO date/time") from e


@router.get("/api/jobs")
def list_jobs(
    request: Request,
    type: str | None = Query(None, description="comma-separated job types"),
    status: str | None = Query(None, description="comma-separated statuses, or 'active'"),
    recording_id: str | None = None,
    dataset_id: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Every job on this machine (not only the selected recording's), newest first, with
    status counts and the scheduler's capacity and the machine's real limits."""
    from aerodoppler.jobs.scheduler import capacity_for
    from aerodoppler.resources import disk_free_bytes, snapshot

    q = get_queue(request)
    settings: Settings = request.app.state.settings
    statuses = list(ACTIVE) if status == "active" else (status.split(",") if status else None)
    jobs = q.list(
        types=type.split(",") if type else None,
        statuses=statuses,
        recording_id=recording_id,
        dataset_id=dataset_id,
        created_after=_ts(created_after),
        created_before=_ts(created_before),
        limit=limit,
    )
    recs = {r.id: r.original_filename for r in request.app.state.store.list_recordings()}
    views = []
    for j in jobs:
        v = job_view(j)
        v["recording"] = recs.get(j.recording_id or "")
        views.append(v)
    cap = capacity_for(settings)
    live = q.list(statuses=["claimed", "running"], limit=1000)
    counts = q.counts()
    return {
        "jobs": views,
        "counts": {
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0) + counts.get("claimed", 0),
            "failed": counts.get("failed", 0),
            "interrupted": counts.get("interrupted", 0),
            "by_status": counts,
        },
        "capacity": {
            "class_limits": cap.class_limits,
            "memory_budget_mb": round(cap.memory_budget_mb),
            "memory_reserved_mb": round(sum(j.memory_mb for j in live)),
            "running_by_class": {
                c: sum(1 for j in live if j.resource_class == c) for c in cap.class_limits
            },
            "scheduler": _scheduler_state(q),
        },
        "resources": {
            **snapshot().as_dict(),
            "data_dir": str(settings.data_dir),
            "disk_free_gb": round(disk_free_bytes(settings.data_dir) / 1e9, 1),
        },
    }


def _scheduler_state(q: SqliteJobQueue) -> dict[str, Any]:
    with q._conn() as c:
        row = c.execute("SELECT * FROM scheduler_lock WHERE name='main'").fetchone()
    if not row:
        return {
            "active": False,
            "note": "no scheduler is running: queued jobs wait until `aerodoppler serve` (or `aerodoppler jobs scheduler`) runs",
        }
    alive = row["lease_expires_at"] > time.time()
    return {"active": alive, "owner": row["owner"], "pid": row["pid"]}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request, debug: bool = False) -> dict[str, Any]:
    j = get_queue(request).get(job_id)
    if j is None:
        raise api_error(404, "not_found", "job not found")
    return job_view(j, debug=debug)


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    q = get_queue(request)
    j = q.get(job_id)
    if j is None:
        raise api_error(404, "not_found", "job not found")
    if j.status in TERMINAL:
        raise api_error(409, "not_active", f"this job is already {j.status}")
    j = q.request_cancel(job_id)
    assert j is not None
    return job_view(j)


@router.post("/api/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: str, request: Request) -> dict[str, Any]:
    q = get_queue(request)
    j = q.get(job_id)
    if j is None:
        raise api_error(404, "not_found", "job not found")
    if j.status not in ("failed", "interrupted", "cancelled"):
        raise api_error(
            409,
            "not_retryable",
            f"only failed, interrupted or cancelled jobs can be retried (this one is {j.status})",
        )
    if j.mode != "worker":
        raise api_error(
            409, "not_retryable", "inline CLI jobs are retried by running the command again"
        )
    from aerodoppler.jobs.service import retry_job as _retry

    j2 = _retry(request.app.state.settings, q, job_id)
    assert j2 is not None
    return job_view(j2)


@router.get("/api/jobs/{job_id}/logs")
def job_logs(
    job_id: str, request: Request, lines: int = Query(200, ge=1, le=5000)
) -> dict[str, Any]:
    j = get_queue(request).get(job_id)
    if j is None:
        raise api_error(404, "not_found", "job not found")
    p = Path(j.log_path) if j.log_path else None
    text = p.read_text(errors="replace").splitlines()[-lines:] if p and p.exists() else []
    return {"job_id": job_id, "lines": [scrub(t) for t in text]}


@router.get("/api/jobs/{job_id}/events")
async def job_events(
    job_id: str,
    request: Request,
    after: int = Query(0, ge=0, description="return events with seq > after"),
    format: str = Query("sse", pattern="^(sse|json)$"),
) -> Any:
    """Progress events. SSE by default (ids are the global event sequence; reconnecting with
    ``Last-Event-ID`` resumes exactly after the last event received); ``format=json`` returns
    the events after ``after`` plus the job's current state, for polling clients."""
    q = get_queue(request)
    j = q.get(job_id)
    if j is None:
        raise api_error(404, "not_found", "job not found")
    last_id = request.headers.get("last-event-id")
    start = int(last_id) if last_id and last_id.isdigit() else after
    if format == "json":
        evs = q.events(job_id, after=start)
        return {"job": job_view(j), "events": evs, "last_seq": evs[-1]["seq"] if evs else start}

    async def gen():
        seq = start
        yield {"event": "snapshot", "id": str(seq), "data": json.dumps(job_view(j), default=str)}
        while True:
            if await request.is_disconnected():
                break
            for e in q.events(job_id, after=seq):
                seq = e["seq"]
                yield {"event": e["kind"], "id": str(seq), "data": json.dumps(e, default=str)}
            cur = q.get(job_id)
            if cur is None or cur.status in TERMINAL:
                yield {
                    "event": "done",
                    "id": str(seq),
                    "data": json.dumps(job_view(cur) if cur else {}, default=str),
                }
                break
            await asyncio.sleep(1.0)

    return EventSourceResponse(gen(), ping=15)
