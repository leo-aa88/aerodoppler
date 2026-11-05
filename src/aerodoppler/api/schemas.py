"""API request/response models (SPEC §11). Strict validation, explicit permission fields."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from aerodoppler.models import (
    Abstention,
    BBox,
    DopplerEvent,
    LocationCandidate,
    QualitySettings,
    Recording,
    SearchMode,
    StageCounter,
)


class HealthResponse(BaseModel):
    status: str
    version: str
    offline: bool
    data_dir: str
    host: str
    authorization_notice: str


class RecordingCreateResponse(BaseModel):
    recording: Recording
    warnings: list[str] = Field(default_factory=list)


class RecordingList(BaseModel):
    recordings: list[Recording]


class SpectrogramResponse(BaseModel):
    recording_id: str
    times: list[float]
    freqs: list[float]
    values: list[list[float]]
    layer: str
    params: dict[str, Any]
    pooled: list[int]
    synthetic: bool


class ManualEventRequest(BaseModel):
    points: list[tuple[float, float]] = Field(
        ..., min_length=3, description="(relative_t_s, frequency_hz)"
    )
    harmonic: int = Field(default=1, ge=1, le=12)
    uncertainty_hz: float = Field(default=1.0, gt=0, le=50)
    annotated_by: str = Field(default="operator", max_length=80)

    @field_validator("points")
    @classmethod
    def _finite(cls, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for t, f in pts:
            if not (0 <= t <= 1e6) or not (1.0 <= f <= 20000.0):
                raise ValueError("ridge point out of range")
        return pts


class EventList(BaseModel):
    events: list[DopplerEvent]
    diagnostics: dict[str, Any] | None = None


class EventReview(BaseModel):
    accepted: bool = Field(..., description="true: I checked it and it is an aircraft pass")
    note: str | None = Field(default=None, max_length=500)


class DetectRequest(BaseModel):
    threshold_z: float = Field(default=2.5, ge=0.5, le=20)
    min_duration_s: float = Field(default=40.0, ge=5, le=3600)
    save: bool = True


class FlightDatasetInfo(BaseModel):
    dataset_id: str
    provider: str | None
    date_utc: str | None
    bbox: list[float] | None
    segments: int | None
    aircraft: int | None
    synthetic: bool
    license: str | None
    storage_bytes: int | None


class FlightDatasetList(BaseModel):
    datasets: list[FlightDatasetInfo]


class SegmentRequest(BaseModel):
    start_s: float = Field(..., ge=0, description="offset into the source file, seconds")
    duration_s: float | None = Field(
        default=None, gt=0, description="length; default = up to the analysis limit"
    )


class FlightImportRequest(BaseModel):
    provider: str = Field(default="adsblol", pattern="^(adsblol|local)$")
    date_utc: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    bbox: BBox
    margin_km: float = Field(default=15.0, ge=0, le=200)
    storage_budget_gb: float | None = Field(default=None, ge=0)
    confirm_download: bool = False
    local_path: str | None = None
    exclude_jets: bool = False
    memory_budget_mb: float | None = Field(default=None, ge=256, le=65536)
    workers: int = Field(default=0, ge=0, le=64)
    force: bool = False  # re-import even when the dataset already exists


class FlightImportResponse(BaseModel):
    dataset_id: str
    summary: dict[str, Any]
    filter_counts: dict[str, Any]
    storage_bytes: int
    license: str


class SearchCreateRequest(BaseModel):
    mode: SearchMode
    recording_id: str
    event_ids: list[str] = Field(..., min_length=1, max_length=8)
    date_utc: str | None = None
    start_utc: str | None = None
    start_uncertainty_s: float | None = Field(default=None, ge=0, le=86400)
    bbox: BBox
    flight_dataset_ids: list[str] = Field(default_factory=list)
    operator_authorized: bool = True
    national_scale_confirmed: bool = False
    max_cpu_hours: float | None = Field(default=1.0, gt=0, le=48)
    quality_settings: QualitySettings = Field(default_factory=QualitySettings)
    seed: int = 0
    allow_date_mismatch: bool = False
    allow_synthetic_mix: bool = False


class BlindSearchRequest(BaseModel):
    """Search without an area: the scope is the extent of the selected flight datasets."""

    mode: SearchMode
    recording_id: str
    event_ids: list[str] = Field(..., min_length=1, max_length=8)
    date_utc: str | None = None
    start_utc: str | None = None
    start_uncertainty_s: float | None = Field(default=None, ge=0, le=86400)
    flight_dataset_ids: list[str] = Field(default_factory=list)
    budget_s: float = Field(default=3600.0, ge=60, le=86400)
    allow_synthetic_mix: bool = False


class SearchStatus(BaseModel):
    id: str
    recording_id: str
    status: str
    stage: str | None
    created_at: str
    updated_at: str
    progress: dict[str, Any]
    error: str | None
    cancel_requested: bool
    cpu_seconds: float | None
    peak_rss_mb: float | None
    request: dict[str, Any]


class CandidatesResponse(BaseModel):
    run_id: str
    status: str
    candidates: list[LocationCandidate]
    stage_counters: list[StageCounter]
    abstention: Abstention | None
    warnings: list[str]
    score_label: str
    synthetic: bool
    confidence_grade: str = "low"
    gates: list[dict[str, Any]] = Field(default_factory=list)
    alternatives: list[dict[str, Any]] = Field(default_factory=list)


class PurgeResponse(BaseModel):
    recording_id: str
    removed: list[str]


class ErrorResponse(BaseModel):
    detail: str
