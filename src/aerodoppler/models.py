"""Core data contracts (SPEC §5).

Units: UTC unix seconds for absolute time, seconds for recording-relative time, metres and
m/s for geometry, hertz for frequency, WGS84 latitude/longitude in degrees.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def unix_to_iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def iso_to_unix(s: str) -> float:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


class AltitudeDatum(StrEnum):
    """Vertical reference of an altitude value."""

    BAROMETRIC_STD = "barometric_std_atm"  # pressure altitude, 1013.25 hPa reference
    GEOMETRIC_HAE = "geometric_wgs84_hae"  # GNSS height above WGS84 ellipsoid
    ORTHOMETRIC = "orthometric_geoid"  # height above geoid (DEM-style "MSL")
    UNKNOWN = "unknown"
    GROUND = "on_ground"


class AltitudeUnit(StrEnum):
    FEET = "ft"
    METRES = "m"


class PositionSource(StrEnum):
    ADSB = "adsb"
    MLAT = "mlat"
    TISB = "tisb"
    ADSR = "adsr"
    OTHER = "other"
    SYNTHETIC = "synthetic"


class BBox(BaseModel):
    """Geographic bounding box: west, south, east, north in degrees."""

    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def _check(self) -> BBox:
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("bbox longitude order must be west < east within [-180, 180]")
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("bbox latitude order must be south < north within [-90, 90]")
        return self

    @classmethod
    def parse(cls, text: str) -> BBox:
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must be WEST,SOUTH,EAST,NORTH")
        return cls(west=parts[0], south=parts[1], east=parts[2], north=parts[3])

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def expanded_km(self, margin_km: float) -> BBox:
        import math

        dlat = margin_km / 111.195
        clat = 0.5 * (self.south + self.north)
        dlon = margin_km / (111.195 * max(math.cos(math.radians(clat)), 1e-3))
        return BBox(
            west=max(-180.0, self.west - dlon),
            south=max(-90.0, self.south - dlat),
            east=min(180.0, self.east + dlon),
            north=min(90.0, self.north + dlat),
        )

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    def approx_area_km2(self) -> float:
        import math

        clat = math.radians(0.5 * (self.south + self.north))
        w = (self.east - self.west) * 111.195 * math.cos(clat)
        h = (self.north - self.south) * 111.195
        return abs(w * h)


class FlightSample(BaseModel):
    source: str
    aircraft_icao24: str
    utc_unix_s: float
    latitude: float
    longitude: float
    altitude_value: float | None
    altitude_unit: AltitudeUnit = AltitudeUnit.FEET
    altitude_datum: AltitudeDatum = AltitudeDatum.UNKNOWN
    groundspeed_mps: float | None = None
    true_track_deg: float | None = None
    vertical_rate_mps: float | None = None
    aircraft_type: str | None = None
    position_source: PositionSource | None = None
    quality_flags: int = 0
    # raw provenance of both ADS-B altitudes (feet) so the datum choice is auditable
    alt_baro_ft: float | None = None
    alt_geom_ft: float | None = None
    callsign: str | None = None


class FlightSegment(BaseModel):
    """A contiguous, kinematically plausible run of samples for one aircraft."""

    stable_id: str
    source: str
    aircraft_icao24: str
    first_utc: float
    last_utc: float
    n_samples: int
    completeness_score: float = Field(ge=0.0, le=1.0)
    aircraft_type: str | None = None
    callsign: str | None = None
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    min_alt_m: float | None = None
    max_alt_m: float | None = None
    max_speed_mps: float | None = None
    flags: list[str] = Field(default_factory=list)


class Recording(BaseModel):
    id: str
    sha256: str
    original_filename: str
    duration_s: float
    sample_rate_hz: int
    channels: int
    # The analysis copy may be one interval of a longer source file. ``duration_s`` is the
    # analysed interval; it starts ``segment_start_s`` seconds into the source. Event times are
    # relative to the analysed interval; source time = event time + segment_start_s.
    source_duration_s: float | None = None
    truncated: bool = False  # the source has audio after the analysed interval
    segment_start_s: float = 0.0
    parent_recording_id: str | None = None  # set when this is another segment of a source
    source_start_time_utc: str | None = None  # UTC at the first sample of the SOURCE file
    # UTC at the first sample of the ANALYSED interval (= source start + segment_start_s)
    start_time_utc: str | None = None
    start_time_uncertainty_s: float | None = None
    recording_date_utc: str | None = None
    # What the recording's time is based on, so nothing pretends to be more exact than it is:
    # verified_start (UTC start known ± uncertainty), date_only (UTC day known),
    # date_range (day between earliest and latest), upper_bound_only (e.g. a video's
    # publication date: recorded on or before it), unknown.
    time_basis: str = "unknown"
    date_earliest_utc: str | None = None
    date_latest_utc: str | None = None
    owner_permission_confirmed: bool = True
    created_at: str = Field(default_factory=utc_now_iso)
    synthetic: bool = False
    notes: str | None = None


class RidgePoint(BaseModel):
    relative_t_s: float
    frequency_hz: float
    uncertainty_hz: float = 1.0


class DopplerEvent(BaseModel):
    id: str | None = None
    recording_id: str
    audio_start_s: float
    audio_end_s: float
    ridge_points: list[RidgePoint]
    candidate_harmonic_numbers: list[int] = Field(default_factory=lambda: [1])
    f0_prior: float | None = None
    detection_method: str = "manual"
    detection_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    annotated_by: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    ridge_harmonic: int = Field(
        default=1, ge=1, description="harmonic number the ridge points belong to"
    )
    extra: dict[str, Any] = Field(default_factory=dict)
    # computed when served by the API (search/events.assess_event); not part of the stored row
    assessment: dict[str, Any] | None = Field(default=None, exclude=False)

    @field_validator("ridge_points")
    @classmethod
    def _sorted_points(cls, pts: list[RidgePoint]) -> list[RidgePoint]:
        return sorted(pts, key=lambda p: p.relative_t_s)

    @model_validator(mode="after")
    def _check_window(self) -> DopplerEvent:
        if self.audio_end_s <= self.audio_start_s:
            raise ValueError("audio_end_s must exceed audio_start_s")
        return self


class SearchMode(StrEnum):
    UNKNOWN_TIME = "unknown-time"  # Mode A
    KNOWN_TIME = "known-time"  # Mode B


class QualitySettings(BaseModel):
    coarse_grid_m: float = 200.0
    refine_grid_m: float = 100.0
    fine_grid_m: float = 25.0
    top_k_refine: int = 5
    event_separation_tolerance_s: float = 20.0
    alignment_sweep_s: float = 20.0
    alignment_step_s: float = 1.0
    f0_step_fraction: float = 0.004
    harmonics: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    max_slant_m: float = 12000.0
    speed_of_sound_mps: float | None = None
    temperature_c: float | None = None
    silence_constraint: bool = True
    silence_audible_m: float = 6000.0
    drift_fraction_max: float = 0.02
    prefer_propeller_types: bool = True
    max_candidates_coarse: int = 400_000
    coarse_t0_step_s: float = 2.0
    coarse_f0_step_fraction: float = 0.015
    coarse_polish_span: float = 1.0  # coarse-stage f0 polish range in coarse f0 steps (0 = off)
    coarse_keep_top: int = 400
    ultra_keep_top: int = 40_000  # pairs surviving the ultra-coarse screen (all pairs scored)
    ultra_t0_step_s: float = 4.0
    max_segments: int = 20_000  # memory guard: above it the search refuses (no silent subset)
    max_assignments: int = 20_000_000  # candidates held as arrays; memory guard only
    screen_budget_s: float = 1800.0  # wall-clock budget of the anytime screen (prior order)
    screen_batch: int = 0  # candidates per screening batch; 0 = max(20000, 4000 x workers)
    # numba batch scorer for the screen and the coarse stage (False = reference path)
    batched_screen: bool = True
    workers: int = 0  # 0 = cpu_count - 1
    ground_elevation_m: float | None = None  # explicit fallback when no DEM is available
    ground_elevation_sigma_m: float | None = None
    mic_height_m: float = 1.5


class SearchRequest(BaseModel):
    mode: SearchMode
    recording_id: str
    event_ids: list[str]
    date_utc: str | None = None
    time_window_utc: tuple[str, str] | None = None
    start_utc: str | None = None
    start_uncertainty_s: float | None = None
    bbox: BBox | None = None
    flight_dataset_ids: list[str] = Field(default_factory=list)
    operator_authorized: bool = True
    national_scale_confirmed: bool = False
    max_cpu_hours: float | None = 1.0
    quality_settings: QualitySettings = Field(default_factory=QualitySettings)
    seed: int = 0
    # development overrides; never set by the default UI flow
    allow_date_mismatch: bool = False
    allow_synthetic_mix: bool = False

    @model_validator(mode="after")
    def _mode_requirements(self) -> SearchRequest:
        if self.mode == SearchMode.UNKNOWN_TIME:
            if not self.date_utc:
                raise ValueError("unknown-time mode requires date_utc")
            if len(self.event_ids) < 2:
                raise ValueError("unknown-time mode requires at least two events")
        if self.mode == SearchMode.KNOWN_TIME:
            if not self.start_utc:
                raise ValueError("known-time mode requires start_utc")
            if len(self.event_ids) < 1:
                raise ValueError("known-time mode requires at least one event")
        if self.bbox is None:
            raise ValueError("bbox is required (national-scale search is opt-in and bounded)")
        return self


class ScoreComponents(BaseModel):
    per_event: list[float]
    coverage: list[float]
    silence_penalty: float = 0.0
    drift_penalty: float = 0.0
    type_prior: float = 0.0


class LocationCandidate(BaseModel):
    lat: float
    lon: float
    ground_elevation_m: float | None
    ground_elevation_datum: str | None = None
    flight_segment_ids: list[str]
    inferred_start_utc: str
    global_time_shift_s: float
    source_fundamental_hz_per_event: list[float | None]
    score_total: float
    score_components: ScoreComponents
    flags: list[str] = Field(default_factory=list)
    rank: int
    nominal_grid_spacing_m: float
    estimated_uncertainty_m: float | None = None
    uncertainty_label: str = "score-landscape extent (uncalibrated)"
    candidate_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StageCounter(BaseModel):
    stage: str
    candidates_in: int
    candidates_out: int
    note: str = ""
    elapsed_s: float = 0.0


class Abstention(BaseModel):
    abstained: bool
    reason: str | None = None
    detail: str | None = None


class Gate(BaseModel):
    name: str
    passed: bool
    detail: str


class SearchResult(BaseModel):
    run_id: str
    request: SearchRequest
    candidates: list[LocationCandidate]
    stage_counters: list[StageCounter]
    abstention: Abstention
    warnings: list[str] = Field(default_factory=list)
    score_label: str = "relative fit score (uncalibrated)"
    manifest: dict[str, Any] = Field(default_factory=dict)
    confidence_grade: str = "low"  # high / medium / low — a graded verdict, not a probability
    gates: list[Gate] = Field(default_factory=list)
    # distinct locations (>= 1 km from the top-1) whose score is within the high-grade margin
    # of the winner: the answer is "one of these", not the top-1 alone
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
