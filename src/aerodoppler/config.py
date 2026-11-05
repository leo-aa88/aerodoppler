"""Runtime configuration for AeroDoppler.

All settings can be overridden with ``AERODOPPLER_*`` environment variables or a ``.env``
file. Defaults are local-first and privacy preserving: the server binds to localhost, no
data leaves the machine, and heavy jobs are bounded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AERODOPPLER_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Field(
        default=Path("data"), description="Root for raw/normalized/dem/recordings/runs"
    )
    host: str = "127.0.0.1"
    port: int = 8000
    offline: bool = Field(default=False, description="Refuse every network access when true")
    max_upload_mb: int = 512
    max_audio_duration_s: float = 7200.0
    # ---- job scheduler (see docs/jobs.md) ----
    jobs_max_import: int = Field(default=1, description="concurrent flight-archive imports")
    jobs_max_analysis: int = Field(default=1, description="concurrent searches/benchmarks")
    jobs_max_light: int = Field(
        default=2, description="concurrent ingest/spectrogram/detection jobs"
    )
    jobs_memory_budget_mb: float | None = Field(
        default=None,
        description="memory all running jobs together may reserve; None = 70 % of the RAM this "
        "process may use (cgroup/WSL limits included)",
    )
    jobs_max_cpu_hours: float = Field(
        default=12.0, description="CPU-time budget per job (all its processes)"
    )
    jobs_min_free_disk_mb: float = Field(
        default=2000.0, description="jobs stop when free disk drops below this"
    )
    jobs_retry_base_s: float = Field(
        default=30.0,
        description="first automatic retry delay for transient failures (doubles, max 30 min)",
    )
    jobs_scheduler_in_api: bool = Field(
        default=True, description="run the job scheduler inside `aerodoppler serve`"
    )
    import_memory_budget_mb: float | None = Field(
        default=None,
        description="peak RAM per import incl. parser processes; None = half of available, max 4096",
    )
    search_memory_budget_mb: float | None = Field(
        default=None,
        description="peak RAM (PSS, workers included) per search; None = half of available, max 8192",
    )
    github_token: str | None = Field(
        default=None, description="optional GitHub token for the ADSB.lol release lookups"
    )
    speed_of_sound_temp_c: float = 20.0
    adsb_gap_split_s: float = 30.0
    adsb_bbox_margin_km: float = 15.0
    analysis_sample_rate_hz: int = 2000
    stft_window_s: float = 4.096
    stft_hop_s: float = 0.25
    max_analysis_freq_hz: float = 900.0
    allow_national_search: bool = Field(
        default=False, description="National-scale searches require explicit operator confirmation"
    )
    basemap_tiles_enabled: bool = Field(
        default=False,
        description="When true the UI may load remote basemap tiles (reveals the viewport to the tile host)",
    )

    # ---- derived paths -------------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def dem_dir(self) -> Path:
        return self.data_dir / "dem"

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "aerodoppler.sqlite3"

    def ensure_dirs(self) -> None:
        for p in (
            self.raw_dir,
            self.normalized_dir,
            self.dem_dir,
            self.recordings_dir,
            self.runs_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
