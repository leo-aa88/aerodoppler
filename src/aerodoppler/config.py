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
    max_concurrent_jobs: int = 2
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
