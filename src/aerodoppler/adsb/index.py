"""Partitioned Parquet storage and DuckDB spatial-temporal queries (SPEC §6.7, §10.2).

Layout under ``data/normalized/<dataset_id>/``::

    manifest.json
    segments.parquet                     # one row per FlightSegment (the index)
    samples/date=YYYY-MM-DD/region=n40w075/part-0.parquet

Queries never load all samples: the segment index is filtered first (time window, bbox,
speed bins, class), then only the matching segments' samples are read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from aerodoppler.adsb.segments import SEGMENT_SCHEMA
from aerodoppler.models import BBox


@dataclass
class FlightDataset:
    dataset_id: str
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def segments_path(self) -> Path:
        return self.root / "segments.parquet"

    @property
    def samples_dir(self) -> Path:
        return self.root / "samples"

    def manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def exists(self) -> bool:
        return self.segments_path.exists()


def write_dataset(
    root: Path, samples: pl.DataFrame, segments: pl.DataFrame, manifest: dict[str, Any]
) -> FlightDataset:
    root.mkdir(parents=True, exist_ok=True)
    ds = FlightDataset(manifest["dataset_id"], root)
    if samples.is_empty():
        pl.DataFrame(schema=SEGMENT_SCHEMA).write_parquet(ds.segments_path)
    else:
        seg_region = segments.select(["segment_id", "date_utc", "region"])
        s = samples.join(seg_region, on="segment_id", how="left")
        for (date, region), part in s.group_by(["date_utc", "region"], maintain_order=True):
            d = ds.samples_dir / f"date={date}" / f"region={region}"
            d.mkdir(parents=True, exist_ok=True)
            part.drop(["date_utc", "region"]).write_parquet(
                d / "part-0.parquet", compression="zstd"
            )
        segments.write_parquet(ds.segments_path)
    manifest = dict(manifest)
    manifest["storage_bytes"] = sum(p.stat().st_size for p in root.rglob("*.parquet"))
    ds.manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return ds


def list_datasets(normalized_dir: Path) -> list[FlightDataset]:
    out = []
    if not normalized_dir.exists():
        return out
    for d in sorted(normalized_dir.iterdir()):
        if (d / "manifest.json").exists():
            out.append(FlightDataset(d.name, d))
    return out


class FlightIndex:
    """DuckDB-backed query layer over one or more datasets."""

    def __init__(self, datasets: list[FlightDataset]):
        self.datasets = [d for d in datasets if d.exists()]
        self.con = duckdb.connect(database=":memory:")

    def _segments_glob(self) -> list[str]:
        return [str(d.segments_path) for d in self.datasets]

    def _samples_globs(self) -> list[str]:
        return [
            str(d.samples_dir / "**" / "*.parquet") for d in self.datasets if d.samples_dir.exists()
        ]

    def segments(self) -> pl.DataFrame:
        if not self.datasets:
            return pl.DataFrame(schema=SEGMENT_SCHEMA)
        return pl.concat([pl.read_parquet(p) for p in self._segments_glob()])

    def query_segments(
        self,
        t_min: float,
        t_max: float,
        bbox: BBox | None = None,
        min_speed_mps: float | None = None,
        max_speed_mps: float | None = None,
        exclude_ground: bool = True,
        require_altitude: bool = True,
        classes: list[str] | None = None,
    ) -> pl.DataFrame:
        if not self.datasets:
            return pl.DataFrame(schema=SEGMENT_SCHEMA)
        conds = ["last_utc >= ?", "first_utc <= ?"]
        params: list[Any] = [t_min, t_max]
        if bbox is not None:
            conds += ["max_lat >= ?", "min_lat <= ?", "max_lon >= ?", "min_lon <= ?"]
            params += [bbox.south, bbox.north, bbox.west, bbox.east]
        if exclude_ground:
            conds.append("NOT on_ground")
        if require_altitude:
            conds.append("n_with_altitude >= 2")
        if min_speed_mps is not None:
            conds.append("max_speed_mps >= ?")
            params.append(min_speed_mps)
        if max_speed_mps is not None:
            conds.append("median_speed_mps <= ?")
            params.append(max_speed_mps)
        if classes:
            conds.append("aircraft_class IN (" + ",".join("?" for _ in classes) + ")")
            params += classes
        files = "[" + ",".join(f"'{p}'" for p in self._segments_glob()) + "]"
        sql = (
            f"SELECT * FROM read_parquet({files}) WHERE "
            + " AND ".join(conds)
            + " ORDER BY first_utc"
        )
        return pl.from_arrow(self.con.sql(sql, params=params).to_arrow_table())  # type: ignore[return-value]

    def load_samples(self, segment_ids: list[str]) -> pl.DataFrame:
        globs = self._samples_globs()
        if not globs or not segment_ids:
            return pl.DataFrame()
        files = "[" + ",".join(f"'{g}'" for g in globs) + "]"
        placeholders = ",".join("?" for _ in segment_ids)
        sql = f"SELECT * FROM read_parquet({files}, hive_partitioning=true, union_by_name=true) WHERE segment_id IN ({placeholders}) ORDER BY segment_id, t"
        return pl.from_arrow(self.con.sql(sql, params=list(segment_ids)).to_arrow_table())  # type: ignore[return-value]

    def coverage_summary(self) -> dict[str, Any]:
        seg = self.segments()
        if seg.is_empty():
            return {"segments": 0}
        air = seg.filter(~pl.col("on_ground"))
        return {
            "datasets": [d.dataset_id for d in self.datasets],
            "segments": seg.height,
            "airborne_segments": air.height,
            "aircraft": seg["icao24"].n_unique(),
            "first_utc": float(seg["first_utc"].min()),
            "last_utc": float(seg["last_utc"].max()),
            "classes": {k: int(v) for k, v in air.group_by("aircraft_class").len().iter_rows()},
            "missing_altitude_segments": int(
                seg.filter(pl.col("flags").str.contains("missing_altitude")).height
            ),
        }
