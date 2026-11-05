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


class DatasetWriter:
    """Incremental, atomic dataset writer.

    Parts are written into ``<root>.partial/`` (samples per date/region as ``part-<k>``) and
    the directory replaces ``<root>`` only in :meth:`finalize`, so a crash or cancellation
    never leaves a half-written dataset where the index would find it. Only the segment table
    (one row per segment) is held in memory; samples are written as each part arrives.
    """

    def __init__(self, root: Path, dataset_id: str):
        import shutil

        self.root = root
        self.dataset_id = dataset_id
        self.tmp = root.with_name(root.name + ".partial")
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True)
        self._segments: list[pl.DataFrame] = []
        self.summary: dict[str, Any] = {
            "samples": 0,
            "aircraft": 0,
            "first_utc": None,
            "last_utc": None,
            "lat_range": None,
            "lon_range": None,
            "hour_histogram": {},
        }

    def add(
        self, samples: pl.DataFrame, segments: pl.DataFrame, part: int, day0_unix: float | None
    ) -> None:
        """Add one part. Parts must hold disjoint aircraft (the importer buckets by icao24)."""
        if not segments.is_empty():
            self._segments.append(segments)
        if samples.is_empty():
            return
        seg_region = segments.select(["segment_id", "date_utc", "region"])
        s = samples.join(seg_region, on="segment_id", how="left")
        for (date, region), grp in s.group_by(["date_utc", "region"], maintain_order=True):
            d = self.tmp / "samples" / f"date={date}" / f"region={region}"
            d.mkdir(parents=True, exist_ok=True)
            grp.drop(["date_utc", "region"]).write_parquet(
                d / f"part-{part}.parquet", compression="zstd"
            )
        sm = self.summary
        sm["samples"] += samples.height
        sm["aircraft"] += samples["icao24"].n_unique()
        t_lo, t_hi = float(samples["t"].min()), float(samples["t"].max())
        sm["first_utc"] = t_lo if sm["first_utc"] is None else min(sm["first_utc"], t_lo)
        sm["last_utc"] = t_hi if sm["last_utc"] is None else max(sm["last_utc"], t_hi)
        for key, col in (("lat_range", "lat"), ("lon_range", "lon")):
            lo, hi = float(samples[col].min()), float(samples[col].max())
            cur = sm[key]
            sm[key] = [lo, hi] if cur is None else [min(cur[0], lo), max(cur[1], hi)]
        if day0_unix is not None:
            hist = (
                samples.select(((pl.col("t") - day0_unix) // 3600).cast(pl.Int64).alias("h"))
                .group_by("h")
                .len()
            )
            for h, n in hist.iter_rows():
                sm["hour_histogram"][str(int(h))] = sm["hour_histogram"].get(str(int(h)), 0) + int(
                    n
                )

    def segments(self) -> pl.DataFrame:
        return pl.concat(self._segments) if self._segments else pl.DataFrame(schema=SEGMENT_SCHEMA)

    def finalize(self, manifest: dict[str, Any]) -> FlightDataset:
        import shutil

        self.segments().write_parquet(self.tmp / "segments.parquet")
        manifest = dict(manifest)
        manifest["storage_bytes"] = sum(p.stat().st_size for p in self.tmp.rglob("*.parquet"))
        (self.tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        old = self.root.with_name(self.root.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        if self.root.exists():
            self.root.rename(old)
        self.tmp.rename(self.root)
        if old.exists():
            shutil.rmtree(old)
        return FlightDataset(self.dataset_id, self.root)

    def abort(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)


def write_dataset(
    root: Path, samples: pl.DataFrame, segments: pl.DataFrame, manifest: dict[str, Any]
) -> FlightDataset:
    """Write a whole in-memory dataset (synthetic/test data) through :class:`DatasetWriter`."""
    w = DatasetWriter(root, manifest["dataset_id"])
    w.add(samples, segments, part=0, day0_unix=None)
    return w.finalize(manifest)


def list_datasets(normalized_dir: Path) -> list[FlightDataset]:
    out = []
    if not normalized_dir.exists():
        return out
    for d in sorted(normalized_dir.iterdir()):
        if d.name.endswith((".partial", ".old")):
            continue
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
