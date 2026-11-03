"""Flight archive import orchestration: fetch → stream traces → parallel parse/filter →
segments → partitioned Parquet + manifest (SPEC §6, §5 reproducibility manifest)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import multiprocessing as mp
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from aerodoppler import __version__
from aerodoppler.adsb.index import FlightDataset, write_dataset
from aerodoppler.adsb.normalize import (
    TraceParseStats,
    empty_samples,
    load_trace_bytes,
    normalize_trace,
)
from aerodoppler.adsb.providers import ArchiveLocation, FlightProvider
from aerodoppler.adsb.segments import SegmentConfig, split_segments
from aerodoppler.models import BBox, iso_to_unix
from aerodoppler.util import git_sha

log = logging.getLogger(__name__)


@dataclass
class ImportRequest:
    date_utc: str
    bbox: BBox
    margin_km: float = 15.0
    storage_budget_gb: float | None = None
    confirm_download: bool = False
    workers: int = max(1, mp.cpu_count() - 1)
    gap_split_s: float = 30.0
    include_adjacent_days: bool = False
    source_label: str | None = None
    extra_notes: list[str] = field(default_factory=list)


def dataset_id_for(provider: str, date_utc: str, bbox: BBox, margin_km: float) -> str:
    h = hashlib.sha1(f"{bbox.as_tuple()}|{margin_km}".encode()).hexdigest()[:8]
    return f"{provider}_{date_utc}_{h}"


# --- worker --------------------------------------------------------------------------------


def _parse_batch(
    args: tuple[list[tuple[str, bytes]], str, tuple[float, float, float, float], float, float],
) -> tuple[bytes | None, dict[str, Any]]:
    items, source, bbox_t, t_min, t_max = args
    bbox = BBox(west=bbox_t[0], south=bbox_t[1], east=bbox_t[2], north=bbox_t[3])
    stats = TraceParseStats()
    frames: list[pl.DataFrame] = []
    for name, data in items:
        try:
            obj = load_trace_bytes(data)
        except Exception as e:  # corrupt member: count and continue
            stats.files += 1
            stats.files_failed += 1
            if len(stats.errors) < 20:
                stats.errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        df = normalize_trace(obj, source, bbox, stats)
        if df is not None and not df.is_empty():
            df = df.filter((pl.col("t") >= t_min) & (pl.col("t") <= t_max))
            if not df.is_empty():
                frames.append(df)
    out = pl.concat(frames) if frames else None
    payload = out.serialize() if out is not None else None
    return payload, stats.as_dict()


def _batches(it: Iterable[tuple[str, bytes]], size: int) -> Iterator[list[tuple[str, bytes]]]:
    batch: list[tuple[str, bytes]] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_traces_parallel(
    traces: Iterable[tuple[str, bytes]],
    source: str,
    bbox: BBox,
    t_min: float,
    t_max: float,
    workers: int = 4,
    batch_size: int = 64,
    progress: bool = True,
) -> tuple[pl.DataFrame, TraceParseStats]:
    from tqdm import tqdm

    stats = TraceParseStats()
    frames: list[pl.DataFrame] = []
    bar = tqdm(desc="traces", unit="file", disable=not progress)
    args = ((b, source, bbox.as_tuple(), t_min, t_max) for b in _batches(traces, batch_size))
    if workers <= 1:
        results: Iterable[tuple[bytes | None, dict[str, Any]]] = map(_parse_batch, args)
        for payload, st in results:
            _absorb(payload, st, frames, stats, bar)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for payload, st in pool.imap_unordered(_parse_batch, args, chunksize=1):
                _absorb(payload, st, frames, stats, bar)
    bar.close()
    samples = pl.concat(frames) if frames else empty_samples()
    return samples, stats


def _absorb(
    payload: bytes | None,
    st: dict[str, Any],
    frames: list[pl.DataFrame],
    stats: TraceParseStats,
    bar,
) -> None:
    sub = TraceParseStats(**{k: v for k, v in st.items()})
    stats.merge(sub)
    bar.update(sub.files)
    if payload is not None:
        import io

        frames.append(pl.DataFrame.deserialize(io.BytesIO(payload)))


# --- orchestrator ---------------------------------------------------------------------------


class StorageBudgetExceeded(RuntimeError):
    pass


def import_flights(
    provider: FlightProvider,
    req: ImportRequest,
    raw_dir: Path,
    normalized_dir: Path,
    progress: bool = True,
) -> FlightDataset:
    """Import one UTC day for a bbox. Raises StorageBudgetExceeded unless the download size is
    within ``storage_budget_gb`` or ``confirm_download`` is set (SPEC §6.1: never download a
    whole country silently)."""
    source = req.source_label or provider.name
    locate_kwargs: dict[str, Any] = {}
    if provider.name == "adsblol":
        locate_kwargs["dest_dir"] = raw_dir / "adsblol"
    location: ArchiveLocation = provider.locate(req.date_utc, **locate_kwargs)
    total = location.total_bytes
    need_download = any(a.local_path is None or not a.local_path.exists() for a in location.assets)
    if need_download and total is not None:
        gb = total / 1e9
        budget_ok = req.storage_budget_gb is not None and gb <= req.storage_budget_gb
        if not (budget_ok or req.confirm_download):
            raise StorageBudgetExceeded(
                f"{provider.name} archive for {req.date_utc} is {gb:.2f} GB (whole-sky daily dump; cannot be subset remotely). "
                f"Re-run with --confirm-download or --storage-budget-gb >= {gb:.2f}."
            )
    dest = raw_dir / provider.name
    location = provider.fetch(location, dest, progress=progress)
    search_bbox = req.bbox.expanded_km(req.margin_km)
    day0 = iso_to_unix(req.date_utc + "T00:00:00Z")
    t_min, t_max = day0 - 3600.0, day0 + 86400.0 + 3600.0  # tolerate midnight-crossing traces
    samples, stats = parse_traces_parallel(
        provider.iter_traces(location),
        source,
        search_bbox,
        t_min,
        t_max,
        workers=req.workers,
        progress=progress,
    )
    seg_cfg = SegmentConfig(gap_split_s=req.gap_split_s)
    samples, segments = split_segments(samples, seg_cfg)
    ds_id = dataset_id_for(provider.name, req.date_utc, req.bbox, req.margin_km)
    manifest = build_manifest(
        ds_id, provider, location, req, search_bbox, stats, samples, segments, seg_cfg
    )
    ds = write_dataset(normalized_dir / ds_id, samples, segments, manifest)
    log.info("dataset %s written: %d samples, %d segments", ds_id, samples.height, segments.height)
    return ds


def build_manifest(
    ds_id: str,
    provider: FlightProvider,
    location: ArchiveLocation,
    req: ImportRequest,
    search_bbox: BBox,
    stats: TraceParseStats,
    samples: pl.DataFrame,
    segments: pl.DataFrame,
    seg_cfg: SegmentConfig,
) -> dict[str, Any]:
    air = segments.filter(~pl.col("on_ground")) if not segments.is_empty() else segments
    cov: dict[str, Any] = {}
    if not samples.is_empty():
        cov = {
            "first_utc": float(samples["t"].min()),
            "last_utc": float(samples["t"].max()),
            "lat_range": [float(samples["lat"].min()), float(samples["lat"].max())],
            "lon_range": [float(samples["lon"].min()), float(samples["lon"].max())],
            "hour_histogram": {
                str(int(k)): int(v)
                for k, v in samples.with_columns(
                    ((pl.col("t") - iso_to_unix(req.date_utc + "T00:00:00Z")) // 3600).alias("h")
                )
                .group_by("h")
                .len()
                .sort("h")
                .iter_rows()
            },
        }
    return {
        "dataset_id": ds_id,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "aerodoppler_version": __version__,
        "git_sha": git_sha(),
        "provider": provider.describe(),
        "license": location.license,
        "attribution": location.attribution,
        "provider_notes": location.notes,
        "sources": [
            {
                "name": a.name,
                "url": a.url,
                "size_bytes": a.size_bytes,
                "sha256": a.sha256,
                "retrieved_at": a.retrieved_at,
                "local_path": str(a.local_path) if a.local_path else None,
            }
            for a in location.assets
        ],
        "request": {
            "date_utc": req.date_utc,
            "bbox": req.bbox.as_tuple(),
            "margin_km": req.margin_km,
            "search_bbox": search_bbox.as_tuple(),
            "gap_split_s": seg_cfg.gap_split_s,
            "max_speed_mps": seg_cfg.max_speed_mps,
            "workers": req.workers,
        },
        "filter_counts": stats.as_dict(),
        "summary": {
            "samples": samples.height,
            "aircraft": samples["icao24"].n_unique() if not samples.is_empty() else 0,
            "segments": segments.height,
            "airborne_segments": air.height if not segments.is_empty() else 0,
            "segments_missing_altitude": int(
                segments.filter(pl.col("flags").str.contains("missing_altitude")).height
            )
            if not segments.is_empty()
            else 0,
            "segments_baro_only": int(
                segments.filter(pl.col("flags").str.contains("baro_altitude_only")).height
            )
            if not segments.is_empty()
            else 0,
            "segments_with_large_gap": int(
                segments.filter(pl.col("flags").str.contains("large_internal_gap")).height
            )
            if not segments.is_empty()
            else 0,
            "aircraft_classes": {
                str(k): int(v) for k, v in air.group_by("aircraft_class").len().iter_rows()
            }
            if not segments.is_empty()
            else {},
            "unknown_type_segments": int(air.filter(pl.col("aircraft_type").is_null()).height)
            if not segments.is_empty()
            else 0,
        },
        "coverage": cov,
        "notes": req.extra_notes,
    }


def import_samples_dataframe(
    samples: pl.DataFrame,
    ds_id: str,
    normalized_dir: Path,
    notes: list[str],
    gap_split_s: float = 30.0,
    synthetic: bool = True,
    extra_manifest: dict[str, Any] | None = None,
) -> FlightDataset:
    """Write an already-normalised sample table (e.g. synthetic flights) as a dataset."""
    seg_cfg = SegmentConfig(gap_split_s=gap_split_s)
    samples, segments = split_segments(samples, seg_cfg)
    manifest = {
        "dataset_id": ds_id,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "aerodoppler_version": __version__,
        "git_sha": git_sha(),
        "provider": {"provider": "synthetic" if synthetic else "dataframe"},
        "license": "synthetic (generated locally)" if synthetic else "n/a",
        "attribution": "SYNTHETIC DEMO DATA — not real aircraft" if synthetic else "",
        "synthetic": synthetic,
        "sources": [],
        "request": {"gap_split_s": gap_split_s},
        "filter_counts": {},
        "summary": {
            "samples": samples.height,
            "segments": segments.height,
            "aircraft": samples["icao24"].n_unique() if not samples.is_empty() else 0,
        },
        "notes": notes,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    return write_dataset(normalized_dir / ds_id, samples, segments, manifest)


def samples_from_flight_samples(samples: list) -> pl.DataFrame:
    """Convert a list of :class:`FlightSample` (e.g. synthetic) to the normalised table."""
    from aerodoppler.adsb.normalize import SAMPLE_SCHEMA

    rows = []
    for s in samples:
        rows.append(
            {
                "source": s.source,
                "icao24": s.aircraft_icao24,
                "t": s.utc_unix_s,
                "lat": s.latitude,
                "lon": s.longitude,
                "alt_baro_ft": s.alt_baro_ft,
                "alt_geom_ft": s.alt_geom_ft,
                "altitude_value_ft": s.alt_geom_ft if s.alt_geom_ft is not None else s.alt_baro_ft,
                "altitude_datum": (
                    "geometric_wgs84_hae" if s.alt_geom_ft is not None else str(s.altitude_datum)
                ),
                "on_ground": False,
                "gs_mps": s.groundspeed_mps,
                "track_deg": s.true_track_deg,
                "vertical_rate_mps": s.vertical_rate_mps,
                "flags": s.quality_flags,
                "stale": False,
                "new_leg": False,
                "position_source": str(s.position_source) if s.position_source else None,
                "callsign": s.callsign,
                "aircraft_type": s.aircraft_type,
                "aircraft_desc": None,
                "registration": None,
                "db_flags": None,
                "category": None,
            }
        )
    return pl.DataFrame(rows, schema=SAMPLE_SCHEMA) if rows else empty_samples()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
