"""Flight archive import orchestration: fetch → stream traces → bounded parallel parse/filter →
staged Parquet parts → per-aircraft-bucket segmentation → atomic partitioned Parquet +
manifest (SPEC §6, §5 reproducibility manifest).

Memory is bounded by design: the tar is streamed, at most ``2 x workers`` batches are in
flight, parsed samples go to disk immediately, and segmentation handles at most
``ROWS_PER_BUCKET`` samples at a time. The worker count comes from the memory budget, not
from ``cpu_count()``."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import multiprocessing as mp
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from aerodoppler import __version__
from aerodoppler.adsb.index import DatasetWriter, FlightDataset, write_dataset
from aerodoppler.adsb.normalize import (
    TraceParseStats,
    classify_aircraft_type,
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
    workers: int = 0  # 0 = derive from the memory budget (see aerodoppler.resources)
    memory_budget_mb: float | None = None  # None = half of available memory, capped at 4 GB
    gap_split_s: float = 30.0
    include_adjacent_days: bool = False
    exclude_jets: bool = False  # drop aircraft classified 'jet' while parsing (whole-sky imports)
    source_label: str | None = None
    extra_notes: list[str] = field(default_factory=list)


def dataset_id_for(
    provider: str, date_utc: str, bbox: BBox, margin_km: float, exclude_jets: bool = False
) -> str:
    key = f"{bbox.as_tuple()}|{margin_km}" + ("|nojets" if exclude_jets else "")
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"{provider}_{date_utc}_{h}"


ProgressFn = Callable[[str, dict[str, Any]], None]
CancelFn = Callable[[], bool]


class ImportCancelled(RuntimeError):
    pass


class MemoryBudgetExceeded(RuntimeError):
    pass


class StorageBudgetExceeded(RuntimeError):
    pass


# Measured 2026-11-05 on the 2026-11-03 archive (IMPLEMENTATION_REPORT.md, import memory): 11
# parsers + parent peaked at 4.6 GB, i.e. ~370 MB per spawned parser (Python + Polars + orjson +
# one batch of decoded traces).
PER_WORKER_MB = 380.0
PARENT_BASE_MB = 500.0
BATCH_SIZE = 128
ROWS_PER_BUCKET = 3_000_000  # segmentation works on at most ~this many samples at a time


# --- worker --------------------------------------------------------------------------------


def _parse_batch_to_file(
    args: tuple[
        list[tuple[str, bytes]], str, tuple[float, float, float, float], float, float, str, bool
    ],
) -> tuple[str | None, int, dict[str, Any]]:
    """Parse one batch of traces and write the kept samples to ``out_path`` (Parquet).

    The DataFrame never travels back through the pipe: the parent only receives the path,
    the row count and the counters, so its memory does not grow with the archive."""
    items, source, bbox_t, t_min, t_max, out_path, exclude_jets = args
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
        if df is None or df.is_empty():
            continue
        df = df.filter((pl.col("t") >= t_min) & (pl.col("t") <= t_max))
        if exclude_jets and not df.is_empty():
            tc = df["aircraft_type"][0]
            desc = df["aircraft_desc"][0]
            cat = df["category"][0]
            if classify_aircraft_type(tc, desc, cat) == "jet":
                stats.points_kept -= df.height
                stats.aircraft_kept -= 1
                continue
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return None, 0, stats.as_dict()
    out = pl.concat(frames)
    out.write_parquet(out_path, compression="lz4")
    return out_path, out.height, stats.as_dict()


def _batches(it: Iterable[tuple[str, bytes]], size: int) -> Iterator[list[tuple[str, bytes]]]:
    batch: list[tuple[str, bytes]] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class ParseOutcome:
    stats: TraceParseStats
    parts: list[Path]
    rows: int
    workers: int
    memory_budget_mb: float
    peak_tree_rss_mb: float
    throttled: bool


def parse_traces_streaming(
    traces: Iterable[tuple[str, bytes]],
    source: str,
    bbox: BBox,
    t_min: float,
    t_max: float,
    staging: Path,
    workers: int,
    memory_budget_mb: float,
    exclude_jets: bool = False,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    batch_size: int = BATCH_SIZE,
) -> ParseOutcome:
    """Parse traces with bounded memory.

    At most ``2 * workers`` batches are in flight (the archive is read only as fast as the
    parsers consume it), parsed samples go straight to Parquet parts under ``staging``, and
    the RSS of this process plus its workers is checked after every batch: above the budget
    new submissions pause until the pool drains (throttle); above 1.5x the budget the import
    stops with :class:`MemoryBudgetExceeded` rather than pushing the machine into swap/OOM.
    """
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from aerodoppler.resources import tree_rss_mb

    staging.mkdir(parents=True, exist_ok=True)
    stats = TraceParseStats()
    parts: list[Path] = []
    rows = 0
    peak = 0.0
    throttled = False
    seq = 0
    last_report = 0.0
    t_start = time.time()

    def absorb(res: tuple[str | None, int, dict[str, Any]]) -> None:
        nonlocal rows
        path, n, st = res
        stats.merge(TraceParseStats(**st))
        if path is not None:
            parts.append(Path(path))
            rows += n

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.time()
        if on_progress and (force or now - last_report >= 1.0):
            last_report = now
            on_progress(
                "parse",
                {
                    "files_parsed": stats.files,
                    "files_failed": stats.files_failed,
                    "aircraft_kept": stats.aircraft_kept,
                    "samples_kept": rows,
                    "points_total": stats.points_total,
                    "workers": workers,
                    "tree_rss_mb": round(peak_now, 1),
                    "peak_tree_rss_mb": round(peak, 1),
                    "memory_budget_mb": round(memory_budget_mb),
                    "throttled": throttled,
                    "elapsed_s": round(now - t_start, 1),
                },
            )

    def check(tag: str) -> None:
        if should_cancel and should_cancel():
            raise ImportCancelled(f"import cancelled during {tag}")

    peak_now = 0.0
    args_iter = (
        (b, source, bbox.as_tuple(), t_min, t_max, None, exclude_jets)
        for b in _batches(traces, batch_size)
    )

    def with_path(a: tuple) -> tuple:
        nonlocal seq
        seq += 1
        return (*a[:5], str(staging / f"part-{seq:06d}.parquet"), a[6])

    if workers <= 1:
        for a in args_iter:
            check("parse")
            absorb(_parse_batch_to_file(with_path(a)))
            peak_now = tree_rss_mb()
            peak = max(peak, peak_now)
            if peak_now > 1.5 * memory_budget_mb:
                raise MemoryBudgetExceeded(
                    f"import used {peak_now:.0f} MB, above 1.5x its {memory_budget_mb:.0f} MB budget"
                )
            report()
    else:
        ctx = mp.get_context("spawn")
        limit = 2 * workers
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            inflight: set = set()
            try:
                for a in args_iter:
                    while len(inflight) >= limit:
                        done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                        for f in done:
                            absorb(f.result())
                        peak_now = tree_rss_mb()
                        peak = max(peak, peak_now)
                        if peak_now > 1.5 * memory_budget_mb:
                            raise MemoryBudgetExceeded(
                                f"import used {peak_now:.0f} MB (parser pool included), above "
                                f"1.5x its {memory_budget_mb:.0f} MB budget; retry with fewer "
                                "workers or a larger --memory-budget-mb"
                            )
                        if peak_now > memory_budget_mb:
                            # throttle: let the pool drain before reading more of the archive
                            throttled = True
                            limit = max(1, workers // 2)
                        report()
                    check("parse")
                    inflight.add(ex.submit(_parse_batch_to_file, with_path(a)))
                for f in inflight:
                    absorb(f.result())
            except BaseException:
                for f in inflight:
                    f.cancel()
                ex.shutdown(wait=True, cancel_futures=True)
                raise
    peak = max(peak, tree_rss_mb())
    report(force=True)
    return ParseOutcome(stats, parts, rows, workers, memory_budget_mb, peak, throttled)


def _segment_staged(
    parts: list[Path],
    rows_total: int,
    staging: Path,
    seg_cfg: SegmentConfig,
    writer: DatasetWriter,
    day0: float,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> int:
    """Split staged samples into segments, at most ~ROWS_PER_BUCKET samples at a time.

    Small imports are segmented in one go. Large ones are first redistributed by aircraft
    (icao24 hash) into buckets in a single streaming pass, so every aircraft's samples are
    in exactly one bucket and each bucket is segmented and written on its own."""
    n_buckets = max(1, -(-rows_total // ROWS_PER_BUCKET))
    if not parts:
        return n_buckets
    if n_buckets == 1:
        samples = pl.read_parquet(parts)
        s, seg = split_segments(samples, seg_cfg)
        writer.add(s, seg, part=0, day0_unix=day0)
        return 1
    bdir = staging / "buckets"
    for i, p in enumerate(parts):
        if should_cancel and should_cancel():
            raise ImportCancelled("import cancelled during bucketing")
        df = pl.read_parquet(p).with_columns(
            (pl.col("icao24").hash(seed=0) % n_buckets).alias("_bucket")
        )
        for key, grp in df.partition_by("_bucket", as_dict=True).items():
            k = int(key[0]) if isinstance(key, tuple) else int(key)
            d = bdir / f"b{k:03d}"
            d.mkdir(parents=True, exist_ok=True)
            grp.drop("_bucket").write_parquet(d / f"{i:06d}.parquet", compression="lz4")
        p.unlink()
        if on_progress and i % 50 == 0:
            on_progress(
                "bucket", {"parts_done": i + 1, "parts_total": len(parts), "buckets": n_buckets}
            )
    for k in range(n_buckets):
        if should_cancel and should_cancel():
            raise ImportCancelled("import cancelled during segmentation")
        d = bdir / f"b{k:03d}"
        if not d.exists():
            continue
        samples = pl.read_parquet(sorted(d.glob("*.parquet")))
        s, seg = split_segments(samples, seg_cfg)
        writer.add(s, seg, part=k, day0_unix=day0)
        del samples, s, seg
        if on_progress:
            on_progress("segment", {"bucket": k + 1, "buckets": n_buckets})
    return n_buckets


# --- orchestrator ---------------------------------------------------------------------------


def import_flights(
    provider: FlightProvider,
    req: ImportRequest,
    raw_dir: Path,
    normalized_dir: Path,
    progress: bool = True,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> FlightDataset:
    """Import one UTC day for a bbox with bounded memory. Raises StorageBudgetExceeded unless
    the download size is within ``storage_budget_gb`` or ``confirm_download`` is set (SPEC
    §6.1: never download a whole country silently)."""
    import shutil

    from aerodoppler.resources import default_memory_budget_mb, recommend_workers

    source = req.source_label or provider.name
    locate_kwargs: dict[str, Any] = {}
    if provider.name == "adsblol":
        locate_kwargs["dest_dir"] = raw_dir / "adsblol"
    if on_progress:
        on_progress("locate", {"date_utc": req.date_utc, "provider": provider.name})
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
    fetch_kwargs: dict[str, Any] = {}
    if provider.name == "adsblol":
        fetch_kwargs = {"on_progress": on_progress, "should_cancel": should_cancel}
    location = provider.fetch(location, dest, progress=progress, **fetch_kwargs)
    search_bbox = req.bbox.expanded_km(req.margin_km)
    day0 = iso_to_unix(req.date_utc + "T00:00:00Z")
    t_min, t_max = day0 - 3600.0, day0 + 86400.0 + 3600.0  # tolerate midnight-crossing traces
    budget = req.memory_budget_mb or default_memory_budget_mb()
    workers = req.workers or recommend_workers(budget, PER_WORKER_MB, PARENT_BASE_MB)
    ds_id = dataset_id_for(provider.name, req.date_utc, req.bbox, req.margin_km, req.exclude_jets)
    staging = normalized_dir / f".staging_{ds_id}"
    if staging.exists():
        shutil.rmtree(staging)
    writer = DatasetWriter(normalized_dir / ds_id, ds_id)
    try:
        parsed = parse_traces_streaming(
            provider.iter_traces(location),
            source,
            search_bbox,
            t_min,
            t_max,
            staging,
            workers=workers,
            memory_budget_mb=budget,
            exclude_jets=req.exclude_jets,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
        seg_cfg = SegmentConfig(gap_split_s=req.gap_split_s)
        n_buckets = _segment_staged(
            parsed.parts, parsed.rows, staging, seg_cfg, writer, day0, on_progress, should_cancel
        )
        if on_progress:
            on_progress("write", {"segments": writer.segments().height})
        manifest = build_manifest(
            ds_id,
            provider,
            location,
            req,
            search_bbox,
            parsed.stats,
            writer.summary,
            writer.segments(),
            seg_cfg,
        )
        manifest["resources"] = {
            "workers": parsed.workers,
            "memory_budget_mb": round(parsed.memory_budget_mb),
            "peak_tree_rss_mb_parse": round(parsed.peak_tree_rss_mb),
            "throttled": parsed.throttled,
            "segmentation_buckets": n_buckets,
        }
        ds = writer.finalize(manifest)
    except BaseException:
        writer.abort()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log.info(
        "dataset %s written: %d samples, %d segments",
        ds_id,
        writer.summary["samples"],
        writer.segments().height,
    )
    return ds


def build_manifest(
    ds_id: str,
    provider: FlightProvider,
    location: ArchiveLocation,
    req: ImportRequest,
    search_bbox: BBox,
    stats: TraceParseStats,
    sample_summary: dict[str, Any],
    segments: pl.DataFrame,
    seg_cfg: SegmentConfig,
) -> dict[str, Any]:
    air = segments.filter(~pl.col("on_ground")) if not segments.is_empty() else segments
    cov: dict[str, Any] = {}
    if sample_summary.get("samples"):
        cov = {
            "first_utc": sample_summary["first_utc"],
            "last_utc": sample_summary["last_utc"],
            "lat_range": sample_summary["lat_range"],
            "lon_range": sample_summary["lon_range"],
            "hour_histogram": dict(
                sorted(sample_summary["hour_histogram"].items(), key=lambda kv: int(kv[0]))
            ),
        }

    def n_flag(flag: str) -> int:
        if segments.is_empty():
            return 0
        return int(segments.filter(pl.col("flags").str.contains(flag)).height)

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
                "sha256_verified_against": a.expected_sha256 and "provider digest",
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
            "exclude_jets": req.exclude_jets,
        },
        "filter_counts": stats.as_dict(),
        "summary": {
            "samples": sample_summary.get("samples", 0),
            "aircraft": sample_summary.get("aircraft", 0),
            "segments": segments.height,
            "airborne_segments": air.height if not segments.is_empty() else 0,
            "segments_missing_altitude": n_flag("missing_altitude"),
            "segments_baro_only": n_flag("baro_altitude_only"),
            "segments_with_large_gap": n_flag("large_internal_gap"),
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
