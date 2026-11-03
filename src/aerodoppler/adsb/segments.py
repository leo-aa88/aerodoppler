"""Split normalised samples into kinematically plausible flight segments (SPEC §6.4) and
build trajectories from them (SPEC §6.5).

Split rules (all configurable): time gap > ``gap_split_s``; implied horizontal speed >
``max_speed_mps`` (teleportation / ICAO reuse); readsb ``new_leg`` flag; transition to or
from the ground. On-ground samples are not interpolated (they form separate flagged
segments that the search never uses). Low-quality segments are kept and flagged, never
silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from aerodoppler.adsb.normalize import classify_aircraft_type
from aerodoppler.geo.altitude import to_ellipsoidal_height, to_metres
from aerodoppler.geo.coordinates import haversine_m
from aerodoppler.models import AltitudeDatum, AltitudeUnit, unix_to_iso
from aerodoppler.physics.trajectory import Trajectory

SEGMENT_SCHEMA: dict[str, pl.DataType] = {
    "segment_id": pl.Utf8,
    "source": pl.Utf8,
    "icao24": pl.Utf8,
    "ordinal": pl.Int32,
    "first_utc": pl.Float64,
    "last_utc": pl.Float64,
    "n_samples": pl.Int32,
    "n_with_altitude": pl.Int32,
    "completeness_score": pl.Float64,
    "max_gap_s": pl.Float64,
    "median_cadence_s": pl.Float64,
    "aircraft_type": pl.Utf8,
    "aircraft_class": pl.Utf8,
    "callsign": pl.Utf8,
    "min_lat": pl.Float64,
    "max_lat": pl.Float64,
    "min_lon": pl.Float64,
    "max_lon": pl.Float64,
    "min_alt_m": pl.Float64,
    "max_alt_m": pl.Float64,
    "max_speed_mps": pl.Float64,
    "median_speed_mps": pl.Float64,
    "altitude_datum": pl.Utf8,
    "on_ground": pl.Boolean,
    "flags": pl.Utf8,  # comma separated
    "date_utc": pl.Utf8,
    "region": pl.Utf8,
}


@dataclass
class SegmentConfig:
    gap_split_s: float = 30.0
    max_speed_mps: float = 350.0
    min_samples: int = 3
    max_interp_gap_s: float = 120.0


def region_cell(lat: float, lon: float) -> str:
    la = int(np.floor(lat))
    lo = int(np.floor(lon))
    return f"{'n' if la >= 0 else 's'}{abs(la):02d}{'e' if lo >= 0 else 'w'}{abs(lo):03d}"


def split_segments(
    samples: pl.DataFrame, cfg: SegmentConfig | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(samples_with_segment_id, segments)``."""
    cfg = cfg or SegmentConfig()
    if samples.is_empty():
        return samples.with_columns(pl.lit(None, dtype=pl.Utf8).alias("segment_id")), pl.DataFrame(
            schema=SEGMENT_SCHEMA
        )
    df = samples.sort(["source", "icao24", "t"]).unique(
        subset=["source", "icao24", "t"], keep="first", maintain_order=True
    )
    seg_ids: list[str] = []
    seg_rows: list[dict] = []
    out_parts: list[pl.DataFrame] = []
    for (source, icao), grp in df.group_by(["source", "icao24"], maintain_order=True):
        t = grp["t"].to_numpy()
        lat = grp["lat"].to_numpy()
        lon = grp["lon"].to_numpy()
        ground = grp["on_ground"].to_numpy()
        new_leg = grp["new_leg"].to_numpy()
        n = len(t)
        cut = np.zeros(n, dtype=bool)
        if n > 1:
            dt = np.diff(t)
            dist = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
            implied = dist / np.maximum(dt, 1e-3)
            cut[1:] = (
                (dt > cfg.gap_split_s)
                | (implied > cfg.max_speed_mps)
                | (ground[1:] != ground[:-1])
                | new_leg[1:]
            )
        seg_index = np.cumsum(cut)
        ids = np.empty(n, dtype=object)
        for k in np.unique(seg_index):
            m = seg_index == k
            idx = np.where(m)[0]
            first = float(t[idx[0]])
            sid = f"{source}:{icao}:{unix_to_iso(first)}:{int(k)}"
            ids[m] = sid
            sub = grp[int(idx[0]) : int(idx[-1]) + 1]
            seg_rows.append(_segment_summary(sid, str(source), str(icao), int(k), sub, cfg))
        out_parts.append(grp.with_columns(pl.Series("segment_id", list(ids), dtype=pl.Utf8)))
        seg_ids.extend(ids.tolist())
    samples_out = (
        pl.concat(out_parts)
        if out_parts
        else df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("segment_id"))
    )
    segments = (
        pl.DataFrame(seg_rows, schema=SEGMENT_SCHEMA)
        if seg_rows
        else pl.DataFrame(schema=SEGMENT_SCHEMA)
    )
    return samples_out, segments


def _segment_summary(
    sid: str, source: str, icao: str, ordinal: int, sub: pl.DataFrame, cfg: SegmentConfig
) -> dict:
    t = sub["t"].to_numpy()
    n = len(t)
    flags: list[str] = []
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    gaps = np.diff(t) if n > 1 else np.array([0.0])
    big = gaps[gaps > 15.0].sum() if n > 1 else 0.0
    completeness = float(np.clip(1.0 - big / dur, 0.0, 1.0)) if dur > 0 else 0.0
    on_ground = bool(sub["on_ground"].all())
    alt_ft = sub["altitude_value_ft"].to_numpy()
    datum_col = sub["altitude_datum"].to_list()
    has_alt = np.isfinite(alt_ft) & ~sub["on_ground"].to_numpy()
    n_alt = int(has_alt.sum())
    if n < cfg.min_samples:
        flags.append("too_few_samples")
    if n_alt < 2 and not on_ground:
        flags.append("missing_altitude")
    if on_ground:
        flags.append("on_ground")
    if n > 1 and gaps.max() > cfg.max_interp_gap_s:
        flags.append("large_internal_gap")
    if bool(sub["stale"].mean() > 0.5) if n else False:
        flags.append("mostly_stale")
    datums = {d for d, h in zip(datum_col, has_alt, strict=False) if h}
    if AltitudeDatum.GEOMETRIC_HAE.value in datums and len(datums) > 1:
        flags.append("mixed_altitude_datum")
    datum = (
        AltitudeDatum.GEOMETRIC_HAE.value
        if AltitudeDatum.GEOMETRIC_HAE.value in datums
        else (
            AltitudeDatum.BAROMETRIC_STD.value
            if AltitudeDatum.BAROMETRIC_STD.value in datums
            else AltitudeDatum.UNKNOWN.value
        )
    )
    if datum == AltitudeDatum.BAROMETRIC_STD.value:
        flags.append("baro_altitude_only")
    gs = sub["gs_mps"].to_numpy()
    gs = gs[np.isfinite(gs)]
    lat = sub["lat"].to_numpy()
    lon = sub["lon"].to_numpy()
    if n > 1:
        d = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        imp = d / np.maximum(gaps, 1e-3)
        max_speed = float(max(imp.max(), gs.max() if gs.size else 0.0))
        med_speed = float(np.median(imp))
    else:
        max_speed = float(gs.max()) if gs.size else float("nan")
        med_speed = max_speed
    type_code = sub["aircraft_type"][0]
    desc = sub["aircraft_desc"][0]
    cats = [c for c in sub["category"].to_list() if c]
    cls = classify_aircraft_type(type_code, desc, cats[-1] if cats else None)
    calls = [c for c in sub["callsign"].to_list() if c]
    alt_m = to_metres(alt_ft[has_alt], AltitudeUnit.FEET) if n_alt else np.array([])
    return {
        "segment_id": sid,
        "source": source,
        "icao24": icao,
        "ordinal": ordinal,
        "first_utc": float(t[0]),
        "last_utc": float(t[-1]),
        "n_samples": n,
        "n_with_altitude": n_alt,
        "completeness_score": completeness,
        "max_gap_s": float(gaps.max()) if n > 1 else 0.0,
        "median_cadence_s": float(np.median(gaps)) if n > 1 else float("nan"),
        "aircraft_type": type_code,
        "aircraft_class": cls,
        "callsign": calls[-1] if calls else None,
        "min_lat": float(lat.min()),
        "max_lat": float(lat.max()),
        "min_lon": float(lon.min()),
        "max_lon": float(lon.max()),
        "min_alt_m": float(alt_m.min()) if alt_m.size else None,
        "max_alt_m": float(alt_m.max()) if alt_m.size else None,
        "max_speed_mps": max_speed,
        "median_speed_mps": med_speed,
        "altitude_datum": datum,
        "on_ground": on_ground,
        "flags": ",".join(flags),
        "date_utc": unix_to_iso(float(t[0]))[:10],
        "region": region_cell(float(lat[0]), float(lon[0])),
    }


@dataclass
class TrajectoryBuild:
    trajectory: Trajectory | None
    altitude_sigma_m: float
    datum: str
    n_used: int
    n_dropped: int
    flags: list[str]


def trajectory_from_samples(
    seg_samples: pl.DataFrame, segment_id: str, max_interp_gap_s: float = 120.0
) -> TrajectoryBuild:
    """Build an ECEF trajectory from one segment's samples with explicit datum handling."""
    sub = seg_samples.sort("t")
    t = sub["t"].to_numpy()
    lat = sub["lat"].to_numpy()
    lon = sub["lon"].to_numpy()
    alt_ft = sub["altitude_value_ft"].to_numpy()
    datum = np.array(sub["altitude_datum"].to_list(), dtype=object)
    ground = sub["on_ground"].to_numpy()
    ok = np.isfinite(alt_ft) & ~ground
    flags: list[str] = []
    n_dropped = int((~ok).sum())
    if ok.sum() < 2:
        return TrajectoryBuild(None, float("nan"), "none", 0, n_dropped, ["unusable_no_altitude"])
    # per-sample datum conversion (geometric preferred where present, else barometric)
    h = np.empty(ok.sum())
    sig = np.empty(ok.sum())
    tt, la, lo, af, dm = t[ok], lat[ok], lon[ok], alt_ft[ok], datum[ok]
    for d in np.unique(dm):
        m = dm == d
        conv = to_ellipsoidal_height(to_metres(af[m], AltitudeUnit.FEET), str(d), la[m], lo[m])
        h[m] = conv.h_m
        sig[m] = conv.sigma_m
    if n_dropped:
        flags.append(f"dropped_{n_dropped}_samples_without_altitude")
    datums = set(dm.tolist())
    if len(datums) > 1:
        flags.append("mixed_altitude_datum")
    primary = (
        AltitudeDatum.GEOMETRIC_HAE.value
        if AltitudeDatum.GEOMETRIC_HAE.value in datums
        else sorted(datums)[0]
    )
    traj = Trajectory.from_geodetic(
        tt, la, lo, h, max_gap_s=max_interp_gap_s, segment_id=segment_id
    )
    return TrajectoryBuild(traj, float(np.max(sig)), str(primary), int(ok.sum()), n_dropped, flags)
