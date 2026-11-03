"""Coarse candidate generation and pruning (SPEC §10.2).

Pipeline: query flight segments overlapping the time window / bbox / speed bins → build
trajectories → 200 m ground grid in a local frame (with DEM elevation → observer HAE) →
per segment, delayed CPA at every grid point, keeping only points inside the slant-distance
bounds (sparse table) → Mode A: pairs with ``tau_B - tau_A ≈ Δt ± tol``; Mode B: ``tau ≈ t0 +
s_event ± (sigma_t0 + tol)``. Every stage records its candidate counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from aerodoppler.adsb.index import FlightIndex
from aerodoppler.adsb.segments import TrajectoryBuild, trajectory_from_samples
from aerodoppler.geo.altitude import ground_point_hae
from aerodoppler.geo.coordinates import geodetic_to_ecef
from aerodoppler.geo.grids import CandidateGrid, regular_grid
from aerodoppler.models import BBox, StageCounter
from aerodoppler.physics.doppler import closest_approach_batch
from aerodoppler.search.events import EventFeatures

CLASS_PRIOR = {"piston": 0.0, "turboprop": 0.0, "helicopter": -0.1, "unknown": -0.1, "jet": -0.4}
MIC_HEIGHT_M = 1.5


@dataclass
class SegmentInfo:
    segment_id: str
    traj_build: TrajectoryBuild
    aircraft_class: str
    aircraft_type: str | None
    callsign: str | None
    icao24: str
    class_prior: float
    flags: list[str] = field(default_factory=list)


@dataclass
class GridWithElevation:
    grid: CandidateGrid
    ground_m: NDArray[np.float64]
    ground_sigma_m: NDArray[np.float64]
    observer_hae_m: NDArray[np.float64]
    observer_ecef: NDArray[np.float64]
    dem_flags: list[str]
    dem_source: str


@dataclass
class SparseCPA:
    """CPA results restricted to grid points within the slant bounds of one segment."""

    seg_idx: NDArray[np.int64]
    point_idx: NDArray[np.int64]
    t_emit: NDArray[np.float64]
    slant: NDArray[np.float64]
    tau: NDArray[np.float64]  # delayed CPA (reception time)
    speed: NDArray[np.float64]


@dataclass
class Assignment:
    """One candidate hypothesis: a grid point plus one segment per event."""

    point_idx: int
    seg_idx: tuple[int, ...]
    tau: tuple[float, ...]
    slant: tuple[float, ...]
    t0_center: float
    class_prior: float


@dataclass
class PreselectResult:
    segments: list[SegmentInfo]
    grid: GridWithElevation
    assignments: list[Assignment]
    counters: list[StageCounter]
    warnings: list[str]


def _counter(stage: str, n_in: int, n_out: int, t0: float, note: str = "") -> StageCounter:
    return StageCounter(
        stage=stage,
        candidates_in=n_in,
        candidates_out=n_out,
        note=note,
        elapsed_s=round(time.perf_counter() - t0, 3),
    )


def load_segments(
    index: FlightIndex,
    t_min: float,
    t_max: float,
    bbox: BBox,
    v_min: float,
    v_max: float,
    max_segments: int,
    counters: list[StageCounter],
    warnings: list[str],
) -> list[SegmentInfo]:
    t0 = time.perf_counter()
    segs = index.query_segments(t_min, t_max, bbox, min_speed_mps=v_min, max_speed_mps=v_max * 1.5)
    total = index.query_segments(t_min, t_max, bbox).height
    counters.append(
        _counter(
            "segments_time_bbox_speed",
            total,
            segs.height,
            t0,
            f"speed bins [{v_min:.0f},{v_max:.0f}] m/s (median<=1.5x max)",
        )
    )
    if segs.height > max_segments:
        warnings.append(f"{segs.height} segments exceed cap {max_segments}; keeping the longest")
        segs = (
            segs.with_columns((pl.col("last_utc") - pl.col("first_utc")).alias("dur"))
            .sort("dur", descending=True)
            .head(max_segments)
        )
    t1 = time.perf_counter()
    samples = index.load_samples(segs["segment_id"].to_list()) if segs.height else pl.DataFrame()
    infos: list[SegmentInfo] = []
    dropped = 0
    for row in segs.iter_rows(named=True):
        sid = row["segment_id"]
        sub = samples.filter(pl.col("segment_id") == sid) if samples.height else pl.DataFrame()
        if sub.height < 2:
            dropped += 1
            continue
        tb = trajectory_from_samples(sub, sid)
        if tb.trajectory is None:
            dropped += 1
            continue
        cls = row.get("aircraft_class") or "unknown"
        infos.append(
            SegmentInfo(
                sid,
                tb,
                cls,
                row.get("aircraft_type"),
                row.get("callsign"),
                row["icao24"],
                CLASS_PRIOR.get(cls, -0.1),
                tb.flags + (row.get("flags") or "").split(","),
            )
        )
    counters.append(
        _counter(
            "trajectories_built",
            segs.height,
            len(infos),
            t1,
            f"{dropped} segments unusable (no altitude / too short)",
        )
    )
    return infos


def build_grid(
    bbox: BBox, spacing_m: float, dem, max_points: int, counters: list[StageCounter]
) -> GridWithElevation:
    t0 = time.perf_counter()
    grid = regular_grid(bbox, spacing_m, max_points=max_points)
    elev = dem.elevation(grid.lat, grid.lon)
    ground = elev.elevation_m
    bad = ~np.isfinite(ground)
    if bad.any():
        grid = grid.subset(~bad)
        ground = ground[~bad]
        sig = elev.sigma_m[~bad]
    else:
        sig = elev.sigma_m
    hae = ground_point_hae(ground, grid.lat, grid.lon, MIC_HEIGHT_M)
    ecef = geodetic_to_ecef(grid.lat, grid.lon, hae)
    counters.append(
        _counter(
            "grid_points",
            int(len(grid) + bad.sum()),
            len(grid),
            t0,
            f"{spacing_m:.0f} m spacing; {int(bad.sum())} points without DEM data dropped",
        )
    )
    return GridWithElevation(grid, ground, np.asarray(sig), hae, ecef, elev.flags, elev.source)


def sparse_cpa(
    segments: list[SegmentInfo],
    grid: GridWithElevation,
    c_mps: float,
    d_min: float,
    d_max: float,
    counters: list[StageCounter],
    coarse_dt_s: float = 10.0,
) -> SparseCPA:
    t0 = time.perf_counter()
    parts: list[tuple[np.ndarray, ...]] = []
    for si, seg in enumerate(segments):
        traj = seg.traj_build.trajectory
        assert traj is not None
        coarse_traj = traj.subsample(coarse_dt_s) if coarse_dt_s > 0 else traj
        te, r, tau = closest_approach_batch(coarse_traj, grid.observer_ecef, c_mps)
        m = (r >= d_min) & (r <= d_max) & np.isfinite(r)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        v = np.linalg.norm(traj.velocity(te[idx]), axis=1)
        parts.append((np.full(len(idx), si), idx, te[idx], r[idx], tau[idx], v))
    if not parts:
        empty = np.empty(0)
        res = SparseCPA(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), empty, empty, empty, empty
        )
    else:
        res = SparseCPA(*(np.concatenate([p[i] for p in parts]) for i in range(6)))  # type: ignore[arg-type]
        res.seg_idx = res.seg_idx.astype(np.int64)
        res.point_idx = res.point_idx.astype(np.int64)
    counters.append(
        _counter(
            "cpa_within_slant_bounds",
            len(segments) * len(grid.grid),
            len(res.tau),
            t0,
            f"slant in [{d_min:.0f}, {d_max:.0f}] m",
        )
    )
    return res


def mode_a_pairs(
    sp: SparseCPA,
    segments: list[SegmentInfo],
    feats: list[EventFeatures],
    delta_t_s: float,
    tol_s: float,
    counters: list[StageCounter],
    max_assignments: int,
    warnings: list[str],
) -> list[Assignment]:
    """Two-event pairing: tau_2 - tau_1 = Δt ± tol, with per-event speed/slant bounds."""
    t0 = time.perf_counter()
    if len(sp.tau) == 0:
        counters.append(_counter("mode_a_pairs", 0, 0, t0, "no CPA entries"))
        return []
    f1, f2 = feats[0], feats[1]
    ok1 = (
        (sp.speed >= f1.v_min_mps)
        & (sp.speed <= f1.v_max_mps)
        & (sp.slant >= f1.d_min_m)
        & (sp.slant <= f1.d_max_m)
    )
    ok2 = (
        (sp.speed >= f2.v_min_mps)
        & (sp.speed <= f2.v_max_mps)
        & (sp.slant >= f2.d_min_m)
        & (sp.slant <= f2.d_max_m)
    )
    a_idx = np.flatnonzero(ok1)
    b_idx = np.flatnonzero(ok2)
    counters.append(
        _counter(
            "event_bounds_filter",
            len(sp.tau),
            int(len(a_idx) + len(b_idx)),
            t0,
            "speed/slant bounds per event (entries usable as event 1 + as event 2)",
        )
    )
    t1 = time.perf_counter()
    if len(a_idx) == 0 or len(b_idx) == 0:
        return []
    tbase = float(np.min(sp.tau))
    key_b = sp.point_idx[b_idx].astype(np.float64) * 1e6 + (sp.tau[b_idx] - tbase)
    order = np.argsort(key_b)
    key_b_sorted = key_b[order]
    b_sorted = b_idx[order]
    key_a = sp.point_idx[a_idx].astype(np.float64) * 1e6 + (sp.tau[a_idx] - tbase)
    lo = np.searchsorted(key_b_sorted, key_a + delta_t_s - tol_s, side="left")
    hi = np.searchsorted(key_b_sorted, key_a + delta_t_s + tol_s, side="right")
    out: list[Assignment] = []
    n_pairs_total = int(np.sum(hi - lo))
    for ia, lo_j, hi_j in zip(a_idx, lo, hi, strict=True):
        for j in range(lo_j, hi_j):
            ib = b_sorted[j]
            if sp.seg_idx[ia] == sp.seg_idx[ib]:
                continue  # same flight cannot be both events
            seg_a, seg_b = segments[sp.seg_idx[ia]], segments[sp.seg_idx[ib]]
            prior = 0.5 * (seg_a.class_prior + seg_b.class_prior)
            out.append(
                Assignment(
                    int(sp.point_idx[ia]),
                    (int(sp.seg_idx[ia]), int(sp.seg_idx[ib])),
                    (float(sp.tau[ia]), float(sp.tau[ib])),
                    (float(sp.slant[ia]), float(sp.slant[ib])),
                    float(sp.tau[ia]) - f1.t_center_s,
                    prior,
                )
            )
            if len(out) >= max_assignments:
                warnings.append(
                    f"assignment cap {max_assignments} reached in Mode A pairing; results may be incomplete"
                )
                break
        if len(out) >= max_assignments:
            break
    counters.append(
        _counter(
            "mode_a_pairs",
            n_pairs_total,
            len(out),
            t1,
            f"Δt={delta_t_s:.1f}s ± {tol_s:.1f}s; same-flight pairs removed",
        )
    )
    return out


def mode_b_matches(
    sp: SparseCPA,
    segments: list[SegmentInfo],
    feats: list[EventFeatures],
    t0_known: float,
    sigma_t0: float,
    tol_s: float,
    counters: list[StageCounter],
    max_assignments: int,
    warnings: list[str],
) -> list[Assignment]:
    """Known-time matching: for each event, tau within t0 + s_centre ± (sigma + tol); a
    candidate point needs one matching segment per event (segments distinct across events)."""
    t0 = time.perf_counter()
    if len(sp.tau) == 0:
        counters.append(_counter("mode_b_matches", 0, 0, t0, "no CPA entries"))
        return []
    per_event: list[dict[int, list[int]]] = []
    n_entries = 0
    for f in feats:
        target = t0_known + f.t_center_s
        ok = (
            (np.abs(sp.tau - target) <= sigma_t0 + tol_s)
            & (sp.speed >= f.v_min_mps)
            & (sp.speed <= f.v_max_mps)
            & (sp.slant >= f.d_min_m)
            & (sp.slant <= f.d_max_m)
        )
        d: dict[int, list[int]] = {}
        for i in np.flatnonzero(ok):
            d.setdefault(int(sp.point_idx[i]), []).append(int(i))
        n_entries += int(ok.sum())
        per_event.append(d)
    points = set(per_event[0].keys())
    for d in per_event[1:]:
        points &= set(d.keys())
    out: list[Assignment] = []
    for p in sorted(points):
        combos: list[list[int]] = [[]]
        for d in per_event:
            combos = [
                c + [i]
                for c in combos
                for i in d[p]
                if sp.seg_idx[i] not in {sp.seg_idx[j] for j in c}
            ]
        for c in combos:
            prior = float(np.mean([segments[sp.seg_idx[i]].class_prior for i in c]))
            out.append(
                Assignment(
                    p,
                    tuple(int(sp.seg_idx[i]) for i in c),
                    tuple(float(sp.tau[i]) for i in c),
                    tuple(float(sp.slant[i]) for i in c),
                    t0_known,
                    prior,
                )
            )
            if len(out) >= max_assignments:
                warnings.append(f"assignment cap {max_assignments} reached in Mode B matching")
                break
        if len(out) >= max_assignments:
            break
    counters.append(
        _counter(
            "mode_b_matches",
            n_entries,
            len(out),
            t0,
            f"|tau - (t0 + s)| <= {sigma_t0:.1f}+{tol_s:.1f} s",
        )
    )
    return out


def summarize_segments(segments: list[SegmentInfo]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": s.segment_id,
            "icao24": s.icao24,
            "type": s.aircraft_type,
            "class": s.aircraft_class,
            "callsign": s.callsign,
            "datum": s.traj_build.datum,
            "alt_sigma_m": s.traj_build.altitude_sigma_m,
            "flags": [f for f in s.flags if f],
        }
        for s in segments
    ]
