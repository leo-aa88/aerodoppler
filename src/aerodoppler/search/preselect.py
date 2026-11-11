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
from numba import njit
from numpy.typing import NDArray

from aerodoppler.adsb.index import FlightIndex
from aerodoppler.adsb.segments import TrajectoryBuild, trajectory_from_samples
from aerodoppler.geo.altitude import ground_point_hae
from aerodoppler.geo.coordinates import geodetic_to_ecef
from aerodoppler.geo.grids import CandidateGrid, regular_grid
from aerodoppler.models import BBox, StageCounter
from aerodoppler.physics.doppler import closest_approach_batch
from aerodoppler.search.events import EventFeatures

PAIR_BLOCK = 4_000_000  # Mode A pairs materialised at a time (bounds the temporaries)
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
    """CPA results restricted to grid points within the slant bounds of one segment.

    Indices are int32 (32 bytes per entry in all): on dense traffic this table holds tens of
    millions of entries, and the candidate pool refers into it instead of copying it."""

    seg_idx: NDArray[np.int32]
    point_idx: NDArray[np.int32]
    slant: NDArray[np.float64]
    tau: NDArray[np.float64]  # delayed CPA (reception time)
    speed: NDArray[np.float64]

    @classmethod
    def empty(cls) -> SparseCPA:
        z = np.empty(0)
        return cls(np.empty(0, np.int32), np.empty(0, np.int32), z, z.copy(), z.copy())


@dataclass
class Assignment:
    """One candidate hypothesis: a grid point plus one segment per event."""

    point_idx: int
    seg_idx: tuple[int, ...]
    tau: tuple[float, ...]
    slant: tuple[float, ...]
    t0_center: float
    class_prior: float


class TooManySegments(ValueError):
    pass


@dataclass
class AssignmentPool:
    """Every candidate hypothesis, best physical-consistency prior first, as compact arrays.

    A candidate is one sparse-CPA entry per event (``ent``, int32 when the table allows), so
    the pool costs 8 bytes per event plus its prior instead of copies of point, segments,
    times and distances; :meth:`rows` gathers those for a slice (identical values).
    ``Assignment`` objects are created only for the candidates being scored (:meth:`take`),
    so the pool can hold tens of millions of candidates; the search scores them in this order
    until its budget is spent and reports how far it got (anytime search, see SearchEngine).
    ``cost`` is the consistency prior (sum over events of |log(slant / slant implied by the
    Doppler slope)|; lower = more consistent) — a priority, not a score bound. The recording
    clock origin is ``tau of event 1 - centre of event 1`` (Mode A) or the known start (Mode B)."""

    sp: SparseCPA
    ent: np.ndarray  # (n, k) sparse-CPA entry per event
    cost: np.ndarray
    seg_prior: np.ndarray  # class prior per segment
    event1_centre_s: float = 0.0  # Mode A: t0 = tau_1 - centre_1
    t0_known: float | None = None  # Mode B: t0 is the known start for every candidate

    def __len__(self) -> int:
        return int(len(self.ent))

    def rows(self, sel: slice | np.ndarray) -> tuple[np.ndarray, ...]:
        """(point_idx, seg, tau, slant, t0_center, class_prior) of the selected candidates."""
        e = self.ent[sel]
        sp = self.sp
        point_idx = sp.point_idx[e[:, 0]].astype(np.int64)
        seg = sp.seg_idx[e].astype(np.int64)
        tau = sp.tau[e]
        slant = sp.slant[e]
        if self.t0_known is not None:
            t0c = np.full(len(e), float(self.t0_known))
        else:
            t0c = tau[:, 0] - self.event1_centre_s
        prior = self.seg_prior[seg].mean(axis=1)
        return point_idx, seg, tau, slant, t0c, prior

    def take(self, a: int, b: int) -> list[Assignment]:
        return self._objects(slice(a, min(b, len(self))))

    def take_indices(self, idx: np.ndarray) -> list[Assignment]:
        return self._objects(np.asarray(idx, dtype=np.int64))

    def _objects(self, sel: slice | np.ndarray) -> list[Assignment]:
        point_idx, seg, tau, slant, t0c, prior = self.rows(sel)
        return [
            Assignment(
                int(point_idx[i]),
                tuple(int(x) for x in seg[i]),
                tuple(float(x) for x in tau[i]),
                tuple(float(x) for x in slant[i]),
                float(t0c[i]),
                float(prior[i]),
            )
            for i in range(len(point_idx))
        ]

    @classmethod
    def empty(cls, k: int) -> AssignmentPool:
        return cls(SparseCPA.empty(), np.zeros((0, k), np.int32), np.zeros(0), np.zeros(0))


def _index_dtype(n: int) -> type:
    return np.int32 if n < np.iinfo(np.int32).max else np.int64


def _slope_prior(sp: SparseCPA, idx: np.ndarray, f: EventFeatures, c_mps: float) -> np.ndarray:
    """|log(geometric slant / slant implied by the event's Doppler slope and the flight speed)|."""
    if f.slope_max_hz_per_s <= 1e-4:
        return np.zeros(len(idx))
    d_est = f.f_center_hz * sp.speed[idx] ** 2 / (c_mps * f.slope_max_hz_per_s)
    return np.abs(np.log(np.maximum(sp.slant[idx], 1.0) / np.maximum(d_est, 1.0)))


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
        # keeping "the longest" (as before) silently dropped aircraft for a reason unrelated to
        # the evidence; refuse instead and say how to make the problem smaller
        raise TooManySegments(
            f"{segs.height} flight segments match the area, time window and speed bounds, above "
            f"the {max_segments} this search holds in memory; shrink the search area or the time "
            "window (or raise quality_settings.max_segments if the machine has the memory)"
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


@njit(cache=True)
def _near_chunk_boxes(X, lo, hi, r_max):  # pragma: no cover - numba
    """True for observers within ``r_max`` of at least one axis-aligned chunk box. A point
    farther than ``r_max`` from every box is farther than ``r_max`` from the whole track, so
    skipping it never drops a CPA entry inside the slant bounds."""
    M = X.shape[0]
    C = lo.shape[0]
    r2 = r_max * r_max
    out = np.zeros(M, dtype=np.bool_)
    for m in range(M):
        for c in range(C):
            d2 = 0.0
            for k in range(3):
                v = X[m, k]
                if v < lo[c, k]:
                    d2 += (lo[c, k] - v) ** 2
                elif v > hi[c, k]:
                    d2 += (v - hi[c, k]) ** 2
            if d2 <= r2:
                out[m] = True
                break
    return out


def _chunk_boxes(pos: NDArray[np.float64], k: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Bounding boxes of consecutive chunks of ``k`` samples; consecutive chunks share their
    boundary sample so every track segment lies inside one box."""
    n = len(pos)
    starts = np.arange(0, max(n - 1, 1), k)
    lo = np.array([pos[s : min(s + k + 1, n)].min(axis=0) for s in starts])
    hi = np.array([pos[s : min(s + k + 1, n)].max(axis=0) for s in starts])
    return lo, hi


def sparse_cpa(
    segments: list[SegmentInfo],
    grid: GridWithElevation,
    c_mps: float,
    d_min: float,
    d_max: float,
    counters: list[StageCounter],
    coarse_dt_s: float = 10.0,
    event_bounds: list[tuple[float, float, float, float]] | None = None,
) -> SparseCPA:
    """Closest approach of every segment to every grid point, kept within the slant bounds.

    Lossless pruning: (1) grid points farther than ``d_max`` from every chunk box of a track
    are not evaluated (their CPA cannot be inside the bounds); (2) with ``event_bounds`` —
    one ``(v_min, v_max, d_min, d_max)`` per event — an entry is kept only if at least one
    event could use it, which is exactly the filter every consumer applies afterwards."""
    t0 = time.perf_counter()
    parts: list[list[np.ndarray | None]] = []
    X_all = grid.observer_ecef
    n_eval = 0
    n_in_bounds = 0
    for si, seg in enumerate(segments):
        traj = seg.traj_build.trajectory
        assert traj is not None
        coarse_traj = traj.subsample(coarse_dt_s) if coarse_dt_s > 0 else traj
        lo, hi = _chunk_boxes(coarse_traj.pos)
        near = np.flatnonzero(_near_chunk_boxes(X_all, lo, hi, d_max))
        if near.size == 0:
            continue
        n_eval += int(near.size)
        te, r, tau = closest_approach_batch(coarse_traj, X_all[near], c_mps)
        m = (r >= d_min) & (r <= d_max) & np.isfinite(r)
        if not m.any():
            continue
        n_in_bounds += int(m.sum())
        sel = np.flatnonzero(m)
        v = np.linalg.norm(traj.velocity(te[sel]), axis=1)
        if event_bounds:
            keep = np.zeros(len(sel), dtype=bool)
            for v_lo, v_hi, r_lo, r_hi in event_bounds:
                keep |= (v >= v_lo) & (v <= v_hi) & (r[sel] >= r_lo) & (r[sel] <= r_hi)
            sel, v = sel[keep], v[keep]
            if sel.size == 0:
                continue
        parts.append(
            [
                np.full(len(sel), si, dtype=np.int32),
                near[sel].astype(np.int32),
                r[sel],
                tau[sel],
                v,
            ]
        )
    if not parts:
        res = SparseCPA.empty()
    else:
        # one field at a time, dropping the parts of each field once joined (peak memory)
        cols = []
        for i in range(5):
            cols.append(np.concatenate([p[i] for p in parts]))
            for p in parts:
                p[i] = None
        res = SparseCPA(*cols)
    n_all = len(segments) * len(grid.grid)
    note = (
        f"slant in [{d_min:.0f}, {d_max:.0f}] m; {n_eval} of {n_all} segment-point pairs "
        f"evaluated (the rest are farther than {d_max:.0f} m from the track)"
    )
    if event_bounds:
        note += (
            f"; {len(res.tau)} of {n_in_bounds} in-bounds entries usable by at least one event "
            "(speed and slant bounds)"
        )
    counters.append(_counter("cpa_within_slant_bounds", n_all, len(res.tau), t0, note))
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
    c_mps: float = 343.0,
) -> AssignmentPool:
    """Two-event pairing: tau_2 - tau_1 = Δt ± tol, with per-event speed/slant bounds.

    Fully vectorised. Returns every valid pair ordered by the *physical consistency prior*
    (slant range implied by each event's Doppler slope and the flight's CPA speed vs. the
    geometric slant range). ``max_assignments`` only guards memory; when it bites, the best
    prior is kept and a warning marks the search incomplete.
    """
    t0 = time.perf_counter()
    if len(sp.tau) == 0:
        counters.append(_counter("mode_a_pairs", 0, 0, t0, "no CPA entries"))
        return AssignmentPool.empty(2)
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
        counters.append(_counter("mode_a_pairs", 0, 0, t1, "no entries pass the event bounds"))
        return AssignmentPool.empty(2)
    # every array below is as long as the CPA table (tens of millions of entries on dense
    # traffic): build each in place and drop it as soon as it is used (peak memory)
    idt = _index_dtype(len(sp.tau))
    del ok1, ok2
    tbase = float(np.min(sp.tau))
    key_b = sp.point_idx[b_idx].astype(np.float64)
    key_b *= 1e6
    key_b += sp.tau[b_idx] - tbase
    order = np.argsort(key_b)
    key_b = key_b[order]  # sorted keys
    b_sorted = b_idx[order].astype(idt)
    del order, b_idx
    key_a = sp.point_idx[a_idx].astype(np.float64)
    key_a *= 1e6
    key_a += sp.tau[a_idx] - tbase
    q = key_a + delta_t_s
    q -= tol_s
    lo = np.searchsorted(key_b, q, side="left")
    q = key_a + delta_t_s
    q += tol_s
    counts = np.searchsorted(key_b, q, side="right")
    del q, key_a, key_b
    counts -= lo
    n_pairs_total = int(counts.sum())
    if n_pairs_total == 0:
        counters.append(
            _counter(
                "mode_a_pairs", 0, 0, t1, f"Δt={delta_t_s:.1f}s ± {tol_s:.1f}s: no pair matches"
            )
        )
        return AssignmentPool.empty(2)
    # materialise the (ia, ib) pairs block by block over the event-1 entries (vectorised, with
    # temporaries bounded by the block) into arrays sized for every pair
    a_idx = a_idx.astype(idt)
    cum = np.cumsum(counts)
    ia = np.empty(n_pairs_total, dtype=idt)
    ib = np.empty(n_pairs_total, dtype=idt)
    cost = np.empty(n_pairs_total)
    n_valid = 0
    start = 0
    while start < len(a_idx):
        done_before = int(cum[start - 1]) if start else 0
        end = int(np.searchsorted(cum, done_before + PAIR_BLOCK, side="right"))
        end = min(max(end, start + 1), len(a_idx))
        c = counts[start:end]
        n_c = int(c.sum())
        if n_c:
            offsets = np.arange(n_c) - np.repeat(np.cumsum(c) - c, c)
            ja = a_idx[np.repeat(np.arange(start, end), c)]
            jb = b_sorted[np.repeat(lo[start:end], c) + offsets]
            keep = sp.seg_idx[ja] != sp.seg_idx[jb]
            ja, jb = ja[keep], jb[keep]
            m = len(ja)
            ia[n_valid : n_valid + m] = ja
            ib[n_valid : n_valid + m] = jb
            cost[n_valid : n_valid + m] = _slope_prior(sp, ja, f1, c_mps) + _slope_prior(
                sp, jb, f2, c_mps
            )
            n_valid += m
        start = end
    del a_idx, b_sorted, lo, counts, cum
    n_same = n_pairs_total - n_valid
    ia, ib, cost = ia[:n_valid], ib[:n_valid], cost[:n_valid]
    note = f"Δt={delta_t_s:.1f}s ± {tol_s:.1f}s; {n_same} same-flight pairs removed"
    order = np.argsort(cost, kind="stable")
    if n_valid > max_assignments:
        # memory guard only: the pool is scored in prior order within a time budget anyway
        order = order[:max_assignments]
        warnings.append(
            f"{n_valid} Mode A pairs exceed the cap {max_assignments} that can be held in memory; "
            f"the {max_assignments} with the best slant-distance consistency prior are kept "
            "(shrink the area or the time window for a complete search)"
        )
        note += f"; memory guard {n_valid} -> {max_assignments} by consistency prior"
    ent = np.empty((len(order), 2), dtype=idt)
    ent[:, 0] = ia[order]
    ent[:, 1] = ib[order]
    del ia, ib
    cost = cost[order]
    del order
    pool = AssignmentPool(
        sp,
        ent,
        cost,
        np.array([s.class_prior for s in segments]),
        event1_centre_s=float(f1.t_center_s),
    )
    counters.append(
        _counter(
            "mode_a_pairs", n_pairs_total, len(pool), t1, note + "; ordered by consistency prior"
        )
    )
    return pool


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
    c_mps: float = 343.0,
) -> AssignmentPool:
    """Known-time matching: for each event, tau within t0 + s_centre ± (sigma + tol); a
    candidate point needs one matching segment per event (segments distinct across events).
    Returns every match ordered by the consistency prior (the old cap stopped in grid order,
    i.e. arbitrarily); ``max_assignments`` only guards memory."""
    t0 = time.perf_counter()
    k = len(feats)
    if len(sp.tau) == 0:
        counters.append(_counter("mode_b_matches", 0, 0, t0, "no CPA entries"))
        return AssignmentPool.empty(k)
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
    rows: list[list[int]] = []  # every entry of a row is at the same grid point
    for p in sorted(points):
        combos: list[list[int]] = [[]]
        for d in per_event:
            combos = [
                c + [i]
                for c in combos
                for i in d[p]
                if sp.seg_idx[i] not in {sp.seg_idx[j] for j in c}
            ]
        rows.extend(combos)
    if not rows:
        counters.append(
            _counter("mode_b_matches", n_entries, 0, t0, "no point matches every event")
        )
        return AssignmentPool.empty(k)
    ent = np.array(rows, dtype=np.int64)  # (n, k) CPA-entry indices
    cost = sum(_slope_prior(sp, ent[:, e], feats[e], c_mps) for e in range(k))
    order = np.argsort(cost, kind="stable")
    note = f"|tau - (t0 + s)| <= {sigma_t0:.1f}+{tol_s:.1f} s; ordered by consistency prior"
    if len(order) > max_assignments:
        warnings.append(
            f"{len(order)} Mode B matches exceed the cap {max_assignments} that can be held in "
            "memory; the best consistency prior is kept"
        )
        order = order[:max_assignments]
    ent, cost = ent[order], cost[order]
    pool = AssignmentPool(
        sp,
        ent.astype(_index_dtype(len(sp.tau))),
        cost,
        np.array([s.class_prior for s in segments]),
        t0_known=float(t0_known),
    )
    counters.append(_counter("mode_b_matches", n_entries, len(pool), t0, note))
    return pool


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
