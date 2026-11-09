"""Search without a geographic area ("blind", within the imported flight data).

No location prior is used: not a search area, not a file name, not a title. The only
geographic limit is the extent of the flight archive(s) imported for the day, and it is
reported as the search scope. Three stages, all budgeted and reported:

A. **Support** — where could the recording possibly be? From every propeller-class flight of
   the day whose speed fits the events, decimated positions are joined in time: an
   unknown-time recording with two passes Δt apart needs two different aircraft whose closest
   approaches are Δt (± tolerance) apart and close enough together that one listener could
   hear both (each within ``max_slant_m``). A known start time instead needs, for every event, an
   aircraft passing at that event's time. The midpoints of such pairs mark candidate cells.
B. **Coarse search** — neighbouring cells are merged into clusters; the existing engine searches
   each cluster with a coarse grid and a short screen budget, best-supported first, until the
   overall budget is spent.
C. **Refine** — the best clusters are searched again at the normal resolution.

The result says how much of the supported area was searched; a cluster that was never searched
is never silently treated as "no match". If nothing is supported, or the evidence cannot
separate the best regions, the answer is an explicit abstention.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aerodoppler.adsb.index import FlightDataset, FlightIndex
from aerodoppler.models import BBox

KM_PER_DEG = 111.32


@dataclass
class BlindConfig:
    step_s: float = 20.0  # decimation of flight tracks for the support join
    max_slant_m: float = 8000.0  # a pass is audible within this distance
    cell_km: float = 10.0  # support cells
    cluster_max_km: float = 60.0  # a cluster is searched as one area of at most this size
    coarse_grid_m: float = 1000.0
    coarse_screen_budget_s: float = 120.0
    coarse_keep_top: int = 4000  # screen survivors per coarse tile run (normal search: 40,000)
    refine_clusters: int = 2
    refine_half_km: float = 10.0  # the refinement searches this far around the coarse best
    budget_s: float = 3600.0
    max_alt_m: float = 3000.0
    classes: tuple[str, ...] = ("piston", "turboprop", "helicopter", "unknown")


@dataclass
class Support:
    cells: dict[tuple[int, int], int]
    best: dict[tuple[int, int], tuple[int, int]]  # cell -> (events explained, pairs doing so)
    cell_deg_lat: float
    cell_deg_lon: float
    n_flights: int
    n_positions: int
    n_pairs: int
    scope: BBox | None
    seconds: float
    notes: list[str] = field(default_factory=list)

    @property
    def area_km2(self) -> float:
        return (
            len(self.cells)
            * (self.cell_deg_lat * KM_PER_DEG)
            * (self.cell_deg_lon * KM_PER_DEG * math.cos(math.radians(self._lat_mid())))
        )

    def _lat_mid(self) -> float:
        if not self.cells:
            return 0.0
        return (
            float(np.mean([i for i, _ in self.cells])) * self.cell_deg_lat + 0.5 * self.cell_deg_lat
        )


def dataset_scope(datasets: list[FlightDataset]) -> BBox | None:
    bbs = []
    for d in datasets:
        req = d.manifest().get("request") or {}
        bb = req.get("search_bbox") or req.get("bbox")
        if bb:
            bbs.append(bb)
    if not bbs:
        return None
    return BBox(
        west=min(b[0] for b in bbs),
        south=min(b[1] for b in bbs),
        east=max(b[2] for b in bbs),
        north=max(b[3] for b in bbs),
    )


def decimated_positions(
    index: FlightIndex, t_min: float, t_max: float, v_min: float, v_max: float, cfg: BlindConfig
) -> tuple[np.ndarray, ...]:
    """(segment_index, time_bin, lat, lon) of every eligible flight, one row per step_s bin,
    computed in DuckDB so the day's samples are never loaded into Python."""
    segs = index.query_segments(
        t_min,
        t_max,
        None,
        min_speed_mps=v_min,
        max_speed_mps=v_max * 1.5,
        classes=list(cfg.classes),
    )
    segs = segs.filter(segs["min_alt_m"] < cfg.max_alt_m) if segs.height else segs
    if segs.height == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0), segs
    ids = segs["segment_id"].to_list()
    globs = index._samples_globs()
    files = "[" + ",".join(f"'{g}'" for g in globs) + "]"
    con = index.con
    con.register("blind_ids", segs.select("segment_id").to_arrow())
    sql = (
        f"SELECT s.segment_id, CAST(floor(s.t / {cfg.step_s}) AS BIGINT) AS tb, avg(s.lat) AS lat, "
        f"avg(s.lon) AS lon FROM read_parquet({files}, hive_partitioning=true, union_by_name=true) s "
        f"JOIN blind_ids USING (segment_id) WHERE s.t BETWEEN {t_min} AND {t_max} "
        "GROUP BY s.segment_id, tb"
    )
    tab = con.sql(sql).fetchnumpy()
    seg_index = {sid: k for k, sid in enumerate(ids)}
    seg = np.array([seg_index[x] for x in tab["segment_id"]], dtype=np.int64)
    return seg, tab["tb"].astype(np.int64), tab["lat"].astype(float), tab["lon"].astype(float), segs


def _speed_ok(speed: np.ndarray, feat: Any) -> np.ndarray:
    """Lenient per-event speed check on a flight's median speed (the engine checks the CPA
    speed exactly later)."""
    return (speed >= 0.5 * feat.v_min_mps) & (speed <= 1.5 * feat.v_max_mps)


def _reach_km(feat: Any, speed: np.ndarray, cfg: BlindConfig, c_mps: float = 343.0) -> np.ndarray:
    """Horizontal distance (km) within which a flight of this speed must pass to produce the
    event: the slant range implied by the ridge's steepest slope for THAT flight's speed,
    ``d = f_c v² / (c |df/dt|)``, times the engine's upper slant margin (3); horizontal <=
    slant. Never above ``max_slant_m``. A flat ridge gives ``max_slant_m``."""
    slope = float(getattr(feat, "slope_max_hz_per_s", 0.0) or 0.0)
    if slope <= 1e-4:
        return np.full(speed.shape, cfg.max_slant_m / 1000.0)
    d = 3.0 * float(feat.f_center_hz) * speed**2 / (c_mps * slope)
    return np.minimum(d, cfg.max_slant_m) / 1000.0


def _flatten_hits(src: np.ndarray, hits: Any, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """query_ball_point lists -> (source index, destination index) arrays."""
    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if lens.sum() == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    flat = np.fromiter((j for h in hits for j in h), dtype=np.int64, count=int(lens.sum()))
    return np.repeat(src, lens), dst[flat]


def support_unknown_time(
    datasets: list[FlightDataset],
    day_t0: float,
    feats: list[Any],
    tol_s: float,
    cfg: BlindConfig,
) -> Support:
    """Cells where two different eligible aircraft pass Δt ± tol apart, each fitting its own
    event (speed, slant range implied by the ridge), close enough together for one listener
    to hear both. ``feats[0:2]`` are the pairing events; every further event is checked at each
    pair (another eligible aircraft within reach of the pair's midpoint at that event's time),
    and cells are ranked by how many events their best pair explains, then by pair count."""
    from scipy.spatial import cKDTree

    fa, fb = sorted(feats[:2], key=lambda f: f.t_center_s)
    extras = list(feats[2:])
    delta_t_s = fb.t_center_s - fa.t_center_s
    t0 = time.perf_counter()
    index = FlightIndex(datasets)
    v_min = min(f.v_min_mps for f in feats)
    v_max = max(f.v_max_mps for f in feats)
    seg, tb, lat, lon, segs = decimated_positions(
        index, day_t0 - 3600.0, day_t0 + 86400.0 + 3600.0, v_min, v_max, cfg
    )
    scope = dataset_scope(datasets)
    cell_lat = cfg.cell_km / KM_PER_DEG
    lat_ref = float(np.median(lat)) if lat.size else 0.0
    cell_lon = cfg.cell_km / (KM_PER_DEG * max(math.cos(math.radians(lat_ref)), 0.1))
    cells: dict[tuple[int, int], int] = {}
    best: dict[tuple[int, int], tuple[int, int]] = {}
    n_pairs = 0
    explained_hist: dict[int, int] = {}
    if lat.size:
        speed = segs["median_speed_mps"].to_numpy().astype(float)[seg]
        # local equirectangular km coordinates (adequate for a few-km neighbourhood)
        x = lon * KM_PER_DEG * math.cos(math.radians(lat_ref))
        y = lat * KM_PER_DEG
        order = np.argsort(tb, kind="stable")
        bounds = np.flatnonzero(np.diff(tb[order])) + 1
        by_bin = {int(tb[ch[0]]): ch for ch in np.split(order, bounds)}

        def lags(offset_s: float) -> range:
            lag = offset_s / cfg.step_s
            w = tol_s / cfg.step_s
            return range(int(math.floor(lag - w)), int(math.ceil(lag + w)) + 1)

        def bin_tree(cache: dict, b: int, ok: np.ndarray, reach: np.ndarray):
            if b not in cache:
                ib = by_bin.get(b)
                ib = ib[ok[ib]] if ib is not None else np.zeros(0, np.int64)
                cache[b] = (
                    ib,
                    cKDTree(np.c_[x[ib], y[ib]]) if ib.size else None,
                    float(reach[ib].max()) if ib.size else 0.0,
                )
            return cache[b]

        ok_a, ok_b = _speed_ok(speed, fa), _speed_ok(speed, fb)
        reach_a, reach_b = _reach_km(fa, speed, cfg), _reach_km(fb, speed, cfg)
        trees_b: dict[int, Any] = {}
        A_parts, B_parts = [], []
        for b, ia in by_bin.items():
            ia = ia[ok_a[ia]]
            if ia.size == 0:
                continue
            for L in lags(delta_t_s):
                ibb, tree, rb_max = bin_tree(trees_b, b + L, ok_b, reach_b)
                if tree is None:
                    continue
                hits = tree.query_ball_point(np.c_[x[ia], y[ia]], r=reach_a[ia] + rb_max)
                pa, pb = _flatten_hits(ia, hits, ibb)
                keep = (seg[pa] != seg[pb]) & (
                    np.hypot(x[pa] - x[pb], y[pa] - y[pb]) <= reach_a[pa] + reach_b[pb]
                )
                A_parts.append(pa[keep])
                B_parts.append(pb[keep])
        A = np.concatenate(A_parts) if A_parts else np.zeros(0, np.int64)
        B = np.concatenate(B_parts) if B_parts else np.zeros(0, np.int64)
        n_pairs = int(A.size)
        mx, my = 0.5 * (x[A] + x[B]), 0.5 * (y[A] + y[B])
        half = 0.5 * np.hypot(x[A] - x[B], y[A] - y[B])
        explained = np.full(A.size, 2, dtype=np.int64)
        for fe in extras:
            ok_e, reach_e = _speed_ok(speed, fe), _reach_km(fe, speed, cfg)
            trees_e: dict[int, Any] = {}
            hit_any = np.zeros(A.size, dtype=bool)
            tb_a = tb[A]
            pair_order = np.argsort(tb_a, kind="stable")
            pb_bounds = np.flatnonzero(np.diff(tb_a[pair_order])) + 1
            for grp in np.split(pair_order, pb_bounds) if A.size else []:
                b = int(tb_a[grp[0]])
                for L in lags(fe.t_center_s - fa.t_center_s):
                    ie, tree, re_max = bin_tree(trees_e, b + L, ok_e, reach_e)
                    if tree is None:
                        continue
                    hits = tree.query_ball_point(np.c_[mx[grp], my[grp]], r=re_max + half[grp])
                    pp, pe = _flatten_hits(grp, hits, ie)
                    good = (
                        (seg[pe] != seg[A[pp]])
                        & (seg[pe] != seg[B[pp]])
                        & (np.hypot(x[pe] - mx[pp], y[pe] - my[pp]) <= reach_e[pe] + half[pp])
                    )
                    hit_any[pp[good]] = True
            explained += hit_any
        ci = np.floor(0.5 * (lat[A] + lat[B]) / cell_lat).astype(np.int64)
        cj = np.floor(0.5 * (lon[A] + lon[B]) / cell_lon).astype(np.int64)
        # distinct flight pairs per cell, and per cell the best number of events explained
        uniq = np.unique(np.c_[ci, cj, seg[A], seg[B], explained], axis=0) if A.size else []
        for i, j, _sa, _sb, e in uniq:
            k = (int(i), int(j))
            cells[k] = cells.get(k, 0) + 1
            be, bc = best.get(k, (0, 0))
            if e > be:
                best[k] = (int(e), 1)
            elif e == be:
                best[k] = (be, bc + 1)
        for e in explained.tolist():
            explained_hist[e] = explained_hist.get(e, 0) + 1
    return Support(
        cells=cells,
        best=best,
        cell_deg_lat=cell_lat,
        cell_deg_lon=cell_lon,
        n_flights=int(segs.height),
        n_positions=int(lat.size),
        n_pairs=n_pairs,
        scope=scope,
        seconds=round(time.perf_counter() - t0, 1),
        notes=[
            f"Δt={delta_t_s:.1f}±{tol_s:.0f} s; each flight within 3x the slant range its "
            f"speed and the ridge slope imply (<= {cfg.max_slant_m / 1000:.0f} km); "
            f"{len(extras)} further event(s) checked; pairs by events explained: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(explained_hist.items()))
        ],
    )


def support_known_time(
    datasets: list[FlightDataset],
    t_cpa_abs: list[float],
    sigma_s: float,
    v_min: float,
    v_max: float,
    cfg: BlindConfig,
) -> Support:
    """Cells within reach of an eligible aircraft at every event's time (known start)."""
    t0 = time.perf_counter()
    index = FlightIndex(datasets)
    lo, hi = min(t_cpa_abs) - sigma_s - 60.0, max(t_cpa_abs) + sigma_s + 60.0
    seg, tb, lat, lon, segs = decimated_positions(index, lo, hi, v_min, v_max, cfg)
    scope = dataset_scope(datasets)
    cell_lat = cfg.cell_km / KM_PER_DEG
    lat_ref = float(np.median(lat)) if lat.size else 0.0
    cell_lon = cfg.cell_km / (KM_PER_DEG * max(math.cos(math.radians(lat_ref)), 0.1))
    reach = int(math.ceil(cfg.max_slant_m / 1000.0 / cfg.cell_km))
    per_event: list[dict[tuple[int, int], set]] = []
    for t_e in t_cpa_abs:
        m = np.abs(tb * cfg.step_s - t_e) <= sigma_s + cfg.step_s
        d: dict[tuple[int, int], set] = {}
        for la, lo_, s_ in zip(lat[m], lon[m], seg[m], strict=True):
            ci, cj = int(math.floor(la / cell_lat)), int(math.floor(lo_ / cell_lon))
            for di in range(-reach, reach + 1):
                for dj in range(-reach, reach + 1):
                    d.setdefault((ci + di, cj + dj), set()).add(int(s_))
        per_event.append(d)
    keys = set(per_event[0]) if per_event else set()
    for d in per_event[1:]:
        keys &= set(d)
    cells = {k: min(len(d[k]) for d in per_event) for k in keys}
    return Support(
        cells=cells,
        best={k: (len(per_event), v) for k, v in cells.items()},
        cell_deg_lat=cell_lat,
        cell_deg_lon=cell_lon,
        n_flights=int(segs.height),
        n_positions=int(lat.size),
        n_pairs=len(cells),
        scope=scope,
        seconds=round(time.perf_counter() - t0, 1),
    )


def cluster_cells(sup: Support, cfg: BlindConfig) -> list[dict[str, Any]]:
    """Merge neighbouring support cells (8-connectivity) and cut clusters into tiles of at most
    ``cluster_max_km``; each tile is one bounded engine run. Ordered by the number of events
    the best supporting pair explains, then by rarity of that support."""
    left = set(sup.cells)
    groups: list[list[tuple[int, int]]] = []
    while left:
        start = left.pop()
        comp, stack = [start], [start]
        while stack:
            ci, cj = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (ci + di, cj + dj)
                    if nb in left:
                        left.remove(nb)
                        comp.append(nb)
                        stack.append(nb)
        groups.append(comp)
    span = max(1, int(cfg.cluster_max_km // cfg.cell_km))
    tiles: list[dict[str, Any]] = []
    for comp in groups:
        by_tile: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for ci, cj in comp:
            by_tile.setdefault((ci // span, cj // span), []).append((ci, cj))
        for cells in by_tile.values():
            i0, i1 = min(c[0] for c in cells), max(c[0] for c in cells) + 1
            j0, j1 = min(c[1] for c in cells), max(c[1] for c in cells) + 1
            bbox = BBox(
                west=j0 * sup.cell_deg_lon,
                east=j1 * sup.cell_deg_lon,
                south=i0 * sup.cell_deg_lat,
                north=i1 * sup.cell_deg_lat,
            )
            if sup.scope is not None:  # never search outside the imported data
                bbox = BBox(
                    west=max(bbox.west, sup.scope.west),
                    east=min(bbox.east, sup.scope.east),
                    south=max(bbox.south, sup.scope.south),
                    north=min(bbox.north, sup.scope.north),
                )
                if bbox.west >= bbox.east or bbox.south >= bbox.north:
                    continue
            explained = max(sup.best.get(c, (0, 0))[0] for c in cells)
            tiles.append(
                {
                    "bbox": bbox,
                    "cells": len(cells),
                    "support": int(sum(sup.cells[c] for c in cells)),
                    "events_explained": int(explained),
                    "support_at_best": int(
                        sum(
                            sup.best[c][1] for c in cells if sup.best.get(c, (0, 0))[0] == explained
                        )
                    ),
                }
            )
    # most events explained first; among equals, most supporting pairs first. Both are
    # heuristics: in dense traffic nearly every region explains every event (measured on the
    # Florida 2026-11-03 day: 76 of 96 regions explained all four), so the order mostly
    # decides what a limited budget reaches, and the result reports what was not searched.
    tiles.sort(key=lambda t: (-t["events_explained"], -t["support_at_best"]))
    return tiles


def region_coverage(
    n_tiles: int, searched: list[dict[str, Any]], stopped_by: str, area_km2: float
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    """How much of the supported area was actually searched. A region whose engine run failed
    (e.g. no terrain data offline) counts as NOT searched, never as "no match"."""
    failed = [s for s in searched if s.get("error")]
    kinds: dict[str, int] = {}
    for f in failed:
        k = str(f["error"]).split(":")[0][:60]
        kinds[k] = kinds.get(k, 0) + 1
    coverage = {
        "supported_clusters": n_tiles,
        "searched_clusters": len(searched) - len(failed),
        "failed_clusters": len(failed),
        "failure_kinds": kinds,
        "not_reached_clusters": n_tiles - len(searched),
        "stopped_by": stopped_by,
        "supported_area_km2": round(area_km2),
    }
    return coverage, failed, kinds


ProgressFn = Callable[[str, dict[str, Any]], None]


def _around(tile: BBox, best: dict[str, float], half_km: float) -> BBox:
    """The part of ``tile`` within ``half_km`` of the coarse best point (refinement area)."""
    dlat = half_km / KM_PER_DEG
    dlon = half_km / (KM_PER_DEG * max(math.cos(math.radians(best["lat"])), 0.1))
    return BBox(
        west=max(tile.west, best["lon"] - dlon),
        east=min(tile.east, best["lon"] + dlon),
        south=max(tile.south, best["lat"] - dlat),
        north=min(tile.north, best["lat"] + dlat),
    )


def run_blind_search(
    settings: Any,
    store: Any,
    *,
    recording_id: str,
    event_ids: list[str],
    mode: str,
    date_utc: str | None,
    start_utc: str | None,
    start_uncertainty_s: float | None,
    flight_dataset_ids: list[str],
    out_dir: Any,
    cfg: BlindConfig | None = None,
    workers: int = 0,
    progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
    allow_synthetic_mix: bool = False,
) -> dict[str, Any]:
    """Blind search (see module docstring). Returns a JSON-able result; writes it to
    ``out_dir/blind_result.json`` and every cluster's engine output to ``out_dir/cluster_*``."""
    import json
    from pathlib import Path

    from aerodoppler.models import QualitySettings, SearchMode, SearchRequest, iso_to_unix
    from aerodoppler.physics.propagation import speed_of_sound_mps
    from aerodoppler.search.datasets import resolve_datasets, search_dates
    from aerodoppler.search.engine import SearchEngine, SearchInputError
    from aerodoppler.search.events import measure_event

    cfg = cfg or BlindConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    rec = store.get_recording(recording_id)
    if rec is None:
        raise ValueError(f"recording {recording_id} not found")
    events = [store.get_event(e) for e in event_ids]
    if any(e is None for e in events):
        raise ValueError("an event of this request does not exist")
    c = speed_of_sound_mps(20.0)
    feats = [measure_event(e, c) for e in events]
    v_min = min(f.v_min_mps for f in feats)
    v_max = max(f.v_max_mps for f in feats)
    unc = float(start_uncertainty_s or 0.0)
    ds_ids, ds_warn = resolve_datasets(
        settings.normalized_dir,
        dates=search_dates(date_utc, start_utc, rec.duration_s, unc),
        recording_synthetic=rec.synthetic,
        requested=flight_dataset_ids,
        bbox=None,
        allow_synthetic_mix=allow_synthetic_mix,
    )
    datasets = [FlightDataset(d, settings.normalized_dir / d) for d in ds_ids]

    def note(_stage: str, **info: Any) -> None:
        if progress:
            progress(_stage, info)

    # ---- A: support ------------------------------------------------------------------------
    note("blind_support", datasets=ds_ids)
    if mode == "unknown-time":
        if len(feats) < 2:
            raise ValueError("unknown-time mode needs two events")
        day_t0 = iso_to_unix(f"{date_utc}T00:00:00Z")
        sup = support_unknown_time(datasets, day_t0, feats, 20.0, cfg)
    else:
        t0 = iso_to_unix(str(start_utc))
        sup = support_known_time(
            datasets, [t0 + f.t_center_s for f in feats], unc + 10.0, v_min, v_max, cfg
        )
    tiles = cluster_cells(sup, cfg)
    note(
        "blind_support_done",
        cells=len(sup.cells),
        area_km2=round(sup.area_km2),
        clusters=len(tiles),
        flights=sup.n_flights,
        seconds=sup.seconds,
    )

    def request(bbox: BBox, coarse: bool) -> SearchRequest:
        q = QualitySettings(
            workers=workers,
            coarse_grid_m=cfg.coarse_grid_m if coarse else 200.0,
            screen_budget_s=cfg.coarse_screen_budget_s if coarse else 900.0,
            # a coarse tile run only has to rank regions: a narrower funnel after the screen
            **({"ultra_keep_top": cfg.coarse_keep_top} if coarse else {}),
        )
        return SearchRequest(
            mode=SearchMode(mode),
            recording_id=recording_id,
            # the first two events drive the engine (as in a normal search); further events
            # only order the regions (stage A)
            event_ids=event_ids[:2] if mode == "unknown-time" else event_ids,
            date_utc=date_utc,
            start_utc=start_utc,
            start_uncertainty_s=start_uncertainty_s,
            bbox=bbox,
            flight_dataset_ids=ds_ids,
            quality_settings=q,
            national_scale_confirmed=True,
            allow_synthetic_mix=allow_synthetic_mix,
        )

    def engine_run(bbox: BBox, coarse: bool, tag: str) -> dict[str, Any]:
        def eng_progress(stage: str, info: dict[str, Any]) -> None:
            keep = ("done", "total", "elapsed_s", "scored", "scheduled")
            note("engine", run=tag, stage=stage, **{k: info[k] for k in keep if k in info})

        eng = SearchEngine(store, settings, progress=eng_progress, cancel_check=should_cancel)
        try:
            res = eng.run(request(bbox, coarse), tag, out_dir / tag)
        except SearchInputError as e:
            return {"error": str(e)}
        best = res.candidates[0] if res.candidates else None
        return {
            "abstained": res.abstention.abstained,
            "reason": res.abstention.reason,
            "grade": res.confidence_grade,
            "best_score": best.score_total if best else None,
            "best": {"lat": best.lat, "lon": best.lon} if best else None,
            "result": res,
        }

    # ---- B: coarse search of supported clusters, best supported first, within budget ----------
    searched: list[dict[str, Any]] = []
    stopped_by = "complete"
    refine_reserve = 0.35 * cfg.budget_s
    for k, tile in enumerate(tiles):
        if should_cancel and should_cancel():
            raise KeyboardInterrupt("cancelled")
        if time.perf_counter() - t_start > cfg.budget_s - refine_reserve:
            stopped_by = "budget"
            break
        r = engine_run(tile["bbox"], True, f"cluster_{k:03d}")
        searched.append(
            {
                **tile,
                "k": k,
                **{
                    x: r.get(x)
                    for x in ("abstained", "reason", "grade", "best_score", "best", "error")
                },
                "_res": r.get("result"),
            }
        )
        note(
            "blind_coarse",
            searched=len(searched),
            clusters=len(tiles),
            elapsed_s=round(time.perf_counter() - t_start),
        )
    # ---- C: refine the best clusters at the normal resolution ---------------------------------
    ranked = sorted(
        [s for s in searched if s.get("best_score") is not None and not s.get("abstained")],
        key=lambda s: -s["best_score"],
    )
    refined: list[dict[str, Any]] = []
    for s in ranked[: cfg.refine_clusters]:
        r = engine_run(
            _around(s["bbox"], s["best"], cfg.refine_half_km), False, f"refine_{s['k']:03d}"
        )
        refined.append(
            {
                "k": s["k"],
                **{
                    x: r.get(x)
                    for x in ("abstained", "reason", "grade", "best_score", "best", "error")
                },
                "_res": r.get("result"),
            }
        )
        note("blind_refine", refined=len(refined))
    final = sorted(
        [r for r in refined if r.get("best_score") is not None and not r.get("abstained")],
        key=lambda r: -r["best_score"],
    )
    # ---- verdict -------------------------------------------------------------------------------
    verdict: dict[str, Any]
    coverage, failed, kinds = region_coverage(len(tiles), searched, stopped_by, sup.area_km2)
    if not tiles:
        verdict = {
            "answer": "abstain",
            "reason": "no place in the imported data is supported: no pair of eligible aircraft passes fits the events",
        }
    elif not final:
        verdict = {
            "answer": "abstain",
            "reason": "no supported region produced a match that passes the engine's evidence thresholds"
            + (
                f"; {len(failed)} region(s) could not be searched and "
                f"{len(tiles) - len(searched)} were not reached"
                if failed or len(tiles) > len(searched)
                else ""
            ),
        }
    else:
        top = final[0]
        grade = top["grade"] or "low"
        reasons = []
        region_gap = None
        others = [r for r in ranked if r["k"] != top["k"]]
        if others:
            region_gap = top["best_score"] - others[0]["best_score"]
            if region_gap < 0.5:
                grade, reasons = (
                    "low",
                    reasons
                    + [
                        f"another region scores within {region_gap:.2f} z (ambiguous between regions)"
                    ],
                )
        if stopped_by == "budget":
            grade = "low"
            reasons.append(
                f"only {len(searched)} of {len(tiles)} supported regions were reached before the budget ran out"
            )
        if failed:
            grade = "low"
            reasons.append(
                f"{len(failed)} supported region(s) could not be searched ("
                + ", ".join(f"{k}: {v}" for k, v in kinds.items())
                + ")"
            )
        verdict = {
            "answer": "location",
            "lat": top["best"]["lat"],
            "lon": top["best"]["lon"],
            "score_total": top["best_score"],
            "engine_grade": top["grade"],
            "grade": grade,
            "grade_reasons": reasons,
            "region_gap_z": region_gap,
            "region_bbox": next(s["bbox"] for s in searched if s["k"] == top["k"]).as_tuple(),
        }
    result = {
        "label": "BLIND SEARCH — no geographic prior; scope = extent of the imported flight data",
        "recording_id": recording_id,
        "mode": mode,
        "scope_bbox": sup.scope.as_tuple() if sup.scope else None,
        "datasets": ds_ids,
        "dataset_warnings": ds_warn,
        "support": {
            "cells": len(sup.cells),
            "area_km2": round(sup.area_km2),
            "flights": sup.n_flights,
            "positions": sup.n_positions,
            "pairs": sup.n_pairs,
            "seconds": sup.seconds,
            "notes": sup.notes,
            "config": {
                k: getattr(cfg, k)
                for k in (
                    "step_s",
                    "max_slant_m",
                    "cell_km",
                    "cluster_max_km",
                    "coarse_grid_m",
                    "coarse_screen_budget_s",
                    "coarse_keep_top",
                    "refine_clusters",
                    "refine_half_km",
                    "budget_s",
                )
            },
        },
        "coverage": coverage,
        "clusters": [
            {k: (v.as_tuple() if isinstance(v, BBox) else v) for k, v in s.items() if k != "_res"}
            for s in searched
        ],
        "refined": [{k: v for k, v in r.items() if k != "_res"} for r in refined],
        "verdict": verdict,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
    }
    (out_dir / "blind_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
