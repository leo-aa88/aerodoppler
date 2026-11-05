"""Search orchestration: Mode A (unknown time) and Mode B (known time) end to end
(SPEC §10, §12). Runs as a cancellable, progress-reporting job and writes a reproducible run
directory (candidates, curves, heatmap, flight tracks, manifest)."""

from __future__ import annotations

import datetime as dt
import json
import logging
import multiprocessing as mp
import os
import re
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from aerodoppler import __version__
from aerodoppler.adsb.index import FlightDataset, FlightIndex
from aerodoppler.audio.spectrogram import Spectrogram
from aerodoppler.config import Settings
from aerodoppler.geo.coordinates import geodetic_to_ecef, haversine_m
from aerodoppler.geo.dem import DEMError, FlatDEM, open_dem_for_bbox
from aerodoppler.jobs.store import Store
from aerodoppler.models import (
    Abstention,
    DopplerEvent,
    Gate,
    LocationCandidate,
    ScoreComponents,
    SearchMode,
    SearchRequest,
    SearchResult,
    StageCounter,
    iso_to_unix,
    unix_to_iso,
)
from aerodoppler.physics.doppler import closest_approach
from aerodoppler.physics.propagation import SoundSpeed
from aerodoppler.search.events import EventFeatures, measure_event
from aerodoppler.search.preselect import (
    Assignment,
    GridWithElevation,
    SegmentInfo,
    build_grid,
    load_segments,
    mode_a_pairs,
    mode_b_matches,
    sparse_cpa,
    summarize_segments,
)
from aerodoppler.search.rank import RankConfig, ambiguity_flags, decide_abstention
from aerodoppler.search.refine import (
    ScoredPoint,
    distinct_maxima,
    landscape_uncertainty_m,
    refinement_points,
)
from aerodoppler.search.score import (
    EventInputs,
    LockedResult,
    ScoreConfig,
    locked_score,
    predicted_curve,
)
from aerodoppler.search.silence import silence_penalty
from aerodoppler.util import git_sha

log = logging.getLogger(__name__)


class SearchCancelled(RuntimeError):
    pass


# warnings emitted by preselection when a hard cap discarded candidates unscored
_TRUNCATION = re.compile(r"exceeds? (the )?cap|cap \d+ reached|cap reached")


class SearchInputError(ValueError):
    pass


@dataclass
class EngineContext:
    spec: Spectrogram
    events: list[DopplerEvent]
    feats: list[EventFeatures]
    c: SoundSpeed
    segments: list[SegmentInfo]
    grid: GridWithElevation
    score_cfg: ScoreConfig
    coarse_cfg: ScoreConfig
    mode: SearchMode
    t0_known: float | None
    sigma_t0: float
    sweep_s: float
    step_s: float
    coarse_step_s: float
    dem: Any
    silence: bool
    silence_audible_m: float
    curve_dt_s: float = 0.5
    ultra_cfg: ScoreConfig | None = None
    ultra_step_s: float = 4.0
    assignments: list[Assignment] = field(default_factory=list)
    extra_tasks: list[tuple[Assignment, np.ndarray]] = field(default_factory=list)


@dataclass
class ScoredAssignment:
    assignment: Assignment
    locked: LockedResult
    observer_ecef: np.ndarray
    lat: float
    lon: float
    east: float
    north: float
    ground_m: float
    silence_pen: float = 0.0
    silence_log: list[dict[str, Any]] = field(default_factory=list)
    total: float = 0.0
    stage: str = "coarse"
    spacing_m: float = 200.0

    def key(self) -> tuple[int, ...]:
        return self.assignment.seg_idx


# ---------------------------------------------------------------------------------------
# worker-side scoring (fork-shared context)
# ---------------------------------------------------------------------------------------

_CTX: EngineContext | None = None
UNEXPLAINED_EVENT_SCORE = 1.0


def attach_extra_events(
    assignments: list[Assignment], sp, feats: list[EventFeatures], tol_s: float
) -> int:
    """For every assignment, explain events 3..k with the sparse-CPA entry at the same grid
    point whose delayed CPA is closest to ``t0_center + centre_k`` (within tol and the event's
    speed/slant bounds). Unexplained slots get seg_idx -1. Returns the number of unexplained."""
    order = np.lexsort((sp.tau, sp.point_idx))
    pts = sp.point_idx[order]
    taus = sp.tau[order]
    starts = np.searchsorted(pts, np.arange(int(pts.max()) + 2) if len(pts) else np.array([0]))
    n_unexpl = 0
    for a in assignments:
        p = a.point_idx
        lo, hi = (int(starts[p]), int(starts[p + 1])) if p + 1 < len(starts) else (0, 0)
        seg_extra, tau_extra, slant_extra = [], [], []
        for f in feats[2:]:
            target = a.t0_center + f.t_center_s
            best, best_d = -1, tol_s
            if hi > lo:
                j = lo + int(np.searchsorted(taus[lo:hi], target))
                for cand in (j - 1, j):
                    if lo <= cand < hi:
                        i = order[cand]
                        d = abs(sp.tau[i] - target)
                        if (
                            d <= best_d
                            and f.v_min_mps <= sp.speed[i] <= f.v_max_mps
                            and f.d_min_m <= sp.slant[i] <= f.d_max_m
                            and int(sp.seg_idx[i]) not in a.seg_idx
                        ):
                            best, best_d = i, d
            if best < 0:
                n_unexpl += 1
                seg_extra.append(-1)
                tau_extra.append(float("nan"))
                slant_extra.append(float("nan"))
            else:
                seg_extra.append(int(sp.seg_idx[best]))
                tau_extra.append(float(sp.tau[best]))
                slant_extra.append(float(sp.slant[best]))
        a.seg_idx = a.seg_idx + tuple(seg_extra)
        a.tau = a.tau + tuple(tau_extra)
        a.slant = a.slant + tuple(slant_extra)
    return n_unexpl


def _t0_candidates(
    ctx: EngineContext, t0_center: float, coarse: bool, ultra: bool = False
) -> np.ndarray:
    half = ctx.sigma_t0 if ctx.mode == SearchMode.KNOWN_TIME else ctx.sweep_s
    step = ctx.ultra_step_s if ultra else (ctx.coarse_step_s if coarse else ctx.step_s)
    n = int(np.floor(half / step))
    return t0_center + step * np.arange(-n, n + 1)


def _score_point(
    ctx: EngineContext,
    assignment: Assignment,
    observer_ecef: np.ndarray,
    coarse: bool,
    taus: tuple[float, ...] | None = None,
    ultra: bool = False,
    t0_override: np.ndarray | None = None,
) -> LockedResult:
    if ultra:
        cfg = ctx.ultra_cfg or ctx.coarse_cfg
    else:
        cfg = ctx.coarse_cfg if coarse else ctx.score_cfg
    curves, inputs = [], []
    taus = taus or assignment.tau
    unexplained: list[int] = []
    for e, (si, tau, f) in enumerate(zip(assignment.seg_idx, taus, ctx.feats, strict=True)):
        if si < 0:  # extra event with no flight at the right time for this hypothesis
            unexplained.append(e)
            continue
        traj = ctx.segments[si].traj_build.trajectory
        assert traj is not None
        half = 0.5 * (f.t_end_s - f.t_start_s) + 30.0
        # at most ~600 curve samples per event (150 in the ultra-coarse screen)
        max_pts = 150.0 if ultra else 600.0
        dt = max(2.0 if ultra else ctx.curve_dt_s, (2.0 * half) / max_pts)
        curves.append(predicted_curve(traj, observer_ecef, ctx.c.c_mps, tau, half, dt))
        ev = ctx.events[e]
        inputs.append(
            EventInputs(
                f.t_center_s,
                f.t_start_s,
                f.t_end_s,
                f.f0_lo_hz,
                f.f0_hi_hz,
                weight=1.0,
                event_id=f.event_id,
                ridge_t=np.array([p.relative_t_s for p in ev.ridge_points]),
                ridge_f=np.array([p.frequency_hz for p in ev.ridge_points]),
                ridge_harmonic=ev.ridge_harmonic,
                plateau_ratio=f.excursion_local,
            )
        )
    t0c = assignment.t0_center if ctx.mode == SearchMode.UNKNOWN_TIME else float(ctx.t0_known)
    t0s = t0_override if t0_override is not None else _t0_candidates(ctx, t0c, coarse, ultra)
    res = locked_score(ctx.spec, curves, inputs, t0s, cfg)
    if unexplained:
        # an unexplained detected event weakens the hypothesis: it enters the mean with a
        # neutral score (archives do miss aircraft) but cannot be compensated by others
        from aerodoppler.search.score import EventScore

        explained = [e.score for e in res.per_event]
        all_scores = explained + [UNEXPLAINED_EVENT_SCORE] * len(unexplained)
        combined = float(np.mean(all_scores))
        if len(explained) > 1:
            combined -= cfg.imbalance_lambda * 0.5 * float(max(explained) - min(explained))
        res.combined = combined
        per_event = list(res.per_event)
        for e in unexplained:
            per_event.insert(
                e, EventScore(UNEXPLAINED_EVENT_SCORE, float("nan"), 0.0, 0.0, 0.0, 0.0, 0)
            )
        res.per_event = per_event
    return res


def _pool_worker_init() -> None:
    """Scoring workers die on SIGTERM. A Python SIGTERM handler inherited through fork runs
    only in a process's main thread, but numba starts threads in the worker: when the signal
    lands on one of those, the main thread stays blocked on the pool's task-queue lock and
    ``Pool.terminate()`` waits for it forever (observed: a benchmark hung 28 min after its
    fine scoring)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _worker_score(args: tuple[int, int]) -> tuple[int, LockedResult]:
    i, level = args  # level: 0 = fine, 1 = coarse, 2 = ultra-coarse
    assert _CTX is not None
    if i >= 0:
        a = _CTX.assignments[i]
        x = _CTX.grid.observer_ecef[a.point_idx]
    else:  # refinement task
        a, x = _CTX.extra_tasks[-1 - i]
    r = _score_point(_CTX, a, x, coarse=level >= 1, ultra=level == 2)
    r.shift_scores = np.empty(0)  # keep the pickled payload small
    r.shifts = np.empty(0)
    return i, r


# ---------------------------------------------------------------------------------------


class SearchEngine:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        self.store = store
        self.settings = settings
        self.progress = progress or (lambda stage, info: None)
        self.cancel_check = cancel_check or (lambda: False)
        self.counters: list[StageCounter] = []
        self.warnings: list[str] = []
        self.t_start = time.perf_counter()
        self._completeness: dict[str, Any] = {}

    # ---- helpers -----------------------------------------------------------------------
    def _check_cancel(self) -> None:
        if self.cancel_check():
            raise SearchCancelled("cancelled by operator")

    def _stage(self, name: str, **info: Any) -> None:
        self._check_cancel()
        info = dict(info)
        info["elapsed_s"] = round(time.perf_counter() - self.t_start, 2)
        info["rss_mb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
        self.progress(name, info)
        log.info("stage %s %s", name, json.dumps(info, default=str))

    # ---- main --------------------------------------------------------------------------
    def run(
        self, req: SearchRequest, run_id: str, out_dir: Path, offline: bool | None = None
    ) -> SearchResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        offline = self.settings.offline if offline is None else offline
        q = req.quality_settings
        rec = self.store.get_recording(req.recording_id)
        if rec is None:
            raise SearchInputError(f"recording {req.recording_id} not found")
        self._stage("load_inputs", recording=rec.id, synthetic=rec.synthetic)
        from aerodoppler.audio.service import ensure_spectrogram

        spec = ensure_spectrogram(self.store, self.settings, rec.id)
        events = []
        for eid in req.event_ids:
            ev = self.store.get_event(eid)
            if ev is None or ev.recording_id != rec.id:
                raise SearchInputError(f"event {eid} not found for recording {rec.id}")
            events.append(ev)
        c = (
            SoundSpeed.from_temperature(q.temperature_c)
            if q.speed_of_sound_mps is None
            else SoundSpeed(q.speed_of_sound_mps, 3.0, "operator supplied")
        )
        feats = [measure_event(ev, c.c_mps) for ev in events]
        # the first two events of the request are the pairing events (sorted by time); any
        # further events are "extra" events explained at the shared t0 (Mode A)
        pair_order = np.argsort([f.t_center_s for f in feats[:2]])
        events = [events[i] for i in pair_order] + events[2:]
        feats = [feats[i] for i in pair_order] + feats[2:]
        for f in feats:
            for fl in f.flags:
                self.warnings.append(f"event {f.event_id}: {fl}")
        self._stage(
            "events_measured",
            events=[
                {
                    "id": f.event_id,
                    "t_center_s": f.t_center_s,
                    "f_center_hz": f.f_center_hz,
                    "v_min": f.v_min_mps,
                    "v_max": f.v_max_mps,
                    "d_min": f.d_min_m,
                    "d_max": f.d_max_m,
                    "f0_range": [f.f0_lo_hz, f.f0_hi_hz],
                }
                for f in feats
            ],
        )

        assert req.bbox is not None
        bbox = req.bbox
        area = bbox.approx_area_km2()
        n_est = area * 1e6 / (q.coarse_grid_m**2)
        if n_est > q.max_candidates_coarse and not req.national_scale_confirmed:
            raise SearchInputError(
                f"bbox of {area:.0f} km² needs ~{n_est:.0f} grid points at {q.coarse_grid_m:.0f} m (> {q.max_candidates_coarse}); shrink the bbox, coarsen the grid or confirm a large-scale search explicitly"
            )

        # time window
        if req.mode == SearchMode.UNKNOWN_TIME:
            assert req.date_utc
            day0 = iso_to_unix(req.date_utc + "T00:00:00Z")
            t_min, t_max = day0 - rec.duration_s - 1800.0, day0 + 86400.0 + rec.duration_s + 1800.0
            t0_known, sigma_t0 = None, 0.0
            if len(feats) < 2:
                raise SearchInputError("unknown-time mode needs two events")
        else:
            assert req.start_utc
            t0_known = iso_to_unix(req.start_utc)
            sigma_t0 = float(
                req.start_uncertainty_s
                if req.start_uncertainty_s is not None
                else (rec.start_time_uncertainty_s or 30.0)
            )
            t_min, t_max = t0_known - sigma_t0 - 600.0, t0_known + rec.duration_s + sigma_t0 + 600.0

        # datasets / index: explicit, date-compatible, same kind (real/SYNTHETIC) as the
        # recording; never "every imported archive" (see search/datasets.py)
        from aerodoppler.search.datasets import (
            DatasetSelectionError,
            resolve_datasets,
            search_dates,
        )

        try:
            ds_ids, ds_warnings = resolve_datasets(
                self.settings.normalized_dir,
                dates=search_dates(
                    req.date_utc,
                    req.start_utc,
                    rec.duration_s,
                    float(req.start_uncertainty_s or rec.start_time_uncertainty_s or 0.0),
                ),
                recording_synthetic=rec.synthetic,
                requested=req.flight_dataset_ids,
                bbox=req.bbox,
                allow_date_mismatch=req.allow_date_mismatch,
                allow_synthetic_mix=req.allow_synthetic_mix,
            )
        except DatasetSelectionError as e:
            raise SearchInputError(str(e)) from e
        self.warnings.extend(ds_warnings)
        datasets = [FlightDataset(d, self.settings.normalized_dir / d) for d in ds_ids]
        datasets = [d for d in datasets if d.exists()]
        if not datasets:
            raise SearchInputError("no flight datasets available; import flights first")
        index = FlightIndex(datasets)
        synthetic_data = any(d.manifest().get("synthetic") for d in datasets) or rec.synthetic

        # DEM
        try:
            if q.ground_elevation_m is not None:
                dem: Any = FlatDEM(q.ground_elevation_m, q.ground_elevation_sigma_m or 30.0)
                self.warnings.append(
                    "flat ground elevation assumed (operator supplied); terrain not modelled"
                )
            else:
                dem = open_dem_for_bbox(self.settings.dem_dir, bbox, offline=offline)
        except DEMError as e:
            raise SearchInputError(f"DEM unavailable: {e}") from e

        v_min = min(f.v_min_mps for f in feats)
        v_max = max(f.v_max_mps for f in feats)
        d_min = min(f.d_min_m for f in feats)
        d_max = min(max(f.d_max_m for f in feats), q.max_slant_m)
        self._stage(
            "query_segments",
            t_min=unix_to_iso(t_min),
            t_max=unix_to_iso(t_max),
            v_bounds=[v_min, v_max],
            slant_bounds=[d_min, d_max],
        )
        segments = load_segments(
            index,
            t_min,
            t_max,
            bbox.expanded_km(self.settings.adsb_bbox_margin_km),
            v_min,
            v_max,
            q.max_segments,
            self.counters,
            self.warnings,
        )
        self._stage("segments_loaded", n=len(segments))
        if not segments:
            return self._finish(
                req,
                run_id,
                out_dir,
                rec,
                spec,
                events,
                feats,
                c,
                [],
                None,
                [],
                [],
                decide_abstention(0, 0, None, None, RankConfig(), []),
                synthetic_data,
                dem,
                datasets,
                t0_known,
            )

        try:
            grid = build_grid(bbox, q.coarse_grid_m, dem, q.max_candidates_coarse, self.counters)
        except ValueError as e:
            raise SearchInputError(
                f"search area too large for a {q.coarse_grid_m:.0f} m grid: {e}"
            ) from e
        self._stage(
            "grid_built", n_points=len(grid.grid), dem=grid.dem_source, dem_flags=grid.dem_flags
        )
        sp = sparse_cpa(segments, grid, c.c_mps, d_min, d_max, self.counters)
        self._stage("cpa_computed", entries=int(len(sp.tau)))
        if req.mode == SearchMode.UNKNOWN_TIME:
            delta_t = feats[1].t_center_s - feats[0].t_center_s
            assignments = mode_a_pairs(
                sp,
                segments,
                feats,
                delta_t,
                q.event_separation_tolerance_s,
                self.counters,
                q.max_assignments,
                self.warnings,
                c_mps=c.c_mps,
            )
        else:
            assignments = mode_b_matches(
                sp,
                segments,
                feats,
                float(t0_known),
                sigma_t0,
                q.event_separation_tolerance_s,
                self.counters,
                q.max_assignments,
                self.warnings,
            )
        if req.mode == SearchMode.UNKNOWN_TIME and len(feats) > 2 and assignments:
            n_unexpl = attach_extra_events(assignments, sp, feats, q.event_separation_tolerance_s)
            self.counters.append(
                StageCounter(
                    stage="extra_events_attached",
                    candidates_in=len(assignments),
                    candidates_out=len(assignments),
                    note=f"{len(feats) - 2} additional event(s) explained by the flight whose delayed CPA matches t0 + centre (± tol); {n_unexpl} event slots left unexplained",
                )
            )
        self._stage("candidates_generated", n=len(assignments))
        if not assignments:
            return self._finish(
                req,
                run_id,
                out_dir,
                rec,
                spec,
                events,
                feats,
                c,
                segments,
                grid,
                [],
                [],
                decide_abstention(len(segments), 0, None, None, RankConfig(), []),
                synthetic_data,
                dem,
                datasets,
                t0_known,
            )

        score_cfg = ScoreConfig(
            f0_step_fraction=q.f0_step_fraction,
            harmonics=tuple(q.harmonics),
            harmonic_weights=tuple([1.0, 0.8, 0.6, 0.45, 0.35, 0.3][: len(q.harmonics)]),
            alphas=tuple(k * q.drift_fraction_max / 600.0 for k in (-1.0, -0.5, 0.0, 0.5, 1.0)),
        )
        coarse_cfg = ScoreConfig(
            f0_step_fraction=q.coarse_f0_step_fraction,
            harmonics=tuple(q.harmonics[:3]),
            harmonic_weights=tuple([1.0, 0.8, 0.6][: len(q.harmonics[:3])]),
            alphas=(0.0,),
        )
        ctx = EngineContext(
            spec,
            events,
            feats,
            c,
            segments,
            grid,
            score_cfg,
            coarse_cfg,
            req.mode,
            t0_known,
            sigma_t0,
            q.alignment_sweep_s,
            q.alignment_step_s,
            q.coarse_t0_step_s,
            dem,
            q.silence_constraint and req.mode == SearchMode.UNKNOWN_TIME,
            q.silence_audible_m,
            curve_dt_s=2 * spec.hop_s,
            assignments=assignments,
        )

        # ---- ultra-coarse screen (every pair; ~10 µs each) ----------------------------------
        self._ctx = ctx
        self._quality = q
        ctx.ultra_cfg = ScoreConfig(
            f0_step_fraction=0.03,
            f0_halfwidth_fraction=0.03,
            harmonics=tuple(q.harmonics[:2]),
            harmonic_weights=(1.0, 0.8)[: len(q.harmonics[:2])],
            alphas=(0.0,),
            f0_polish_step_fraction=0.0,
        )
        ctx.ultra_step_s = q.ultra_t0_step_s
        coarse_idx = list(range(len(assignments)))
        if len(assignments) > q.ultra_keep_top:
            t_u = time.perf_counter()
            ultra_results = self._parallel_scores(
                ctx, coarse_idx, coarse=True, workers=q.workers, ultra=True
            )
            ultra_scores = np.array([r.combined if r is not None else -9.0 for r in ultra_results])
            order_u = np.argsort(-ultra_scores)
            coarse_idx = [int(i) for i in order_u[: q.ultra_keep_top] if ultra_scores[i] > -5]
            dropped_u = order_u[q.ultra_keep_top :]
            self._completeness["ultra"] = {
                "scheduled": int(len(assignments)),
                "kept": int(len(coarse_idx)),
                "best_discarded": float(ultra_scores[dropped_u].max()) if len(dropped_u) else None,
                "best_kept": float(ultra_scores[order_u[0]]),
            }
            self.counters.append(
                StageCounter(
                    stage="ultra_coarse_screen",
                    candidates_in=len(assignments),
                    candidates_out=len(coarse_idx),
                    note=f"all pairs scored with 2 harmonics, 3 f0 values, t0 step {q.ultra_t0_step_s}s, <=150 curve samples; top {q.ultra_keep_top} kept",
                    elapsed_s=round(time.perf_counter() - t_u, 2),
                )
            )
            self._stage(
                "ultra_coarse_screened", kept=len(coarse_idx), best=float(np.max(ultra_scores))
            )
        # ---- coarse scoring -------------------------------------------------------------
        t_c = time.perf_counter()
        coarse_results = self._parallel_scores(ctx, coarse_idx, coarse=True, workers=q.workers)
        coarse_scores = np.array([r.combined if r is not None else -9.0 for r in coarse_results])
        self.counters.append(
            StageCounter(
                stage="coarse_scoring",
                candidates_in=len(coarse_idx),
                candidates_out=int(np.sum(np.isfinite(coarse_scores) & (coarse_scores > -5))),
                note=f"t0 step {q.coarse_t0_step_s}s, f0 step {q.coarse_f0_step_fraction:.3f}",
                elapsed_s=round(time.perf_counter() - t_c, 2),
            )
        )
        self._stage(
            "coarse_scored", best=float(np.nanmax(coarse_scores)) if len(coarse_scores) else None
        )
        order_c = np.argsort(-coarse_scores)
        keep = [int(i) for i in order_c[: q.coarse_keep_top] if coarse_scores[i] > -5]
        dropped_c = order_c[q.coarse_keep_top :]
        finite_dropped = coarse_scores[dropped_c] if len(dropped_c) else np.array([])
        finite_dropped = finite_dropped[finite_dropped > -5]
        self._completeness["coarse"] = {
            "scheduled": int(len(coarse_idx)),
            "kept": int(len(keep)),
            "best_discarded": float(finite_dropped.max()) if finite_dropped.size else None,
            "best_kept": float(coarse_scores[order_c[0]]) if len(order_c) else None,
        }

        # ---- fine scoring + silence -----------------------------------------------------
        t_f = time.perf_counter()
        scored: list[ScoredAssignment] = []
        fine_results = self._parallel_scores(ctx, keep, coarse=False, workers=q.workers)
        for i in keep:
            self._check_cancel()
            a = assignments[i]
            res = fine_results[i]
            if res is None:
                continue
            x = grid.observer_ecef[a.point_idx]
            sa = ScoredAssignment(
                a,
                res,
                x,
                float(grid.grid.lat[a.point_idx]),
                float(grid.grid.lon[a.point_idx]),
                float(grid.grid.east[a.point_idx]),
                float(grid.grid.north[a.point_idx]),
                float(grid.ground_m[a.point_idx]),
                spacing_m=q.coarse_grid_m,
            )
            scored.append(sa)
        self._apply_penalties(ctx, scored, feats, q)
        scored.sort(key=lambda s: s.total, reverse=True)
        self.counters.append(
            StageCounter(
                stage="fine_scoring",
                candidates_in=len(keep),
                candidates_out=len(scored),
                note="full f0/t0 sweep + drift model + silence term",
                elapsed_s=round(time.perf_counter() - t_f, 2),
            )
        )
        self._stage("fine_scored", best=scored[0].total if scored else None)

        # ---- refinement ----------------------------------------------------------------
        t_r = time.perf_counter()
        all_points: list[ScoredPoint] = [self._as_point(s) for s in scored]
        basins = distinct_maxima(all_points, min_separation_m=3 * q.coarse_grid_m, k=q.top_k_refine)
        refined: list[ScoredAssignment] = []
        for b_rank, b in enumerate(basins):
            self._check_cancel()
            sa0: ScoredAssignment = b.detail["sa"]
            best = sa0
            stages = [(1.5 * q.coarse_grid_m, q.refine_grid_m)]
            if b_rank < 2:  # finest grid only for the two leading basins (cost control)
                stages.append((1.5 * q.refine_grid_m, q.fine_grid_m))
            for half, spacing in stages:
                g, ground, ecef = refinement_points(
                    grid.grid, best.east, best.north, half, spacing, dem
                )
                tasks: list[tuple[Assignment, np.ndarray]] = []
                for j in range(len(g)):
                    taus = tuple(
                        closest_approach(
                            segments[si].traj_build.trajectory, ecef[j], c.c_mps
                        ).t_receive_cpa
                        if si >= 0
                        else float("nan")
                        for si in sa0.assignment.seg_idx
                    )  # type: ignore[arg-type]
                    tasks.append(
                        (
                            Assignment(
                                -1,
                                sa0.assignment.seg_idx,
                                taus,
                                sa0.assignment.slant,
                                taus[0] - feats[0].t_center_s,
                                sa0.assignment.class_prior,
                            ),
                            ecef[j],
                        )
                    )
                ctx.extra_tasks = tasks
                results = self._parallel_scores(
                    ctx,
                    [-1 - j for j in range(len(tasks))],
                    coarse=False,
                    workers=q.workers,
                    extra=True,
                )
                local: list[ScoredAssignment] = []
                for j in range(len(g)):
                    res = results[j]
                    if res is None:
                        continue
                    a2 = tasks[j][0]
                    local.append(
                        ScoredAssignment(
                            a2,
                            res,
                            ecef[j],
                            float(g.lat[j]),
                            float(g.lon[j]),
                            float(g.east[j]),
                            float(g.north[j]),
                            float(ground[j]),
                            stage=f"refine_{spacing:.0f}m",
                            spacing_m=spacing,
                        )
                    )
                self._apply_penalties(ctx, local, feats, q, silence_log_from=sa0)
                all_points.extend(self._as_point(s) for s in local)
                best = max(local + [best], key=lambda s: s.total)
            refined.append(best)
        refined.sort(key=lambda s: s.total, reverse=True)
        # basins that converged to the same place are merged (keep the best)
        merged: list[ScoredAssignment] = []
        for sa in refined:
            if all(
                np.hypot(sa.east - m.east, sa.north - m.north) >= 1.5 * q.coarse_grid_m
                for m in merged
            ):
                merged.append(sa)
        refined = merged
        self.counters.append(
            StageCounter(
                stage="refinement",
                candidates_in=len(basins),
                candidates_out=len(refined),
                note=f"{q.refine_grid_m:.0f} m then {q.fine_grid_m:.0f} m grids around {len(basins)} distinct basins",
                elapsed_s=round(time.perf_counter() - t_r, 2),
            )
        )
        self._stage("refined", best=refined[0].total if refined else None)

        # ---- rank / abstain ------------------------------------------------------------
        rank_cfg = RankConfig()
        best_sa = refined[0] if refined else None
        best_cov = (
            float(np.mean([e.coverage for e in best_sa.locked.per_event])) if best_sa else None
        )
        abst = decide_abstention(
            len(segments),
            len(assignments),
            best_sa.total if best_sa else None,
            best_cov,
            rank_cfg,
            [],
        )
        return self._finish(
            req,
            run_id,
            out_dir,
            rec,
            spec,
            events,
            feats,
            c,
            segments,
            grid,
            refined,
            all_points,
            abst,
            synthetic_data,
            dem,
            datasets,
            t0_known,
            scored_coarse=scored,
            coarse_scores=(assignments, coarse_scores),
        )

    # ---- scoring helpers ---------------------------------------------------------------
    def _parallel_scores(
        self,
        ctx: EngineContext,
        idx: list[int],
        coarse: bool,
        workers: int,
        extra: bool = False,
        ultra: bool = False,
    ) -> list[LockedResult | None]:
        """Score assignments (idx >= 0) or refinement tasks (idx < 0 → ctx.extra_tasks) in a
        fork-based pool sharing the spectrogram; returns results positioned by index."""
        global _CTX
        _CTX = ctx
        n_out = len(ctx.extra_tasks) if extra else len(ctx.assignments)
        out: list[LockedResult | None] = [None] * n_out
        n_workers = workers or max(1, mp.cpu_count() - 1)
        level = 2 if ultra else (1 if coarse else 0)
        tasks = [(i, level) for i in idx]
        done = 0
        if ultra:
            label = "ultra_coarse_progress"
        elif coarse:
            label = "coarse_scoring_progress"
        else:
            label = "refine_progress" if extra else "fine_scoring_progress"
        every = 20000 if ultra else (2000 if coarse else 100)

        def put(i: int, r: LockedResult) -> None:
            out[(-1 - i) if i < 0 else i] = r

        if n_workers > 1 and len(tasks) > 16 and hasattr(os, "fork"):
            ctx_mp = mp.get_context("fork")
            with ctx_mp.Pool(n_workers, initializer=_pool_worker_init) as pool:
                for i, r in pool.imap_unordered(
                    _worker_score, tasks, chunksize=max(1, min(64, len(tasks) // (n_workers * 4)))
                ):
                    put(i, r)
                    done += 1
                    if done % every == 0:
                        self._stage(label, done=done, total=len(tasks))
                # normal end: let the workers exit on the sentinel instead of signalling them
                pool.close()
                pool.join()
        else:
            for t in tasks:
                i, r = _worker_score(t)
                put(i, r)
                done += 1
                if done % every == 0:
                    self._stage(label, done=done, total=len(tasks))
        return out

    def _apply_penalties(
        self,
        ctx: EngineContext,
        scored: list[ScoredAssignment],
        feats: list[EventFeatures],
        q,
        silence_log_from: ScoredAssignment | None = None,
    ) -> None:
        for sa in scored:
            pen, slog = 0.0, []
            if ctx.silence and len(feats) >= 2:
                if (
                    silence_log_from is not None
                    and silence_log_from.silence_log is not None
                    and silence_log_from.stage == "coarse"
                ):
                    pen, slog = silence_log_from.silence_pen, silence_log_from.silence_log
                else:
                    gap = (feats[0].t_end_s, feats[1].t_start_s)
                    f0r = (
                        min(f.f0_lo_hz for f in feats) * 0.8,
                        max(f.f0_hi_hz for f in feats) * 1.2,
                    )
                    heard = [x for x in sa.assignment.slant if np.isfinite(x)]
                    pen, slog = silence_penalty(
                        ctx.spec,
                        ctx.segments,
                        set(sa.assignment.seg_idx),
                        sa.observer_ecef,
                        sa.locked.t0_abs,
                        gap,
                        ctx.c.c_mps,
                        ctx.silence_audible_m,
                        f0r,
                        ctx.score_cfg,
                        heard_slant_m=min(heard) if heard else None,
                    )
            sa.silence_pen, sa.silence_log = pen, slog
            sa.total = (
                sa.locked.combined
                - pen
                + (sa.assignment.class_prior if q.prefer_propeller_types else 0.0)
            )

    def _track_support(self, ctx: EngineContext, winner: ScoredAssignment) -> tuple[bool, str]:
        """The matched tracks must physically support the fitted CPA: emission CPA strictly
        inside the sampled interval, no large sample gap near it, no clamped (glitch) velocity,
        and the altitude datum is reported."""
        ok, notes = True, []
        for si, tau in zip(winner.assignment.seg_idx, winner.assignment.tau, strict=True):
            if si < 0 or not np.isfinite(tau):
                continue
            seg = ctx.segments[si]
            traj = seg.traj_build.trajectory
            assert traj is not None
            cpa = closest_approach(traj, winner.observer_ecef, ctx.c.c_mps)
            te = cpa.t_emit_cpa
            edge = min(te - traj.t_start, traj.t_end - te)
            near = (traj.t >= te - 60.0) & (traj.t <= te + 60.0)
            gaps = np.diff(traj.t[near]) if near.sum() > 1 else np.array([0.0])
            speed = float(np.linalg.norm(traj.velocity(np.array([te]))[0]))
            problems = []
            if edge < 5.0:
                problems.append(f"CPA {edge:.0f} s from the track end (unsupported)")
            if gaps.size and gaps.max() > 30.0:
                problems.append(f"{gaps.max():.0f} s sample gap near CPA")
            if speed >= 399.0:
                problems.append("clamped glitch velocity at CPA")
            if not (15.0 <= speed <= 350.0):
                problems.append(f"implausible speed {speed:.0f} m/s at CPA")
            gmax = float(gaps.max()) if gaps.size else 0.0
            notes.append(
                f"{seg.icao24}: CPA {edge:.0f} s inside track, max gap near CPA {gmax:.0f} s, speed {speed:.0f} m/s, datum {seg.traj_build.datum}"
                + (" — " + "; ".join(problems) if problems else "")
            )
            if problems:
                ok = False
        return ok, "; ".join(notes) if notes else "no matched tracks"

    def _completeness_gate(self) -> tuple[bool, str]:
        """The funnel must not have discarded a competitor comparable to the winner."""
        notes, ok = [], True
        for stage in ("ultra", "coarse"):
            c = self._completeness.get(stage)
            if not c:
                continue
            bd, bk = c.get("best_discarded"), c.get("best_kept")
            notes.append(
                f"{stage}: {c['kept']}/{c['scheduled']} kept, best discarded {bd if bd is None else round(bd, 2)} vs best kept {bk if bk is None else round(bk, 2)}"
            )
            if bd is not None and bk is not None and bd >= bk - 0.25:
                ok = False
        caps = [w for w in self.warnings if _TRUNCATION.search(w)]
        if caps:
            ok = False
            notes.append(
                "hard cap hit, so some candidates were never scored and this is NOT an "
                "exhaustive search: " + "; ".join(caps)
            )
        return ok, "; ".join(notes) if notes else "no truncation"

    def _sensitivity(
        self,
        ctx: EngineContext,
        winner: ScoredAssignment,
        q,
        c_sigma_mps: float = 5.0,
        alt_bias_m: float = 150.0,
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Re-refine the winner on a ±100 m / 25 m grid under declared perturbations (speed of
        sound ±5 m/s; observer-vs-flight height bias ±150 m) and report how far the answer
        moves and how much it loses. Engineering scenarios, not a confidence region."""
        from aerodoppler.geo.coordinates import ecef_to_geodetic, haversine_m
        from aerodoppler.physics.propagation import SoundSpeed

        lat0, lon0, h0 = ecef_to_geodetic(winner.observer_ecef)
        scenarios = [
            ("nominal", 0.0, 0.0),
            ("c-5", -c_sigma_mps, 0.0),
            ("c+5", c_sigma_mps, 0.0),
            ("height-150", 0.0, -alt_bias_m),
            ("height+150", 0.0, alt_bias_m),
        ]
        base_c = ctx.c
        results = []
        nominal_total = None
        try:
            for name, dc, dh in scenarios:
                ctx.c = SoundSpeed(base_c.c_mps + dc, base_c.sigma_mps, base_c.note)
                frame = ctx.grid.grid.frame
                centre = geodetic_to_ecef(float(lat0), float(lon0), float(h0) + dh).reshape(3)
                e0, n0 = frame.ecef_to_enu(centre)[:2]
                offs = 25.0 * np.arange(-4, 5)
                ee, nn = np.meshgrid(e0 + offs, n0 + offs)
                lat, lon = frame.east_north_grid_to_geodetic(ee.ravel(), nn.ravel())
                pts = geodetic_to_ecef(lat, lon, np.full(lat.shape, float(h0) + dh))
                best = None
                for j in range(len(lat)):
                    taus = tuple(
                        closest_approach(
                            ctx.segments[si].traj_build.trajectory, pts[j], ctx.c.c_mps
                        ).t_receive_cpa
                        if si >= 0
                        else float("nan")
                        for si in winner.assignment.seg_idx
                    )
                    a2 = Assignment(
                        -1,
                        winner.assignment.seg_idx,
                        taus,
                        winner.assignment.slant,
                        taus[0] - ctx.feats[0].t_center_s,
                        winner.assignment.class_prior,
                    )
                    res = _score_point(ctx, a2, pts[j], coarse=False, taus=taus)
                    tot = (
                        res.combined
                        + (winner.assignment.class_prior if q.prefer_propeller_types else 0.0)
                        - winner.silence_pen
                    )
                    if best is None or tot > best[0]:
                        best = (tot, float(lat[j]), float(lon[j]))
                assert best is not None
                shift = float(haversine_m(best[1], best[2], winner.lat, winner.lon))
                if name == "nominal":
                    nominal_total = best[0]
                results.append(
                    {
                        "scenario": name,
                        "score": round(best[0], 3),
                        "shift_m": round(shift, 1),
                        "lat": best[1],
                        "lon": best[2],
                    }
                )
        finally:
            ctx.c = base_c
        max_shift = max(r["shift_m"] for r in results)
        max_drop = max((nominal_total or 0.0) - r["score"] for r in results)
        tol = max(300.0, float(winner.spacing_m) * 4)
        ok = max_shift <= tol and max_drop <= 1.0
        detail = (
            f"max position shift {max_shift:.0f} m (tolerance {tol:.0f} m), max score drop {max_drop:.2f} z over "
            + ", ".join(
                f"{r['scenario']}: {r['shift_m']:.0f} m / {r['score']:.2f}" for r in results
            )
        )
        return ok, detail, results

    def _confidence_gates(
        self,
        ctx: EngineContext,
        winner: ScoredAssignment,
        rivals: list[ScoredAssignment],
        landscape_radius_m: float | None,
        seed: int,
    ) -> tuple[str, list[dict]]:
        """Graded verdict for the top hypothesis (see rank.gate_results)."""
        from aerodoppler.search.events import assess_event
        from aerodoppler.search.rank import gate_results

        # fresh scoring in-process to get per-event shift curves (workers drop them)
        full = _score_point(
            ctx, winner.assignment, winner.observer_ecef, coarse=False, taus=winner.assignment.tau
        )
        shifts = full.shifts
        own_offsets: list[float | None] = []
        for arr in full.per_event_shift_scores:
            if len(arr) == len(shifts) and np.isfinite(arr).any():
                own_offsets.append(
                    float(shifts[int(np.argmax(arr))] - shifts[full.per_event[0].shift_index])
                )
            else:
                own_offsets.append(None)
        # chance-match null: the same curves at random clock offsets that are far from the
        # fitted clock but still keep every event's window inside the recording. An offset that
        # pushes a curve out of the recording scores an empty window (~0), which is not a
        # chance match; short clips used to get exactly that (or no sample at all) and the gate
        # then compared the score with a meaningless null. Now the sample count is reported
        # and the gate fails visibly when too few valid offsets exist.
        extents = self._curve_extents(ctx, winner, full.t0_abs)
        # one null per event: each event's own curve misaligned against its own window. (A
        # single shared shift would have to satisfy every event's band at once; with windows of
        # different lengths those bands barely overlap or not at all.)
        null_mean, null_std, null_n, null_diag = [], [], [], []
        for i in range(len(full.per_event)):
            offs_i, diag_i = self._null_offsets(ctx, seed + i, extents, event_index=i)
            null_diag.append(diag_i)
            a = np.array([])
            if offs_i.size and extents[i] is not None:
                null = _score_point(
                    ctx,
                    winner.assignment,
                    winner.observer_ecef,
                    coarse=False,
                    taus=winner.assignment.tau,
                    t0_override=full.t0_abs + offs_i,
                )
                arr = null.per_event_shift_scores[i]
                a = arr[np.isfinite(arr) & (arr > -5)]
            null_n.append(int(a.size))
            null_mean.append(float(a.mean()) if a.size else 0.0)
            null_std.append(float(a.std()) if a.size > 1 else 1.0)
        # rival with different flights
        wset = set(winner.assignment.seg_idx)
        riv = [r.total for r in rivals if set(r.assignment.seg_idx) != wset]
        margin = (winner.total - max(riv)) if riv else None
        far = [
            r.total for r in rivals if haversine_m(r.lat, r.lon, winner.lat, winner.lon) >= 1000.0
        ]
        margin_loc = (winner.total - max(far)) if far else None
        # distinct alternative locations (>= 1 km away) that the winner does not clearly beat:
        # reported alongside the grade so a thin win is shown as "one of these", not as one spot
        self._alternatives = self._distinct_alternatives(ctx, winner, rivals)
        track = self._track_support(ctx, winner)
        complete = self._completeness_gate()
        sens_ok, sens_detail, sens_rows = self._sensitivity(ctx, winner, self._quality)
        self._sensitivity_rows = sens_rows
        return gate_results(
            n_events=len(ctx.feats),
            track_support=track,
            completeness=complete,
            sensitivity=(sens_ok, sens_detail),
            mode_known_time=ctx.mode == SearchMode.KNOWN_TIME,
            per_event_scores=[e.score for e in full.per_event],
            per_event_coverage=[e.coverage for e in full.per_event],
            null_mean=null_mean,
            null_std=null_std,
            null_n=null_n,
            null_diagnostics=null_diag,
            event_evidence=[assess_event(ev, ctx.c.c_mps) for ev in ctx.events],
            event_flags=[list(f.flags) for f in ctx.feats],
            own_clock_offsets_s=own_offsets,
            margin_distinct_flights_z=margin,
            margin_distinct_location_z=margin_loc,
            unheard_close_pass_penalty=winner.silence_pen if ctx.silence else None,
            landscape_radius_m=landscape_radius_m,
        )

    @staticmethod
    def _curve_extents(
        ctx: EngineContext, winner: ScoredAssignment, t0_abs: float
    ) -> list[tuple[float, float] | None]:
        """Recording-relative time span of each event's predicted curve at the fitted clock
        (the curve covers the event window plus 30 s, centred on the delayed CPA)."""
        out: list[tuple[float, float] | None] = []
        for si, tau, f in zip(
            winner.assignment.seg_idx, winner.assignment.tau, ctx.feats, strict=True
        ):
            if si < 0:
                out.append(None)
                continue
            half = 0.5 * (f.t_end_s - f.t_start_s) + 30.0
            c = float(tau) - t0_abs
            out.append((c - half, c + half))
        return out

    @staticmethod
    def _null_offsets(
        ctx: EngineContext,
        seed: int,
        extents: list[tuple[float, float] | None] | None = None,
        event_index: int | None = None,
        n_wanted: int = 48,
        min_coverage: float = 0.4,
        min_offset_s: float = 20.0,
        spacing_s: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Clock offsets for the chance-match null: the same predicted curves, misaligned.

        Scoring only looks at each event's own window, so a shifted curve must still cover at
        least ``min_coverage`` of that window, and the shift must be at least half the window
        (and ``min_offset_s``) to be a genuinely different alignment. The admissible offsets
        form bands of a few tens of seconds around the fitted clock; drawing uniformly over
        the whole recording (the previous sampler) almost never hit them on long recordings,
        which is why a 6,380 s clip reported "0 valid samples". Offsets are drawn from the
        admissible set itself, at least ``spacing_s`` apart. Returns the offsets and a
        diagnostic describing the admissible set (for the gate text)."""
        t_lo, t_hi = float(ctx.spec.times[0]), float(ctx.spec.times[-1])
        windows = [(float(f.t_start_s), float(f.t_end_s)) for f in ctx.feats]
        if extents is None:
            extents = [(a - 30.0, b + 30.0) for a, b in windows]
        if event_index is not None:  # this event's null only
            windows, extents = [windows[event_index]], [extents[event_index]]
        reach = max(
            [(e[1] - e[0]) + (b - a) for e, (a, b) in zip(extents, windows, strict=True) if e]
            or [t_hi - t_lo]
        )
        grid = np.arange(-reach, reach + 0.5 * spacing_s, 0.5)
        ok = np.ones(grid.size, dtype=bool)
        for e, (a, b) in zip(extents, windows, strict=True):
            dur = max(b - a, 1e-6)
            ok &= np.abs(grid) >= max(min_offset_s, 0.5 * dur)
            if e is None:
                continue
            c0, c1 = e
            # shifting the clock by +off moves the curve to [c0 - off, c1 - off]
            cov = np.minimum(c1 - grid, min(b, t_hi)) - np.maximum(c0 - grid, max(a, t_lo))
            ok &= cov >= min_coverage * dur
        admissible = grid[ok]
        diag: dict[str, Any] = {
            "admissible_s": float(admissible.size * 0.5),
            "band": [float(np.abs(admissible).min()), float(np.abs(admissible).max())]
            if admissible.size
            else None,
            "rule": f"shift >= max({min_offset_s:.0f} s, half the event window) and >= "
            f"{min_coverage:.0%} of the window still covered by the predicted curve",
        }
        if admissible.size == 0:
            return np.array([], dtype=float), diag
        rng = np.random.default_rng(seed)
        order = rng.permutation(admissible)
        keep: list[float] = []
        for off in order:
            if all(abs(off - k) >= spacing_s for k in keep):
                keep.append(float(off))
                if len(keep) >= n_wanted:
                    break
        diag["drawn"] = len(keep)
        return np.array(sorted(keep), dtype=float), diag

    def _distinct_alternatives(
        self,
        ctx: EngineContext,
        winner: ScoredAssignment,
        rivals: list[ScoredAssignment],
        margin_z: float = 0.5,
        min_distance_m: float = 1000.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in sorted(rivals, key=lambda r: -r.total):
            d = haversine_m(r.lat, r.lon, winner.lat, winner.lon)
            if d < min_distance_m or r.total < winner.total - margin_z:
                continue
            if any(haversine_m(r.lat, r.lon, o["lat"], o["lon"]) < min_distance_m for o in out):
                continue
            out.append(
                {
                    "lat": float(r.lat),
                    "lon": float(r.lon),
                    "score_total": round(float(r.total), 4),
                    "gap_z": round(float(winner.total - r.total), 4),
                    "distance_from_top1_m": round(float(d)),
                    "icao24": sorted({ctx.segments[i].icao24 for i in r.assignment.seg_idx}),
                }
            )
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _as_point(sa: ScoredAssignment) -> ScoredPoint:
        return ScoredPoint(sa.lat, sa.lon, sa.east, sa.north, sa.ground_m, sa.total, {"sa": sa})

    # ---- output ------------------------------------------------------------------------
    def _finish(
        self,
        req,
        run_id,
        out_dir,
        rec,
        spec,
        events,
        feats,
        c,
        segments,
        grid,
        refined: list[ScoredAssignment],
        all_points: list[ScoredPoint],
        abst: Abstention,
        synthetic: bool,
        dem,
        datasets,
        t0_known,
        scored_coarse: list[ScoredAssignment] | None = None,
        coarse_scores=None,
    ) -> SearchResult:
        cands: list[LocationCandidate] = []
        flags_common: list[str] = []
        if synthetic:
            flags_common.append("SYNTHETIC DEMO")
        if grid is not None:
            flags_common.extend(grid.dem_flags)
        distinct_scores = [s.total for s in refined]
        amb = ambiguity_flags(distinct_scores, RankConfig())
        for rank, sa in enumerate(refined, start=1):
            seg_ids = [
                segments[i].segment_id if i >= 0 else "(unexplained)" for i in sa.assignment.seg_idx
            ]
            flags = list(flags_common) + (amb if rank <= 2 else [])
            n_unexpl = sum(1 for i in sa.assignment.seg_idx if i < 0)
            if n_unexpl:
                flags.append(
                    f"{n_unexpl} detected event(s) not explained by any flight at this hypothesis"
                )
            for i in sa.assignment.seg_idx:
                if i < 0:
                    continue
                if segments[i].traj_build.datum != "geometric_wgs84_hae":
                    flags.append(
                        f"{segments[i].segment_id}: altitude datum {segments[i].traj_build.datum} (sigma {segments[i].traj_build.altitude_sigma_m:.0f} m)"
                    )
                if segments[i].aircraft_class == "jet":
                    flags.append(
                        f"{segments[i].segment_id}: classified jet (prior {segments[i].class_prior})"
                    )
                if segments[i].aircraft_class == "unknown":
                    flags.append(
                        f"{segments[i].segment_id}: aircraft type unknown (not treated as jet)"
                    )
            unc = (
                landscape_uncertainty_m(all_points, self._as_point(sa), drop=0.15)
                if all_points
                else None
            )
            cands.append(
                LocationCandidate(
                    lat=sa.lat,
                    lon=sa.lon,
                    ground_elevation_m=sa.ground_m,
                    ground_elevation_datum=getattr(dem, "vertical_datum", "assumed_orthometric"),
                    flight_segment_ids=seg_ids,
                    inferred_start_utc=unix_to_iso(sa.locked.t0_abs),
                    global_time_shift_s=float(
                        sa.locked.t0_abs
                        - (
                            sa.assignment.t0_center
                            if req.mode == SearchMode.UNKNOWN_TIME
                            else t0_known
                        )
                    ),
                    source_fundamental_hz_per_event=[
                        e.f0_hz if np.isfinite(e.f0_hz) else None for e in sa.locked.per_event
                    ],
                    score_total=sa.total,
                    score_components=ScoreComponents(
                        per_event=[e.score for e in sa.locked.per_event],
                        coverage=[e.coverage for e in sa.locked.per_event],
                        silence_penalty=sa.silence_pen,
                        drift_penalty=float(sum(e.drift_penalty for e in sa.locked.per_event)),
                        type_prior=sa.assignment.class_prior,
                    ),
                    flags=flags,
                    rank=rank,
                    nominal_grid_spacing_m=sa.spacing_m,
                    estimated_uncertainty_m=unc,
                    candidate_id=f"cand_{rank:03d}",
                    extra={
                        "segments": [
                            {
                                "segment_id": segments[i].segment_id,
                                "icao24": segments[i].icao24,
                                "type": segments[i].aircraft_type,
                                "class": segments[i].aircraft_class,
                                "callsign": segments[i].callsign,
                            }
                            if i >= 0
                            else {
                                "segment_id": None,
                                "icao24": None,
                                "type": None,
                                "class": "unexplained",
                                "callsign": None,
                            }
                            for i in sa.assignment.seg_idx
                        ],
                        "tau_abs": [
                            unix_to_iso(t) if np.isfinite(t) else None for t in sa.assignment.tau
                        ],
                        "slant_m": [
                            float(x) if np.isfinite(x) else None for x in sa.assignment.slant
                        ],
                        "alpha_per_s": [e.alpha_per_s for e in sa.locked.per_event],
                        "silence_log": sa.silence_log,
                        "stage": sa.stage,
                    },
                )
            )
        manifest = self._manifest(
            req, run_id, rec, spec, events, feats, c, datasets, dem, synthetic
        )
        grade, gates = "low", []
        ctx = getattr(self, "_ctx", None)
        if refined and ctx is not None and not abst.abstained:
            try:
                rivals = list(scored_coarse or []) + list(refined[1:])
                grade, gates = self._confidence_gates(
                    ctx, refined[0], rivals, cands[0].estimated_uncertainty_m, req.seed
                )
            except Exception as e:  # gates are diagnostics; never fail the run
                self.warnings.append(
                    f"confidence gates could not be evaluated: {type(e).__name__}: {e}"
                )
        manifest["sensitivity_profile"] = getattr(self, "_sensitivity_rows", [])
        manifest["search_completeness"] = self._completeness
        result = SearchResult(
            run_id=run_id,
            request=req,
            candidates=cands,
            stage_counters=self.counters,
            abstention=abst,
            warnings=self.warnings,
            manifest=manifest,
            confidence_grade=grade,
            gates=[Gate(**g) for g in gates],
            alternatives=getattr(self, "_alternatives", []),
        )
        self._export(
            out_dir,
            result,
            refined,
            all_points,
            segments,
            spec,
            feats,
            events,
            scored_coarse,
            coarse_scores,
            grid,
        )
        self._stage(
            "finished", abstained=abst.abstained, reason=abst.reason, n_candidates=len(cands)
        )
        return result

    def _manifest(
        self, req, run_id, rec, spec, events, feats, c, datasets, dem, synthetic
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "aerodoppler_version": __version__,
            "git_sha": git_sha(),
            "seed": req.seed,
            "recording": {
                "id": rec.id,
                "sha256": rec.sha256,
                "duration_s": rec.duration_s,
                "sample_rate_hz": rec.sample_rate_hz,
                "synthetic": rec.synthetic,
                "start_time_utc": rec.start_time_utc,
                "start_time_uncertainty_s": rec.start_time_uncertainty_s,
                "recording_date_utc": rec.recording_date_utc,
            },
            "spectrogram": spec.params,
            "events": [e.model_dump() for e in events],
            "event_features": [f.__dict__ for f in feats],
            "speed_of_sound": c.__dict__,
            "datasets": [{"dataset_id": d.dataset_id, "manifest": d.manifest()} for d in datasets],
            "dem": dem.describe() if hasattr(dem, "describe") else str(dem),
            "settings": {
                k: v for k, v in self.settings.model_dump().items() if k not in {"data_dir"}
            },
            "request": req.model_dump(mode="json"),
            "synthetic_data": synthetic,
            "score_label": "relative fit score (uncalibrated); not a probability",
            "elapsed_s": round(time.perf_counter() - self.t_start, 2),
            "peak_rss_mb": round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1),
        }

    def _export(
        self,
        out_dir: Path,
        result: SearchResult,
        refined,
        all_points,
        segments,
        spec,
        feats,
        events,
        scored_coarse,
        coarse_scores,
        grid,
    ) -> None:
        (out_dir / "result.json").write_text(result.model_dump_json(indent=2))
        (out_dir / "manifest.json").write_text(json.dumps(result.manifest, indent=2, default=str))
        # heatmap: every evaluated point
        rows = ["lat,lon,east_m,north_m,score,stage"]
        for p in all_points:
            rows.append(
                f"{p.lat:.6f},{p.lon:.6f},{p.east:.1f},{p.north:.1f},{p.score:.4f},{p.detail['sa'].stage}"
            )
        if coarse_scores is not None and grid is not None:
            assignments, cs = coarse_scores
            for a, s in zip(assignments, cs, strict=True):
                if s > -5:
                    rows.append(
                        f"{grid.grid.lat[a.point_idx]:.6f},{grid.grid.lon[a.point_idx]:.6f},{grid.grid.east[a.point_idx]:.1f},{grid.grid.north[a.point_idx]:.1f},{s:.4f},coarse_all"
                    )
        (out_dir / "heatmap.csv").write_text("\n".join(rows) + "\n")
        # curves per candidate (predicted at best f0 / t0, plus observed ridge points)
        cdir = out_dir / "curves"
        cdir.mkdir(exist_ok=True)
        c_mps = result.manifest["speed_of_sound"]["c_mps"]
        for cand, sa in zip(result.candidates, refined, strict=True):
            payload = {"candidate_id": cand.candidate_id, "t0_abs": sa.locked.t0_abs, "events": []}
            for e_idx, (si, tau) in enumerate(
                zip(sa.assignment.seg_idx, sa.assignment.tau, strict=True)
            ):
                if si < 0:
                    payload["events"].append(
                        {"event_id": feats[e_idx].event_id, "segment_id": None, "unexplained": True}
                    )
                    continue
                traj = segments[si].traj_build.trajectory
                f = feats[e_idx]
                half = 0.5 * (f.t_end_s - f.t_start_s) + 60.0
                curve = predicted_curve(traj, sa.observer_ecef, c_mps, tau, half, spec.hop_s)
                es = sa.locked.per_event[e_idx]
                t_rel = curve.t_receive_abs - sa.locked.t0_abs
                m = (t_rel >= f.t_start_s - 30) & (t_rel <= f.t_end_s + 30)
                payload["events"].append(
                    {
                        "event_id": f.event_id,
                        "segment_id": segments[si].segment_id,
                        "f0_hz": es.f0_hz,
                        "alpha_per_s": es.alpha_per_s,
                        "score": es.score,
                        "coverage": es.coverage,
                        "t_rel_s": np.round(t_rel[m], 3).tolist(),
                        "doppler_factor": np.round(curve.factor[m], 6).tolist(),
                        "slant_range_m": np.round(curve.slant_range_m[m], 1).tolist(),
                        "predicted_hz_by_harmonic": {
                            str(h): np.round(
                                es.f0_hz
                                * (1 + es.alpha_per_s * (t_rel[m] - f.t_center_s))
                                * h
                                * curve.factor[m],
                                3,
                            ).tolist()
                            for h in result.request.quality_settings.harmonics
                        },
                        "observed_ridge": {
                            "t_rel_s": [p.relative_t_s for p in events[e_idx].ridge_points],
                            "frequency_hz": [p.frequency_hz for p in events[e_idx].ridge_points],
                            "harmonic": events[e_idx].ridge_harmonic,
                        },
                        "delayed_cpa_utc": unix_to_iso(tau),
                        "delayed_cpa_rel_s": tau - sa.locked.t0_abs,
                    }
                )
            (cdir / f"{cand.candidate_id}.json").write_text(json.dumps(payload))
        # flight tracks for the map (segments used by any candidate + all considered)
        used = {i for sa in refined for i in sa.assignment.seg_idx if i >= 0}
        tracks = []
        for i, s in enumerate(segments):
            traj = s.traj_build.trajectory
            if traj is None:
                continue
            lat, lon, h = traj.geodetic()
            step = max(1, len(lat) // 400)
            tracks.append(
                {
                    "segment_id": s.segment_id,
                    "icao24": s.icao24,
                    "type": s.aircraft_type,
                    "class": s.aircraft_class,
                    "callsign": s.callsign,
                    "used": i in used,
                    "lat": np.round(lat[::step], 5).tolist(),
                    "lon": np.round(lon[::step], 5).tolist(),
                    "alt_hae_m": np.round(h[::step], 0).tolist(),
                    "t": np.round(traj.t[::step], 1).tolist(),
                }
            )
        (out_dir / "tracks.json").write_text(json.dumps(tracks))
        (out_dir / "segments.json").write_text(
            json.dumps(summarize_segments(segments), default=str)
        )
