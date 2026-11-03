"""Search orchestration: Mode A (unknown time) and Mode B (known time) end to end
(SPEC §10, §12). Runs as a cancellable, progress-reporting job and writes a reproducible run
directory (candidates, curves, heatmap, flight tracks, manifest)."""

from __future__ import annotations

import datetime as dt
import json
import logging
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from aerodoppler import __version__
from aerodoppler.adsb.index import FlightDataset, FlightIndex, list_datasets
from aerodoppler.audio.spectrogram import Spectrogram
from aerodoppler.config import Settings
from aerodoppler.geo.dem import DEMError, FlatDEM, open_dem_for_bbox
from aerodoppler.jobs.store import Store
from aerodoppler.models import (
    Abstention,
    DopplerEvent,
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
) -> LockedResult:
    if ultra:
        cfg = ctx.ultra_cfg or ctx.coarse_cfg
    else:
        cfg = ctx.coarse_cfg if coarse else ctx.score_cfg
    curves, inputs = [], []
    taus = taus or assignment.tau
    for e, (si, tau, f) in enumerate(zip(assignment.seg_idx, taus, ctx.feats, strict=True)):
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
    return locked_score(ctx.spec, curves, inputs, _t0_candidates(ctx, t0c, coarse, ultra), cfg)


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
        order = np.argsort([f.t_center_s for f in feats])
        events = [events[i] for i in order]
        feats = [feats[i] for i in order]
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

        # datasets / index
        datasets = (
            [FlightDataset(d, self.settings.normalized_dir / d) for d in req.flight_dataset_ids]
            if req.flight_dataset_ids
            else list_datasets(self.settings.normalized_dir)
        )
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

        grid = build_grid(bbox, q.coarse_grid_m, dem, q.max_candidates_coarse, self.counters)
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
            coarse_idx = [
                int(i)
                for i in np.argsort(-ultra_scores)[: q.ultra_keep_top]
                if ultra_scores[i] > -5
            ]
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
        keep = np.argsort(-coarse_scores)[: q.coarse_keep_top]
        keep = [int(i) for i in keep if coarse_scores[i] > -5]

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
            with ctx_mp.Pool(n_workers) as pool:
                for i, r in pool.imap_unordered(
                    _worker_score, tasks, chunksize=max(1, min(64, len(tasks) // (n_workers * 4)))
                ):
                    put(i, r)
                    done += 1
                    if done % every == 0:
                        self._stage(label, done=done, total=len(tasks))
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
                    )
            sa.silence_pen, sa.silence_log = pen, slog
            sa.total = (
                sa.locked.combined
                - pen
                + (sa.assignment.class_prior if q.prefer_propeller_types else 0.0)
            )

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
            seg_ids = [segments[i].segment_id for i in sa.assignment.seg_idx]
            flags = list(flags_common) + (amb if rank <= 2 else [])
            for i in sa.assignment.seg_idx:
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
                    source_fundamental_hz_per_event=[e.f0_hz for e in sa.locked.per_event],
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
                            for i in sa.assignment.seg_idx
                        ],
                        "tau_abs": [unix_to_iso(t) for t in sa.assignment.tau],
                        "slant_m": list(sa.assignment.slant),
                        "alpha_per_s": [e.alpha_per_s for e in sa.locked.per_event],
                        "silence_log": sa.silence_log,
                        "stage": sa.stage,
                    },
                )
            )
        manifest = self._manifest(
            req, run_id, rec, spec, events, feats, c, datasets, dem, synthetic
        )
        result = SearchResult(
            run_id=run_id,
            request=req,
            candidates=cands,
            stage_counters=self.counters,
            abstention=abst,
            warnings=self.warnings,
            manifest=manifest,
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
        used = {i for sa in refined for i in sa.assignment.seg_idx}
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
