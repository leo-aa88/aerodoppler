"""Cheap pre-flight workload estimate for a search ("dry run").

Reads only the segment index (no scoring, no CPA computation), so it answers in about a
second even for large areas. Numbers are counts and caps, not runtimes: the one runtime given
is a measured reference run, labelled as such, never an ETA for this search.
"""

from __future__ import annotations

from typing import Any

from aerodoppler.adsb.index import FlightDataset, FlightIndex
from aerodoppler.config import Settings
from aerodoppler.models import Recording, SearchMode, SearchRequest, iso_to_unix
from aerodoppler.resources import default_memory_budget_mb, recommend_workers, snapshot
from aerodoppler.search.datasets import resolve_datasets, search_dates

# measured on the 2026-11-04 dense-traffic benchmark (IMPLEMENTATION_REPORT.md §M7)
REFERENCE = (
    "reference: ~1,250 airborne segments in a 36×33 km area with a 200 m grid took 6–15 min "
    "with 8 workers and peaked at ~3.4 GB"
)


def estimate_workload(settings: Settings, rec: Recording, req: SearchRequest) -> dict[str, Any]:
    q = req.quality_settings
    unc = float(req.start_uncertainty_s or 0.0)
    ds_ids, warnings = resolve_datasets(
        settings.normalized_dir,
        dates=search_dates(req.date_utc, req.start_utc, rec.duration_s, unc),
        recording_synthetic=rec.synthetic,
        requested=req.flight_dataset_ids,
        bbox=req.bbox,
        allow_date_mismatch=req.allow_date_mismatch,
        allow_synthetic_mix=req.allow_synthetic_mix,
    )
    if req.mode == SearchMode.KNOWN_TIME and req.start_utc:
        t0 = iso_to_unix(req.start_utc)
        t_min, t_max = t0 - unc - 600.0, t0 + rec.duration_s + unc + 600.0
    else:
        day0 = iso_to_unix(f"{req.date_utc}T00:00:00Z")
        t_min, t_max = day0 - 3600.0, day0 + 86400.0 + 3600.0
    assert req.bbox is not None
    area_km2 = req.bbox.approx_area_km2()
    index = FlightIndex([FlightDataset(d, settings.normalized_dir / d) for d in ds_ids])
    margin_bbox = req.bbox.expanded_km(settings.adsb_bbox_margin_km)
    segs = index.query_segments(t_min, t_max, margin_bbox)
    n_seg = segs.height
    grid_points = int(area_km2 * 1e6 / max(q.coarse_grid_m, 1.0) ** 2)
    budget = settings.search_memory_budget_mb or default_memory_budget_mb(cap_mb=8192.0)
    workers = q.workers or recommend_workers(budget, 250.0, base_mb=2000.0)
    notes: list[str] = []
    if grid_points > q.max_candidates_coarse:
        notes.append(
            f"{grid_points:,} grid points exceed the {q.max_candidates_coarse:,} limit: the search "
            "will refuse to run; use a coarser grid or a smaller area"
        )
    if n_seg > q.max_segments:
        notes.append(
            f"{n_seg:,} flight segments exceed the {q.max_segments:,} the search holds in memory: "
            "it will refuse to run; shrink the area or the time window"
        )
    if req.mode == SearchMode.UNKNOWN_TIME:
        notes.append(
            "unknown-time mode pairs every aircraft pass with every other one whose time "
            f"separation matches the events (± {q.event_separation_tolerance_s:.0f} s); work grows "
            "roughly with the square of the number of segments. Candidates are scored in order of "
            f"physical consistency for up to {q.screen_budget_s / 60:.0f} min; if that budget runs "
            "out the result says how much was scored and is marked not exhaustive"
        )
    feasible = n_seg <= q.max_segments and grid_points <= q.max_candidates_coarse
    return {
        "datasets": ds_ids,
        "dataset_warnings": warnings,
        "time_window_utc": [t_min, t_max],
        "area_km2": round(area_km2, 1),
        "grid_points": grid_points,
        "segments_in_window_and_area": n_seg,
        "segments_cap": q.max_segments,
        "assignment_cap": q.max_assignments,
        "workers": workers,
        "memory_budget_mb": round(budget),
        "machine": snapshot().as_dict(),
        "within_caps": feasible,
        "notes": notes,
        "reference_run": REFERENCE,
    }
