"""Hierarchical spatial refinement (SPEC §10.4): 200 m → 100 m → 25 m around the top
spatially distinct basins, recomputing terrain and the physical curves at every point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerodoppler.geo.altitude import ground_point_hae
from aerodoppler.geo.coordinates import geodetic_to_ecef
from aerodoppler.geo.grids import CandidateGrid, local_refinement_grid
from aerodoppler.search.preselect import MIC_HEIGHT_M


@dataclass
class ScoredPoint:
    lat: float
    lon: float
    east: float
    north: float
    ground_m: float
    score: float
    detail: dict[str, Any]


def distinct_maxima(
    points: list[ScoredPoint], min_separation_m: float, k: int
) -> list[ScoredPoint]:
    """Greedy non-maximum suppression: best-first, dropping points closer than the separation."""
    chosen: list[ScoredPoint] = []
    for p in sorted(points, key=lambda q: q.score, reverse=True):
        if all(np.hypot(p.east - c.east, p.north - c.north) >= min_separation_m for c in chosen):
            chosen.append(p)
        if len(chosen) >= k:
            break
    return chosen


def refinement_points(
    grid: CandidateGrid,
    centre_east: float,
    centre_north: float,
    half_width_m: float,
    spacing_m: float,
    dem,
) -> tuple[CandidateGrid, NDArray[np.float64], NDArray[np.float64]]:
    g = local_refinement_grid(grid.frame, centre_east, centre_north, half_width_m, spacing_m)
    elev = dem.elevation(g.lat, g.lon)
    ground = elev.elevation_m
    ok = np.isfinite(ground)
    g = g.subset(ok)
    ground = ground[ok]
    hae = ground_point_hae(ground, g.lat, g.lon, MIC_HEIGHT_M)
    return g, ground, geodetic_to_ecef(g.lat, g.lon, hae)


def landscape_uncertainty_m(points: list[ScoredPoint], best: ScoredPoint, drop: float) -> float:
    """Radius (m) of the region whose score stays within ``drop`` of the best — a description
    of the score landscape, explicitly NOT a calibrated confidence region."""
    near = [p for p in points if p.score >= best.score - drop]
    if len(near) <= 1:
        return 0.0
    d = [float(np.hypot(p.east - best.east, p.north - best.north)) for p in near]
    return float(max(d))
