"""Candidate ground-point grids (SPEC §10.2.3, §12): generated on demand in a local metric
frame, restricted to flight corridors, refined hierarchically; never a CONUS-wide array."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry import LineString, Point

from aerodoppler.geo.coordinates import LocalFrame
from aerodoppler.models import BBox


@dataclass
class CandidateGrid:
    frame: LocalFrame
    east: NDArray[np.float64]
    north: NDArray[np.float64]
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    spacing_m: float

    def __len__(self) -> int:
        return int(self.east.size)

    def subset(self, mask: NDArray[np.bool_]) -> CandidateGrid:
        return CandidateGrid(
            self.frame,
            self.east[mask],
            self.north[mask],
            self.lat[mask],
            self.lon[mask],
            self.spacing_m,
        )


def frame_for_bbox(bbox: BBox) -> LocalFrame:
    return LocalFrame(0.5 * (bbox.south + bbox.north), 0.5 * (bbox.west + bbox.east), 0.0)


def regular_grid(
    bbox: BBox, spacing_m: float, frame: LocalFrame | None = None, max_points: int = 2_000_000
) -> CandidateGrid:
    frame = frame or frame_for_bbox(bbox)
    corners = frame.geodetic_to_enu(
        [bbox.south, bbox.south, bbox.north, bbox.north],
        [bbox.west, bbox.east, bbox.west, bbox.east],
        [0, 0, 0, 0],
    )
    e0, e1 = corners[:, 0].min(), corners[:, 0].max()
    n0, n1 = corners[:, 1].min(), corners[:, 1].max()
    ne = int(np.floor((e1 - e0) / spacing_m)) + 1
    nn = int(np.floor((n1 - n0) / spacing_m)) + 1
    if ne * nn > max_points:
        raise ValueError(
            f"grid would have {ne * nn} points (> {max_points}); use a smaller bbox or corridor masking at a coarser spacing"
        )
    ee, nn_ = np.meshgrid(e0 + spacing_m * np.arange(ne), n0 + spacing_m * np.arange(nn))
    east = ee.ravel()
    north = nn_.ravel()
    lat, lon = frame.east_north_grid_to_geodetic(east, north)
    inside = (lat >= bbox.south) & (lat <= bbox.north) & (lon >= bbox.west) & (lon <= bbox.east)
    return CandidateGrid(frame, east[inside], north[inside], lat[inside], lon[inside], spacing_m)


def corridor_mask(
    grid: CandidateGrid, polylines_enu: list[NDArray[np.float64]], max_distance_m: float
) -> NDArray[np.bool_]:
    """True for grid points within ``max_distance_m`` (horizontal) of any polyline."""
    lines = [LineString(p[:, :2]) for p in polylines_enu if len(p) >= 2]
    if not lines:
        return np.zeros(len(grid), dtype=bool)
    tree = STRtree(lines)
    pts = [Point(e, n) for e, n in zip(grid.east, grid.north, strict=False)]
    hits = tree.query(pts, predicate="dwithin", distance=max_distance_m)
    mask = np.zeros(len(grid), dtype=bool)
    mask[np.unique(hits[0])] = True
    return mask


def local_refinement_grid(
    frame: LocalFrame,
    centre_east: float,
    centre_north: float,
    half_width_m: float,
    spacing_m: float,
) -> CandidateGrid:
    n = int(np.floor(half_width_m / spacing_m))
    offs = spacing_m * np.arange(-n, n + 1)
    ee, nn = np.meshgrid(centre_east + offs, centre_north + offs)
    east, north = ee.ravel(), nn.ravel()
    lat, lon = frame.east_north_grid_to_geodetic(east, north)
    return CandidateGrid(frame, east, north, lat, lon, spacing_m)


def merge_grids(grids: list[CandidateGrid], dedupe_m: float) -> CandidateGrid:
    frame = grids[0].frame
    east = np.concatenate([g.east for g in grids])
    north = np.concatenate([g.north for g in grids])
    key = np.round(east / dedupe_m).astype(np.int64) * 1_000_003 + np.round(
        north / dedupe_m
    ).astype(np.int64)
    _, idx = np.unique(key, return_index=True)
    idx = np.sort(idx)
    lat, lon = frame.east_north_grid_to_geodetic(east[idx], north[idx])
    return CandidateGrid(frame, east[idx], north[idx], lat, lon, min(g.spacing_m for g in grids))
