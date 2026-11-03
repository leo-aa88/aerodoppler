"""Coordinate conversions.

Internal geometry uses ECEF XYZ metres (WGS84). All pyproj transformers are built with
``always_xy=True`` so axis order is explicit: (longitude, latitude, height).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import Transformer

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 1.0 - (WGS84_B / WGS84_A) ** 2


@lru_cache(maxsize=1)
def _geodetic_to_ecef_transformer() -> Transformer:
    return Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


@lru_cache(maxsize=1)
def _ecef_to_geodetic_transformer() -> Transformer:
    return Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


def geodetic_to_ecef(lat_deg: ArrayLike, lon_deg: ArrayLike, h_m: ArrayLike) -> NDArray[np.float64]:
    """WGS84 geodetic (lat, lon, ellipsoidal height) -> ECEF (N, 3) metres."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    lat, lon, h = np.broadcast_arrays(lat, lon, h)
    x, y, z = _geodetic_to_ecef_transformer().transform(lon, lat, h)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


def ecef_to_geodetic(
    xyz: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ECEF (N, 3) -> (lat_deg, lon_deg, h_m)."""
    arr = np.asarray(xyz, dtype=np.float64)
    lon, lat, h = _ecef_to_geodetic_transformer().transform(arr[..., 0], arr[..., 1], arr[..., 2])
    return np.asarray(lat), np.asarray(lon), np.asarray(h)


def enu_basis(lat_deg: float, lon_deg: float) -> NDArray[np.float64]:
    """Rows are the ECEF unit vectors of local East, North, Up at the given point."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    return np.stack([east, north, up])


class LocalFrame:
    """A local ENU tangent frame centred on a reference geodetic point.

    Suitable for grids of a few tens of kilometres (errors stay well below the candidate
    grid spacing). Conversions go through exact ECEF rotation, not a flat-earth
    approximation, so the only error is the curvature of the ellipsoid relative to the
    tangent plane, which affects only the *Up* component (~ d^2 / 2R).
    """

    def __init__(self, lat0: float, lon0: float, h0: float = 0.0):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.h0 = float(h0)
        self.origin = geodetic_to_ecef(lat0, lon0, h0).reshape(3)
        self.basis = enu_basis(lat0, lon0)

    def ecef_to_enu(self, xyz: ArrayLike) -> NDArray[np.float64]:
        d = np.asarray(xyz, dtype=np.float64) - self.origin
        return d @ self.basis.T

    def enu_to_ecef(self, enu: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(enu, dtype=np.float64) @ self.basis + self.origin

    def geodetic_to_enu(self, lat: ArrayLike, lon: ArrayLike, h: ArrayLike) -> NDArray[np.float64]:
        return self.ecef_to_enu(geodetic_to_ecef(lat, lon, h))

    def enu_to_geodetic(
        self, enu: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return ecef_to_geodetic(self.enu_to_ecef(enu))

    def east_north_grid_to_geodetic(
        self, east_m: ArrayLike, north_m: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map planar (E, N) grid coordinates at ellipsoid height h0 to lat/lon.

        The tangent-plane point is projected back onto the ellipsoid by iterating height
        so that returned lat/lon lie on a surface of constant height h0.
        """
        e = np.asarray(east_m, dtype=np.float64)
        n = np.asarray(north_m, dtype=np.float64)
        up = np.zeros_like(e)
        for _ in range(3):
            lat, lon, h = self.enu_to_geodetic(np.stack([e, n, up], axis=-1))
            up = up - (h - self.h0)
        return lat, lon


def haversine_m(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> NDArray[np.float64]:
    """Great-circle distance on a sphere of mean radius; adequate for sanity checks."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return np.asarray(2 * r * np.arcsin(np.sqrt(a)))


def ecef_distance_m(lat1: float, lon1: float, lat2: float, lon2: float, h: float = 0.0) -> float:
    a = geodetic_to_ecef(lat1, lon1, h)
    b = geodetic_to_ecef(lat2, lon2, h)
    return float(np.linalg.norm(a - b))
