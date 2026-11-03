"""Digital elevation models (SPEC §7).

Sources: (a) cached local GeoTIFF tiles, (b) USGS 3DEP via the National Map ImageServer
export (returns a GeoTIFF for a bbox; NAVD88 orthometric metres), (c) Copernicus DEM GLO-30
public COGs on AWS (EGM2008 orthometric metres), (d) a clearly labelled synthetic DEM for
offline demos, (e) :class:`FlatDEM` — an *explicit*, operator-supplied constant elevation
with an uncertainty, flagged in every result. A missing DEM raises :class:`DEMError`;
nothing ever silently returns zero ground elevation.

All DEM heights are treated as orthometric (geoid) heights and converted to WGS84 ellipsoidal
height with :func:`aerodoppler.geo.altitude.ground_point_hae` before geometry.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds

from aerodoppler.models import BBox


class DEMError(RuntimeError):
    """Raised when elevation data is missing or invalid for the requested area."""


@dataclass
class ElevationResult:
    elevation_m: NDArray[np.float64]  # orthometric metres
    sigma_m: NDArray[np.float64]
    datum: str
    source: str
    flags: list[str]


class FlatDEM:
    """Operator-declared constant ground elevation (used only when no raster is available)."""

    def __init__(self, elevation_m: float, sigma_m: float, note: str = "operator supplied"):
        self.elev = float(elevation_m)
        self.sigma = float(sigma_m)
        self.note = note

    def elevation(self, lat, lon) -> ElevationResult:
        lat = np.asarray(lat, dtype=np.float64)
        return ElevationResult(
            np.full(lat.shape, self.elev),
            np.full(lat.shape, self.sigma),
            "assumed_orthometric",
            f"flat_dem:{self.note}",
            ["flat_dem_assumed"],
        )

    def describe(self) -> dict[str, Any]:
        return {"type": "flat", "elevation_m": self.elev, "sigma_m": self.sigma, "note": self.note}


class GeoTiffDEM:
    """One or more GeoTIFF tiles in EPSG:4326 (lat/lon). Reads the covering window once per
    query and interpolates bilinearly; NaN/nodata pixels raise or are flagged."""

    def __init__(
        self,
        paths: list[Path],
        vertical_datum: str = "orthometric_geoid",
        sigma_m: float = 3.0,
        source: str = "geotiff",
    ):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise DEMError("no DEM tiles supplied")
        self.tiles = []
        for p in self.paths:
            if not p.exists():
                raise DEMError(f"DEM tile missing: {p}")
            with rasterio.open(p) as ds:
                if ds.crs is None or not ds.crs.is_geographic:
                    raise DEMError(f"DEM {p} must be in a geographic (lat/lon) CRS; got {ds.crs}")
                self.tiles.append((p, ds.bounds, ds.nodata, dict(ds.tags())))
        self.vertical_datum = vertical_datum
        self.sigma_m = sigma_m
        self.source = source
        self.synthetic = any(t[3].get("AERODOPPLER_SYNTHETIC") == "true" for t in self.tiles)

    def covers(self, bbox: BBox) -> bool:
        for _, b, _, _ in self.tiles:
            if (
                b.left <= bbox.west
                and b.right >= bbox.east
                and b.bottom <= bbox.south
                and b.top >= bbox.north
            ):
                return True
        return False

    def elevation(self, lat, lon) -> ElevationResult:
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        out = np.full(lat.shape, np.nan)
        flags: list[str] = []
        for p, b, nodata, _ in self.tiles:
            m = (
                (lat >= b.bottom)
                & (lat <= b.top)
                & (lon >= b.left)
                & (lon <= b.right)
                & np.isnan(out)
            )
            if not m.any():
                continue
            with rasterio.open(p) as ds:
                pad = 2 * max(abs(ds.res[0]), abs(ds.res[1]))
                w = window_from_bounds(
                    max(b.left, lon[m].min() - pad),
                    max(b.bottom, lat[m].min() - pad),
                    min(b.right, lon[m].max() + pad),
                    min(b.top, lat[m].max() + pad),
                    ds.transform,
                )
                w = w.round_offsets().round_lengths()
                arr = ds.read(1, window=w, masked=True).astype(np.float64)
                arr = np.ma.filled(arr, np.nan)
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                tr = ds.window_transform(w)
            # fractional pixel coordinates (pixel centres at +0.5)
            cols = (lon[m] - tr.c) / tr.a - 0.5
            rows = (lat[m] - tr.f) / tr.e - 0.5
            out[m] = _bilinear(arr, rows, cols)
        bad = ~np.isfinite(out)
        if bad.any():
            flags.append(f"dem_nodata_at_{int(bad.sum())}_points")
        if self.synthetic:
            flags.append("synthetic_dem")
        return ElevationResult(
            out, np.full(lat.shape, self.sigma_m), self.vertical_datum, self.source, flags
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "geotiff",
            "tiles": [str(p) for p in self.paths],
            "vertical_datum": self.vertical_datum,
            "sigma_m": self.sigma_m,
            "source": self.source,
            "synthetic": self.synthetic,
        }


def _bilinear(
    arr: NDArray[np.float64], rows: NDArray[np.float64], cols: NDArray[np.float64]
) -> NDArray[np.float64]:
    h, w = arr.shape
    r0 = np.clip(np.floor(rows).astype(int), 0, h - 1)
    c0 = np.clip(np.floor(cols).astype(int), 0, w - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    fr = np.clip(rows - r0, 0, 1)
    fc = np.clip(cols - c0, 0, 1)
    a = (
        arr[r0, c0] * (1 - fr) * (1 - fc)
        + arr[r0, c1] * (1 - fr) * fc
        + arr[r1, c0] * fr * (1 - fc)
        + arr[r1, c1] * fr * fc
    )
    # fall back to nearest when a neighbour is nodata
    nearest = arr[
        np.clip(np.rint(rows).astype(int), 0, h - 1), np.clip(np.rint(cols).astype(int), 0, w - 1)
    ]
    return np.where(np.isfinite(a), a, nearest)


# ---------------------------------------------------------------------------------------
# Tile acquisition
# ---------------------------------------------------------------------------------------

USGS_3DEP_IMAGESERVER = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)
COPERNICUS_S3 = "https://copernicus-dem-30m.s3.amazonaws.com"


def dem_cache_name(prefix: str, bbox: BBox) -> str:
    w, s, e, n = bbox.as_tuple()
    return f"{prefix}_{w:.3f}_{s:.3f}_{e:.3f}_{n:.3f}.tif".replace("-", "m")


def fetch_usgs_3dep(
    bbox: BBox,
    out_path: Path,
    resolution_m: float = 30.0,
    timeout_s: float = 120.0,
    offline: bool = False,
) -> Path:
    """Download a 3DEP GeoTIFF for a bbox (NAVD88 metres). Tiles the request at <= 2000 px."""
    import httpx

    if out_path.exists():
        return out_path
    if offline:
        raise DEMError(f"offline mode: 3DEP tile {out_path.name} not cached")
    clat = math.radians(0.5 * (bbox.south + bbox.north))
    px_deg = resolution_m / 111195.0
    width = int(math.ceil((bbox.east - bbox.west) * math.cos(clat) / px_deg))
    height = int(math.ceil((bbox.north - bbox.south) / px_deg))
    if width > 2000 or height > 2000:
        raise DEMError(
            f"3DEP request too large ({width}x{height} px); shrink the bbox or use coarser resolution"
        )
    params = {
        "bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    r = httpx.get(USGS_3DEP_IMAGESERVER, params=params, timeout=timeout_s, follow_redirects=True)
    if r.status_code != 200 or r.content[:4] not in (b"II*\x00", b"MM\x00*"):
        raise DEMError(f"3DEP export failed (status {r.status_code}): {r.text[:200]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.tif")
    tmp.write_bytes(r.content)
    _stamp_tags(
        tmp,
        out_path,
        {
            "AERODOPPLER_SOURCE": "USGS 3DEP ImageServer exportImage",
            "AERODOPPLER_VERTICAL_DATUM": "NAVD88 (orthometric)",
            "AERODOPPLER_RETRIEVED": dt.datetime.now(dt.UTC).isoformat(),
            "AERODOPPLER_URL": str(r.url),
        },
    )
    return out_path


def copernicus_tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def fetch_copernicus_glo30(
    bbox: BBox, out_dir: Path, timeout_s: float = 300.0, offline: bool = False
) -> list[Path]:
    """Download the 1°x1° Copernicus GLO-30 COG tiles covering the bbox (EGM2008 metres)."""
    import httpx

    paths = []
    for lat in range(int(math.floor(bbox.south)), int(math.floor(bbox.north)) + 1):
        for lon in range(int(math.floor(bbox.west)), int(math.floor(bbox.east)) + 1):
            name = copernicus_tile_name(lat, lon)
            out = out_dir / f"{name}.tif"
            if not out.exists():
                if offline:
                    raise DEMError(f"offline mode: Copernicus tile {name} not cached")
                url = f"{COPERNICUS_S3}/{name}/{name}.tif"
                with httpx.stream("GET", url, timeout=timeout_s, follow_redirects=True) as r:
                    if r.status_code == 404:
                        raise DEMError(
                            f"Copernicus tile {name} does not exist (ocean or outside coverage)"
                        )
                    r.raise_for_status()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".tmp.tif")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_bytes(1 << 20):
                            fh.write(chunk)
                _stamp_tags(
                    tmp,
                    out,
                    {
                        "AERODOPPLER_SOURCE": "Copernicus DEM GLO-30 (AWS public COG)",
                        "AERODOPPLER_VERTICAL_DATUM": "EGM2008 (orthometric)",
                        "AERODOPPLER_RETRIEVED": dt.datetime.now(dt.UTC).isoformat(),
                        "AERODOPPLER_URL": url,
                    },
                )
            paths.append(out)
    return paths


def _stamp_tags(tmp: Path, out: Path, tags: dict[str, str]) -> None:
    try:
        with rasterio.open(tmp, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as ds:
            ds.update_tags(**tags)
    except Exception:  # read-only or COG layout: provenance goes to the sidecar only
        pass
    tmp.rename(out)
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(tags, indent=2))


def make_synthetic_dem(
    bbox: BBox,
    out_path: Path,
    base_elevation_m: float = 120.0,
    amplitude_m: float = 40.0,
    resolution_deg: float = 0.0005,
    seed: int = 0,
) -> Path:
    """Smooth rolling synthetic terrain, clearly tagged SYNTHETIC."""
    rng = np.random.default_rng(seed)
    w = int(math.ceil((bbox.east - bbox.west) / resolution_deg))
    h = int(math.ceil((bbox.north - bbox.south) / resolution_deg))
    y, x = np.mgrid[0:h, 0:w]
    terrain = np.full((h, w), base_elevation_m, dtype=np.float32)
    for _ in range(6):
        kx, ky = rng.uniform(0.5, 3.0, 2) * 2 * np.pi
        ph = rng.uniform(0, 2 * np.pi, 2)
        terrain += (
            amplitude_m / 6 * np.sin(kx * x / w + ph[0]) * np.cos(ky * y / h + ph[1])
        ).astype(np.float32)
    transform = from_bounds(bbox.west, bbox.south, bbox.east, bbox.north, w, h)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
    ) as ds:
        ds.write(terrain, 1)
        ds.update_tags(
            AERODOPPLER_SYNTHETIC="true",
            AERODOPPLER_SOURCE="synthetic rolling terrain (make_synthetic_dem)",
            AERODOPPLER_VERTICAL_DATUM="assumed orthometric",
            AERODOPPLER_SEED=str(seed),
        )
    return out_path


def open_dem_for_bbox(
    dem_dir: Path, bbox: BBox, prefer: str = "auto", offline: bool = False
) -> GeoTiffDEM:
    """Find or fetch DEM coverage for a bbox: local tiles first, then 3DEP (US), then Copernicus."""
    tiles = sorted(dem_dir.glob("*.tif")) if dem_dir.exists() else []
    if tiles:
        try:
            dem = GeoTiffDEM(tiles, source="local_geotiff")
            if dem.covers(bbox):
                return dem
        except DEMError:
            pass
    in_conus = (
        -125.5 <= bbox.west and bbox.east <= -66.0 and 24.0 <= bbox.south and bbox.north <= 49.5
    )
    errors = []
    if prefer in ("auto", "3dep") and in_conus:
        try:
            p = fetch_usgs_3dep(bbox, dem_dir / dem_cache_name("usgs3dep", bbox), offline=offline)
            return GeoTiffDEM(
                [p], vertical_datum="orthometric_geoid", sigma_m=2.0, source="usgs_3dep"
            )
        except DEMError as e:
            errors.append(str(e))
    if prefer in ("auto", "copernicus"):
        try:
            ps = fetch_copernicus_glo30(bbox, dem_dir / "copernicus", offline=offline)
            return GeoTiffDEM(
                ps, vertical_datum="orthometric_geoid", sigma_m=4.0, source="copernicus_glo30"
            )
        except (DEMError, OSError) as e:
            errors.append(str(e))
    raise DEMError(
        "no DEM available for bbox "
        + str(bbox.as_tuple())
        + ": "
        + "; ".join(errors)
        + ". Provide a GeoTIFF in "
        + str(dem_dir)
        + " or an explicit --ground-elevation with --ground-elevation-sigma."
    )
