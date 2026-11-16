from pathlib import Path

import numpy as np
import pytest

from aerodoppler.geo.altitude import ground_point_hae
from aerodoppler.geo.dem import DEMError, FlatDEM, GeoTiffDEM, make_synthetic_dem, open_dem_for_bbox
from aerodoppler.geo.grids import corridor_mask, local_refinement_grid, merge_grids, regular_grid
from aerodoppler.models import BBox

BB = BBox(west=-75.2, south=39.9, east=-74.8, north=40.1)


def test_synthetic_dem_is_labelled_and_sampled(tmp_path: Path):
    p = make_synthetic_dem(BB, tmp_path / "syn.tif", base_elevation_m=120.0, amplitude_m=40.0)
    dem = GeoTiffDEM([p])
    assert dem.synthetic and dem.covers(BB)
    r = dem.elevation([40.0, 39.95], [-75.0, -74.9])
    assert np.all(np.isfinite(r.elevation_m)) and "synthetic_dem" in r.flags
    assert 60 < r.elevation_m.min() and r.elevation_m.max() < 180
    hae = ground_point_hae(r.elevation_m, [40.0, 39.95], [-75.0, -74.9])
    assert np.all(hae < r.elevation_m)  # negative geoid undulation in the US east
    outside = dem.elevation([45.0], [-75.0])
    assert np.isnan(outside.elevation_m).all() and any("nodata" in f for f in outside.flags)


def test_missing_dem_raises_not_zero(tmp_path: Path):
    with pytest.raises(DEMError):
        GeoTiffDEM([tmp_path / "nope.tif"])
    with pytest.raises(DEMError):
        open_dem_for_bbox(tmp_path, BBox(west=10, south=50, east=10.1, north=50.1), offline=True)
    flat = FlatDEM(50.0, 20.0)
    r = flat.elevation([1.0, 2.0], [3.0, 4.0])
    assert "flat_dem_assumed" in r.flags and np.all(r.sigma_m == 20.0)


def test_regular_grid_spacing_and_corridor_mask():
    g = regular_grid(BB, 200.0)
    assert len(g) > 1000
    d = np.diff(np.sort(np.unique(np.round(g.east))))
    assert np.allclose(d[d > 1], 200.0, atol=1.0)
    assert np.all((g.lat >= BB.south) & (g.lat <= BB.north))
    # corridor: a straight east-west line through the frame origin
    line = np.array([[-30000.0, 0.0, 1000.0], [30000.0, 0.0, 1000.0]])
    m = corridor_mask(g, [line], 1000.0)
    assert m.any() and np.all(np.abs(g.north[m]) <= 1000.0 + 1e-6)
    assert np.all(np.abs(g.north[~m]) > 1000.0 - 200.0)
    fine = local_refinement_grid(g.frame, 0.0, 0.0, 300.0, 100.0)
    assert len(fine) == 49
    merged = merge_grids([g.subset(m), fine], dedupe_m=10.0)
    assert len(merged) <= len(g.subset(m)) + len(fine)


def test_grid_size_guard():
    with pytest.raises(ValueError):
        regular_grid(BBox(west=-125, south=25, east=-66, north=49), 200.0)


@pytest.mark.network
def test_usgs_3dep_and_copernicus_online(tmp_path: Path):
    from aerodoppler.geo.dem import fetch_copernicus_glo30, fetch_usgs_3dep

    bb = BBox(west=-81.2, south=29.0, east=-81.1, north=29.1)
    p = fetch_usgs_3dep(bb, tmp_path / "3dep.tif")
    d1 = GeoTiffDEM([p]).elevation([29.05], [-81.15]).elevation_m[0]
    ps = fetch_copernicus_glo30(bb, tmp_path / "cop")
    d2 = GeoTiffDEM(ps).elevation([29.05], [-81.15]).elevation_m[0]
    assert abs(d1 - d2) < 5.0 and 0 < d1 < 60


def _real_tile(path: Path, bbox: BBox, elevation_m: float) -> Path:
    """A plain (untagged, i.e. not SYNTHETIC) flat GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds

    w = h = 40
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(bbox.west, bbox.south, bbox.east, bbox.north, w, h),
    ) as ds:
        ds.write(np.full((h, w), elevation_m, np.float32), 1)
    return path


def _elev(dem) -> float:
    return float(dem.elevation([40.0], [-75.0]).elevation_m[0])


def test_real_searches_never_use_synthetic_tiles(tmp_path: Path):
    """Regression: every synthetic scene writes a tile over the same box and the mosaic used
    whichever sorted first, also for a real recording searched there."""
    make_synthetic_dem(BB, tmp_path / "a_synthetic.tif", seed=1)
    with pytest.raises(DEMError, match="SYNTHETIC tile"):
        open_dem_for_bbox(tmp_path, BB, offline=True)
    _real_tile(tmp_path / "z_real.tif", BB, 55.0)
    dem = open_dem_for_bbox(tmp_path, BB, offline=True)
    assert not dem.synthetic and _elev(dem) == pytest.approx(55.0)


def test_a_synthetic_search_uses_exactly_its_own_tile(tmp_path: Path):
    a = make_synthetic_dem(BB, tmp_path / "synthetic_a.tif", seed=1)
    b = make_synthetic_dem(BB, tmp_path / "synthetic_b.tif", seed=2)
    want_b = _elev(GeoTiffDEM([b]))
    assert abs(_elev(GeoTiffDEM([a])) - want_b) > 0.5  # the two scenes' terrains differ here
    dem = open_dem_for_bbox(tmp_path, BB, offline=True, synthetic_tile="synthetic_b.tif")
    assert dem.synthetic and _elev(dem) == pytest.approx(want_b)
    # without a declaration two overlapping synthetic tiles are an error, not a guess
    with pytest.raises(DEMError, match="2 SYNTHETIC DEM tiles overlap"):
        open_dem_for_bbox(tmp_path, BB, offline=True, allow_synthetic=True)
    a.unlink()
    only_b = open_dem_for_bbox(tmp_path, BB, offline=True, allow_synthetic=True)
    assert _elev(only_b) == pytest.approx(want_b)


def test_a_declared_synthetic_tile_must_exist_be_synthetic_and_cover(tmp_path: Path):
    with pytest.raises(DEMError, match="missing"):
        open_dem_for_bbox(tmp_path, BB, offline=True, synthetic_tile="nope.tif")
    _real_tile(tmp_path / "real.tif", BB, 10.0)
    with pytest.raises(DEMError, match="not tagged SYNTHETIC"):
        open_dem_for_bbox(tmp_path, BB, offline=True, synthetic_tile="real.tif")
    small = BBox(west=-75.1, south=39.95, east=-75.0, north=40.0)
    make_synthetic_dem(small, tmp_path / "small.tif")
    with pytest.raises(DEMError, match="does not cover"):
        open_dem_for_bbox(tmp_path, BB, offline=True, synthetic_tile="small.tif")
