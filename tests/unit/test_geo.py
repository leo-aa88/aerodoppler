import numpy as np
import pytest

from aerodoppler.geo.altitude import get_geoid, to_ellipsoidal_height, to_metres
from aerodoppler.geo.coordinates import LocalFrame, ecef_to_geodetic, geodetic_to_ecef, haversine_m
from aerodoppler.models import AltitudeDatum, AltitudeUnit, BBox


def test_wgs84_ecef_round_trip():
    lat = np.array([0.0, 45.0, -33.9, 71.2])
    lon = np.array([0.0, -75.0, 151.2, -156.8])
    h = np.array([0.0, 1000.0, 50.0, 12000.0])
    xyz = geodetic_to_ecef(lat, lon, h)
    assert np.allclose(np.linalg.norm(xyz[0]), 6378137.0)
    la, lo, hh = ecef_to_geodetic(xyz)
    assert (
        np.allclose(la, lat, atol=1e-9)
        and np.allclose(lo, lon, atol=1e-9)
        and np.allclose(hh, h, atol=1e-6)
    )


def test_axis_order_is_explicit():
    # lat=10, lon=20 must not be confused with lat=20, lon=10
    a = geodetic_to_ecef(10.0, 20.0, 0.0)
    b = geodetic_to_ecef(20.0, 10.0, 0.0)
    assert np.linalg.norm(a - b) > 1e6
    assert a[2] == pytest.approx(6378137.0 * np.sin(np.radians(10.0)) * (1 - 0.00669438), rel=1e-2)


def test_local_frame_round_trip_and_scale():
    f = LocalFrame(40.0, -75.0, 100.0)
    enu = np.array([[1000.0, 2000.0, 50.0], [-3000.0, 500.0, -10.0]])
    back = f.ecef_to_enu(f.enu_to_ecef(enu))
    assert np.allclose(back, enu, atol=1e-6)
    lat, lon = f.east_north_grid_to_geodetic(np.array([1000.0]), np.array([0.0]))
    d = haversine_m(40.0, -75.0, lat, lon)
    assert d[0] == pytest.approx(1000.0, abs=4.0)  # haversine uses a sphere
    _, _, h = f.enu_to_geodetic(np.array([[1000.0, 0.0, 0.0]]))
    assert h[0] == pytest.approx(100.0 + 1000.0**2 / (2 * 6.37e6), abs=0.05)


def test_feet_to_metres_and_datums():
    assert to_metres(1000.0, AltitudeUnit.FEET) == pytest.approx(304.8)
    assert to_metres(5.0, "m") == 5.0
    with pytest.raises(ValueError):
        to_metres(1.0, "cubits")
    g = get_geoid()
    # Published EGM96 undulation checks (NGA): ~17.2 m at (0,0); strongly negative south of India.
    assert g.undulation_m(0.0, 0.0) == pytest.approx(17.16, abs=0.5)
    assert g.undulation_m(0.0, 78.0) < -80.0
    assert -40.0 < g.undulation_m(40.0, -75.0) < -25.0
    hae = to_ellipsoidal_height(1000.0, AltitudeDatum.ORTHOMETRIC, 40.0, -75.0)
    assert hae.h_m[()] == pytest.approx(1000.0 + g.undulation_m(40.0, -75.0), abs=1e-6)
    geom = to_ellipsoidal_height(1000.0, AltitudeDatum.GEOMETRIC_HAE, 40.0, -75.0)
    assert geom.h_m[()] == 1000.0 and geom.sigma_m[()] == 0.0
    baro = to_ellipsoidal_height(1000.0, AltitudeDatum.BAROMETRIC_STD, 40.0, -75.0)
    assert baro.sigma_m[()] > 50.0
    unk = to_ellipsoidal_height(1000.0, "unknown", 40.0, -75.0)
    assert unk.sigma_m[()] > baro.sigma_m[()]
    with pytest.raises(ValueError):
        to_ellipsoidal_height(0.0, AltitudeDatum.GROUND, 40.0, -75.0)


def test_bbox_validation_and_margin():
    b = BBox.parse("-75.5,39.8,-74.5,40.2")
    assert b.contains(40.0, -75.0) and not b.contains(41.0, -75.0)
    e = b.expanded_km(15.0)
    assert e.north - b.north == pytest.approx(15.0 / 111.195, rel=1e-3)
    with pytest.raises(ValueError):
        BBox.parse("-74.5,39.8,-75.5,40.2")
    assert 7000 < b.approx_area_km2() < 9000 or 3000 < b.approx_area_km2() < 5000
