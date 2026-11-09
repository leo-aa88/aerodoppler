"""Blind search building blocks: the support join, clustering and the API entry (offline,
flight data generated in the test)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from aerodoppler.adsb.ingest import import_samples_dataframe
from aerodoppler.adsb.normalize import SAMPLE_SCHEMA
from aerodoppler.config import Settings
from aerodoppler.models import BBox
from aerodoppler.search.blind import (
    KM_PER_DEG,
    BlindConfig,
    Support,
    _around,
    cluster_cells,
    region_coverage,
    support_unknown_time,
)

DAY0 = 1793664000.0  # 2026-11-03T00:00:00Z
P = (28.50, -80.80)  # where the two true passes cross


def _track(icao: str, lat0: float, lon0: float, heading_deg: float, t_cpa: float) -> pl.DataFrame:
    """A straight 50 m/s piston track passing (lat0, lon0) at t_cpa, ±10 min, every 5 s."""
    t = np.arange(t_cpa - 600.0, t_cpa + 600.0, 5.0)
    d_km = 0.05 * (t - t_cpa)
    h = math.radians(heading_deg)
    lat = lat0 + d_km * math.cos(h) / KM_PER_DEG
    lon = lon0 + d_km * math.sin(h) / (KM_PER_DEG * math.cos(math.radians(lat0)))
    n = t.size
    return pl.DataFrame(
        {
            "source": ["t"] * n,
            "icao24": [icao] * n,
            "t": t,
            "lat": lat,
            "lon": lon,
            "alt_baro_ft": [1500.0] * n,
            "alt_geom_ft": [1500.0] * n,
            "altitude_value_ft": [1500.0] * n,
            "altitude_datum": ["geometric_wgs84_hae"] * n,
            "on_ground": [False] * n,
            "gs_mps": [50.0] * n,
            "track_deg": [heading_deg] * n,
            "vertical_rate_mps": [0.0] * n,
            "flags": [None] * n,
            "stale": [False] * n,
            "new_leg": [False] * n,
            "position_source": [None] * n,
            "callsign": [None] * n,
            "aircraft_type": ["C172"] * n,
            "aircraft_desc": [None] * n,
            "registration": [None] * n,
            "db_flags": [None] * n,
            "category": [None] * n,
        },
        schema_overrides={
            "flags": pl.Utf8,
            "position_source": pl.Utf8,
            "callsign": pl.Utf8,
            "aircraft_desc": pl.Utf8,
            "registration": pl.Utf8,
            "db_flags": pl.Int64,
            "category": pl.Utf8,
        },
    )


@pytest.fixture()
def flights(tmp_path):
    s = Settings(data_dir=tmp_path / "data", offline=True)
    s.ensure_dirs()
    tracks = [
        _track("aaa001", *P, 0.0, DAY0 + 36000.0),  # northbound over P
        _track("aaa002", P[0] + 0.01, P[1], 90.0, DAY0 + 36300.0),  # eastbound 300 s later
        # decoys 150 km south: the same 300 s separation, but 40 km apart (no single listener)
        _track("bbb001", 27.10, -80.80, 0.0, DAY0 + 36000.0),
        _track("bbb002", 27.10, -80.39, 0.0, DAY0 + 36300.0),
        # a third aircraft over P 500 s after the first: only P explains a third event
        _track("aaa003", P[0] - 0.01, P[1], 270.0, DAY0 + 36500.0),
        # a valid decoy pair (same geometry as at P) 220 km south, with no third aircraft
        _track("ccc001", 26.50, -80.80, 0.0, DAY0 + 36000.0),
        _track("ccc002", 26.51, -80.80, 90.0, DAY0 + 36300.0),
    ]
    df = pl.concat(tracks)
    df = df.cast({k: v for k, v in SAMPLE_SCHEMA.items() if k in df.columns})
    ds = import_samples_dataframe(
        df,
        "real_2026-11-03",
        s.normalized_dir,
        notes=[],
        synthetic=False,
        extra_manifest={"request": {"date_utc": "2026-11-03", "bbox": [-82.0, 26.0, -79.0, 30.0]}},
    )
    return s, ds


def _feat(t: float, slope: float = 0.0) -> SimpleNamespace:
    """Event stand-in; with f_c = 100 Hz a 50 m/s flight's reach is 3 f v²/(c slope) = 2.19/slope km."""
    return SimpleNamespace(
        t_center_s=t, v_min_mps=30.0, v_max_mps=70.0, f_center_hz=100.0, slope_max_hz_per_s=slope
    )


def _cell_of(sup: Support, lat: float, lon: float) -> tuple[int, int]:
    return int(math.floor(lat / sup.cell_deg_lat)), int(math.floor(lon / sup.cell_deg_lon))


def test_support_marks_only_where_two_passes_fit(flights):
    _s, ds = flights
    sup = support_unknown_time([ds], DAY0, [_feat(100.0), _feat(400.0)], 20.0, BlindConfig())
    assert sup.cells, "the true crossing must be supported"
    lat_c = [(i + 0.5) * sup.cell_deg_lat for i, _ in sup.cells]
    assert not [la for la in lat_c if 26.9 < la < 27.3], "40 km apart: no listener hears both"
    near = {
        _cell_of(sup, P[0] + di * sup.cell_deg_lat, P[1] + dj * sup.cell_deg_lon)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
    }
    assert near & set(sup.cells)
    assert sup.scope == BBox(west=-82.0, south=26.0, east=-79.0, north=30.0)


def test_a_third_event_ranks_the_region_that_explains_it_first(flights):
    _s, ds = flights
    feats = [_feat(100.0), _feat(400.0), _feat(600.0)]
    sup = support_unknown_time([ds], DAY0, feats, 20.0, BlindConfig())
    tiles = cluster_cells(sup, BlindConfig())
    assert tiles[0]["events_explained"] == 3
    b = tiles[0]["bbox"]
    assert b.south <= P[0] <= b.north and b.west <= P[1] <= b.east
    decoy = [t for t in tiles if t["bbox"].north < 27.0]
    assert decoy and all(t["events_explained"] == 2 for t in decoy)


def test_support_is_empty_for_a_separation_no_pair_has(flights):
    _s, ds = flights
    sup = support_unknown_time([ds], DAY0, [_feat(100.0), _feat(2100.0)], 20.0, BlindConfig())
    assert not sup.cells


def test_support_uses_each_events_slant_bound(flights):
    """Two passes 1.1 km apart fit ridges implying ~3 km reach; ~150 m reach excludes them, and
    the order in which the events are given does not matter."""
    _s, ds = flights
    near = support_unknown_time(
        [ds], DAY0, [_feat(400.0, 0.73), _feat(100.0, 0.73)], 20.0, BlindConfig()
    )
    assert near.cells
    far = support_unknown_time(
        [ds], DAY0, [_feat(100.0, 14.6), _feat(400.0, 14.6)], 20.0, BlindConfig()
    )
    assert not far.cells


def test_clusters_are_tiled_clipped_and_ordered_by_events_explained():
    cfg = BlindConfig(cell_km=10.0, cluster_max_km=30.0)
    lat_deg = 10.0 / KM_PER_DEG
    # a 7x1 strip of cells (70 km) and one isolated cell with more support
    cells = {(280 + i, -800): 1 for i in range(7)}
    cells[(300, -790)] = 50
    sup = Support(
        cells=cells,
        best={k: (2, v) for k, v in cells.items()},
        cell_deg_lat=lat_deg,
        cell_deg_lon=lat_deg,
        n_flights=0,
        n_positions=0,
        n_pairs=0,
        scope=BBox(west=-90.0, south=25.0, east=-70.0, north=31.4),
        seconds=0.0,
    )
    cells_best = {k: (2, v) for k, v in cells.items()}
    cells_best[(300, -790)] = (3, 50)  # explains a third event: first
    sup.best = cells_best
    tiles = cluster_cells(sup, cfg)
    assert tiles[0]["support"] == 50 and tiles[0]["events_explained"] == 3
    strip = tiles[1:]
    assert sum(t["cells"] for t in strip) == 7
    assert all(t["cells"] <= 3 for t in strip), "tiles are cut at cluster_max_km"
    assert all(t["bbox"].north <= 31.4 for t in tiles), "never outside the imported data"


def test_refinement_area_stays_inside_the_tile():
    tile = BBox(west=-81.0, south=28.0, east=-80.5, north=28.6)
    a = _around(tile, {"lat": 28.58, "lon": -80.9}, 10.0)
    assert a.north == 28.6 and a.south == pytest.approx(28.58 - 10.0 / KM_PER_DEG)
    assert tile.west <= a.west < a.east <= tile.east


def test_blind_search_api_validates_before_queueing(flights):
    from aerodoppler.api.app import create_app

    s, _ds = flights
    with TestClient(create_app(s, web_dir=s.data_dir / "no-web")) as c:
        r = c.post(
            "/api/searches/blind",
            json={"mode": "unknown-time", "recording_id": "rec_x", "event_ids": ["evt_a"]},
        )
        assert r.status_code == 422
        r = c.post(
            "/api/searches/blind",
            json={
                "mode": "unknown-time",
                "recording_id": "rec_missing",
                "event_ids": ["evt_a", "evt_b"],
                "date_utc": "2026-11-03",
            },
        )
        assert r.status_code == 404


def test_failed_regions_count_as_not_searched():
    searched = [
        {"k": 0, "best_score": 4.2},
        {"k": 1, "error": "DEM unavailable: no DEM available for bbox (…)"},
        {"k": 2, "error": "DEM unavailable: no DEM available for bbox (…)"},
    ]
    cov, failed, kinds = region_coverage(5, searched, "budget", 1234.4)
    assert cov["searched_clusters"] == 1 and cov["failed_clusters"] == 2
    assert cov["not_reached_clusters"] == 2 and cov["stopped_by"] == "budget"
    assert kinds == {"DEM unavailable": 2} and len(failed) == 2
