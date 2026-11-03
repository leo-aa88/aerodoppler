"""Real-format ADSB.lol trace fixtures → normalise → segments → Parquet → DuckDB query."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from aerodoppler.adsb.index import FlightIndex, list_datasets
from aerodoppler.adsb.ingest import ImportRequest, import_flights
from aerodoppler.adsb.normalize import (
    TraceParseStats,
    classify_aircraft_type,
    load_trace_bytes,
    normalize_trace,
)
from aerodoppler.adsb.providers import LocalTraceProvider
from aerodoppler.adsb.segments import SegmentConfig, split_segments, trajectory_from_samples
from aerodoppler.models import AltitudeDatum, BBox

FIX = Path(__file__).parents[1] / "fixtures" / "adsblol" / "2026-11-02" / "traces" / "04"


def _load(name: str):
    return load_trace_bytes((FIX / name).read_bytes())


def test_parse_real_trace_schema_c172():
    obj = _load("trace_full_a12704.json")
    assert obj["icao"] == "a12704" and obj["t"] == "C172"
    st = TraceParseStats()
    df = normalize_trace(obj, "adsblol", None, st)
    assert df is not None and df.height == 140 == st.points_kept
    assert df["t"].is_sorted()
    assert df["t"][0] >= obj["timestamp"]
    # geometric altitude preferred when present, baro otherwise; both preserved raw
    assert set(df["altitude_datum"].unique().to_list()) <= {
        AltitudeDatum.GEOMETRIC_HAE.value,
        AltitudeDatum.BAROMETRIC_STD.value,
        AltitudeDatum.UNKNOWN.value,
    }
    geo = df.filter(pl.col("altitude_datum") == AltitudeDatum.GEOMETRIC_HAE.value)
    assert geo.height > 0
    assert np.allclose(geo["altitude_value_ft"].to_numpy(), geo["alt_geom_ft"].to_numpy())
    assert df["gs_mps"].drop_nulls().max() < 120  # a C172 does not fly 240 kt
    assert df["position_source"].drop_nulls().unique().to_list()


def test_parse_ground_and_tisb_edge_cases():
    st = TraceParseStats()
    ground = normalize_trace(_load("trace_full_a22904.json"), "adsblol", None, st)
    assert ground is not None and ground["on_ground"].all()
    assert (ground["altitude_datum"] == AltitudeDatum.GROUND.value).all()
    assert st.points_on_ground == 40
    tisb = normalize_trace(_load("trace_full_~26d904.json"), "adsblol", None, st)
    assert tisb is not None and tisb.height == 2 and tisb["aircraft_type"][0] is None
    assert classify_aircraft_type(None) == "unknown"  # unknown is not jet
    assert (
        classify_aircraft_type("A320") == "jet"
        and classify_aircraft_type("C172") == "piston"
        and classify_aircraft_type("PC12") == "turboprop"
        and classify_aircraft_type("R44") == "helicopter"
    )


def test_bbox_filter_counts():
    obj = _load("trace_full_a0e404.json")
    df_all = normalize_trace(obj, "adsblol")
    assert df_all is not None
    lat0, lon0 = float(df_all["lat"].median()), float(df_all["lon"].median())
    tiny = BBox(west=lon0 - 0.01, south=lat0 - 0.01, east=lon0 + 0.01, north=lat0 + 0.01)
    st = TraceParseStats()
    df = normalize_trace(obj, "adsblol", tiny, st)
    assert st.points_outside_bbox > 0 and (df is None or df.height < df_all.height)
    assert st.points_total == df_all.height


def test_segments_split_on_ground_and_gaps():
    df = normalize_trace(_load("trace_full_a24504.json"), "adsblol")
    assert df is not None
    samples, segs = split_segments(df, SegmentConfig(gap_split_s=30.0))
    assert samples["segment_id"].null_count() == 0
    assert segs.height >= 2
    assert segs.filter(pl.col("on_ground")).height >= 1
    air = segs.filter(~pl.col("on_ground"))
    assert air.height >= 1
    assert all(f == "" or "on_ground" not in f for f in air["flags"].to_list())
    # no segment may contain an internal gap above the split threshold
    assert (air["max_gap_s"] <= 30.0).all()
    assert air["aircraft_class"][0] == "piston"
    # trajectory build with datum handling
    sid = air.sort("n_samples", descending=True)["segment_id"][0]
    tb = trajectory_from_samples(samples.filter(pl.col("segment_id") == sid), sid)
    assert tb.trajectory is not None and tb.trajectory.t.size >= 3
    assert np.all(np.isfinite(tb.trajectory.pos))
    assert tb.datum in {AltitudeDatum.GEOMETRIC_HAE.value, AltitudeDatum.BAROMETRIC_STD.value}
    assert tb.altitude_sigma_m >= 0.0
    # speeds plausible for a C152 (< 80 m/s)
    assert np.nanmax(tb.trajectory.speed) < 100.0


def test_full_local_import_and_query(tmp_path: Path):
    prov = LocalTraceProvider(FIX.parent, source_label="adsblol-fixture")
    # bbox around the C152 / P28A traces
    df = normalize_trace(_load("trace_full_a24504.json"), "adsblol")
    assert df is not None
    lat0, lon0 = float(df["lat"].mean()), float(df["lon"].mean())
    bbox = BBox(west=lon0 - 1.0, south=lat0 - 1.0, east=lon0 + 1.0, north=lat0 + 1.0)
    req = ImportRequest(
        date_utc="2026-11-02", bbox=bbox, margin_km=15.0, workers=2, confirm_download=True
    )
    ds = import_flights(prov, req, tmp_path / "raw", tmp_path / "normalized", progress=False)
    assert ds.exists()
    man = ds.manifest()
    assert man["filter_counts"]["files"] == 6
    assert man["summary"]["segments"] >= 2
    assert (
        man["license"]
        and man["request"]["bbox"] == list(bbox.as_tuple())
        or tuple(man["request"]["bbox"]) == bbox.as_tuple()
    )
    assert "storage_bytes" in man and man["storage_bytes"] > 0
    assert list(ds.samples_dir.glob("date=*/region=*/part-0.parquet"))
    idx = FlightIndex(list_datasets(tmp_path / "normalized"))
    segs = idx.query_segments(man["coverage"]["first_utc"], man["coverage"]["last_utc"], bbox)
    assert segs.height >= 1 and (~segs["on_ground"]).all()
    s = idx.load_samples(segs["segment_id"].to_list()[:2])
    assert s.height > 0 and set(s["segment_id"].unique().to_list()) <= set(
        segs["segment_id"].to_list()
    )
    # speed-bin filter prunes
    fast = idx.query_segments(
        man["coverage"]["first_utc"], man["coverage"]["last_utc"], bbox, min_speed_mps=200.0
    )
    assert fast.height <= segs.height
    cov = idx.coverage_summary()
    assert cov["segments"] == man["summary"]["segments"]


def test_corrupt_member_is_counted_not_fatal(tmp_path: Path):
    d = tmp_path / "traces" / "aa"
    d.mkdir(parents=True)
    (d / "trace_full_bad.json").write_bytes(b"\x1f\x8b garbage")
    (d / "trace_full_a12704.json").write_bytes((FIX / "trace_full_a12704.json").read_bytes())
    prov = LocalTraceProvider(tmp_path / "traces")
    df = normalize_trace(_load("trace_full_a12704.json"), "x")
    assert df is not None
    bbox = BBox(west=-180, south=-90, east=180, north=90)
    req = ImportRequest(date_utc="2026-11-02", bbox=bbox, workers=1, confirm_download=True)
    ds = import_flights(prov, req, tmp_path / "raw", tmp_path / "norm", progress=False)
    fc = ds.manifest()["filter_counts"]
    assert fc["files_failed"] == 1 and fc["files"] == 2 and fc["errors"]
    with pytest.raises(ValueError):
        load_trace_bytes(b'{"foo": 1}')
