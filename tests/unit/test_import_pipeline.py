"""Bounded-memory import pipeline: streaming parse, bucketed segmentation, atomic writes,
memory budget, archive availability and cache verification (offline, fixture data only)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from aerodoppler.adsb import adsblol, ingest
from aerodoppler.adsb.adsblol import AdsbLolProvider, ProviderUnavailable
from aerodoppler.adsb.index import DatasetWriter, list_datasets
from aerodoppler.adsb.ingest import ImportRequest, MemoryBudgetExceeded, import_flights
from aerodoppler.adsb.providers import ArchiveAsset, ArchiveLocation, LocalTraceProvider
from aerodoppler.models import BBox

TRACES = Path(__file__).parents[1] / "fixtures" / "adsblol" / "2026-11-02" / "traces"
WORLD = BBox(west=-180, south=-90, east=180, north=90)


def _import(tmp: Path, **kw) -> dict:
    req = ImportRequest(date_utc="2026-11-02", bbox=WORLD, confirm_download=True, **kw)
    ds = import_flights(LocalTraceProvider(TRACES), req, tmp / "raw", tmp / "norm", progress=False)
    return ds.manifest()


def test_bucketed_segmentation_matches_single_pass(tmp_path, monkeypatch):
    one = _import(tmp_path / "a", workers=1)
    monkeypatch.setattr(ingest, "ROWS_PER_BUCKET", 500)  # force several buckets
    many = _import(tmp_path / "b", workers=2, memory_budget_mb=8000)
    assert many["resources"]["segmentation_buckets"] > 1
    assert one["summary"] == many["summary"]
    assert one["filter_counts"] == many["filter_counts"]
    sa = pl.read_parquet(tmp_path / "a" / "norm" / one["dataset_id"] / "segments.parquet")
    sb = pl.read_parquet(tmp_path / "b" / "norm" / many["dataset_id"] / "segments.parquet")
    assert sorted(sa["segment_id"].to_list()) == sorted(sb["segment_id"].to_list())
    # staging is cleaned up and no partial directory is left behind
    assert not list((tmp_path / "b" / "norm").glob(".staging_*"))
    assert not list((tmp_path / "b" / "norm").glob("*.partial"))


def test_progress_and_resources_are_reported(tmp_path):
    seen: list[str] = []
    req = ImportRequest(date_utc="2026-11-02", bbox=WORLD, confirm_download=True, workers=1)
    ds = import_flights(
        LocalTraceProvider(TRACES),
        req,
        tmp_path / "raw",
        tmp_path / "norm",
        progress=False,
        on_progress=lambda stage, info: seen.append(stage),
    )
    res = ds.manifest()["resources"]
    assert {"locate", "parse", "write"} <= set(seen)
    assert res["workers"] == 1 and res["peak_tree_rss_mb_parse"] > 0


def test_memory_budget_exceeded_aborts_cleanly(tmp_path):
    with pytest.raises(MemoryBudgetExceeded):
        _import(tmp_path, workers=1, memory_budget_mb=1.0)
    norm = tmp_path / "norm"
    assert not list(norm.glob("*.partial")) and not list(norm.glob(".staging_*"))
    assert list_datasets(norm) == []


def test_cancel_during_parse(tmp_path):
    req = ImportRequest(date_utc="2026-11-02", bbox=WORLD, confirm_download=True, workers=1)
    with pytest.raises(ingest.ImportCancelled):
        import_flights(
            LocalTraceProvider(TRACES),
            req,
            tmp_path / "raw",
            tmp_path / "norm",
            progress=False,
            should_cancel=lambda: True,
        )
    assert list_datasets(tmp_path / "norm") == []


def test_dataset_writer_is_atomic(tmp_path):
    root = tmp_path / "norm" / "ds1"
    w = DatasetWriter(root, "ds1")
    assert (tmp_path / "norm" / "ds1.partial").exists()
    assert list_datasets(tmp_path / "norm") == []  # a partial dataset is never listed
    w.abort()
    assert not (tmp_path / "norm" / "ds1.partial").exists() and not root.exists()


def test_exclude_jets_drops_jets_only(tmp_path):
    all_ = _import(tmp_path / "a", workers=1)
    props = _import(tmp_path / "b", workers=1, exclude_jets=True)
    assert all_["dataset_id"] != props["dataset_id"]
    assert "jet" not in props["summary"]["aircraft_classes"]
    for k, v in props["summary"]["aircraft_classes"].items():
        assert all_["summary"]["aircraft_classes"].get(k) == v


# ---- archive availability ------------------------------------------------------------------

NOW = dt.datetime(2026, 11, 5, 2, 0, tzinfo=dt.UTC)


@pytest.mark.parametrize(
    "day,kind",
    [
        ("2026-11-05", "future_date"),  # today: the day has not ended
        ("2026-11-06", "future_date"),
        ("2026-11-04", "not_yet_published"),  # ended 2 h ago; published ~3.5 h after the end
        ("2026-10-01", "not_found"),
        ("2022-06-01", "before_archive_start"),
        ("1977-07-04", "before_archive_start"),
    ],
)
def test_missing_day_classification(day, kind):
    assert adsblol.availability_kind_for_missing(day, NOW) == kind


def test_offline_without_cache_is_explicit(tmp_path):
    with pytest.raises(ProviderUnavailable) as e:
        AdsbLolProvider(offline=True).locate("2026-11-02", dest_dir=tmp_path)
    assert e.value.kind == "offline_no_cache"


def test_future_day_fails_before_any_network(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("must not call the network")

    monkeypatch.setattr(adsblol.httpx, "get", no_network)
    with pytest.raises(ProviderUnavailable) as e:
        AdsbLolProvider().locate("2999-01-01", dest_dir=tmp_path)
    assert e.value.kind == "future_date"
    assert "has not ended" in str(e.value)


def test_local_cache_ignores_partial_corrupt_and_sidecars(tmp_path):
    d = tmp_path / "2026-11-02"
    d.mkdir()
    for name in (
        "x.tar.aa",
        "x.tar.ab",
        "x.tar.ac.part",
        "x.tar.aa.verified.json",
        "x.tar.ab.corrupt",
        "x.tar",
    ):
        (d / name).write_bytes(b"0")
    names = [a.name for a in AdsbLolProvider._local_assets(tmp_path, "2026-11-02")]
    assert names == ["x.tar", "x.tar.aa", "x.tar.ab"]


def test_sha256_is_cached_by_size_and_mtime(tmp_path, monkeypatch):
    f = tmp_path / "part.tar.aa"
    f.write_bytes(b"abc")
    calls = []
    real = adsblol.sha256_file
    monkeypatch.setattr(adsblol, "sha256_file", lambda p: calls.append(p) or real(p))
    h1 = adsblol._cached_sha256(f)
    h2 = adsblol._cached_sha256(f)
    assert h1 == h2 and len(calls) == 1
    f.write_bytes(b"abcd")  # size changes: re-hash
    adsblol._cached_sha256(f)
    assert len(calls) == 2


def _location(tmp: Path, payload: bytes, digest: str | None) -> ArchiveLocation:
    return ArchiveLocation(
        "adsblol",
        "2026-11-02",
        [
            ArchiveAsset(
                "x.tar", "https://example.invalid/x.tar", len(payload), expected_sha256=digest
            )
        ],
        "ODbL-1.0",
        "test",
    )


def test_corrupt_download_is_kept_and_refetched(tmp_path, monkeypatch):
    import hashlib

    good = b"good archive bytes"
    bad = b"bad! archive bytes"
    assert len(good) == len(bad)
    served = [bad, good]

    def fake_download(
        self, url, target, expected, progress, max_retries, on_bytes=None, should_cancel=None
    ):
        target.write_bytes(served.pop(0))

    monkeypatch.setattr(AdsbLolProvider, "_download_resumable", fake_download)
    loc = _location(tmp_path, good, hashlib.sha256(good).hexdigest())
    out = AdsbLolProvider().fetch(loc, tmp_path, progress=False)
    d = tmp_path / "2026-11-02"
    assert (d / "x.tar").read_bytes() == good
    assert (d / "x.tar.corrupt").read_bytes() == bad  # bad copy kept, not deleted
    assert out.assets[0].sha256 == hashlib.sha256(good).hexdigest()


def test_insufficient_disk_is_reported_before_download(tmp_path, monkeypatch):
    import aerodoppler.resources as res

    monkeypatch.setattr(res, "disk_free_bytes", lambda p: 1_000)
    with pytest.raises(ProviderUnavailable) as e:
        AdsbLolProvider().fetch(_location(tmp_path, b"x" * 10, None), tmp_path, progress=False)
    assert e.value.kind == "insufficient_disk"


def test_retrieval_metadata_written(tmp_path, monkeypatch):
    def fake_download(
        self, url, target, expected, progress, max_retries, on_bytes=None, should_cancel=None
    ):
        target.write_bytes(b"0123456789")

    monkeypatch.setattr(AdsbLolProvider, "_download_resumable", fake_download)
    AdsbLolProvider().fetch(_location(tmp_path, b"0123456789", None), tmp_path, progress=False)
    meta = json.loads((tmp_path / "2026-11-02" / "retrieval.json").read_text())
    assert meta[0]["sha256"] and meta[0]["name"] == "x.tar"
