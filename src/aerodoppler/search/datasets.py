"""Explicit flight-dataset selection for a search.

A search must never run silently across every imported archive: days that do not cover the
recording add only false candidates, and mixing SYNTHETIC with real archives would make a
result meaningless. Without an explicit list, the date-compatible archives of the same kind
(real for a real recording, synthetic for a synthetic one) are chosen; an explicit list is
checked for date and kind, and mismatches need an explicit override.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aerodoppler.adsb.index import FlightDataset, list_datasets
from aerodoppler.models import BBox


class DatasetSelectionError(ValueError):
    def __init__(
        self, message: str, code: str, field: str = "flight_dataset_ids", action: str = ""
    ):
        super().__init__(message)
        self.code = code
        self.field = field
        self.action = action


@dataclass
class DatasetInfo:
    dataset_id: str
    date_utc: str | None
    synthetic: bool
    bbox: tuple[float, float, float, float] | None
    coverage_bbox: tuple[float, float, float, float] | None
    manifest: dict[str, Any] = field(repr=False, default_factory=dict)


def describe(ds: FlightDataset) -> DatasetInfo:
    m = ds.manifest()
    req = m.get("request") or {}
    bb = req.get("search_bbox") or req.get("bbox")
    return DatasetInfo(
        dataset_id=ds.dataset_id,
        date_utc=req.get("date_utc") or m.get("date_utc"),
        synthetic=bool(m.get("synthetic", False)),
        bbox=tuple(req["bbox"]) if req.get("bbox") else None,  # type: ignore[arg-type]
        coverage_bbox=tuple(bb) if bb else None,  # type: ignore[arg-type]
        manifest=m,
    )


def search_dates(
    date_utc: str | None, start_utc: str | None, duration_s: float, uncertainty_s: float
) -> set[str]:
    """UTC days the recording can overlap: the date (Mode A, any time of day) or the days
    touched by [start - uncertainty, start + duration + uncertainty] (Mode B)."""
    if start_utc:
        t = dt.datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
        lo = t - dt.timedelta(seconds=uncertainty_s)
        hi = t + dt.timedelta(seconds=duration_s + uncertainty_s)
        days = set()
        d = lo.date()
        while d <= hi.date():
            days.add(d.isoformat())
            d += dt.timedelta(days=1)
        return days
    if date_utc:
        return {date_utc}
    return set()


def _overlaps(a: tuple[float, float, float, float], b: BBox) -> bool:
    w, s, e, n = a
    return not (e < b.west or w > b.east or n < b.south or s > b.north)


def resolve_datasets(
    normalized_dir: Path,
    *,
    dates: set[str],
    recording_synthetic: bool,
    requested: list[str],
    bbox: BBox | None,
    allow_date_mismatch: bool = False,
    allow_synthetic_mix: bool = False,
) -> tuple[list[str], list[str]]:
    """Return ``(dataset_ids, warnings)`` or raise :class:`DatasetSelectionError`."""
    available = {d.dataset_id: describe(d) for d in list_datasets(normalized_dir) if d.exists()}
    kind = "SYNTHETIC" if recording_synthetic else "real"
    warnings: list[str] = []
    if requested:
        chosen = []
        for did in requested:
            info = available.get(did)
            if info is None:
                raise DatasetSelectionError(
                    f"flight dataset {did} is not imported",
                    "dataset_not_found",
                    action="pick one from the list of imported datasets or import that day",
                )
            if info.synthetic != recording_synthetic and not allow_synthetic_mix:
                raise DatasetSelectionError(
                    f"{did} is {'SYNTHETIC' if info.synthetic else 'real'} flight data but the "
                    f"recording is {kind}; mixing them makes the result meaningless",
                    "synthetic_mix",
                    action="deselect it (an explicit development override exists: allow_synthetic_mix)",
                )
            if dates and info.date_utc not in dates and not allow_date_mismatch:
                raise DatasetSelectionError(
                    f"{did} covers {info.date_utc}, but the recording can only be on "
                    f"{', '.join(sorted(dates))} (UTC)",
                    "date_mismatch",
                    action="select the archive of the recording's UTC date, or correct the date",
                )
            chosen.append(info)
    else:
        chosen = [
            i
            for i in available.values()
            if i.synthetic == recording_synthetic and (not dates or i.date_utc in dates)
        ]
        if not chosen:
            have = sorted({i.date_utc for i in available.values() if i.date_utc})
            raise DatasetSelectionError(
                f"no imported {kind} flight archive covers {', '.join(sorted(dates)) or 'the recording date'} (UTC)"
                + (f"; imported days: {', '.join(have)}" if have else "; nothing is imported yet"),
                "no_dataset_for_date",
                action="import the ADS-B archive for that UTC day first (Import page → Flight archive)",
            )
    if bbox is not None:
        outside = [
            i.dataset_id for i in chosen if i.coverage_bbox and not _overlaps(i.coverage_bbox, bbox)
        ]
        if outside and len(outside) == len(chosen):
            raise DatasetSelectionError(
                "the search area lies outside every selected flight dataset's imported area "
                f"({', '.join(outside)}): no aircraft could match",
                "bbox_outside_data",
                field="bbox",
                action="import flights for this area, or move the search area into the imported one",
            )
        for did in outside:
            warnings.append(f"{did} does not overlap the search area and contributes nothing")
        partial = [
            i.dataset_id
            for i in chosen
            if i.coverage_bbox
            and _overlaps(i.coverage_bbox, bbox)
            and not (
                i.coverage_bbox[0] <= bbox.west
                and i.coverage_bbox[1] <= bbox.south
                and i.coverage_bbox[2] >= bbox.east
                and i.coverage_bbox[3] >= bbox.north
            )
        ]
        for did in partial:
            warnings.append(
                f"the search area extends beyond {did}'s imported area: aircraft outside it are "
                "missing, so locations near the edge are less reliable"
            )
    return [i.dataset_id for i in chosen], warnings
