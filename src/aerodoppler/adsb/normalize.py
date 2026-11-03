"""readsb trace normalisation (SPEC §6.2).

Verified against the readsb documentation (README-json.md, "trace jsons") and the live
ADSB.lol archive of 2026-11-02. Each ``trace`` entry is::

    [seconds_after_timestamp, lat, lon, alt_baro_ft | "ground" | null, gs_knots | null,
     track_deg | null, flags, vertical_rate_fpm | null, aircraft_obj | null,
     position_type | null, alt_geom_ft | null, geom_rate_fpm | null, ias | null, roll | null]

flags bitfield: 1 = stale position, 2 = start of new leg, 4 = vertical rate geometric,
8 = altitude (index 3) is geometric rather than barometric. Files generated before 2022 have
only the first 9 columns; the parser tolerates that.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import orjson
import polars as pl

from aerodoppler.models import AltitudeDatum, BBox

KNOTS_TO_MPS = 0.514444
FPM_TO_MPS = 0.3048 / 60.0

SAMPLE_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.Utf8,
    "icao24": pl.Utf8,
    "t": pl.Float64,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "alt_baro_ft": pl.Float64,
    "alt_geom_ft": pl.Float64,
    "altitude_value_ft": pl.Float64,
    "altitude_datum": pl.Utf8,
    "on_ground": pl.Boolean,
    "gs_mps": pl.Float64,
    "track_deg": pl.Float64,
    "vertical_rate_mps": pl.Float64,
    "flags": pl.Int32,
    "stale": pl.Boolean,
    "new_leg": pl.Boolean,
    "position_source": pl.Utf8,
    "callsign": pl.Utf8,
    "aircraft_type": pl.Utf8,
    "aircraft_desc": pl.Utf8,
    "registration": pl.Utf8,
    "db_flags": pl.Int32,
    "category": pl.Utf8,
}


@dataclass
class TraceParseStats:
    files: int = 0
    files_failed: int = 0
    aircraft_kept: int = 0
    points_total: int = 0
    points_kept: int = 0
    points_invalid_position: int = 0
    points_on_ground: int = 0
    points_missing_altitude: int = 0
    points_outside_bbox: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: TraceParseStats) -> None:
        for k in (
            "files",
            "files_failed",
            "aircraft_kept",
            "points_total",
            "points_kept",
            "points_invalid_position",
            "points_on_ground",
            "points_missing_altitude",
            "points_outside_bbox",
        ):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.errors.extend(other.errors[: max(0, 50 - len(self.errors))])

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["errors"] = d["errors"][:50]
        return d


def load_trace_bytes(data: bytes) -> dict[str, Any]:
    """Decode a (possibly gzip-compressed) trace file into a dict."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    obj = orjson.loads(data)
    if not isinstance(obj, dict) or "trace" not in obj or "timestamp" not in obj:
        raise ValueError("not a readsb trace object")
    return obj


def _f(v: Any) -> float:
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def normalize_trace(
    obj: dict[str, Any],
    source: str,
    bbox: BBox | None = None,
    stats: TraceParseStats | None = None,
) -> pl.DataFrame | None:
    """Convert one trace object into normalised rows; returns None when nothing is kept."""
    st = stats if stats is not None else TraceParseStats()
    st.files += 1
    icao = str(obj.get("icao", "")).lower()
    base = float(obj["timestamp"])
    trace = obj.get("trace") or []
    st.points_total += len(trace)
    if not trace or not icao:
        return None
    ac_type = obj.get("t")
    desc = obj.get("desc")
    reg = obj.get("r")
    db_flags = obj.get("dbFlags")
    n = len(trace)
    t = np.empty(n)
    lat = np.empty(n)
    lon = np.empty(n)
    alt_baro = np.full(n, np.nan)
    alt_geom = np.full(n, np.nan)
    on_ground = np.zeros(n, dtype=bool)
    gs = np.full(n, np.nan)
    trk = np.full(n, np.nan)
    vr = np.full(n, np.nan)
    flags = np.zeros(n, dtype=np.int32)
    pos_type: list[str | None] = [None] * n
    callsign: list[str | None] = [None] * n
    category: list[str | None] = [None] * n
    last_callsign: str | None = None
    last_category: str | None = None
    for i, p in enumerate(trace):
        t[i] = base + _f(p[0])
        lat[i] = _f(p[1])
        lon[i] = _f(p[2])
        a = p[3] if len(p) > 3 else None
        if a == "ground":
            on_ground[i] = True
        else:
            alt_baro[i] = _f(a)
        gs[i] = _f(p[4]) * KNOTS_TO_MPS if len(p) > 4 and p[4] is not None else np.nan
        trk[i] = _f(p[5]) if len(p) > 5 else np.nan
        fl = int(p[6]) if len(p) > 6 and p[6] is not None else 0
        flags[i] = fl
        vr[i] = _f(p[7]) * FPM_TO_MPS if len(p) > 7 and p[7] is not None else np.nan
        ac = p[8] if len(p) > 8 else None
        if isinstance(ac, dict):
            fl_name = ac.get("flight")
            if isinstance(fl_name, str) and fl_name.strip():
                last_callsign = fl_name.strip()
            cat = ac.get("category")
            if isinstance(cat, str):
                last_category = cat
        callsign[i] = last_callsign
        category[i] = last_category
        pos_type[i] = p[9] if len(p) > 9 and isinstance(p[9], str) else None
        if len(p) > 10 and p[10] is not None:
            alt_geom[i] = _f(p[10])
        if fl & 8 and not on_ground[i] and np.isnan(alt_geom[i]):
            # altitude column is geometric for this record
            alt_geom[i] = alt_baro[i]
            alt_baro[i] = np.nan
    # preferred altitude + datum
    datum = np.full(n, AltitudeDatum.UNKNOWN.value, dtype=object)
    alt_val = np.full(n, np.nan)
    has_geom = ~np.isnan(alt_geom)
    has_baro = ~np.isnan(alt_baro)
    alt_val[has_geom] = alt_geom[has_geom]
    datum[has_geom] = AltitudeDatum.GEOMETRIC_HAE.value
    only_baro = has_baro & ~has_geom
    alt_val[only_baro] = alt_baro[only_baro]
    datum[only_baro] = AltitudeDatum.BAROMETRIC_STD.value
    datum[on_ground] = AltitudeDatum.GROUND.value
    valid_pos = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (np.abs(lat) <= 90)
        & (np.abs(lon) <= 180)
        & np.isfinite(t)
    )
    st.points_invalid_position += int((~valid_pos).sum())
    keep = valid_pos.copy()
    if bbox is not None:
        inside = (lat >= bbox.south) & (lat <= bbox.north) & (lon >= bbox.west) & (lon <= bbox.east)
        st.points_outside_bbox += int((keep & ~inside).sum())
        keep &= inside
    st.points_on_ground += int((keep & on_ground).sum())
    st.points_missing_altitude += int((keep & ~on_ground & np.isnan(alt_val)).sum())
    if not keep.any():
        return None
    idx = np.where(keep)[0]
    st.points_kept += len(idx)
    st.aircraft_kept += 1
    df = pl.DataFrame(
        {
            "source": [source] * len(idx),
            "icao24": [icao] * len(idx),
            "t": t[idx],
            "lat": lat[idx],
            "lon": lon[idx],
            "alt_baro_ft": alt_baro[idx],
            "alt_geom_ft": alt_geom[idx],
            "altitude_value_ft": alt_val[idx],
            "altitude_datum": [str(d) for d in datum[idx]],
            "on_ground": on_ground[idx],
            "gs_mps": gs[idx],
            "track_deg": trk[idx],
            "vertical_rate_mps": vr[idx],
            "flags": flags[idx],
            "stale": (flags[idx] & 1) > 0,
            "new_leg": (flags[idx] & 2) > 0,
            "position_source": [pos_type[i] for i in idx],
            "callsign": [callsign[i] for i in idx],
            "aircraft_type": [ac_type if isinstance(ac_type, str) else None] * len(idx),
            "aircraft_desc": [desc if isinstance(desc, str) else None] * len(idx),
            "registration": [reg if isinstance(reg, str) else None] * len(idx),
            "db_flags": [int(db_flags) if isinstance(db_flags, int) else None] * len(idx),
            "category": [category[i] for i in idx],
        },
        schema=SAMPLE_SCHEMA,
    )
    return df


def empty_samples() -> pl.DataFrame:
    return pl.DataFrame(schema=SAMPLE_SCHEMA)


# ---------------------------------------------------------------------------------------
# Aircraft type classification (heuristic; unknown is NOT jet)
# ---------------------------------------------------------------------------------------

_PISTON_PREFIXES = (
    "C1",
    "C2",
    "C3",
    "P28",
    "PA",
    "SR2",
    "BE3",
    "BE5",
    "BE7",
    "BE9",
    "M20",
    "DA4",
    "DA2",
    "RV",
    "AA5",
    "GLST",
    "C77",
    "C82",
    "DR40",
    "P210",
    "TB",
    "AC11",
    "LNC",
    "COL",
    "EVOT",
    "G2",
    "G3",
    "G5",
    "HUSK",
    "CH7",
    "J3",
    "PTS",
    "DV20",
    "FDCT",
    "EUPA",
    "P32",
    "P46",
    "BL8",
    "BE23",
    "BE24",
    "BE33",
    "BE35",
    "BE36",
    "BE55",
    "BE58",
    "BE60",
    "BE76",
    "C25",
    "C30",
    "C33",
    "C34",
    "C40",
    "C41",
    "C42",
    "C50",
    "PA3",
    "PA4",
    "PA6",
)
_TURBOPROP = {
    "C208",
    "C408",
    "PC12",
    "PC6T",
    "PC21",
    "TBM7",
    "TBM8",
    "TBM9",
    "BE20",
    "BE30",
    "B350",
    "BE10",
    "BE9L",
    "BE9T",
    "DHC6",
    "DH8A",
    "DH8B",
    "DH8C",
    "DH8D",
    "AT43",
    "AT45",
    "AT72",
    "AT75",
    "AT76",
    "SW4",
    "SF34",
    "E120",
    "D328",
    "C212",
    "P180",
    "PAY1",
    "PAY2",
    "PAY3",
    "PAY4",
    "SB20",
    "KODI",
    "C441",
    "MU2",
    "AC90",
    "AC95",
    "PC7",
    "PC9",
    "T6",
    "AT8T",
    "AT5T",
    "AT6T",
    "PA46",
    "P46T",
    "M600",
    "EPIC",
    "L410",
    "C295",
    "C130",
    "C30J",
    "P3",
    "E2",
    "A400",
    "DHC2",
    "DHC3",
    "DHC7",
    "ATP",
    "F27",
    "F50",
    "AN2",
    "AN26",
    "AN12",
    "Y12",
    "SH33",
    "SH36",
    "DO28",
    "BN2P",
    "TRIS",
    "JS31",
    "JS32",
    "JS41",
    "SC7",
    "L188",
    "CVLT",
    "PRIM",
}
_HELI_PREFIXES = (
    "R22",
    "R44",
    "R66",
    "EC",
    "AS3",
    "AS5",
    "H60",
    "H1",
    "B06",
    "B07",
    "B40",
    "B41",
    "B42",
    "B43",
    "B21",
    "B47",
    "UH1",
    "A109",
    "A119",
    "A139",
    "A169",
    "A189",
    "S76",
    "S92",
    "S61",
    "MD5",
    "MD6",
    "H47",
    "CH47",
    "EH10",
    "BK17",
    "HUCO",
    "ALO",
    "GAZL",
    "LYNX",
    "ENST",
    "SCHW",
    "H135",
    "H145",
    "H155",
    "H160",
    "H175",
    "H225",
    "AW",
)


def classify_aircraft_type(
    type_code: str | None, desc: str | None = None, category: str | None = None
) -> str:
    """Return one of 'piston', 'turboprop', 'helicopter', 'jet', 'unknown'.

    Heuristic designed for prioritisation only. Unknown types are never treated as jets.
    """
    if type_code:
        tc = type_code.upper().strip()
        if tc in _TURBOPROP:
            return "turboprop"
        if tc.startswith(_HELI_PREFIXES):
            return "helicopter"
        if tc.startswith(_PISTON_PREFIXES):
            return "piston"
        if tc.startswith(
            (
                "A3",
                "A2",
                "A1",
                "A38",
                "B7",
                "B3",
                "B4",
                "B5",
                "B6",
                "B2",
                "E1",
                "E7",
                "E5",
                "E3",
                "E9",
                "CRJ",
                "C25",
                "C5",
                "C56",
                "C68",
                "C70",
                "C75",
                "GLF",
                "GL",
                "LJ",
                "H25",
                "FA",
                "F2",
                "F9",
                "DC",
                "MD8",
                "MD9",
                "MD1",
                "BCS",
                "SU9",
                "RJ",
                "AN1",
                "IL",
                "TU",
                "CL",
                "PRM",
                "EA50",
                "HDJT",
                "SF50",
                "C510",
                "C525",
            )
        ):
            return "jet"
    if desc:
        d = desc.upper()
        if any(
            k in d
            for k in (
                "CESSNA",
                "PIPER",
                "CIRRUS",
                "BEECH",
                "MOONEY",
                "DIAMOND",
                "ROBIN",
                "GRUMMAN",
                "VANS",
                "RV-",
            )
        ):
            return "piston"
        if "HELICOPTER" in d or "ROTORCRAFT" in d:
            return "helicopter"
        if "JET" in d or "AIRBUS" in d or "BOEING" in d:
            return "jet"
    if category in {"A1"}:
        return "piston"  # light (< 7000 kg) — usually piston/small turboprop
    if category in {"A7"}:
        return "helicopter"
    if category in {"A4", "A5"}:
        return "jet"  # high-vortex / heavy
    return "unknown"
