"""Altitude units, vertical datums and the EGM96 geoid.

Why this matters (SPEC §5): ADS-B *barometric* altitude is a pressure altitude referenced to
the ISA standard atmosphere; ADS-B *geometric* altitude (``alt_geom``) is GNSS height above
the WGS84 ellipsoid (HAE); DEMs such as USGS 3DEP (NAVD88) and Copernicus GLO-30 (EGM2008)
are orthometric heights above a geoid. These differ by tens of metres (geoid undulation) and,
for barometric altitudes, by an additional weather-dependent offset. All 3D slant-range
geometry in AeroDoppler is done in **WGS84 ellipsoidal height**; this module converts every
input to that reference and reports an uncertainty for each conversion.

Geoid: a full-resolution (15 arcmin) EGM96 undulation grid is embedded in the package
(``geo/data/egm96_15min_cm.npz``, derived from the public-domain NGA grid distributed by
PROJ). NAVD88 and EGM2008 differ from EGM96 by typically < 1-2 m, which is included in the
reported uncertainty rather than modelled explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerodoppler.models import AltitudeDatum, AltitudeUnit

FEET_TO_M = 0.3048

# Uncertainty budget (1-sigma-ish, metres) attached to each datum conversion.
GEOID_MODEL_SIGMA_M = 1.5  # EGM96 vs NAVD88/EGM2008 local differences, interpolation
BARO_SIGMA_M = 60.0  # pressure altitude vs true altitude without QNH/temperature data
UNKNOWN_DATUM_SIGMA_M = 80.0


class Geoid:
    """Bilinear EGM96 undulation N(lat, lon) in metres: h_ellipsoid = H_orthometric + N."""

    def __init__(self, undulation_cm: NDArray[np.int16], lat0: float, lon0: float, step_deg: float):
        self.n = undulation_cm.astype(np.float32) / 100.0
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.step = float(step_deg)
        self.rows, self.cols = self.n.shape

    def undulation_m(self, lat: ArrayLike, lon: ArrayLike) -> NDArray[np.float64]:
        lat_a = np.asarray(lat, dtype=np.float64)
        lon_a = np.asarray(lon, dtype=np.float64)
        lon_a = ((lon_a + 180.0) % 360.0) - 180.0
        r = (self.lat0 - lat_a) / self.step
        c = (lon_a - self.lon0) / self.step
        r0 = np.clip(np.floor(r).astype(int), 0, self.rows - 2)
        c0 = np.floor(c).astype(int) % self.cols
        c1 = (c0 + 1) % self.cols
        fr = np.clip(r - r0, 0.0, 1.0)
        fc = c - np.floor(c)
        n00 = self.n[r0, c0]
        n01 = self.n[r0, c1]
        n10 = self.n[r0 + 1, c0]
        n11 = self.n[r0 + 1, c1]
        top = n00 * (1 - fc) + n01 * fc
        bot = n10 * (1 - fc) + n11 * fc
        return np.asarray(top * (1 - fr) + bot * fr, dtype=np.float64)


@lru_cache(maxsize=1)
def get_geoid() -> Geoid:
    with resources.files("aerodoppler.geo.data").joinpath("egm96_15min_cm.npz").open("rb") as fh:
        z = np.load(fh)
        return Geoid(z["undulation_cm"], float(z["lat0"]), float(z["lon0"]), float(z["step_deg"]))


def to_metres(value: ArrayLike, unit: AltitudeUnit | str) -> NDArray[np.float64]:
    v = np.asarray(value, dtype=np.float64)
    if str(unit) == AltitudeUnit.FEET:
        return v * FEET_TO_M
    if str(unit) == AltitudeUnit.METRES:
        return v
    raise ValueError(f"unknown altitude unit {unit!r}")


@dataclass(frozen=True)
class HeightHAE:
    """Height above the WGS84 ellipsoid together with the conversion uncertainty."""

    h_m: NDArray[np.float64]
    sigma_m: NDArray[np.float64]
    datum_in: AltitudeDatum
    note: str


def to_ellipsoidal_height(
    value_m: ArrayLike,
    datum: AltitudeDatum | str,
    lat: ArrayLike,
    lon: ArrayLike,
) -> HeightHAE:
    """Normalise an altitude to WGS84 ellipsoidal height.

    Raises for ``on_ground`` (callers must exclude those samples) and returns a large
    uncertainty for ``unknown`` instead of guessing silently.
    """
    datum = AltitudeDatum(str(datum))
    v = np.asarray(value_m, dtype=np.float64)
    lat_a = np.asarray(lat, dtype=np.float64)
    lon_a = np.asarray(lon, dtype=np.float64)
    if datum == AltitudeDatum.GEOMETRIC_HAE:
        return HeightHAE(v, np.zeros_like(v), datum, "geometric HAE used directly")
    if datum == AltitudeDatum.GROUND:
        raise ValueError(
            "on-ground samples have no usable altitude; exclude them before conversion"
        )
    n = get_geoid().undulation_m(lat_a, lon_a)
    if datum == AltitudeDatum.ORTHOMETRIC:
        return HeightHAE(
            v + n, np.full_like(v, GEOID_MODEL_SIGMA_M), datum, "orthometric + EGM96 undulation"
        )
    if datum == AltitudeDatum.BAROMETRIC_STD:
        # Pressure altitude approximates orthometric height only under ISA conditions; the
        # residual (QNH deviation ~8 m/hPa, non-standard temperature) is carried as sigma.
        return HeightHAE(
            v + n,
            np.full_like(v, float(np.hypot(BARO_SIGMA_M, GEOID_MODEL_SIGMA_M))),
            datum,
            "barometric (ISA) treated as orthometric + EGM96 undulation; weather offset unmodelled",
        )
    return HeightHAE(
        v + n,
        np.full_like(v, UNKNOWN_DATUM_SIGMA_M),
        AltitudeDatum.UNKNOWN,
        "unknown datum assumed orthometric; large uncertainty attached",
    )


def ground_point_hae(
    ground_elev_ortho_m: ArrayLike, lat: ArrayLike, lon: ArrayLike, mic_height_m: float = 1.5
) -> NDArray[np.float64]:
    """Observer ellipsoidal height from DEM orthometric ground elevation plus mic height."""
    g = np.asarray(ground_elev_ortho_m, dtype=np.float64)
    return g + get_geoid().undulation_m(lat, lon) + mic_height_m
