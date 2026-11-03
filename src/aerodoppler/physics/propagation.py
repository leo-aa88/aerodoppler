"""Acoustic propagation: speed of sound and retarded-time delay.

Model (SPEC §9): a stationary observer at ECEF ``x`` receives at ``t_receive = t_emit +
r(t_emit)/c`` where ``r`` is the slant range at *emission* time. Wind, refraction and
temperature gradients are not modelled; they are represented as an uncertainty on ``c``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_TEMPERATURE_C = 20.0


def speed_of_sound_mps(temperature_c: float | None = None) -> float:
    """Dry-air approximation c = 331.3 + 0.606 * T (m/s); 343.4 m/s at 20 °C."""
    t = DEFAULT_TEMPERATURE_C if temperature_c is None else float(temperature_c)
    if not (-80.0 <= t <= 60.0):
        raise ValueError(f"temperature {t} °C outside the plausible range [-80, 60]")
    return 331.3 + 0.606 * t


@dataclass(frozen=True)
class SoundSpeed:
    """Effective speed of sound with an uncertainty used when temperature/wind are unknown."""

    c_mps: float
    sigma_mps: float
    note: str

    @classmethod
    def from_temperature(
        cls, temperature_c: float | None, wind_sigma_mps: float = 5.0
    ) -> SoundSpeed:
        if temperature_c is None:
            # +-15 °C spread -> +-9 m/s; add typical unmodelled wind component.
            return cls(
                speed_of_sound_mps(None),
                float(np.hypot(9.0, wind_sigma_mps)),
                "temperature unknown; default 20 °C",
            )
        return cls(
            speed_of_sound_mps(temperature_c),
            float(np.hypot(1.0, wind_sigma_mps)),
            f"temperature {temperature_c} °C",
        )


def propagation_delay_s(slant_range_m: ArrayLike, c_mps: float) -> NDArray[np.float64]:
    r = np.asarray(slant_range_m, dtype=np.float64)
    if c_mps <= 0:
        raise ValueError("speed of sound must be positive")
    return r / c_mps


def reception_time(
    t_emit: ArrayLike, slant_range_m: ArrayLike, c_mps: float
) -> NDArray[np.float64]:
    return np.asarray(t_emit, dtype=np.float64) + propagation_delay_s(slant_range_m, c_mps)


def doppler_factor(v_toward_mps: ArrayLike, c_mps: float) -> NDArray[np.float64]:
    """f_receive / f_source = c / (c - v_toward); approaching (v_toward>0) raises frequency."""
    v = np.asarray(v_toward_mps, dtype=np.float64)
    if np.any(np.abs(v) >= c_mps):
        raise ValueError("supersonic radial velocity: Doppler model valid only for |v_toward| < c")
    return c_mps / (c_mps - v)
