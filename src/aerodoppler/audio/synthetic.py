"""Independent synthetic-audio simulator (SPEC §9: avoid the "inverse crime").

This module deliberately does **not** use the Doppler formula or the piecewise-linear
trajectory code in :mod:`aerodoppler.physics`. It synthesises the received pressure wave
from first principles:

* the aircraft follows a *smooth* (cubic-spline or analytic) path ``p(t)``;
* the source emits a harmonic series with phase ``phi(t_e) = 2*pi * int f0(tau) dtau``;
* each output sample at receiver time ``t_r`` is obtained by solving the retarded-time
  equation ``t_r = t_e + |p(t_e) - x| / c`` for ``t_e`` (Newton iteration) and evaluating
  ``sum_h a_h sin(h * phi(t_e)) / r(t_e)``.

The Doppler shift therefore *emerges* from the retarded phase; nothing in the forward model
is reused. The simulated flights can also be exported as ADS-B-like samples so they flow
through the real ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from aerodoppler.geo.altitude import get_geoid
from aerodoppler.geo.coordinates import LocalFrame, ecef_to_geodetic
from aerodoppler.models import AltitudeDatum, AltitudeUnit, FlightSample, PositionSource


@dataclass
class SyntheticFlight:
    name: str
    icao24: str
    t_start: float  # absolute unix s
    t_end: float
    path_ecef: Any  # callable t -> (N,3)
    f0_hz: float
    drift_per_s: float = 0.0  # fractional f0 drift per second (linear)
    harmonic_amps: tuple[float, ...] = (1.0, 0.7, 0.5, 0.35, 0.25)
    level_at_1km: float = 1.0  # pressure amplitude of harmonic 1 at 1 km
    aircraft_type: str | None = "C172"
    audible: bool = True
    meta: dict = field(default_factory=dict)

    def f0(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.f0_hz * (1.0 + self.drift_per_s * (t - self.t_start))

    def phase(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """2*pi * integral of f0 from t_start (closed form for linear drift)."""
        dt = t - self.t_start
        return 2 * np.pi * self.f0_hz * (dt + 0.5 * self.drift_per_s * dt**2)


# ---------------------------------------------------------------------------------------
# Path builders (ENU waypoints -> smooth ECEF path)
# ---------------------------------------------------------------------------------------


def spline_path_from_enu(frame: LocalFrame, times: NDArray[np.float64], enu: NDArray[np.float64]):
    """Cubic spline in ECEF through ENU waypoints; linear extrapolation beyond the ends."""
    ecef = frame.enu_to_ecef(enu)
    cs = CubicSpline(times, ecef, axis=0, bc_type="natural", extrapolate=True)

    def path(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return cs(np.asarray(t, dtype=np.float64))

    path.spline = cs  # type: ignore[attr-defined]
    return path


def straight_path(
    frame: LocalFrame,
    t_cpa: float,
    cpa_enu: NDArray[np.float64],
    heading_deg: float,
    speed_mps: float,
):
    """Analytic straight, level, constant-speed path (ENU), closest to the frame origin at t_cpa."""
    h = np.radians(heading_deg)
    direction = np.array([np.sin(h), np.cos(h), 0.0])
    c0 = np.asarray(cpa_enu, dtype=np.float64)

    def path(t: NDArray[np.float64]) -> NDArray[np.float64]:
        tt = np.asarray(t, dtype=np.float64)[:, None]
        return frame.enu_to_ecef(c0[None, :] + direction[None, :] * speed_mps * (tt - t_cpa))

    return path


def turning_path(
    frame: LocalFrame,
    t0: float,
    start_enu: NDArray[np.float64],
    heading_deg: float,
    speed_mps: float,
    duration_s: float,
    turn_rate_deg_s: float,
    turn_start_s: float,
    turn_duration_s: float,
    climb_rate_mps: float = 0.0,
    step_s: float = 2.0,
):
    """Straight -> coordinated turn -> straight, integrated then splined (non-straight path)."""
    n = int(duration_s / step_s) + 1
    ts = t0 + step_s * np.arange(n)
    hdg = np.radians(heading_deg) + np.radians(turn_rate_deg_s) * np.clip(
        ts - t0 - turn_start_s, 0, turn_duration_s
    )
    vx, vy = speed_mps * np.sin(hdg), speed_mps * np.cos(hdg)
    x = start_enu[0] + np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * step_s)])
    y = start_enu[1] + np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * step_s)])
    z = start_enu[2] + climb_rate_mps * (ts - t0)
    return spline_path_from_enu(frame, ts, np.stack([x, y, z], axis=1))


# ---------------------------------------------------------------------------------------
# Retarded-time synthesis
# ---------------------------------------------------------------------------------------


def solve_emission_times(
    path,
    observer_ecef: NDArray[np.float64],
    t_receive: NDArray[np.float64],
    c: float,
    iters: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Newton solve of g(t_e) = t_e + |p(t_e) - x|/c - t_r = 0. Returns (t_emit, range)."""
    x = observer_ecef.reshape(1, 3)
    te = t_receive.copy()
    r = np.linalg.norm(path(te) - x, axis=1)
    te = t_receive - r / c
    h = 1e-3
    for _ in range(iters):
        p = path(te)
        d = p - x
        r = np.linalg.norm(d, axis=1)
        g = te + r / c - t_receive
        # dr/dt by symmetric difference of the path (keeps the solver independent of velocity code)
        r_p = np.linalg.norm(path(te + h) - x, axis=1)
        r_m = np.linalg.norm(path(te - h) - x, axis=1)
        drdt = (r_p - r_m) / (2 * h)
        te = te - g / (1.0 + drdt / c)
    r = np.linalg.norm(path(te) - x, axis=1)
    return te, r


def synthesize_flight_signal(
    flight: SyntheticFlight,
    observer_ecef: NDArray[np.float64],
    t_receive: NDArray[np.float64],
    c: float,
    coarse_step_s: float = 0.01,
) -> NDArray[np.float64]:
    """Received pressure (arbitrary units) at the given receiver times for one flight."""
    # Solve the retarded-time equation on a coarse receiver grid, then interpolate t_e and
    # 1/r with cubic splines (both are smooth); the phase is evaluated exactly at every sample.
    tr0, tr1 = float(t_receive[0]), float(t_receive[-1])
    grid = np.arange(tr0 - coarse_step_s, tr1 + 2 * coarse_step_s, coarse_step_s)
    te_g, r_g = solve_emission_times(flight.path_ecef, observer_ecef, grid, c)
    te = CubicSpline(grid, te_g)(t_receive)
    inv_r = CubicSpline(grid, 1.0 / np.maximum(r_g, 1.0))(t_receive)
    gate = ((te >= flight.t_start) & (te <= flight.t_end)).astype(np.float64)
    # smooth 5 s on/off ramps so that the start/end of emission do not click
    ramp = np.clip((te - flight.t_start) / 5.0, 0, 1) * np.clip((flight.t_end - te) / 5.0, 0, 1)
    phi = flight.phase(te)
    sig = np.zeros_like(t_receive)
    for h, a in enumerate(flight.harmonic_amps, start=1):
        sig += a * np.sin(h * phi)
    return sig * inv_r * 1000.0 * flight.level_at_1km * gate * ramp


def coloured_noise(
    n: int, fs: float, rng: np.random.Generator, alpha: float = 1.0
) -> NDArray[np.float64]:
    """1/f^alpha noise with unit RMS, generated by spectral shaping of white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.ones_like(freqs)
    shape[1:] = 1.0 / np.power(np.maximum(freqs[1:], 1.0), alpha / 2.0)
    out = np.fft.irfft(spec * shape, n=n)
    return out / (np.std(out) + 1e-12)


@dataclass
class SyntheticScene:
    observer_lat: float
    observer_lon: float
    observer_h_hae: float
    frame: LocalFrame
    flights: list[SyntheticFlight]
    t_start_abs: float
    duration_s: float
    fs: int
    c_mps: float

    @property
    def observer_ecef(self) -> NDArray[np.float64]:
        from aerodoppler.geo.coordinates import geodetic_to_ecef

        return geodetic_to_ecef(self.observer_lat, self.observer_lon, self.observer_h_hae).reshape(
            3
        )


def synthesize_scene(
    scene: SyntheticScene,
    rng: np.random.Generator,
    snr_db_at_cpa: float = 10.0,
    hum_hz: float | None = 60.0,
    hum_level: float = 0.3,
    gust_modulation: float = 0.2,
) -> tuple[NDArray[np.float32], dict]:
    """Return (audio float32 in [-1,1], truth dict)."""
    n = int(round(scene.duration_s * scene.fs))
    t_r = scene.t_start_abs + np.arange(n) / scene.fs
    x = scene.observer_ecef
    total = np.zeros(n)
    truth: dict = {"flights": []}
    peak_rms = 0.0
    for fl in scene.flights:
        if not fl.audible:
            truth["flights"].append({"name": fl.name, "audible": False})
            continue
        s = synthesize_flight_signal(fl, x, t_r, scene.c_mps)
        # signal RMS in a 20 s window around its loudest point
        win = int(10 * scene.fs)
        k = int(np.argmax(np.abs(s)))
        seg = s[max(0, k - win) : k + win]
        peak_rms = max(peak_rms, float(np.sqrt(np.mean(seg**2))))
        total += s
        truth["flights"].append(
            {
                "name": fl.name,
                "icao24": fl.icao24,
                "f0_hz": fl.f0_hz,
                "drift_per_s": fl.drift_per_s,
                "audible": True,
            }
        )
    if peak_rms <= 0:
        peak_rms = 1.0
    noise = coloured_noise(n, scene.fs, rng, alpha=1.0) * peak_rms / (10 ** (snr_db_at_cpa / 20))
    if hum_hz:
        hum = sum(
            np.sin(2 * np.pi * hum_hz * k * (t_r - t_r[0]) + rng.uniform(0, 2 * np.pi)) / k
            for k in (1, 2, 3)
        )
        noise += hum_level * peak_rms * hum
    if gust_modulation > 0:
        slow = coloured_noise(n, scene.fs, rng, alpha=2.0)
        slow = slow / (np.max(np.abs(slow)) + 1e-9)
        total *= 1.0 + gust_modulation * slow
    audio = total + noise
    audio = audio / (np.max(np.abs(audio)) * 1.05 + 1e-12)
    truth.update(
        {
            "observer_lat": scene.observer_lat,
            "observer_lon": scene.observer_lon,
            "observer_h_hae": scene.observer_h_hae,
            "t_start_abs": scene.t_start_abs,
            "duration_s": scene.duration_s,
            "fs": scene.fs,
            "c_mps": scene.c_mps,
            "snr_db_at_cpa": snr_db_at_cpa,
            "synthetic": True,
        }
    )
    return audio.astype(np.float32), truth


# ---------------------------------------------------------------------------------------
# ADS-B-like export
# ---------------------------------------------------------------------------------------


def flight_to_adsb_samples(
    flight: SyntheticFlight,
    rng: np.random.Generator,
    cadence_s: float = 6.0,
    jitter_s: float = 2.0,
    baro_offset_m: float = 25.0,
    position_noise_m: float = 5.0,
    dropout_fraction: float = 0.05,
    gap: tuple[float, float] | None = None,
    source: str = "synthetic",
) -> list[FlightSample]:
    """Sample the smooth path like an ADS-B receiver would: irregular cadence, feet, both
    barometric (offset from true) and geometric altitudes, random dropouts, optional gap."""
    t = flight.t_start
    times = []
    while t <= flight.t_end:
        times.append(t)
        t += cadence_s + rng.uniform(-jitter_s, jitter_s)
    ta = np.array(times)
    keep = rng.uniform(size=len(ta)) >= dropout_fraction
    if gap is not None:
        keep &= ~((ta >= gap[0]) & (ta <= gap[1]))
    ta = ta[keep]
    pos = flight.path_ecef(ta)
    pos = pos + rng.normal(scale=position_noise_m, size=pos.shape)
    lat, lon, h = ecef_to_geodetic(pos)
    n_geoid = get_geoid().undulation_m(lat, lon)
    alt_geom_ft = h / 0.3048
    alt_baro_ft = (h - n_geoid + baro_offset_m) / 0.3048
    vel = flight.path_ecef(ta + 0.5) - flight.path_ecef(ta - 0.5)
    samples = []
    for i in range(len(ta)):
        samples.append(
            FlightSample(
                source=source,
                aircraft_icao24=flight.icao24,
                utc_unix_s=float(ta[i]),
                latitude=float(lat[i]),
                longitude=float(lon[i]),
                altitude_value=float(round(alt_baro_ft[i] / 25) * 25),
                altitude_unit=AltitudeUnit.FEET,
                altitude_datum=AltitudeDatum.BAROMETRIC_STD,
                groundspeed_mps=float(np.linalg.norm(vel[i])),
                aircraft_type=flight.aircraft_type,
                position_source=PositionSource.SYNTHETIC,
                quality_flags=0,
                alt_baro_ft=float(round(alt_baro_ft[i] / 25) * 25),
                alt_geom_ft=float(round(alt_geom_ft[i] / 25) * 25),
                callsign=flight.name,
            )
        )
    return samples
