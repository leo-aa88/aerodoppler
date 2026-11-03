"""Semi-synthetic validation on REAL ADS-B trajectories (M7 without field recordings).

Audio is synthesised (independent retarded-time simulator) for real aircraft tracks from an
imported ADSB.lol dataset, as heard from a chosen observer position; the search then has to
find that position among *all* real traffic of the day, with a real DEM. This exercises real
sampling cadence, real manoeuvres, real altitude datums and real distractor density, but it is
NOT a real-recording benchmark: propagation, source spectra and noise are simulated. Every
output is labelled accordingly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline

from aerodoppler.adsb.index import FlightDataset, FlightIndex
from aerodoppler.adsb.segments import trajectory_from_samples
from aerodoppler.audio.decode import write_wav
from aerodoppler.audio.synthetic import SyntheticFlight, SyntheticScene, synthesize_scene
from aerodoppler.geo.altitude import ground_point_hae
from aerodoppler.geo.coordinates import LocalFrame, geodetic_to_ecef
from aerodoppler.models import BBox, unix_to_iso
from aerodoppler.physics.doppler import closest_approach
from aerodoppler.physics.propagation import speed_of_sound_mps


@dataclass
class RealTrajectoryCase:
    name: str
    observer_lat: float
    observer_lon: float
    seed: int = 1
    min_separation_s: float = 300.0
    max_separation_s: float = 1500.0
    max_cpa_m: float = 3000.0
    audible_cpa_m: float = 6000.0
    snr_db: float = 10.0
    fs: int = 8000
    classes: tuple[str, ...] = ("piston", "turboprop", "helicopter")


def _spline_path(traj):
    cs = CubicSpline(traj.t, traj.pos, axis=0, bc_type="natural", extrapolate=True)

    def path(t):
        return cs(np.asarray(t, dtype=np.float64))

    return path


def build_real_trajectory_scenario(
    case: RealTrajectoryCase, dataset: FlightDataset, dem, out_dir: Path, c_mps: float | None = None
) -> dict[str, Any]:
    """Pick two real low-flying propeller aircraft passing near the observer, synthesise the
    recording from every audible real aircraft in the window, and write wav + truth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(case.seed)
    c = c_mps or speed_of_sound_mps(20.0)
    idx = FlightIndex([dataset])
    man = dataset.manifest()
    req_bbox = man["request"]["bbox"]
    bbox = BBox(west=req_bbox[0], south=req_bbox[1], east=req_bbox[2], north=req_bbox[3])
    ground = float(dem.elevation([case.observer_lat], [case.observer_lon]).elevation_m[0])
    if not np.isfinite(ground):
        raise ValueError("DEM has no elevation at the observer")
    obs_hae = float(ground_point_hae(ground, case.observer_lat, case.observer_lon, 1.5))
    x = geodetic_to_ecef(case.observer_lat, case.observer_lon, obs_hae).reshape(3)
    segs = idx.query_segments(
        man["coverage"]["first_utc"], man["coverage"]["last_utc"], bbox, classes=list(case.classes)
    )
    segs = segs.filter(
        (pl.col("last_utc") - pl.col("first_utc") >= 240.0)
        & (pl.col("min_alt_m") < 3000.0)
        & (pl.col("median_speed_mps") >= 35.0)
    )
    samples = idx.load_samples(segs["segment_id"].to_list())
    passes = []
    trajs = {}
    for row in segs.iter_rows(named=True):
        sid = row["segment_id"]
        tb = trajectory_from_samples(samples.filter(pl.col("segment_id") == sid), sid)
        if tb.trajectory is None or tb.trajectory.t_end - tb.trajectory.t_start < 240:
            continue
        cpa = closest_approach(tb.trajectory, x, c)
        if (
            cpa.slant_range_m <= case.audible_cpa_m
            and tb.trajectory.t_start + 60 < cpa.t_emit_cpa < tb.trajectory.t_end - 60
        ):
            passes.append(
                {
                    "segment_id": sid,
                    "icao24": row["icao24"],
                    "type": row["aircraft_type"],
                    "class": row["aircraft_class"],
                    "t_cpa": cpa.t_emit_cpa,
                    "tau": cpa.t_receive_cpa,
                    "slant": cpa.slant_range_m,
                    "speed": cpa.speed_mps,
                }
            )
            trajs[sid] = tb.trajectory
    passes.sort(key=lambda p: p["t_cpa"])
    close = [p for p in passes if p["slant"] <= case.max_cpa_m]
    pairs = []
    for i, a in enumerate(close):
        for b in close[i + 1 :]:
            d = b["tau"] - a["tau"]
            if case.min_separation_s <= d <= case.max_separation_s and a["icao24"] != b["icao24"]:
                pairs.append((a, b))
    if not pairs:
        raise ValueError(
            f"no suitable pair near ({case.observer_lat}, {case.observer_lon}); {len(close)} close passes"
        )
    a, b = pairs[int(rng.integers(len(pairs)))]
    t_rec0 = a["tau"] - rng.uniform(150.0, 260.0)
    duration = (b["tau"] - t_rec0) + rng.uniform(250.0, 400.0)
    frame = LocalFrame(case.observer_lat, case.observer_lon, obs_hae)
    flights: list[SyntheticFlight] = []
    truth_flights = []
    for p in passes:
        if p["tau"] < t_rec0 - 300 or p["tau"] > t_rec0 + duration + 300:
            continue
        traj = trajs[p["segment_id"]]
        f0 = float(rng.uniform(75.0, 135.0))
        level = 1.0 if p["class"] == "piston" else 0.8
        amps = (
            (1.0, 0.65, 0.45, 0.3, 0.2)
            if p["class"] != "helicopter"
            else (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)
        )
        flights.append(
            SyntheticFlight(
                p["icao24"],
                p["icao24"],
                traj.t_start,
                traj.t_end,
                _spline_path(traj),
                f0,
                float(rng.normal(0, 1e-5)),
                amps,
                level,
                p["type"],
                audible=True,
            )
        )
        truth_flights.append({**p, "f0_hz": f0, "chosen": p in (a, b)})
    scene = SyntheticScene(
        case.observer_lat, case.observer_lon, obs_hae, frame, flights, t_rec0, duration, case.fs, c
    )
    audio, truth = synthesize_scene(
        scene, rng, snr_db_at_cpa=case.snr_db, hum_hz=60.0, hum_level=0.4, gust_modulation=0.2
    )
    wav = out_dir / f"realtraj_{case.name}.wav"
    write_wav(wav, audio, case.fs)
    truth.update(
        {
            "label": "SEMI-SYNTHETIC: real ADS-B trajectories, simulated audio",
            "case": case.name,
            "seed": case.seed,
            "dataset_id": dataset.dataset_id,
            "recording_start_utc": unix_to_iso(t_rec0),
            "recording_start_unix": t_rec0,
            "date_utc": man["request"]["date_utc"],
            "bbox": req_bbox,
            "ground_elevation_m": ground,
            "observer_hae_m": obs_hae,
            "chosen_pair": [a, b],
            "audible_flights": truth_flights,
            "event_separation_s": b["tau"] - a["tau"],
            "wav_path": str(wav),
        }
    )
    (out_dir / f"truth_{case.name}.json").write_text(json.dumps(truth, indent=2, default=str))
    return truth
