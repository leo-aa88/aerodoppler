"""Soft silence constraint (SPEC §10.2.7).

Between the two events, other tracked aircraft that would have passed close to the candidate
observer should have left a ridge. For each such flight we predict its Doppler curve, look for
tonal evidence along it (best over a coarse f0 sweep) and, if none is found, add a penalty
weighted by (a) how audible it should have been (slant range vs. ``audible_m``) and (b) its
aircraft class (jets at altitude are often not tonal). Archives miss aircraft and some aircraft
are inaudible, so this is a bounded penalty, never an elimination. Every penalised flight is
logged with its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerodoppler.audio.spectrogram import Spectrogram
from aerodoppler.physics.doppler import closest_approach
from aerodoppler.search.preselect import SegmentInfo
from aerodoppler.search.score import EventInputs, ScoreConfig, predicted_curve, score_event_curve

CLASS_AUDIBILITY = {"piston": 1.0, "turboprop": 0.9, "helicopter": 0.9, "unknown": 0.5, "jet": 0.25}


@dataclass
class SilenceCheck:
    segment_id: str
    aircraft_class: str
    slant_m: float
    tau_abs: float
    t_rel_s: float
    evidence_z: float
    weight: float
    penalty: float


def silence_penalty(
    spec: Spectrogram,
    segments: list[SegmentInfo],
    exclude_seg_idx: set[int],
    observer_ecef: NDArray[np.float64],
    t0_abs: float,
    gap_rel: tuple[float, float],
    c_mps: float,
    audible_m: float,
    f0_range: tuple[float, float],
    cfg: ScoreConfig,
    evidence_threshold_z: float = 1.5,
    penalty_per_flight_z: float = 0.6,
    max_total_z: float = 1.5,
    heard_slant_m: float | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """``heard_slant_m``: the closest slant range among the heard (matched) passes; an unheard
    aircraft that passed much closer than anything heard weighs more (fdop's "unheard close
    pass"), one that passed farther than the heard ones weighs less."""
    lo_abs, hi_abs = t0_abs + gap_rel[0], t0_abs + gap_rel[1]
    if hi_abs - lo_abs < 30.0:
        return 0.0, []
    total = 0.0
    log: list[dict[str, Any]] = []
    for si, seg in enumerate(segments):
        if si in exclude_seg_idx:
            continue
        traj = seg.traj_build.trajectory
        if traj is None or traj.t_end < lo_abs - 600 or traj.t_start > hi_abs + 600:
            continue
        cpa = closest_approach(traj, observer_ecef, c_mps)
        if not (lo_abs <= cpa.t_receive_cpa <= hi_abs) or cpa.slant_range_m > audible_m:
            continue
        aud = (1.0 - cpa.slant_range_m / audible_m) * CLASS_AUDIBILITY.get(seg.aircraft_class, 0.5)
        if heard_slant_m is not None and heard_slant_m > 0:
            aud *= float(np.clip(heard_slant_m / max(cpa.slant_range_m, 1.0), 0.25, 2.0))
        if aud <= 0.05:
            continue
        s_centre = cpa.t_receive_cpa - t0_abs
        half = 60.0
        curve = predicted_curve(traj, observer_ecef, c_mps, cpa.t_receive_cpa, half, spec.hop_s)
        ev = EventInputs(
            s_centre,
            max(gap_rel[0], s_centre - half),
            min(gap_rel[1], s_centre + half),
            f0_range[0],
            f0_range[1],
        )
        coarse = ScoreConfig(
            f0_step_fraction=0.02,
            alphas=(0.0,),
            harmonics=cfg.harmonics[:3],
            harmonic_weights=cfg.harmonic_weights[:3],
        )
        best, _, _, _, _, _ = score_event_curve(spec, curve, ev, np.array([t0_abs]), coarse)
        evidence = float(best[0])
        pen = penalty_per_flight_z * aud if evidence < evidence_threshold_z else 0.0
        total += pen
        log.append(
            SilenceCheck(
                seg.segment_id,
                seg.aircraft_class,
                cpa.slant_range_m,
                cpa.t_receive_cpa,
                s_centre,
                evidence,
                aud,
                pen,
            ).__dict__
        )
    return min(total, max_total_z), log
