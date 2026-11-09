"""The pruned CPA table must equal a brute-force table filtered the way every consumer filters
it (per-event speed and slant bounds): pruning saves work, never drops a usable entry."""

from __future__ import annotations

import numpy as np

from aerodoppler.adsb.segments import TrajectoryBuild
from aerodoppler.geo.coordinates import geodetic_to_ecef
from aerodoppler.geo.grids import regular_grid
from aerodoppler.models import BBox
from aerodoppler.physics.doppler import closest_approach_batch
from aerodoppler.physics.trajectory import Trajectory
from aerodoppler.search.preselect import GridWithElevation, SegmentInfo, sparse_cpa

C = 343.0


def _track(rng, lat0, lon0, speed, heading, alt, t0, n=240, dt=5.0, turn=0.0):
    t = t0 + np.arange(n) * dt
    hd = np.radians(heading + turn * np.arange(n) * dt)
    north = np.cumsum(speed * dt * np.cos(hd)) - speed * dt * n / 2 * np.cos(hd[0])
    east = np.cumsum(speed * dt * np.sin(hd)) - speed * dt * n / 2 * np.sin(hd[0])
    lat = lat0 + north / 111_320.0
    lon = lon0 + east / (111_320.0 * np.cos(np.radians(lat0)))
    pos = geodetic_to_ecef(lat, lon, np.full(n, alt)).reshape(-1, 3)
    return Trajectory.from_ecef(t, pos)


def _setup(seed=4):
    rng = np.random.default_rng(seed)
    bbox = BBox(west=-81.3, south=28.9, east=-81.1, north=29.1)
    g = regular_grid(bbox, 500.0, max_points=100_000)
    hae = np.full(len(g), 10.0)
    grid = GridWithElevation(
        g, np.zeros(len(g)), np.ones(len(g)), hae, geodetic_to_ecef(g.lat, g.lon, hae), [], "flat"
    )
    segs = []
    for k in range(12):
        tr = _track(
            rng,
            float(rng.uniform(28.7, 29.3)),  # some tracks far outside the area
            float(rng.uniform(-81.6, -80.8)),
            float(rng.uniform(30, 90)),
            float(rng.uniform(0, 360)),
            float(rng.uniform(200, 2500)),
            1.79e9 + k * 100,
            turn=float(rng.uniform(-0.3, 0.3)),
        )
        tb = TrajectoryBuild(tr, 10.0, "wgs84", len(tr.t), 0, [])
        segs.append(SegmentInfo(f"s{k}", tb, "piston", "C172", None, f"a{k}", 0.0))
    return grid, segs


def test_pruned_table_equals_brute_force_after_event_filter():
    grid, segs = _setup()
    d_min, d_max = 150.0, 6000.0
    bounds = [(30.0, 60.0, 150.0, 4000.0), (55.0, 120.0, 800.0, 6000.0)]
    got = sparse_cpa(segs, grid, C, d_min, d_max, [], event_bounds=bounds)
    want = set()
    for si, s in enumerate(segs):
        tr = s.traj_build.trajectory.subsample(10.0)
        te, r, tau = closest_approach_batch(tr, grid.observer_ecef, C)
        v = np.linalg.norm(s.traj_build.trajectory.velocity(te), axis=1)
        for p in range(len(r)):
            if not (d_min <= r[p] <= d_max):
                continue
            if any(lo <= v[p] <= hi and rl <= r[p] <= rh for lo, hi, rl, rh in bounds):
                want.add((si, p, round(float(tau[p]), 6), round(float(r[p]), 6)))
    have = {
        (int(a), int(b), round(float(c), 6), round(float(d), 6))
        for a, b, c, d in zip(got.seg_idx, got.point_idx, got.tau, got.slant, strict=True)
    }
    assert want and have == want


def test_far_tracks_are_not_evaluated():
    grid, segs = _setup()
    counters = []
    sparse_cpa(segs, grid, C, 150.0, 3000.0, counters)
    note = counters[-1].note
    evaluated = int(note.split(";")[1].split(" of ")[0])
    assert evaluated < len(segs) * len(grid.grid)
