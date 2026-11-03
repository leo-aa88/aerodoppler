# AeroDoppler — Implementation Report

This file is a running log of what was built, the commands run, observed outputs, decisions,
deviations from SPEC.md, known limitations and the next executable action. It is updated at
each milestone. Nothing here is a claim unless the command output shown was actually observed.

Environment used for development: WSL2 Linux, Python 3.12 (uv-managed), 16 CPU, 39 GB RAM,
FFmpeg 4.4.2 (system), Node 24. Network access was available during development; all tests
and the demo also run with `AERODOPPLER_OFFLINE=true`.

## M0 — Bootstrap (done)

* `pyproject.toml` (uv, hatchling), `Makefile`, `.env.example`, `.gitignore`, package layout
  per SPEC §4, pydantic models (`models.py`) for every SPEC §5 contract, settings
  (`config.py`, `AERODOPPLER_*` env vars, localhost default).
* Lint: `uv run ruff check src tests` → `All checks passed!` (observed).

## M1 — Physics and synthetic oracle (done)

* `physics/propagation.py`: speed of sound 331.3 + 0.606 T, Doppler factor c/(c - v_toward),
  propagation delay; uncertainty object for unknown temperature/wind.
* `physics/trajectory.py`: piecewise-linear ECEF position, smoothed central-difference velocity,
  no extrapolation, NaN inside gaps > `max_gap_s`, duplicate timestamps dropped, NaN rejected.
* `physics/doppler.py`: emission-time-correct forward model, monotonic reception inversion,
  exact (non-snapped) closest approach on the polyline, batched CPA for many observers,
  closed-form straight flyby used as an *independent* reference.
* `geo/coordinates.py`: pyproj EPSG:4979↔4978 with `always_xy=True`, local ENU frame.
* `geo/altitude.py`: feet/metres, datum normalisation to WGS84 ellipsoidal height with an
  uncertainty per datum; full 15-arcmin EGM96 geoid embedded (1.7 MB, public domain NGA grid
  converted from PROJ's `us_nga_egm96_15.tif`; checked: N(0,0)=17.2 m).
* `audio/synthetic.py`: **independent simulator**. It solves the retarded-time equation per
  sample with Newton iteration on a *cubic-spline* path and evaluates the harmonic phase at the
  emission time; no Doppler formula and no piecewise-linear code is shared with the forward
  model. Exports ADS-B-like samples (irregular cadence, feet, baro+geom altitudes with a
  deliberate 25 m baro offset, dropouts, optional gap).
* `audio/spectrogram.py`: `scipy.signal.ShortTimeFFT`, Hann 4.096 s / hop 0.25 s / 2 kHz,
  whitening = frequency block-median background + per-bin long-block temporal median (hum
  removal) + MAD scaling. A first version used `scipy.ndimage.median_filter` along time
  (47 s per 5-minute file); replaced by block medians (1.5 s total).

Tests (observed):

```
$ uv run pytest tests/unit -q
....................                                                     [100%]
20 passed in 9.47s
```

Covered: analytic straight flyby vs forward model (t_receive within 2 ms, f within 2e-4
relative), sign convention, zero radial velocity at CPA, delayed CPA, propagation delay,
monotonic reception and inversion, clock-shift invariance, emission-vs-reception discrepancy
on a curved track (> 0.5 Hz, i.e. > 2 bins), interpolation continuity, gaps/no extrapolation,
duplicates/NaN handling, noisy data finite, zero-speed degenerate, batched CPA = scalar CPA,
WGS84↔ECEF round trip, explicit axis order, ENU frame scale, feet/metres, geoid values,
datum uncertainties, bbox validation. Synthetic oracle: ridge extracted from the independent
simulator's audio matches the forward model (built from ADS-B-like samples of the same flight)
with 90th-percentile error < 0.6 Hz for a straight flyby and < 0.8 Hz for a turning flight
with RPM drift; a straight-line surrogate fits the turning case worse than the true track.
