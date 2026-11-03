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

## M2 — Data: ADSB.lol ingest, QC, Parquet, DEM, indexes (done)

Upstream format verification (2026-11-03): read https://www.adsb.lol/docs/open-data/historical,
readsb `README-json.md` ("trace jsons"), the GitHub release API for
`adsblol/globe_history_2026`, and the first 3 MB of the real `v2026.11.02-planes-readsb-prod-0.tar.aa`
via an HTTP range request. Findings: one release per UTC day and instance; split tar parts
(`.tar.aa` 2,000,000,000 B + `.tar.ab` 1,501,931,008 B for 2026-11-02 prod-0); members
`traces/<last 2 hex>/trace_full_<icao>.json` are gzip-compressed readsb traces with 14 columns
(baro alt or "ground", geometric alt at index 10, flags bit 8 = altitude geometric); licence ODbL 1.0.

Implemented: `adsb/normalize.py` (schema-faithful parser, datum per sample, raw baro+geom
preserved, bbox filter counts), `adsb/segments.py` (split on gap > 30 s, implied speed > 350 m/s,
ground transitions, readsb new-leg flag; flags instead of drops; completeness score),
`adsb/index.py` (Parquet partitioned by date/region + `segments.parquet`, DuckDB queries by
time/bbox/speed/class, sample loading by segment id), `adsb/providers.py` (provider protocol,
streaming split-tar reader, local fixture provider), `adsb/adsblol.py` (GitHub release lookup
with prod→staging fallback, resumable Range downloads with size verification and SHA-256,
offline cache reuse, rate-limit handling), `adsb/ingest.py` (parallel parse with spawn pool,
storage-budget guard, reproducibility manifest with URLs/sizes/hashes/retrieval time/git SHA/
filter counts/coverage), `geo/dem.py` (GeoTIFF tiles, USGS 3DEP ImageServer export, Copernicus
GLO-30 AWS COGs, labelled synthetic DEM, explicit FlatDEM fallback, `DEMError` instead of zero),
`geo/grids.py` (local-frame grids, corridor masking with STRtree, refinement grids).

Real data evidence (observed):

```
$ uv run aerodoppler flights import --provider adsblol --date 2026-11-02 --bbox -81.8,28.6,-80.8,29.6 --confirm-download
traces: 63445file [00:52, 1201.16file/s]
{"dataset_id": "adsblol_2026-11-02_1e2f2f08",
 "summary": {"samples": 357981, "aircraft": 1170, "segments": 3897, "airborne_segments": 2372,
             "segments_missing_altitude": 53, "segments_baro_only": 30, "segments_with_large_gap": 0,
             "aircraft_classes": {"jet": 1432, "unknown": 106, "piston": 741, "turboprop": 47, "helicopter": 46},
             "unknown_type_segments": 130},
 "filter_counts": {"files": 63445, "files_failed": 0, "aircraft_kept": 1170, "points_total": 99248990,
                   "points_kept": 358288, "points_invalid_position": 0, "points_on_ground": 22193,
                   "points_missing_altitude": 1, "points_outside_bbox": 98890702},
 "storage_bytes": 7754849, "license": "ODbL-1.0"}
```

Raw archive: 3.5 GB downloaded once (`data/raw/adsblol/2026-11-02/`), normalised bbox dataset 7.8 MB.
Cadence analysis on the real bbox data: median interval 2.9 s; 0.35 % of intervals > 30 s, of
which 75 % are > 120 s (true coverage breaks), so the SPEC default gap split of 30 s is kept.

DEM evidence (observed): `aerodoppler geo fetch-dem --bbox -81.3,28.9,-81.0,29.2 --source 3dep`
returned a 973×1112 px NAVD88 GeoTIFF in 15 s; `geo elevation -- 29.05 -81.15` → 11.99 m
(3DEP), EGM96 undulation −29.7 m; Copernicus GLO-30 tiles N28/N29 W081/W082 fetched from AWS and
gave 11.86 m at the same point (consistent to 0.13 m).

Decision: OpenSky Trino is not implemented (requires approved research access; not a plain
download). Decision: Copernicus COG tags cannot be rewritten in place; provenance goes to a JSON
sidecar. Limitation: the daily archive can only be fetched whole (≈3.3–3.5 GB/day); the importer
refuses unless `--confirm-download` or a sufficient `--storage-budget-gb` is given and reuses the
cached raw parts for later bboxes.

Tests (observed): `uv run pytest -q -m "not network"` → `32 passed, 1 deselected in 10.95s`
(includes 6 fixture-based ADS-B pipeline integration tests using the six real traces in
`tests/fixtures/adsblol/`, see PROVENANCE.md). The network test for 3DEP/Copernicus is marked
`network`.

## M3 — DSP (in progress)

* `audio/decode.py`: libsndfile for WAV/FLAC/OGG, FFmpeg for MP3/video, SHA-256, duration
  limits, anti-aliased polyphase resampling to the 2 kHz analysis stream, no enhancement.
* `audio/ridges.py`: iterative Viterbi ridge tracker (numba), stationary-tone rejection,
  duration-weighted harmonic grouping, f0 prior at the steepest-slope point.
  Synthetic two-flight + 60 Hz hum test: both events found, harmonics numbered 1,2,3 and 1,2,
  hum rejected, f0 priors 92.7 Hz (true 90) and 124.6 Hz (true 115; turning flight, prior is
  coarse by design). Figure: `docs/img/m3_auto_ridges_synthetic.png` (labelled SYNTHETIC).
  Noise-only input yields zero ridges (abstention path).
