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

## M3 — DSP (done)

* Recording store (`jobs/store.py`, SQLite WAL: recordings, events, searches, audit),
  `audio/service.py` (ingest with SHA-256 + preserved original + 2 kHz analysis copy,
  spectrogram cache, automatic proposals, manual events, purge), CLI `audio ingest|spectrogram|
  detect|add-event|events|list|purge`, API routes.
* Detector fixes found by testing: (1) ridge pieces of one physical ridge returned by separate
  Viterbi iterations are now merged when time-adjacent and frequency-continuous; (2) harmonic
  numbering is duration-weighted and compares plateau continuations correctly (edge-clamped
  interpolation); (3) the f0 prior is read at the steepest-slope point (≈ delayed CPA).
  On the demo recording this yields exactly two events with harmonics {1..5} and {1..4}.
* `pick_event_pair` chooses the two most credible, time-separated events for Mode A.
* Manual annotation works through the CLI, the API (`POST /api/recordings/{id}/events`) and the
  UI (click-to-draw ridge, harmonic number) before and independent of auto detection.

## M4/M5 — Inverse search (done)

`search/events.py` (features and generous speed/slant bounds), `search/preselect.py` (segment
query by time/bbox/speed bins, trajectories, 200 m grid with DEM, sparse CPA table, Mode A pairing
`tau_B − tau_A = Δt ± tol`, Mode B matching), `search/score.py` (numba bilinear curve scoring,
per-shift f0 estimate from ridge points ± 3 % plus a 0.04 % polish pass, RPM-drift sweep with
penalty, **shared t0** chosen from event 1), `search/silence.py` (soft, audibility- and
class-weighted penalty with a per-flight log), `search/refine.py` + `rank.py` (distinct maxima,
100 m → 25 m refinement, basin merge, landscape radius, abstention reasons, ambiguity flag),
`search/engine.py` (fork-pool parallel coarse/fine/refine scoring, stage counters, progress,
cancellation, run directory with result/manifest/heatmap/curves/tracks).

Performance decisions (measured): the first fine stage swept ~100 f0 values × 81 shifts × 3
drifts over 2400 curve points (150 ms/candidate); replacing the f0 sweep by the ridge-derived
estimate (±3 %), sampling curves at 0.5 s and a 1 s shift step brought it to ~40 ms, and the
refinement was parallelised. Demo Mode A end to end: 90–130 s on 6–8 workers (16-core WSL2).

Evidence (observed, `aerodoppler demo`, fully offline, SYNTHETIC two-aircraft scenario with three
distractors, 60 Hz hum, gusts, SNR 10 dB at CPA, synthetic DEM, baro-offset ADS-B samples):

```
[demo] mode_a: abstained=False best error 19.6 m, score 3.392 (relative fit score (uncalibrated)),
       inferred start 2026-09-08T14:07:13.240306Z (truth 2026-09-08T14:07:13.000000Z)
[demo] mode_b: abstained=False best error 187.4 m, score 4.6, inferred start 2026-09-08T14:07:10Z (truth …13Z)
```

(an earlier run of the same demo with seed 7 gave 48.8 m / 108.6 m; differences come from the
coarse grid phase and the 25 m refinement, both reported as `nominal_grid_spacing_m`).
Figures: `data/runs/demo/mode_a_map.png`, `mode_a_curves_best.png`, `mode_b_*`,
`spectrogram_detected_ridges.png` (copied to `docs/img/` for the README).

Locked-alignment regression (`tests/unit/test_locked_scoring.py`): a flight pair whose second CPA
is 60 s inconsistent with the observed Δt scores 0.6–1.5 z lower than the truth with the shared t0,
while an independent shift of event 2 would recover the full ridge score (the forbidden
procedure would make the wrong pair indistinguishable).

Known degeneracy (documented in `docs/theory.md`): a single straight flyby with known time is
ambiguous along-track by ±σ_clock·v and mirror-ambiguous across the track; the integration test
and the demo use ±5–10 s and check that the truth is among the leading distinct candidates.

## M6 — Productization (done)

* FastAPI (`api/`) with all SPEC §11 routes (+ tracks, heatmap, audit, SSE progress), strict
  pydantic validation, id sanitisation, upload/duration limits, localhost default, no-referrer /
  no-store headers, static serving of the Next.js export.
* Subprocess job runner with persistent status, cooperative cancel + SIGTERM escalation,
  concurrency cap, CPU seconds / peak RSS.
* Next.js 15 static export (`web/`): Import, Spectrogram lab (Plotly heatmap, click-to-draw ridge,
  harmonic overlay, Δt readout), Search (Mode A/B, bbox, resource caps, SSE progress, cancel),
  Results (MapLibre map with tracks/candidates/score landscape, ranked table, curve overlays,
  exports), Data & privacy (purge with inline confirmation, audit log). Basemap tiles off by
  default. SYNTHETIC banners wherever synthetic data is shown.
* Dockerfile + docker-compose (port published on 127.0.0.1 only), GitHub Actions CI (lint, fast
  tests, slow synthetic end-to-end, web build), docs (theory, data sources, privacy).
* Screenshots in `docs/img/ui_*.png` were taken from the running application with Playwright
  (`scripts/screenshots.py`).

## M7 — Validation and performance (in progress; see final section for results)

Issues found and fixed while running the benchmarks (each verified by re-running the case):

1. **Dense real traffic made the CPA stage impractical.** 1,323 real segments × 29k grid points
   with the numpy broadcast CPA took > 45 min. A numba kernel plus 10 s trajectory subsampling
   for the coarse screen (refinement keeps the full track) brought it to 18 s; the subsampled CPA
   differs from the exact one by < 5 m / < 0.1 s on the test track (`test_numba_cpa_kernel…`).
2. **Pair cap bias.** With 10.2 M CPA entries the 200k assignment cap truncated pairs in grid
   order. Pairing is now vectorised and, when capped, keeps the pairs with the best slant-distance
   consistency prior; the run carries a warning.
3. **Detector in multi-aircraft audio.** On a semi-synthetic recording with five real aircraft the
   Viterbi path bridged between aircraft and the transitive harmonic grouping produced one
   1,285 s "mega-event" (and 21 fragments), which also made every curve evaluation 48 ms.
   Fixes: split paths at > 1 s sub-threshold stretches, re-join only across ≤ 1.5 s gaps with
   frequency and slope continuity, base-anchored harmonic groups with time-overlap requirement,
   shape-aware event-pair picking, and ≤ 600 curve samples per event. Result: the two genuine
   flybys (Δt 754 s, matching the truth) are the chosen pair.
4. **False match in synthetic case `long_sep_1100`.** The straight first flight has an along-track
   × clock degeneracy; a straight distractor then fitted the second event's long plateaus nearly
   as well as the true turning flight (rank 1 at 15.5 km, truth rank 2 at 49 m, flagged
   "ambiguous"). Fixes: curve samples weighted by local Doppler slope (plateaus 0.3, transition 1)
   and a 5-point RPM-drift grid. Re-run of the same recording: truth rank 1 at **7 m**, score 4.31
   vs 3.64 for the distractor pair (gap 0.67 z, no longer ambiguous).

**Blocker (environment): Docker image build.** `docker build` fails in this WSL2 environment
because the Docker daemon cannot resolve registry hostnames (`lookup registry-1.docker.io on
10.255.255.254:53: i/o timeout`, same for ghcr.io). The `Dockerfile`/`docker-compose.yml` are
provided and follow the tested local install steps, but the container run has **not** been
verified here. The "clean machine" evidence is instead the GitHub Actions workflow
(`.github/workflows/ci.yml`: uv sync, lint, fast + slow offline tests, web build) and the
documented quick start.

### M7 results — synthetic benchmark (final code, observed)

`uv run aerodoppler benchmark --workers 5 --out data/runs/benchmark_synthetic` (12 scenarios;
`data/runs/benchmark_synthetic/synthetic_benchmark.json`). **SYNTHETIC: simulated audio and
flights; measures the inverse engine, not real-world accuracy.** All scenarios share the same
observer and the same two-aircraft layout family (parameters varied: SNR 0/4/8/10 dB at CPA,
CPA distance 600–4,200 m, speeds 62–110 m/s, separation 300/920/1,100 s, straight vs turning
second flight, an ADS-B gap, 3–8 distractors, headings, RPM drift), so geometric diversity is
limited; the semi-synthetic benchmark below uses real tracks.

| metric | Mode A (unknown time, 2 events) | Mode B (known time ±10 s, 2 events) |
|---|---|---|
| cases answered / abstained | 12 / 0 | 12 / 0 |
| top-1 error median / p90 / max | **33.1 m / 45.0 m / 53.0 m** | 7.3 m / 31.2 m / 69.9 m |
| top-1 within 300 m | 12/12 | 12/12 |
| false matches (> 1 km) | 0 | 0 |
| correct flight pair at rank 1 | 12/12 | 12/12 |
| runtime median / max (5 workers, machine shared with another benchmark) | 162 s / 300 s | 72 s / 141 s |

Per case (Mode A top-1 error, runner-up gap in z): base 31 m (1.18), SNR 4 dB 35 m (1.20),
SNR 0 dB 7 m (1.21), far CPA 30 m (1.01), near/fast 53 m (0.80), separation 300 s 31 m (1.14),
separation 1100 s 37 m (0.42, previously a 15 km false match before slope weighting),
straight second flight 7 m (1.44), ADS-B gap 46 m (1.85), 8 distractors 37 m (1.48),
heading mix 7 m (1.73), RPM drift 35 m (1.07). Errors are bounded by the 25 m final grid plus
the 200 m coarse phase; `nominal_grid_spacing_m` reports the actual resolution.

### M7 results — semi-synthetic benchmark on REAL trajectories (observed)

`aerodoppler benchmark --real-trajectories adsblol_2026-11-02_1e2f2f08 --observers … --max-assignments 2000000`
Simulated audio of every real aircraft within 6 km of the observer (random f0 75–135 Hz,
harmonics, 1/r loss, SNR 10 dB at CPA, 60 Hz hum, gusts) rendered from the **real** ADSB.lol
tracks via the independent simulator; the search sees the whole real day (≈1,250 airborne
segments inside the 36×33 km search box) and the real USGS 3DEP DEM. **Still simulated
acoustics; not a real-recording benchmark.** Cases: three observer sites near Daytona Beach,
Florida (ormond 29.10/−81.15, deland 29.03/−81.25, daytona_w 29.17/−81.10) × seeds.

| case | audible aircraft (≤6 km) | closest passes | result (top-1) | flights | runtime |
|---|---|---|---|---|---|
| ormond_s1 | 5 | 0.5, 1.7, 2.5, 4.3, 4.3 km | **13.3 m** | correct pair | 823 s (60k cap) |
| deland_s1 | – | – | **16.1 m** | correct pair | 708 s (60k cap) |
| daytona_w_s1 | 6 (four within 1.1 km in 10 min) | 0.64, 0.68, 0.67, 1.0, 2.3, 5.7 km | **17.8 km (false match, flagged ambiguous, best score 2.95 vs ≥4 in solved cases)** | wrong | 1,235 s (no cap, 626k pairs) |
| seed-2 cases | see final table below | | | | |

Diagnosis of the daytona failure (kept as a known limitation): with four close flybys inside
ten minutes the auto-detected second event is a composite ridge (its global plateau ratio
1.608 is impossible for a 63 m/s aircraft), and the search re-interprets the events as other
aircraft at a different time of day. The ranked list is correctly flagged `ambiguous` and the
scores are markedly lower than in solved cases, but the system does not abstain. Mitigations
implemented along the way (local excursion window, slope weighting, imbalance/coverage
penalties, all-pairs ultra-coarse screen) fixed the other false matches but not this one.
Proposed next step: joint multi-event Mode A (use all 3–4 detected events at one shared
t0 instead of a pair) and operator-guided event selection in the spectrogram lab, which is
already possible manually.

### Performance (observed on the real-trajectory runs, 6 workers, machine shared)

* segments → trajectories: 1,246 real segments in 12 s; sparse CPA (29k grid × 1,246 tracks,
  10 s subsampled): 17 s; Mode A pairing (vectorised) 626k pairs in 6 s.
* ultra-coarse screen: 626k pairs in 836 s (≈8 ms per pair per worker — dominated by per-
  candidate Python/forward-model overhead, not the numba kernel); coarse: 40k in 297 s; fine:
  400 in 35 s; refinement: 29 s. Peak RSS 3.4 GB (holds 626k assignment objects + 10 M CPA
  entries). With the synthetic demo (5 segments) the whole search takes 1.5–2.5 min.
* Obvious optimisations not done: batch the forward model across candidates sharing a segment,
  drop per-candidate Python objects in the ultra screen (structured arrays), and prune by the
  consistency prior only as a tie-breaker.

### Final test summary (observed, final code)

```
$ AERODOPPLER_OFFLINE=true uv run pytest -q -m "not network" --durations=12
52 passed, 1 deselected, 1 warning in 771.71s (0:12:51)
```
(run concurrently with a 6-worker benchmark; alone the suite takes ~8 min.) Slowest:
```
389.12s call     tests/e2e/test_api.py::test_full_flow
246.22s call     tests/integration/test_search_synthetic.py::test_mode_a_unknown_time_recovers_location_and_clock
41.87s setup    tests/integration/test_search_synthetic.py::test_auto_events_found
35.26s call     tests/integration/test_search_synthetic.py::test_mode_b_known_time_single_event
15.79s setup    tests/e2e/test_api.py::test_health_and_privacy_defaults
8.56s call     tests/unit/test_ridges.py::test_detector_finds_both_events_and_rejects_hum
3.06s call     tests/integration/test_search_synthetic.py::test_missing_dem_is_an_explicit_error
2.56s call     tests/integration/test_adsb_pipeline.py::test_full_local_import_and_query
```
Lint: `uv run ruff check src tests` → All checks passed; `ruff format --check` → formatted.
Network-marked test (3DEP + Copernicus online fetch) passes when run with network access
(`uv run pytest -m network`); it is deselected offline.
