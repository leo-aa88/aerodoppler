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

(earlier runs of the same demo during development gave 48.8 m / 108.6 m and 29.9 m / 73 m;
the **final code** gives Mode A **7.3 m** (score 4.21, inferred start 14:07:12.8 vs truth
14:07:13.0) and Mode B **128 m** (±10 s clock, single straight flyby: along-track/mirror
ambiguity, see docs/theory.md). Differences come from the coarse grid phase and the 25 m
refinement, both reported as `nominal_grid_spacing_m`.)
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
Re-run on 2026-11-04 with the ten-gate version (`docs/synthetic_benchmark_2026-11-04.json`):
accuracy unchanged, grades 9 high / 3 medium; details in the reliability pass below.

### M7 results — semi-synthetic benchmark on REAL trajectories (observed, 2026-11-04, 24 cases)

`aerodoppler benchmark --real-trajectories adsblol_2026-11-02_1e2f2f08 --observers <12 sites> --seeds 1,2 --max-assignments 2000000 --oracle-events --workers 8`
(`data/runs/benchmark_realtraj_wide/real_trajectory_benchmark.json`, copied to
`docs/real_trajectory_benchmark_2026-11-04.json`; code 42ad26f, whose `src/` is identical to the
ten-gate commit). Simulated audio of every real aircraft within 6 km of the observer
(smoothing-spline paths through the real ADS-B positions, random f0 75–135 Hz, harmonics, 1/r
loss, SNR 10 dB at CPA, 60 Hz hum, gusts); the search sees the whole real day (≈1,250 airborne
segments inside the 36×33 km box) and the real USGS 3DEP DEM. Twelve observer sites across the
central-Florida box — the three Daytona Beach / DeLand training-area sites of the first run plus
nine grid points between 28.8–29.4°N and 81.05–81.55°W — two seeds each. Site `n2930w8155` had
only one close pass on both seeds and is reported as a scenario error ("no suitable pair"),
leaving 22 answered cases. **Still simulated acoustics; not a real-recording benchmark.** The
"oracle events" column uses the true flights' ridges traced as manual events (the
operator-annotation workflow) and therefore tests the search alone. Grade and alternatives are
the ten-gate output; "alternatives" counts the distinct maxima ≥ 1 km from the top answer that
the result lists.

| case | audible (≤ 6 km) | Δt (s) | auto events: error / grade / alternatives | oracle events: error / grade / alternatives | oracle: flights |
|---|---|---|---|---|---|
| ormond_s1 | 5 | 755 | 7.6 km / low / 5 | 16.7 km / low / 5 | wrong |
| ormond_s2 | 7 | 1443 | 21.9 km / low / 2 | 16.5 km / low / 0 | wrong |
| deland_s1 | 3 | 1053 | 12.1 km / low / 1 | **55 m** / high / 0 | correct |
| deland_s2 | 6 | 1168 | **15 m** / low / 3 | 16.1 km / low / 1 | wrong |
| daytona_w_s1 | 6 | 547 | 15.0 km / low / 0 | 21.2 km / low / 5 | wrong |
| daytona_w_s2 | 4 | 1466 | 18.0 km / low / 0 | 20.9 km / low / 5 | wrong |
| n2920w8115_s1 | 3 | 786 | **23 m** / low / 1 | **33 m** / medium / 0 | correct |
| n2920w8115_s2 | 3 | 786 | 19.2 km / low / 1 | **23 m** / high / 0 | correct |
| n2910w8105_s1 | 4 | 338 | 10.6 km / low / 2 | 11.0 km / low / 1 | wrong |
| n2910w8105_s2 | 7 | 1051 | **12 m** / medium / 0 | **43 m** / low / 1 | correct |
| n2880w8125_s1 | 2 | 636 | 20.0 km / low / 4 | **37 m** / medium / 0 | correct |
| n2880w8125_s2 | 4 | 1143 | **27 m** / low / 0 | 21.0 km / low / 5 | wrong |
| n2880w8145_s1 | 3 | 318 | 16.9 km / low / 5 | **30 m** / medium / 5 | correct |
| n2880w8145_s2 | 5 | 1160 | **14 m** / medium / 0 | **14 m** / high / 0 | correct |
| n2930w8125_s1 | 5 | 679 | 22.7 km / low / 0 | 21.2 km / low / 3 | wrong |
| n2930w8125_s2 | 4 | 445 | 15.2 km / low / 4 | 21.6 km / low / 0 | wrong |
| n2900w8115_s1 | 4 | 340 | 6.5 km / low / 5 | **95 m** / high / 0 | correct |
| n2900w8115_s2 | 6 | 388 | 15.9 km / low / 0 | **16 m** / medium / 1 | correct |
| n2890w8155_s1 | 3 | 601 | **49 m** / high / 0 | **22 m** / high / 0 | correct |
| n2890w8155_s2 | 3 | 601 | **66 m** / high / 0 | **39 m** / high / 0 | correct |
| n2940w8115_s1 | 2 | 855 | 19.3 km / low / 0 | 360 m / high / 0 | correct |
| n2940w8115_s2 | 3 | 675 | 18.5 km / low / 0 | **31 m** / low / 1 | correct |
| n2930w8155_s1, _s2 | 1 | — | scenario error: no suitable pair | — | — |

| | auto events (22 answered) | oracle events (22 answered) |
|---|---|---|
| top-1 within 300 m / beyond 1 km | 7 (32 %) / 15 (68 %) | 12 (55 %) / 9 (41 %) |
| top-1 median | 13.6 km | 75 m |
| best of top-3 within 300 m | 8 (36 %) | 15 (68 %) |
| correct flight pair at rank 1 | not scored (auto events need not map to the true pair) | 13/22 (59 %) |
| grades high / medium / low | 2 / 2 / 18 | 7 / 4 / 11 |
| errors of the answers graded high | 49 m, 66 m | 14, 22, 23, 39, 55, 95, 360 m |
| confidently wrong (high and > 1 km) / medium and > 1 km | 0 / 0 | 0 / 0 |
| correct (< 300 m) but graded low | 3 | 2 |
| wrong (> 1 km) answers that list ≥ 1 alternative | 9/15 | 7/9 |
| runtime median / max (8 workers, machine shared) | 577 s / 918 s | 458 s / 972 s |

Reading. *Accuracy:* unchanged from the first six cases — with blind event detection in dense
training traffic the engine answers correctly about one time in three and otherwise lands
10–23 km away; with the true ridges traced by hand it is right in just over half the cases and
then to tens of metres (oracle median 75 m, best-of-top-3 median 41 m), so the search is not the
main limitation, the composite/fragmentary automatic events are. The six original cases
reproduce to the displayed precision except `ormond_s1` oracle (25 m → 16.7 km), the documented
near-tie. *Gates (the purpose of this re-run):* across 44 answers there is no confidently wrong
one; all 24 answers beyond 1 km are graded low, and the 9 graded high are all within 360 m of
the truth (the 7 oracle ones with the correct aircraft; auto-event runs do not score the pair),
worst 360 m (`n2940w8115_s1` oracle: radius 594 m, distinct-location margin 0.52 z,
just above the 0.5 threshold). The 16 answered cases from the nine new sites were not looked at
while the two new rules were chosen, so for them this is a blind test. The costs: 5 correct
answers graded low, and one wrong answer (`ormond_s2` oracle, 16.5 km) held at low by a single
gate — `track_support` (the winning `a2f0fc` CPA sits at its track end) — while `core_matches`
passed at 15σ from only 11 valid offsets against a chance mean of 3.75, i.e. one event matched
well almost everywhere, and the margins passed (1.76 z / 0.64 z). Among the wrong answers
`track_support` fails in 24/24, `core_matches` in 21/24, `margin_distinct_location` in 16/24,
`radius` and `margin` in 10/24 each, and the remaining gates in 6 or fewer. Wrong answers list an alternative in
16/24 cases; in the other 8 the funnel kept no distinct rival within the reported set, so a
wrong low-graded answer is still shown alone. Earlier development runs (cubic-spline audio
artefact, before local-excursion and whole-window scoring) had scored 13 m and 16 m on
ormond_s1/deland_s1 with automatic events; they are superseded by this table.

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



### M7 results — REAL recordings: Pathak Livermore set (observed, final code)

Data: Zenodo 23424278 (CC BY 4.0; Praveen Pathak), 40 smartphone recordings of light-aircraft
passages over Livermore, CA, May 2026 — found through the dopplerprint project's README. Used
here: the 29 passages on 2026-05-16/17 (the two days whose ADSB.lol archives were imported,
3.5 GB each; the remaining 11 passages are on four other days and were not run). Three further
passages are trilateration outliers (recorded elsewhere) and were excluded.

Protocol (`aerodoppler benchmark --livermore … --datasets adsblol_2026-05-17_c432e6e8,adsblol_2026-05-16_c432e6e8`;
note that the recording clock comes from the dataset's own UTC-stamped ridge, so this validates
the location search given a ±2 s clock, not clock discovery):
recorder position trilaterated from the published per-passage track offsets (37 inliers, rms
0.06 m against the author's own geometry — i.e. the author's coordinate recovered, not an
independent survey); audio sample-zero UTC recovered by sliding the published UTC ridge over our
spectrogram (that is the only use of the published ridge in 25/29 cases); events detected
**blind** by our detectors (template fit + ridge tracker; in 4 cases neither produced an event
overlapping the passage and the published ridge was used with harmonic assumed 1); known-time
Mode B, ±2 s; one event; 4 km box whose centre is offset from the recorder by a seeded 1–2 km;
the whole day's real ADS-B traffic (26k segments in the 95×90 km import box); real USGS 3DEP
terrain. Runtime ≈ 13 s per case (4 workers).

| metric | value |
|---|---|
| cases | 29 (18 answered, 11 abstained: all weak_fit) |
| answered — top-1 error median / p90 / max | **391 m / 4.4 km / 6.7 km** |
| answered — top-1 within 300 m / false match > 1 km | 44 % / 39 % |
| answered — best of top-3 median / within 300 m | 179 m / 67 % |
| answered — correct aircraft at rank 1 | 83 % (15/18) |
| all 29 incl. abstained candidates — top-1 median / within 500 m / > 1 km | 466 m / 55 % / 38 % |
| all 29 — correct aircraft at rank 1 | 83 % (24/29) |
| confidence grades (answered, 9-gate version of 2026-11-03; superseded by the 10-gate re-run below) | high 5 (median 315 m, 40 % ≤ 300 m, 20 % > 1 km, max 3.4 km), medium 1 (3.1 km), low 12 (median 364 m, 42 % > 1 km) |
| gates failing among the 18 answered | margin_distinct_location 12, radius 9, margin (different flights) 4, core_matches 1, track_support 1 |

Reading: a single straight pass with a ±2 s clock pins the aircraft well (83 % right) but
leaves the mirror ambiguity across the track and an along-track slack, which is why "best of
top-3" (179 m) is much better than top-1 (391 m) and why most errors > 1 km are the mirror
solution. The grades only partly separate good from bad answers on single-pass real
recordings: before the distinct-location margin gate the "high" group contained 7 cases with
29 % beyond 1 km; with it (final version) "high" contains 5 cases with 20 % beyond 1 km and the
mirror cases mostly drop to low — but one 3.4 km answer (a wrong aircraft fitted to a
plateau-less fragment, not a mirror) was still graded high; the 2026-11-04 reliability pass
below traces that to a silent null fallback and removes it. "High" must still not be read as
"within a few hundred metres" on one-pass recordings. The three
self-distrust gates added after DOPPLERGEIST (track support, search completeness, sensitivity)
all passed on these cases and therefore did not change their grades. The abstention threshold (`min_score_z` = 1.2, tuned on synthetic audio) is
conservative for noisy phone audio: 5 of the 11 abstained cases had a top-1 within 300 m.
For comparison, dopplerprint reports 34/40 correct aircraft and a 207 m median with the pass
position supplied; our numbers are with blind event detection and an offset box, so they are
not directly comparable. These are the first real-world numbers for this code and remain far
from the ~100 m aspiration; they are reproducible from `docs/livermore_benchmark_2026-11-03.json`.

| case | category | R_min (m) | event source | grade | top-1 err (m) | best of top-3 (m) | aircraft correct |
|---|---|---|---|---|---|---|---|
| N3012N_R40 | V_varying_H_varying | 2158 | auto | medium | 1500 | 1347 | False |
| N310MG_R23 | V_constant_H_varying | 1685 | auto | high | 3127 | 47 | True |
| N916KT_R22 | V_constant_H_constant | 1048 | auto | low | 153 | 153 | True |
| N7RR_R30 | V_constant_H_varying | 3047 | auto | high | 85 | 85 | True |
| N563FX_R21 | V_constant_H_varying | 2100 | auto | low | 6733 | 5733 | False |
| N958JX_R50 | V_constant_H_varying | 5068 | auto | abstained | 4567 | 3817 | False |
| N8085W_R27 | V_constant_H_varying | 1490 | auto | abstained | 64 | 64 | True |
| N394FX_R20 | V_constant_H_constant | 1758 | published ridge fallback | abstained | 545 | 465 | True |
| N413AK_R05 | V_constant_H_varying | 1332 | auto | high | 466 | 176 | True |
| N413AK_R49 | V_constant_H_constant | 3936 | auto | abstained | 5626 | 1881 | False |
| N4004R_R19 | V_constant_H_varying | 1427 | auto | high | 82 | 82 | True |
| N248LF_R14 | V_constant_H_varying | 3607 | auto | medium | 6746 | 297 | True |
| N67WL_R59 | V_constant_H_varying | 3948 | auto | high | 167 | 167 | True |
| N7PG_R60 | V_constant_H_varying | 4759 | published ridge fallback | abstained | 530 | 401 | True |
| N231DT_R57 | V_varying_H_varying | 2484 | auto | medium | 2121 | 1672 | True |
| N172EL_R35 | V_constant_H_varying | 1373 | auto | medium | 1061 | 163 | True |
| N913TC_R15 | V_constant_H_varying | 1981 | auto | abstained | 50 | 50 | True |
| N788BM_R03 | V_constant_H_constant | 1262 | auto | medium | 67 | 67 | True |
| N748SP_R36 | V_varying_H_constant | 1357 | auto | high | 3389 | 3389 | False |
| N60RS_R11 | V_varying_H_constant | 2817 | auto | medium | 115 | 115 | True |
| N61NG_R37 | V_constant_H_varying | 2786 | auto | abstained | 284 | 134 | True |
| N500MG_R17 | V_constant_H_varying | 5687 | auto | medium | 499 | 499 | True |
| N795MM_R13 | V_constant_H_varying | 4930 | auto | medium | 229 | 229 | True |
| N413AK_R16 | V_constant_H_varying | 1628 | published ridge fallback | abstained | 1459 | 243 | True |
| N456CS_R38 | V_varying_H_constant | 2511 | auto | medium | 181 | 181 | True |
| N9393T_R02 | V_constant_H_varying | 3006 | auto | abstained | 92 | 92 | True |
| N86LP_R01 | V_constant_H_varying | 1595 | auto | high | 315 | 315 | True |
| N621KS_R18 | V_constant_H_varying | 3031 | auto | abstained | 2484 | 343 | True |
| N4040_R12 | V_constant_H_constant | 1024 | published ridge fallback | abstained | 145 | 145 | True |


### Reliability pass 2026-11-04 — gates, not accuracy (observed)

Motivation: the roadmap for v0.2 is reliability (know when the answer is untrustworthy) before
new features. Diagnosing the one Livermore answer graded *high* at 3.4 km (N748SP_R36, wrong
aircraft) showed a bug rather than a tuning problem: the chance-match null drew random clock
offsets of 60–900 s, but the clip is 68 s long, so every null sample fell outside the recording,
the null array was empty and the gate silently compared the score with a placeholder `0.00±1.00`
("2.9σ", passed). A second case (N563FX_R21) had the same placeholder. On most other short
clips the "null" was the score of a curve lying entirely outside the recording (≈ 0 with a tiny
spread, hence the 20–50σ figures in the old details), which is not a chance match either.

Changes (gates only; no search, scoring or detection change):

1. **In-recording chance-match null.** Offsets are drawn within ± the recording length and kept
   only if every event's window stays ≥ 40 % inside the recording while shifted by at least
   max(20 s, half its duration); the sample count and the |offset| band are reported. With
   fewer than 8 valid samples the gate fails with "null NOT estimable (clip too short)" instead
   of passing on a made-up null (`engine._null_offsets`, `rank.gate_results(null_n=…)`).
2. **`event_shape` gate** (high only): an event flagged `plateaus_not_reached`,
   `no_descending_slope` or `non_monotonic_ridge` cannot support a high grade — the fragment
   leaves speed and slant range underdetermined, which is exactly how a C172 at 450 m slant
   out-scored the true C172 at 1.36 km in N748SP_R36. The flag already existed as a warning.
3. **`alternatives`** in `SearchResult` and on the Results screen: distinct locations ≥ 1 km from
   the top-1 whose score is within 0.5 z of it, with gap and flights, so a thin win is shown
   as "one of these" rather than a single spot.

Livermore re-run (`docs/livermore_benchmark_2026-11-04.json`, same protocol, same 29 cases;
locations, errors and abstentions are byte-identical to 2026-11-03 since only gates changed):

| | 2026-11-03 | 2026-11-04 |
|---|---|---|
| high: n / median / max / > 1 km / correct aircraft | 5 / 315 m / 3389 m / 1 / 4 of 5 | **3 / 315 m / 466 m / 0 / 3 of 3** |
| medium | 1 (3.1 km) | 3 (85 m, 3.1 km, 3.4 km) |
| low | 12 (median 364 m) | 12 (median 364 m) |
| null not estimable | — (silent placeholder in 2 cases) | 1 case (N563FX_R21, already low), failed visibly |
| cases listing ≥ 1 alternative | — | 13 of 18 (all low/medium; the 3 high have none) |

Per-case changes: N748SP_R36 high → medium (event_shape; its null is now 0.69±0.24 from 48
offsets of 33–48 s), N7RR_R30 high → medium (event_shape; a correct 85 m answer, the price of
the rule), everything else unchanged. Caveats, stated plainly: both rules were chosen while
looking at these 18 answered cases, so the table is not a blind test of them; and on 60–180 s
clips the valid offsets form a narrow band, so the reported σ values (5–76) are a threshold
test, not a significance level — the band is now printed next to them. The gate thresholds
themselves are unchanged.

Synthetic benchmark re-run with the ten gates (`data/runs/benchmark_synthetic_gates`, 5
workers, machine shared with the dense-traffic benchmark; raw JSON in
`docs/synthetic_benchmark_2026-11-04.json`; SYNTHETIC audio and flights): accuracy unchanged.
Mode A 12/12 answered, top-1 median 31.2 m / p90 45.0 m / max 53.0 m (2026-11-03: 33.1 / 45.0 /
53.0 m), eleven per-case errors identical to the metre and `drifting_rpm` 35.0 → 19.6 m; Mode B
12/12, median 7.3 m, p90 21.4 m (was 31.2 m), max 69.9 m, three cases moved by one or two grid
steps. Grades (Mode A): **high 9, medium 3, low 0**. The in-recording null is a real sample on
these 1,500–1,700 s recordings (16–48 valid offsets of 136–997 s per event; per-event
significances 4.4–311σ); `event_shape` and `core_matches` demoted nothing. The three mediums:
`lowsnr_4` and `verylow_0` fail `no_unheard_close_pass` (silence penalty 0.07 z and 0.70 z: a
distractor passes close while the detector, at 4 and 0 dB SNR, found no ridge for it);
`long_sep_1100` misses `margin` and `margin_distinct_location` by a hair (gap 0.498 z, high
needs 0.5) and lists one alternative 15.5 km away (the pair `ad0001`+`ad0100`, the same
hypothesis that was a 15 km false match before slope weighting). The 2026-11-03 run did not
persist per-case grades (result files then lacked the `gates` field), so there is no
before/after grade table for the synthetic set. Runtime median 289 s / max 442 s (was 162 /
300 s; the 8-worker dense-traffic benchmark was running alongside).

Dense-traffic reproducibility note: re-running the documented `ormond_s1` oracle-events case
(same seed, same scenario, same chosen pair) gave 16.7 km instead of 25 m. The documented win
was a near-tie (true pair 4.176 vs a circuit-flying trainer pair 4.168, 0.008 z); this time
the same two hypotheses scored 4.272 vs 4.320 and the order flipped (fine/refinement scores
vary at the 0.1 z level between runs with different worker counts). The new output says so:
grade low, five alternatives listed with the true pair 0.05 z behind. The 2026-11-03 "oracle
2/6" therefore reads as "1 clear + 1 coin-flip"; the widened 12-site dense-traffic benchmark
in §M7 above replaces that table (its gate details predate the printed offset band, behaviour
identical).

### Overhaul 2026-11-05 — reliability, memory, usability, evidence (observed)

Driven by manual testing on real recordings (a 7,484 s file rejected, a WSL freeze during a
large-area import, 17 stale "running" jobs blocking searches, a flat 329.5 Hz ridge searched as
aircraft evidence, an unreadable map, a null "not estimable" on a 6,380 s clip). What changed and
what was measured; the persistent job *queue* (one scheduler for every job type) is the next
task and is not part of this section.

**Import memory (P0).** The old importer fed `multiprocessing.Pool.imap_unordered` from the tar
stream with no backpressure (the pool's feeder thread reads the whole archive into its task
queue), started `cpu_count − 1` spawned parsers, and kept every parsed frame for one final
`pl.concat`. Now: at most `2 × workers` batches in flight, parsers write Parquet parts straight to
a staging directory, segmentation runs per aircraft bucket of ≤ 3 M samples, the worker count is
derived from a memory budget (default: half of available memory, cgroup/WSL-aware, max 4 GB),
the process tree is watched after every batch (throttle above the budget, stop above 1.5×), and
the dataset is written into `<id>.partial/` and renamed only when complete. Measured with
`scripts/profile_import.py` (whole process tree sampled every 0.5 s, kill ceiling 20 GB) on the
2026-11-03 ADSB.lol day (3.53 GB, 68,672 trace files), 16-CPU / 42 GB WSL2 machine, offline:

| region | before (`2313baf`) | after |
|---|---|---|
| Manhattan (−74.15,40.65,−73.85,40.90) | 61 s, peak 4.39 GB, 15 parsers | 61.5 s, peak 3.50 GB, 9 parsers (default budget) |
| CONUS (−125,24,−66,50) | killed by the 20 GB ceiling after 102 s (21.9 GB and rising) | completed in 389 s, peak 4.62 GB, 11 parsers; 53.7 M samples, 685,631 segments, 1.46 GB Parquet |

The CONUS run used the first per-parser estimate (300 MB) and overshot its 4.1 GB budget by
13 % (throttled, not stopped); the estimate was then raised to 380 MB and the Manhattan re-run
stayed under budget. Output equality: sample and filter counts identical to the old importer;
the segment count differed by 4 of 43,943 because duplicate timestamps were resolved by arrival
order (non-deterministic in both versions) — segmentation now sorts on a full key.

**Archive handling.** Release lookups distinguish future day, not yet published (days appeared
≈ 3.5 h after the UTC day ended: 2026-11-03 at 03:27 UTC on 11-04, 2026-11-04 at 03:28 UTC on
11-05), before the archive (no ADSB.lol repository before 2023; first release 2023-02-16), not
found, rate limited, network/outage, offline without cache, corrupt download, insufficient disk;
no other day is ever substituted. Parts are verified against GitHub's published SHA-256 digests
(cached per size+mtime), a failing part is kept as `.corrupt` and fetched again, downloads resume
from `.part`, free disk is checked first.

**Jobs.** Imports are persistent jobs (`imports` table, detached worker process, heartbeat, PID +
creation time, stage/byte/file progress, cancel, retry, log); a duplicate request returns the
running job; one import at a time; stale rows from dead workers become `interrupted` (never
deleted) at API start-up and before every admission; searches use the same status and a PSS-based
memory budget with a budget-derived worker count. Global `/api/jobs` and a Jobs page. Codex's
search-side fixes (process identity, atomic admission) were kept.

**Long recordings.** Files over the 2 h analysis limit are no longer rejected: the first 2 h are
analysed and labelled, further segments are created from the preserved original (hard link) with
FFmpeg input seeking, and each recording records its offset in the source; UTC is shifted by the
offset. A chirp test checks that a 20–25 s segment of an MP3 decodes at the right time.

**Time metadata.** Recordings record their time basis (verified start, date only, date range,
upper bound only — e.g. a video publication date — or unknown); an uncertainty without a start,
a start without a timezone or without an uncertainty, and inconsistent dates are refused.

**Evidence labels.** Every event is measured (descent around the steepest point, excursion,
plateaus, monotonicity) and labelled plausible aircraft Doppler / weak Doppler evidence / tonal
ridge only. Controls on the events already in the local database: real Livermore phone
recordings 221 plausible, 8 weak, 0 tonal-only; hand-traced true passes (simulated over real
tracks) 66/66 plausible; automatic detections in dense simulated traffic 629 / 457 / 212; the
Central Park 13–62 s ridge: tonal-only (0.44 % change). A second finding: the straight-pass
template detector re-measures points inside a ±4 Hz band that follows its own template, so at
noise level its "measured" ridge is the template — a flat 329.5 Hz test tone produced a
"plausible" 364→327 Hz event. Template events now store per-point levels and are judged only
from points at z ≥ 3; on three synthetic scenarios (0/4/10 dB) all 6 true passes stay plausible
and 2 of 3 spurious template events become tonal-only. Searches refuse tonal-only events unless
the operator reviewed and accepted them; a new `event_evidence` gate keeps such results at low
(weak evidence caps at medium). Detection reports what it scanned (always the whole recording),
every rejected candidate with its reason, and next steps.

**Gates and search integrity.** (1) The chance-match null sampled offsets uniformly over the
whole recording, but scoring only looks at each event's own window, so admissible shifts form
bands of a few tens of seconds (24–59 s for the Central Park event); on a 6,380 s clip ≈ 0.5 %
of draws landed there and none scored validly — "0 valid samples, clip too short" was the wrong
diagnosis. Offsets are now drawn from the admissible set (≥ 1 s apart) and the gate reports the
band and the real reason when it is still not estimable. (2) `clock_lock` no longer passes on a
single event (it trivially agrees with its own fitted clock): single-event results can reach
medium, never high. (3) `search_complete` missed two of the three hard caps (segment cap "exceed
cap", Mode B assignment cap) because it string-matched "exceed the cap"; all caps now mark the
search as not exhaustive. (4) A grid over the point cap raised a raw ValueError; it is now an
explicit input error. (5) Searches never run silently over every imported archive: datasets are
resolved to the recording's day and kind (real/SYNTHETIC); mixing or other days need an explicit
development override; an area outside the data is refused. **Before/after benchmark numbers for
(1)–(3) and the evidence gate are pending** (synthetic and Livermore re-runs started after this
commit); until then the 2026-11-04 tables above describe the previous gate version.

**UI.** Guided Import page (readiness checklist, time-basis choice with live UTC conversion,
segment controls, live import progress that survives reloads), Jobs page, Spectrogram lab
(halo-outlined colour-blind-safe ridges 3–4 px, start/steepest/end markers, whole-recording
overview strip, zoom to event or rejected candidate, frequency presets, colormaps, keyboard pan/
zoom, evidence labels and review, detection diagnostics), Search page (modes enabled by the
recording's time basis, evidence per event, explicit dataset selection, presets, advanced
settings with units and help, checklist that disables Run, dry-run workload estimate), Results
(verdict first: "LOW — the coordinates below are NOT a verified location", failed checks in
plain words, time assumption and bounded-search scope, map with only the hypotheses' tracks by
default, layer toggles, labelled lat/lon grid without a basemap, zoom from table). Fixed: the
spectrogram axis title overlapped the controls; Plotly and MapLibre click handlers accumulated on
every redraw (duplicated drawn points). The consent checkboxes/`--authorized` gate were removed
at the owner's request (fields still accepted, not enforced).

**Not done from the overhaul brief (stated plainly).** Blind search without a geographic prior
(P3) and its held-out evaluation harness are not implemented; the bbox is still required.
Score/margin calibration on independent real recordings, hard-negative sound sets (speech,
music, wind, MP3 artefacts) and an anytime budget-aware replacement for the hard candidate caps
are not done (the caps are now at least reported as incomplete search). Historical recordings
before 2023 get a clear "no archive" message; no non-ADS-B path exists. The UI has no automated
contrast/snapshot tests yet (verified by headless-browser screenshots only). Events detected
before this version have no per-point levels, so old template events keep the lenient label.

### Final test summary (observed, final code)

```
$ AERODOPPLER_OFFLINE=true uv run pytest -q -m "not network" --durations=8
56 passed, 1 deselected, 1 warning in 551.10s (0:09:11)
```
(run while a benchmark and the demo were using the machine). 36 unit, 13 integration/e2e-API/CLI
and 7 slow synthetic end-to-end tests. Lint: `uv run ruff check src tests` → All checks passed;
`ruff format --check` → formatted. The network-marked test (3DEP + Copernicus online fetch)
passed when run with network access (`uv run pytest -m network` → 1 passed, 22.5 s).

## Definition of done — status against SPEC §1 and §13

| criterion | status |
|---|---|
| (a) `make demo` offline with recovered location and curve overlays | done (7.3 m Mode A on the final code; PNGs in `data/runs/demo/`) |
| (b) ingest a real archive and own WAV/video | done (ADSB.lol 2026-11-02 Florida bbox; WAV/FLAC/MP3/video via FFmpeg; real-trace fixtures) |
| (c) manual / automatic Doppler events | done (CLI, API, UI click-to-draw; Viterbi proposals) |
| (d) bounded Mode A / Mode B search | done (CLI, API job, UI) |
| (e) inspect / export / reproduce | done (JSON, GeoJSON, manifest with hashes/URLs/git SHA/seed, zip) |
| (f) unit, integration, e2e tests pass | done (52 offline + 1 network) |
| fail clearly without evidence | done (abstention reasons; auth gate; DEM error; input errors) |
| `make lint`, `make test`, `make demo` on a clean machine | lint/test/demo pass here; CI workflow provided; **Docker build not verified** (daemon DNS) |
| screenshots from the running app in README | done (Playwright, `docs/img/ui_*.png`) |
| real-world benchmark (10–20 consented recordings) | **done on 29 published CC-BY recordings** (Livermore set; derived recorder position): answered top-1 median 391 m, best-of-top-3 179 m, 83 % correct aircraft; see M7 |

## Remaining issues (truthful list)

1. **Real-recording validation is limited.** 29 published single-pass recordings (one site,
   phone audio, known-time mode only) give a top-1 median of ~400–500 m with the mirror
   ambiguity dominating; no two-event real recording has been validated. Scores stay labelled
   "relative fit score (uncalibrated)" and `estimated_uncertainty_m` is a score-landscape
   radius, not a confidence region; the abstention threshold is not calibrated for real audio.
2. **Dense GA traffic defeats two-event evidence.** In the Daytona semi-synthetic case (four
   flybys within 1.1 km in ten minutes) the auto-detected events are composites and the search
   returns a false match (flagged ambiguous, low score). Multi-event Mode A (implemented) did not
   rescue this case; better ridge separation for overlapping aircraft and operator-curated
   events are the next steps. Pattern-work aircraft (repeated circuits) produce comb-like CPA
   sequences that match many two-event separations.
3. **Performance at real traffic density.** ~20 min and 3–4 GB RSS for a 36×33 km box with
   ~1,250 segments (all pairs screened). Batching the forward model across candidates and
   replacing per-candidate Python objects would give an order of magnitude; national-scale
   search is opt-in and not optimised.
4. **Docker image build unverified** in this environment (daemon DNS); compose file publishes
   on localhost only.
5. **Single straight flyby, known time** is inherently along-track/mirror ambiguous (documented).
6. **Propagation physics** is limited to a constant speed of sound (temperature-adjustable);
   wind, refraction and terrain occlusion are uncertainty terms only.
7. **OpenSky provider** not implemented (needs approved access); provider interface exists.
8. **Aircraft type classification** is a heuristic table; unknown types are never excluded.
9. The UI has no authentication and must stay on localhost; SSE progress is a 1 s poll.
10. The confidence grade is a rule-based verdict whose thresholds (2σ, ±4 s, 0.5 z, 1 km,
    ±150 m / ±5 m/s) were set by judgement on synthetic data and the Livermore set, not
    calibrated against outcomes. After the 2026-11-04 reliability pass "high" on the Livermore
    single-pass clips is 3 cases, all the right aircraft, worst 466 m — but 18 answered cases
    cannot calibrate anything, and the two new rules were chosen with these cases in view. A
    blind multi-pass real set (roadmap M8) is the only way to test them honestly.

## Next executable actions

```bash
make setup && make test-fast && make demo        # verify locally (offline)
uv run aerodoppler benchmark --out data/runs/benchmark_synthetic            # synthetic suite
uv run aerodoppler benchmark --manifest docs/benchmark_manifest_example.json  # real recordings (edit first)
```

## Exact local startup

```bash
make setup                      # Python deps (uv)
make web-install && make web-build   # optional UI
make demo                       # offline SYNTHETIC end-to-end, writes data/runs/demo/
make serve                      # http://127.0.0.1:8000 (UI + API docs at /docs)
uv run aerodoppler flights import --provider adsblol --date 2026-11-02 --bbox -81.8,28.6,-80.8,29.6 --confirm-download
uv run aerodoppler audio ingest my.wav --date 2026-11-02 --authorized
uv run aerodoppler audio detect <rec-id>
uv run aerodoppler search --recording <rec-id> --mode unknown-time --date 2026-11-02 --bbox -81.3,28.9,-81.0,29.2 --events <e1>,<e2> --authorized
uv run aerodoppler results export <run-id> --format geojson
```

## Ideas adopted from related open implementations (reviewed 2026-11-03)

Two independent re-implementations of the same technique were reviewed at the user's request:
**mjshiggins/dopplertrack** (Python; validated on colsto's Oyster Bay clip at 43 m and on a
12-hour eagle-cam stream) and **Hexay/dopplerprint** (Rust; 40 real Livermore recordings,
median 207 m with the pass position supplied). What was taken over (re-implemented here, no
code copied):

1. **Score over the whole measured window** (dopplertrack): brightness along the modelled
   curve is averaged over the full event window with uncovered parts counting as zero, so a
   flight that explains only half a sweep earns half the score. Implemented in the scoring
   kernel (`valid` fraction multiplies the mean).
2. **Parametric flyover detector** (dopplertrack's `detect`): sliding grid search of an ideal
   straight-pass curve (f_c, v, d, t_c) with Nelder–Mead refinement and per-frame ridge
   re-measurement (`audio/template_detect.py`). It integrates along the whole sweep and found
   every tested Livermore flyover that the frame-by-frame tracker missed (phone audio in wind,
   ridge 1.5–3 z above background).
3. **Real-recording benchmark data** (dopplerprint): the Pathak Livermore set (Zenodo
   23424278, CC BY 4.0, 40 passages with UTC-stamped ridges and KML tracks). The recorder
   position is not published; following dopplerprint's approach it is trilaterated from the
   published per-passage track offsets (our own implementation: 37 inliers, residual rms
   0.06 m against the author's own geometry; 3 passages recorded elsewhere are excluded), and
   the audio clock is recovered by sliding the published ridge over our spectrogram.
   `aerodoppler benchmark --livermore <dir> --datasets …` (`validation/livermore.py`).
4. **Extra flyovers locked to one clock** (both): implemented as multi-event Mode A
   (`attach_extra_events`), kept opt-in because it regressed a dense-traffic case.

5. **Graded confidence gates** (Dicklesworthstone/franken_doppler_finder, reviewed at the
   user's request): five gates — events beat a chance-match null (the same predicted curves at
   48 random clock offsets in the same recording) by ≥ 2σ with coverage ≥ 40 %; every event's own
   best clock within ±4 s of the shared clock; margin ≥ 0.5 z over the best rival that uses
   *different flights*; no penalised unheard close pass; landscape radius ≤ 1 km → high /
   medium / low (`search/rank.py::gate_results`, shown in `result.json`, CLI and UI). On the
   synthetic demo the verdict is **high** (events 20σ and 6σ above chance, margin 3.2 z, radius
   79 m). Three further self-distrust gates follow DOPPLERGEIST's `geometry.py` /
   `finite_search.py` / `matching.py` (reviewed at the user's request): **track_support** (the
   matched tracks must physically support the fitted CPA — emission CPA ≥ 5 s inside the
   sampled interval, no > 30 s gap within ±60 s of it, no clamped glitch velocity, plausible
   speed; datum reported), **search_complete** (retention accounting: the best score discarded
   at the ultra-coarse and coarse cuts must trail the best kept by ≥ 0.25 z and the pair cap
   must not have been hit, otherwise the funnel may have dropped a comparable competitor), and
   **sensitivity_stable** (the winner is re-refined on a ±100 m / 25 m grid under speed of
   sound ±5 m/s and observer-vs-flight height bias ±150 m; the answer must move < max(300 m,
   4 grid cells) and lose < 1 z; the scenario table is saved in the manifest). High needs all
   eight gates; medium additionally requires track_support. DOPPLERGEIST's subset association
   with explicitly unmatched proposals corresponds to our opt-in multi-event mode. Also from fdop: the template detector now requires both halves of a pass to carry
   evidence (rejects hum, clicks and one-sided plateau pieces), and the silence term weighs an
   unheard aircraft by how much closer it passed than the heard ones. fdop's spatio-temporal
   hash join is conceptually what the Δt pairing here does on (grid point, time bucket) keys;
   its emission-time physics and ADS-B cleanup were already present. fdop's own reported
   results (224 m at the reference site, mixed/frequently wrong elsewhere) match our
   experience that dense or sparse real traffic, not the physics, decides success.

6. **1337isnot1337/DOPPLERGEIST** (Python, 141 stars, reviewed at the user's request): the
   most process-heavy of the four (resumable tiled search, disk-backed accounting of discarded
   candidates, extensive benchmark ledgers) but, by its own status table, "no accepted blind
   real localization"; its automatic-discovery runs on the same 40 Livermore clips all returned
   `needs_input / recording_time_required` because it refuses to search without defensible
   clock evidence. That is a fair policy and it sharpens a caveat on our Livermore numbers:
   our known-time runs take the clock from the dataset's own UTC-stamped ridge (a
   dataset-assisted step, equivalent to a recordist supplying the start time to ±2 s), so they
   validate the location search, not clock discovery. DOPPLERGEIST also surveyed AeroSonicDB
   (Zenodo 8368882) and the NASA small-UAV acoustics set and found neither gives recorder
   coordinates/clock precise enough for a 100 m claim — consistent with our dataset search.
   Taken over as design guidance: keep refusing (abstaining) without a date or clock, keep
   competing explanations visible (top-K distinct maxima, ambiguity flag), and report the
   100/500/1,000 m thresholds (now in the Livermore table). Its joint all-flyover matching under
   one clock corresponds to our opt-in multi-event mode.

Ideas noted but not implemented: fdop's per-(event, flight) locus hypotheses instead of a ground
grid (an optimisation), profile-likelihood ranges on fitted curve parameters, evidence in nats
against a calibrated null for ranking (ours uses the null only in the gates), weather-adjusted
sound speed/wind; ICAO Doc 8643 engine classes (OpenSky mirror) or tar1090-db
instead of the heuristic type table; Open-Meteo weather for speed of sound and uniform wind;
log-frequency spectrogram; per-flight-pair margin reporting; AWS Terrarium terrain tiles as a
third DEM source; a contrastive (spectral-neighbourhood) score, which dopplertrack found did
not help.
