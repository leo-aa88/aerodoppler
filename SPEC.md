# AeroDoppler — End-to-End Technical Specification

**Version:** 1.0 (2 November 2026)  
**Audience:** Claude Code / autonomous coding agent  
**Status:** Implementation specification; scientific and engineering assumptions explicitly labeled  
**Primary use:** Reproducible, consent-based research on aircraft-Doppler acoustic geolocation from recordings owned by or provided with permission from the recording owner.

## 0. Instructions to the implementing agent

Build an actual working, testable, locally runnable application from an empty repository. Do not stop at design documents, scaffolding, API stubs, screenshots, mock results, or an unverified claim of success. Implement the complete vertical slice: flight ingestion, audio ingestion, physical modeling, candidate generation, curve scoring, a usable review interface, and reproducible tests.

Work incrementally, run tests after each milestone, and record actual command outputs in `IMPLEMENTATION_REPORT.md`. When source data or a real recording is unavailable, use clearly labeled synthetic fixtures and describe the limitation; never silently fabricate ADS-B observations, actual recordings, benchmarks, confidence, or accuracy. Prefer simple reliable implementations before optimization. Use the public source documentation listed in section 15 and verify API/dataset formats against current upstream documentation before writing importers.

**Privacy requirements:** This is a local-first research and forensics tool for recordings the operator owns or has permission to analyze. Put a prominent authorization acknowledgement in the UI and CLI. Do not implement bulk scraping of other people's recordings, integration with social-media monitoring, automatic identification of recording owners, or covert tracking. By default do not upload recordings, analyzed locations, or derived spectrograms to external services. Bind the web server to localhost. Add audit metadata, purge controls, resource limits, and an explicit abstention state when evidence is insufficient. National-scale data search is an optional, resource-intensive mode behind an explicit operator confirmation, not the default.

## 1. Problem statement and success criteria

Given audio containing one or more weak aircraft tonal signatures and historical ADS-B aircraft trajectories, estimate the recording microphone's stationary location by comparing observed spectrogram ridges with predicted, propagation-corrected Doppler tracks at candidate coordinates. Support both:

- **Mode A (unknown recording start time):** Known UTC *date*, unknown clock time, two or more detected aircraft events with measured relative separation. Jointly identify the aircraft, approximate clock time, and candidate observer position.
- **Mode B (known recording clock time):** Known UTC start time (with explicit uncertainty), one or more aircraft events. Infer location from flight trajectories matching that time.

A result is a ranked set of **hypotheses**, not a guaranteed exact address. Return coordinates, scoring diagnostics, matched flight IDs, predicted-versus-observed curves, alternative hypotheses, coverage warnings, and a calibrated or clearly labeled *uncalibrated* uncertainty estimate. Never assert “within 100 m” from grid spacing alone. The ~100 m demonstrated in the source video is a case study, not a validated general accuracy distribution.

**Definition of done:** A developer can (a) run `make demo` completely offline with synthetic flights/audio and see a recovered location and plotted curve overlays; (b) ingest a real, explicitly selected historical archive and their own WAV/video; (c) manually annotate or automatically propose Doppler events; (d) run a geographically bounded Mode A or B search; (e) inspect, export, and reproduce ranked results; and (f) run passing unit, integration, and end-to-end tests. Fail clearly when no usable aircraft evidence exists.

## 2. Source-derived algorithm and engineering extrapolations

### Directly described in the Colsto video transcript

1. Obtain historical ADS-B positions, velocities, heights, and aircraft type information.
2. Inspect a high-resolution spectrogram for weak propeller-aircraft Doppler ridges and harmonics.
3. Use observed event separation (920 seconds in the video), broad velocity and slant-distance constraints, and delayed closest-point-of-approach (CPA) times to reduce candidate observer coordinates.
4. Use a 200 m candidate grid, refine promising regions to 100 m, and account for local ground elevation using a DEM.
5. Use lack of an intervening aircraft signature as a *silence constraint*.
6. Predict a Doppler curve using the aircraft's real, non-straight trajectory and **retarded emission time**; naïvely using source velocity at sound-reception time produces the wrong shape.
7. Score modeled curves against the spectrogram, sweeping uncertain event-1 frequency and alignment; lock event-2 temporal alignment to the physical difference between the two modeled aircraft events while permitting its own frequency uncertainty.
8. Refine near the best paired-flight candidate; demonstrate an additional single-event, known-time search; visually verify the example camera site.

### NOT established by the transcript; implement and validate ourselves

Automatic aircraft-ridge detection; precise spectrogram settings; filtering thresholds; objective normalization; handling aircraft RPM drift, wind, atmospheric refraction, microphone clocks, incomplete ADS-B coverage, aircraft metadata, and measurement uncertainties; confidence calibration; true-positive/false-positive rates; general applicability across arbitrary audio; all cost and performance numbers. Do not present any of these as the creator's exact implementation.

## 3. Technology choices

**One Python service, one optional web frontend; no microservices in v1.**

- Python 3.12+, `uv`, `numpy`, `scipy` (`ShortTimeFFT`), `soundfile`, FFmpeg, `polars`, `pyarrow`, `duckdb`, `numba`, `pydantic`, `pyproj`, `rasterio`, `shapely`, `scikit-learn`, `fastapi`, `uvicorn`, and `pytest`.
- Local metadata/job state: SQLite; large flight records: partitioned Parquet queried via DuckDB; rasters: georeferenced GeoTIFF. No required hosted database.
- Frontend: Next.js + TypeScript, Plotly for interactive time-frequency plots and curve overlays, MapLibre GL JS for maps. Frontend is for evidence inspection, manual ridge annotation, search setup, status, and result review.
- Background jobs: straightforward subprocess-based local job manager with persistent status/cancellation and bounded concurrency; add Redis only if a concrete need arises.
- `ruff`, `mypy` where sensible, `pytest`, GitHub Actions, Dockerfile plus `docker compose` optional for local reproducibility. `Makefile` provides the canonical commands.
- GPU acceleration is optional. First make Numba CPU kernels correct and profiled; add CUDA only behind an interchangeable computation interface.

## 4. Repository layout

```text
AeroDoppler/
  README.md
  SPEC.md                         # this specification
  IMPLEMENTATION_REPORT.md
  pyproject.toml
  uv.lock
  Makefile
  .env.example
  src/aerodoppler/
    cli.py
    config.py
    models.py
    audio/{decode.py,spectrogram.py,ridges.py,synthetic.py}
    adsb/{providers.py,adsblol.py,normalize.py,segments.py,index.py}
    geo/{coordinates.py,altitude.py,dem.py,grids.py}
    physics/{propagation.py,doppler.py,trajectory.py}
    search/{preselect.py,events.py,silence.py,score.py,refine.py,rank.py}
    jobs/{store.py,runner.py}
    api/{app.py,schemas.py,routes.py}
    validation/{benchmark.py,metrics.py}
  web/                            # Next.js app
  tests/{unit,integration,e2e,fixtures}/
  scripts/
  data/{raw,normalized,dem,recordings,runs}/ # gitignored
  docs/{theory.md,data-sources.md,validation.md,privacy.md}
```

## 5. Data contracts, units, and coordinate conventions

Use UTC ISO-8601 for absolute timestamps, seconds for recording-relative time, meters and m/s for geometry and velocity, hertz for frequency, and WGS84 `(latitude, longitude)` in API responses. Internally use **ECEF XYZ meters** for observer/source geometry and velocity, or a verified ENU tangent frame for a small region. Geographic `pyproj` transforms MUST specify longitude/latitude axis order explicitly (`always_xy=True`).

Types:

```text
FlightSample:
  source, aircraft_icao24, utc_unix_s, latitude, longitude,
  altitude_value, altitude_unit, altitude_datum,
  groundspeed_mps?, true_track_deg?, vertical_rate_mps?,
  aircraft_type?, position_source?, quality_flags?
FlightSegment:
  stable_id = source + icao24 + segment_start_utc + ordinal,
  first_utc, last_utc, samples[], completeness_score
Recording:
  id, sha256, original_filename, duration_s, sample_rate_hz,
  channels, start_time_utc?, start_time_uncertainty_s?,
  recording_date_utc?, owner_permission_confirmed, created_at
DopplerEvent:
  recording_id, audio_start_s, audio_end_s,
  ridge_points[(relative_t_s, frequency_hz, uncertainty_hz)],
  candidate_harmonic_numbers[], f0_prior?, detection_method,
  detection_quality, annotated_by?
SearchRequest:
  mode, recording_id, event_ids[], date_utc?, time_window_utc?,
  bbox?, operator_authorized, max_cpu_hours?, quality_settings
LocationCandidate:
  lat, lon, ground_elevation_m, flight_segment_ids[],
  inferred_start_utc, global_time_shift_s,
  source_fundamental_hz_per_event[],
  score_total, score_components, flags[], rank,
  nominal_grid_spacing_m, estimated_uncertainty_m?
```

Store all input source URLs, retrieval timestamps, provider versions, checksums, license labels, sample-filter counts, input parameters, git SHA, and random seeds in a reproducibility manifest associated with every run.

**Altitude datum warning:** Do not assume all ADS-B altitudes are “above sea level.” Barometric/pressure altitude, geometric/ellipsoidal GNSS altitude, and DEM geoid/vertical datums differ. Record the datum, prefer reliable geometric altitude when provided, and normalize aircraft and observer to the same vertical reference before computing 3D slant distances. If correction data is unavailable, increase altitude uncertainty and flag results. Never silently subtract incompatible quantities.

## 6. Historical flight ingestion

**Primary provider:** ADSB.lol archived daily readsb data, subject to actual archive availability and ODbL obligations. It exposes daily split archives, containing gzipped JSON trace files and daily heatmaps. Implement a provider interface to allow alternate sources. OpenSky historical Trino access is an optional licensed/approved provider; do not rely on ordinary OpenSky REST for arbitrarily old, national-scale state-vector history.

Required ingestion behavior:

1. Accept explicit UTC date(s), geographical bbox, and storage budget. Download or import source archives with integrity checks, resume/retry, and an informative progress indicator. Never download the whole country silently for a tiny bounded search.
2. Parse readsb traces robustly, with fixture tests from real documented schema. Normalize positions/altitudes/track and preserve original field provenance.
3. Apply date/bbox filters; include a safety margin for nearby aircraft (configurable default 15 km; adapt to source audibility). Merge data that crosses midnight when needed.
4. Split tracks at gaps (initial threshold 30 s, configurable), teleportation/kinematic anomalies, ICAO reuse, or invalid positions. Preserve low-quality tracks with flags rather than pretending they are valid.
5. Interpolate physically plausible 3D positions and velocities: begin with piecewise linear position plus smoothed finite-difference velocity; validate against constant-velocity analytic cases. Upgrade to bounded Hermite interpolation only if justified by tests. Never extrapolate across large gaps.
6. Derive aircraft type from a cacheable metadata lookup when available. Jets can be deprioritized for propeller-style tonal events, but **unknown type does not mean jet** and must not be unconditionally discarded.
7. Write partitioned Parquet by UTC date and region, alongside spatial-temporal indexes. Avoid treating callsign as a unique aircraft/flight key.
8. Report flights ingested, rejected, missing altitude, untracked periods, spatial/temporal coverage, and storage size.

Provide an offline test importer for small fixture files, and an online importer that gracefully handles source downtime or rate limits.

## 7. DEM and altitude module

Support (a) cached local GeoTIFF tiles, (b) USGS 3DEP for US bounding boxes, and optionally (c) Copernicus DEM GLO-30. Query DEM elevation at candidate ground points; use raster tiling and caching rather than performing a web request per search coordinate. Explicitly track vertical reference and geoid transformations. A missing/invalid DEM must produce an actionable error or an uncertainty-aware fallback, not a made-up zero ground elevation. Include terrain at the microphone but do not pretend the source method solves detailed acoustic occlusion, wind, or refraction.

## 8. Audio and spectrogram processing

1. Accept WAV/FLAC/MP3 and video supported by FFmpeg. Preserve original files locally; compute SHA-256. Decode to floating-point PCM, preserve original sample rate and channels in metadata. An analysis copy may be downmixed and resampled with anti-alias filtering (e.g., 2 kHz for 0–900 Hz analysis).
2. Preserve weak tonal energy: optional DC removal and anti-aliasing are fine; automatic speech enhancement, noise gates, destructive high-pass filters, and lossy recompression are OFF in the detection path by default.
3. Use a multiresolution STFT. Suggested initial settings for a downsampled 2 kHz stream: Hann window of 4.096 s (8192 samples), hop 0.25–0.5 s, optional longer-window 8.192 s pass. These are *starting settings*, not source-confirmed values. Expose parameters.
4. Store spectrogram intensity and a noise-normalized/whitened representation (robust local median and MAD over appropriate time-frequency neighborhoods). Preserve absolute time offsets to the original audio.
5. UI: pan/zoom time/frequency, contrast, ridge drawing/editing, harmonic overlay, spectrogram export, ridge export, event separation readout.
6. Automatic detector: find long, smoothly varying tonal ridges using peak extraction plus dynamic programming/Viterbi or graph-based ridge linking; reject stationary equipment hum via temporal-frequency variation; group likely harmonics. Return confidence and editable proposals. Manual annotations MUST work before the automatic detector is declared finished.
7. Allow a source fundamental to drift slowly because aircraft engine RPM may change; incorporate an optional low-order drift model with a penalty so it does not absorb arbitrary geometric mismatch. Treat source pitch as a nuisance parameter, not a known constant.
8. If no stable tonal event is detected, return an explicit abstention reason rather than inferring a location from unrelated noise.

## 9. Physical forward model — must be correct and independently tested

For stationary observer at ECEF position `x`, aircraft trajectory `p(t_e)`, source velocity `v(t_e)`, and effective speed of sound `c`, define:

```text
r(t_e, x) = ||p(t_e) - x||
u(t_e, x) = (p(t_e) - x) / r(t_e, x)
v_toward(t_e, x) = - dot(v(t_e), u(t_e, x))
t_receive = t_emit + r(t_emit, x) / c
f_receive(t_receive) = f_source(t_emit) * c / (c - v_toward(t_emit, x))
```

The sign convention is: approaching aircraft -> `v_toward > 0` -> higher received frequency; receding -> lower. Work in subsonic, finite cases. Use geometric/relative radial velocity, not the magnitude of aircraft velocity. For `c`, initial approximation `331.3 + 0.606*T_C` m/s with a documented default near 343 m/s at 20°C; make temperature configurable and model uncertain c when unknown. Wind and propagation gradients are deferred/uncertainty terms until validated.

Correct algorithm: iterate over emission-time track samples (or solve the implicit relation for a given receive time), compute propagation delay and Doppler frequency from **emission-time** geometry, and interpolate the resulting `(t_receive, f_receive)` onto recording-relative spectrogram coordinates. It is WRONG to use aircraft state at `t_receive` and then apply only a constant time translation. Confirm that `t_receive(t_emit)` remains monotonic for valid subsonic tracks.

Define the delayed CPA as the reception time of an emission near minimum slant distance; distinguish CPA emission UTC, delayed CPA UTC, and the corresponding recording-relative measured event center. Numerically refine the CPA near sampled minima rather than assuming the nearest ADS-B sample is exact.

Required unit tests: analytic straight-line flyby; symmetry/approach/recede frequency sign; zero radial velocity at closest approach; propagation delay; receiver-time versus emission-time discrepancy on curved or accelerating trajectories; clock-shift invariance; altitude-datum sanity; interpolation continuity; finite values for noisy data. Include an **independent** synthetic-audio simulator so tests do not only invert their own exact mathematical implementation (“inverse crime”).

## 10. Candidate search, pruning, and scoring

### 10.1 Inputs and operating modes

Use the manually or automatically measured event time ranges, event separation, frequency ridge samples, coarse f0 estimates, velocity bounds, and CPA-distance bounds. Treat all measured values as uncertain. Mode A searches candidate aircraft pairs for the known date while allowing unknown recording clock time. Mode B searches flights near the provided clock time and its uncertainty. Always start with a small bbox in local development; use regional or CONUS scale only after performance profiling.

### 10.2 Fast pruning (coarse stage)

1. Calculate approximate event-wise velocity/distance plausibility from detected Doppler curves. Keep generous bounds when frequency estimates are uncertain; do not over-prune.
2. Search only flight trajectory segments that overlap relevant date/time/space conditions and their possible acoustic footprint. Build an index for aircraft speed bins, UTC time bins, and spatial cells.
3. Generate 200 m candidate ground-grid points only near viable flight corridors or inferred slant-distance annuli, not as an in-memory grid over the entire CONUS.
4. For every candidate, estimate delayed CPA time of the potentially matching flights (including slant distance and DEM elevation).
5. Mode A: find two-flight candidate pairs where the difference in delayed CPA reception times is within the observed event separation `Δt ± tolerance`; the video's example is `Δt = 920 s`, **not a hard-coded constant**. Mode B: match delayed CPA to the known UTC event time ± uncertainties.
6. Apply approximate event-wise speed and slant-distance bounds and flight type preferences. Keep multiple candidate hypotheses.
7. Optional silence constraint: check *other likely audible flights* between events. Use a **soft, coverage- and audibility-weighted penalty**, not unconditional elimination: ADS-B archives may miss aircraft and some tracked aircraft might not be audible. Log each penalized flight and its evidence.
8. Record candidate counts after every filtering stage and never lose all candidates without explaining why.

### 10.3 Physically accurate curve scoring (expensive stage)

For each surviving candidate and its proposed flight assignments, compute emission-time-correct predicted Doppler ridges; sample a whitened spectrogram along them using bilinear interpolation (frequency and recording time). Support harmonic tracks and noise-floor comparison. A baseline objective is robust normalized energy integrated along predicted curves; optional second objective is weighted robust distance to manually traced ridges. Penalize path length bias, missing signal, obviously impossible f0 drift, and poor event coverage. Avoid comparing raw summed brightness between events of different lengths without normalization.

**Locked two-event alignment (critical):**

- Let flight 1's predicted delayed CPA be at absolute UTC `τ1(x)` and flight 2's at `τ2(x)`.
- Fit **one shared recording-clock origin**, `t0`, using event 1 with a small measurement-alignment sweep. An observed recording-relative time `s` corresponds to absolute UTC `t0 + s`.
- Fit uncertain emission fundamental frequencies `f01` and `f02` separately, each within sensible, data-derived bounds.
- Score event 2 at its physically determined temporal position using that *same* `t0`. **Never independently slide event 2 in time to improve its score**, since doing so destroys the relative-time evidence.
- Compute a combined score and retain per-event scores. Compare alternatives using normalized objective values and rank per flight-pair and across flight-pairs.

For known-time Mode B, constrain `t0` to the supplied start timestamp uncertainty; compare multiple candidate aircraft and observer points using a common methodology. Penalize over-flexible models. Report when several distinct locations score nearly as well.

### 10.4 Spatial refinement

Refine the top independent flight-pair/location basins from 200 m to 100 m, then optionally 25 m or continuous local optimization around the best modes. Recompute terrain and the physical curve for each refined point. Report actual optimizer resolution, geographic uncertainty, and score landscape. Use a score heatmap and top-K spatially distinct maxima rather than one point estimate alone.

### 10.5 Calibration and abstention

Do not convert arbitrary score gaps into fabricated Bayesian confidence. Initially label scores as **relative fit scores (uncalibrated)**. Build a calibration set from independently measured recording locations, dates, and aircraft events. Measure top-1/top-K distance error, success/abstention rate, false matches, coverage, and parameter sensitivity; derive confidence/credible regions only after demonstrated calibration. Auto-abstain if evidence, flight coverage, identifiability, or score separation is insufficient.

## 11. API, CLI, and review UI

### CLI examples (exact command spelling is a proposed interface)

```bash
make setup
make test
make demo
uv run aerodoppler flights import --provider adsblol --date YYYY-MM-DD --bbox WEST,SOUTH,EAST,NORTH
uv run aerodoppler audio ingest ./my_recording.wav --date YYYY-MM-DD --authorized
uv run aerodoppler audio spectrogram <recording-id>
uv run aerodoppler search --recording <id> --mode unknown-time --date YYYY-MM-DD --bbox WEST,SOUTH,EAST,NORTH --events <id1>,<id2> --authorized
uv run aerodoppler search --recording <id> --mode known-time --start-utc YYYY-MM-DDTHH:MM:SSZ --bbox WEST,SOUTH,EAST,NORTH --events <id1> --authorized
uv run aerodoppler results export <run-id> --format json
uv run aerodoppler serve --host 127.0.0.1 --port 8000
```

Use an actual date in commands; `YYYY-MM-DD` is a placeholder. The **video transcript says September 8 but does not specify a year**; do not assume the demonstration occurred in 2026, and do not pretend the transcript includes downloadable original test audio.

### HTTP API

```text
GET  /health
POST /api/recordings                     # permission acknowledgement required
GET  /api/recordings/{id}/spectrogram
POST /api/recordings/{id}/events        # manual/imported ridge annotations
GET  /api/flight-datasets
POST /api/flight-datasets/import
POST /api/searches                     # Mode A or B search configuration
GET  /api/searches/{id}                 # status, stage, counters, errors
GET  /api/searches/{id}/candidates
GET  /api/searches/{id}/candidates/{candidateId}/curves
GET  /api/searches/{id}/events          # optional SSE progress
POST /api/searches/{id}/cancel
DELETE /api/recordings/{id}             # purge original and derivatives
GET  /api/searches/{id}/export          # reproducible JSON/GeoJSON
```

Use request/response Pydantic models, strict input validation, file-size limits, audio duration limits, path traversal protection, bounded jobs, and helpful 4xx errors. Network-exposed deployments require authentication and authorization; default deployment must remain localhost-only. Do not transmit private audio or location data to hosted analytics.

### UI screens

1. **Import:** local recording, declared recording time/date uncertainty, authorization checkbox, flight archive status, bbox/map polygon.
2. **Spectrogram lab:** waveform, time-frequency zoom, ridge proposals, editable manual points, harmonic relationships, event pairing and Δt calculation.
3. **Search:** choose Mode A/B, region, job resource cap, displayed pruning stages/candidate counts, cancellable progress.
4. **Results:** map with top geographically distinct candidate positions, flight trajectories, score heatmap, side-by-side predicted/observed spectrogram overlays, flight ID, inferred time, source data quality, and reasons for abstention.
5. **Export/data controls:** JSON, GeoJSON, PNG of diagnostic plots, manifest, delete/purge.

Use real backend data exclusively; if only synthetic data is loaded, visibly label it **SYNTHETIC DEMO** throughout.

## 12. Algorithmic performance and correctness constraints

- Avoid materializing a dense 200 m grid across the entire United States. Generate near-flight candidate cells on demand, partition by region/time, and use coarse-to-fine pruning.
- Batch ECEF geometric calculations with NumPy/Numba. Precompute interpolated flight segments and coarse time-space bins. Cache DEM tiles, flight segments, and spectrograms with checksums and invalidation.
- Run heavy searches as cancellable jobs, with stage progress, counters, elapsed compute time, peak RSS, estimated remaining work when reliable, and explicit CPU/disk budgets.
- Provide a CPU-only deterministic code path. Results with identical inputs and settings must be reproducible within documented numerical tolerances.
- Handle input datasets that have missing/duplicated timestamps, inconsistent altitudes, low data coverage, wrong aircraft classification, audio clock drift, MP3 artifacts, overlapping aircraft, variable RPM, and unmodeled noise. Errors must be visible, not silently ignored.
- Include an `--offline` mode after caches are populated; `make demo` and full tests must run without web access.

## 13. Validation and measurable acceptance gates

### Unit tests (mandatory)

- Analytic constant-speed straight flyby and comparison to forward model within numerical tolerances.
- Correct Doppler frequency sign, retarded receive-time placement, derivative and monotonicity.
- Degenerate zero-speed and near-CPA tests; realistic finite-valued velocities.
- WGS84↔ECEF round trip, meter/feet and vertical reference handling; consistent ground elevation.
- Interpolation with missing data and deliberately corrupted points.
- STFT frequency/time recovery for known test chirps; ridge extraction precision and recall.
- Locked scoring regression where independent alignment of event 2 would produce an incorrect winner.
- Negative tests: no flights, no ridges, no DEM, weak signal, wrong date, wrong time, multi-modal candidates.

### Integration tests (mandatory)

- Load a small, real-format archived ADS-B trace fixture, normalize to Parquet, index, and query.
- Generate synthetic audio using separate simulator logic for two known trajectories and receiver position; inject background noise, amplitude loss, source-frequency drift, codec artifacts, and non-straight maneuvers.
- Run Mode A without recording clock time and Mode B with clock time; verify synthetic ground-truth recovery to within expected search-grid tolerance **for identifiable scenarios**.
- Run end-to-end locally through API and UI without stubbed search results. Export/reimport a run manifest.

### Real-world benchmark (required to claim practical geolocation accuracy)

Collect at least 10–20 owner-consented recordings at surveyed known coordinates, with UTC synchronization, several flight geometries, and diverse SNRs. Keep coordinates hidden from the inference engine until results are committed. Report sample size, error distributions (median, p90), top-K recall, failure/abstention rate, calibration, and runtime. Include difficult negative cases. Treat a median error below 100 m as an **aspirational goal**, not a predeclared result or guarantee. An evaluator must be able to reproduce metrics from saved manifests.

### Exit criteria for release v1

- `make lint`, `make test`, `make demo`, and an end-to-end run pass on a clean machine.
- The demo visually shows **observed vs predicted curves**, flight paths, predicted receiver point, actual synthetic receiver point, error distance, and data provenance.
- Known-time and unknown-time searches both run successfully on at least one identifiable synthetic case each.
- Manual annotation works and auto-detection returns editable proposals. No “works only if we hand-edit a notebook” result.
- Missing information produces explicit errors or abstention. Privacy defaults and deletion controls work.
- README contains exact quick-start, cost/storage caveats, screenshots taken from the running application, data attribution, license and honest benchmark results.

## 14. Milestone sequence and required evidence

**M0 — Bootstrap:** repository, tooling, config, models, CLI, basic server, tests. Evidence: lint and CI passing.

**M1 — Physics and synthetic oracle:** forward propagation, synthetic flights and audio, analytic tests, spectrogram diagnostics. Evidence: predicted-vs-synthetic Doppler overlays and unit tests.

**M2 — Data:** ADSB.lol local/remote ingest, quality control, normalized Parquet, altitude/DEM, indexes. Evidence: one actual sample archived trace is parsed; provenance manifest and sample statistics.

**M3 — DSP:** upload/decode, STFT, annotation UI, manual events, automatic ridge suggestions. Evidence: synthetic ridge extraction plus real or openly licensed example data labeled accurately.

**M4 — Inverse search:** bounded Mode B, candidate generator, single-event forward-curve score, ranking, map output. Evidence: end-to-end synthetic known-time reconstruction.

**M5 — Full paired-flight search:** Mode A unknown time, event spacing, CPA-delay search, velocity/slant constraints, cautious silence term, locked two-event scoring, hierarchical refinement. Evidence: identifiable synthetic unknown-time recovery and competing-flight negative tests.

**M6 — Productization:** fully connected frontend, local job status/cancel, exports, privacy/data deletion, docs, clean-machine container run. Evidence: browser-based end-to-end scenario.

**M7 — Real validation/performance:** owner-consented field recordings, error histograms, resource profiling, bounded search optimizations and CPU fallback. Evidence: reproducible benchmark report with failures reported.

At each milestone, commit working code and log decisions and deviations. If a milestone encounters an external blocker, implement an honest offline fixture path, document the blocker, and continue without claiming the unavailable integration works. No placeholder return values in production search/score code.

## 15. References and provenance

**Primary case study:** Colsto, *I discovered a sound that leaks your exact location (and it's everywhere)* — https://www.youtube.com/watch?v=UytYrhYLOKI . The implementation sequence above is derived from the user-supplied video transcript, particularly 02:28–07:30 (flight/CPA timing), 08:30–13:23 (velocity, slant distance, silence), 13:47–17:53 (nonlinear trajectory and propagation), and 18:17–24:25 (locked scoring and independent known-time validation). This is a demonstration, not a published benchmark for generalized accuracy.

**Current implementation references** (verify compatibility and licenses before coding):

- ADSB.lol historical dataset: https://www.adsb.lol/docs/open-data/historical
- ADSB.lol archive repository for 2026: https://github.com/adsblol/globe_history_2026
- OpenSky archive/access constraints: https://opensky-network.org/data
- OpenSky REST limitations: https://github.com/openskynetwork/opensky-api/blob/master/docs/free/rest.rst
- USGS Elevation Point Query / 3DEP: https://apps.nationalmap.gov/epqs/
- Copernicus DEM: https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM
- SciPy ShortTimeFFT: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html
- pyproj coordinate conversion: https://pyproj4.github.io/pyproj/latest/api/transformer.html

**Licensing:** ADSB.lol archive data is ODbL 1.0; attribution and share-alike obligations may apply to derived databases. Review the actual terms before redistributing data or building a commercial service. OpenSky historical research access and commercial use are restricted according to their stated terms.

## 16. Final instruction to Claude Code

Start at M0 and execute the milestones in order. Choose the minimal complete vertical slice rather than an elaborate but nonfunctional architecture. Do not claim a capability unless you ran a test exercising it. Keep a running `IMPLEMENTATION_REPORT.md` containing commands, observed outputs, fixture provenance, known deficiencies, and the next executable action. End with exact local startup commands, screenshots or diagnostic artifacts generated by your working system, test summary, benchmark results if available, and a truthful remaining-issues list. Do not substitute a textual description for working code.
