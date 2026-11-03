# Validation

This document separates three levels of evidence. Only the third would justify any claim about
real-world geolocation accuracy, and **no real-recording benchmark has been run yet**.

## 1. Unit and integration tests (automated)

`uv run pytest` — see `IMPLEMENTATION_REPORT.md` for the observed outputs.

* Forward model vs closed-form straight flyby, sign conventions, delayed CPA, monotonic
  reception, clock-shift invariance, curved-track emission-vs-reception discrepancy,
  interpolation/gaps/duplicates/NaN, batched CPA.
* Geodesy: WGS84↔ECEF round trip, axis order, ENU frame, EGM96 values, datum uncertainties.
* Synthetic oracle: ridge extracted from the **independent** retarded-time simulator matches the
  forward model built from ADS-B-like samples (p90 < 0.6 Hz straight, < 0.8 Hz turning + drift).
* Detector: two flights + 60 Hz hum → both events, correct harmonic numbering, hum rejected;
  noise-only → no ridges; piece merging; event-pair selection.
* Locked two-event scoring regression.
* Real-schema ADSB.lol fixtures → normalise → segments → Parquet → DuckDB queries; corrupt
  members counted, not fatal.
* DEM: synthetic tile sampling, nodata flags, missing DEM raises, flat fallback flagged; online
  3DEP/Copernicus agreement (network-marked).
* End-to-end synthetic: Mode A recovers location (< 300 m), flight pair, clock origin (< 3 s)
  and f0 (< 1 Hz); Mode B with one event; wrong date / wrong time / missing DEM / unauthorized
  negatives; HTTP API flow with a real subprocess job, exports, purge, path-traversal checks.

## 2. Synthetic and semi-synthetic benchmarks (`aerodoppler benchmark`)

* **Synthetic benchmark** (`data/runs/benchmark_synthetic/synthetic_benchmark.json`): 12
  scenarios varying SNR (0–10 dB at CPA), CPA distance, speed, event separation, straight vs
  turning second flight, an ADS-B gap, more distractors, headings, RPM drift. Metrics: top-1 and
  top-3 error (median, p90), success rate within 300 m, false-match rate (> 1 km), abstention
  rate and reasons, flight-pair correctness, runtime. **Measures the inverse engine on simulated
  audio; not real-world accuracy.**
* **Semi-synthetic real-trajectory benchmark** (`--real-trajectories <dataset>`): simulated audio
  of *real* ADSB.lol tracks (real cadence, manoeuvres, altitude datums) at known observer
  positions in the imported Florida bbox, searched against all 2,372 real airborne segments of
  2026-11-02 with the real USGS 3DEP DEM. Propagation, source spectra and noise are still
  simulated.

Results of both runs are reported in `IMPLEMENTATION_REPORT.md` (section M7) exactly as observed,
including failures and abstentions.

## 3. Real-recording benchmark (required before any accuracy claim) — NOT DONE

Protocol implemented in `aerodoppler benchmark --manifest cases.json`:

```json
{"cases": [{"id": "site01", "audio": "recordings/site01.wav", "owner_consent": true,
            "date_utc": "2026-09-08", "start_utc": null, "start_uncertainty_s": null,
            "bbox": "-75.3,39.8,-74.7,40.2", "datasets": ["adsblol_2026-09-08_xxxxxxxx"],
            "truth_lat": 40.0012, "truth_lon": -75.0034, "truth_icao24": ["a12345", "a67890"],
            "events": null}]}
```

Requirements: 10–20 owner-consented recordings at surveyed coordinates, UTC-synchronised, several
geometries and SNRs, difficult negatives. Truth coordinates are read only after each result is
written. Report sample size, error distribution (median, p90), top-K recall, abstention rate,
calibration and runtime. Until such a run exists every score in this software is labelled
**relative fit score (uncalibrated)** and `estimated_uncertainty_m` is a score-landscape radius,
not a confidence region. A median error below 100 m is an aspiration, not a result.
