# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. `CLAUDE.md` imports this
file, so keep shared instructions here.

## What this is

AeroDoppler estimates where a **stationary recording** was made by matching propeller-aircraft
Doppler ridges in its spectrogram against historical ADS-B trajectories (ADSB.lol). It is
consent-based, local-first research software. `SPEC.md` (also `docs/AeroDoppler_TECHNICAL_SPEC.md`)
is the requirements document; `IMPLEMENTATION_REPORT.md` records decisions, measured results and
known limitations; `docs/theory.md` explains the physics, search and confidence gates.

## Commands

```bash
make setup                 # uv sync --all-extras (Python 3.12, uv)
make lint                  # ruff check + ruff format --check on src, tests, scripts
make format                # ruff format + ruff check --fix
make test-fast             # unit + fixture tests, offline, ~30 s
make test                  # everything incl. slow synthetic end-to-end, ~10 min
make demo                  # offline SYNTHETIC end-to-end run -> data/runs/demo/ (~3 min)
make web-install && make web-build   # Next.js static export -> web/out/
make serve                 # API + UI on http://127.0.0.1:8000
uv run pytest tests/unit/test_physics.py -q          # a single file
AERODOPPLER_OFFLINE=true uv run pytest -q -m "not network"   # what CI runs
```

CI (`.github/workflows/ci.yml`) runs `ruff check src tests`, `ruff format --check src tests`,
the fast tests and the slow tests offline, plus `npm run build` in `web/`. Run `make lint`
before committing; a lint warning that "an unsafe fix is available" is still a failure.

## Layout

- `src/aerodoppler/physics/` — emission-time Doppler forward model, trajectories, CPA (numba).
- `src/aerodoppler/geo/` — WGS84/ECEF/ENU, EGM96 geoid (`geo/data/egm96_15min_cm.npz`), DEM, grids.
- `src/aerodoppler/adsb/` — readsb trace parsing, segments, Parquet/DuckDB index, ADSB.lol provider.
- `src/aerodoppler/audio/` — decode, spectrogram/whitening, Viterbi ridges, template flyover
  detector, independent synthetic simulator (`synthetic.py`), recording service.
- `src/aerodoppler/search/` — event features, candidate funnel (`preselect.py`), scoring
  (`score.py`), silence term, refinement, ranking and confidence gates (`rank.py`), orchestration
  (`engine.py`).
- `src/aerodoppler/api/`, `cli.py` — FastAPI (`api/jobs_routes.py` for `/api/jobs`), Typer CLI.
- `src/aerodoppler/jobs/` — persistent SQLite job queue (`queue.py`), job types (`handlers.py`),
  scheduler (`scheduler.py`), worker process (`worker.py`), submission/inline/migration
  (`service.py`), search execution (`runner.py`), recordings/events store (`store.py`). See
  `docs/jobs.md`. Long work never runs inside a request handler: submit a job instead.
- `src/aerodoppler/validation/` — scenarios, demo, benchmarks (synthetic, real-trajectory, Livermore).
- `web/` — Next.js UI. `tests/unit|integration|e2e/`. `scripts/` — screenshots, release smoke test.

## Invariants — do not break these

- **Physics is emission-time correct:** geometry and Doppler are evaluated at emission time and
  mapped to reception time (`t_r = t_e + r/c`). Never use aircraft state at reception time.
- **One shared recording clock:** in multi-event scoring `t0` is fitted once and every event is
  scored at it. Never let an event slide in time independently (see
  `tests/unit/test_locked_scoring.py`).
- **Vertical datums:** all 3-D geometry uses WGS84 ellipsoidal height. Convert barometric,
  geometric and DEM (orthometric) heights with `geo/altitude.py`; never subtract mixed datums.
- **Scores are uncalibrated relative fit values,** not probabilities. The confidence grade is a
  rule-based verdict. Do not label anything as a probability or a confidence interval.
- **Fail visibly:** missing DEM, no flights, weak evidence → explicit error or abstention, never a
  silent fallback (e.g. never a made-up zero ground elevation).
- **The simulator stays independent** of the forward model (no shared Doppler code), so tests do
  not merely invert their own maths.

## Honesty and evidence

- Never invent or round up results. Report benchmark numbers exactly as observed, including
  failures, and update `IMPLEMENTATION_REPORT.md` when results change.
- Changes to detection, scoring, search or gates need before/after numbers (at least the
  synthetic benchmark: `uv run aerodoppler benchmark --out data/runs/benchmark_synthetic`).
- Synthetic data must be labelled SYNTHETIC wherever it is shown.

## Privacy and scope

- Only work with recordings the owner consented to. Never commit recordings, derived
  spectrograms, or private locations. Fixtures must be synthetic or openly licensed with a
  provenance note.
- Keep the server bound to localhost by default and never add telemetry or uploads of user data.
- Do not add bulk scraping of others' recordings, social-media monitoring, owner identification
  or covert tracking.

## Gotchas

- `/data/` (repo root only) is git-ignored and holds large local caches: ADSB.lol archives
  (~3.5 GB/day), normalised datasets, DEM tiles, recordings and runs. Package data lives in
  `src/aerodoppler/geo/data/`; don't re-introduce an unanchored `data/` ignore rule.
- `src/aerodoppler/web_static/` is the UI bundled for releases; it is git-ignored and included in
  wheels through `[tool.hatch.build] artifacts`. The app finds the UI via `resolve_web_dir()`
  (`$AERODOPPLER_WEB_DIR` → packaged copy → `web/out`).
- MapLibre 6 loads its worker as a separate module. `npm run build`/`dev` copy it to
  `web/public/maplibre-gl-worker.mjs` (git-ignored) and `MapView.tsx` calls `setWorkerUrl`; without
  that the map renders empty. The static-UI route must answer `HEAD` too (Next 16 prefetches).
- `AERODOPPLER_OFFLINE=true` forbids network access; tests that need the network are marked
  `network`. Tests marked `slow` take minutes.
- The version lives only in `src/aerodoppler/__init__.py`; releases stamp it from the git tag
  (see `CONTRIBUTING.md` → Releases). Do not hard-code versions elsewhere.
- Search runs use `multiprocessing` with `fork`; heavy benchmarks on real traffic take 10–25 min
  and several GB of RAM. Forked helpers inherit signal handlers: a handler installed in a job
  process must check `os.getpid()` (multiprocessing stops pools with SIGTERM to the children).
- Job status lives in the `jobs` table; never infer a live worker from a `running` row. Use
  `scheduler.worker_alive` (PID + creation time + command line) and the lease.

## Commits

Short imperative summary line, blank line, then the why. Keep commits focused.
