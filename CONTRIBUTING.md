# Contributing to AeroDoppler

Thanks for your interest. AeroDoppler is research software: correctness, honest measurement and
the privacy of recordists matter more than features. Please read this page before opening an
issue or pull request.

## Ground rules

- **Consent first.** Only work with recordings you own or have the recordist's permission to
  use. Never commit, attach or describe in detail private recordings, spectrograms derived from
  them, or the locations of people who did not consent. Test fixtures must be synthetic or
  openly licensed, with a provenance note (see `tests/fixtures/adsblol/PROVENANCE.md`).
- **Out of scope.** Contributions that add bulk scraping of other people's recordings,
  social-media monitoring, identification of recording owners, covert tracking, or uploads of
  user data to external services will not be accepted. See `docs/privacy.md`.
- **No unmeasured claims.** Do not describe a change as improving accuracy or confidence
  without a benchmark run that shows it. Report regressions as well as gains.
- Be respectful. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting issues

Use the issue forms: **Bug report**, **Feature request**, **Question** or **Real-world result
report**. Security problems go through private reporting, not public issues
(see [SECURITY.md](SECURITY.md)). A wrong location on a real recording is usually a known
limitation rather than a bug; a result report with the setup and the failing gates is the most
useful way to share it.

## Development setup

Requirements: Python 3.12+ via [uv](https://docs.astral.sh/uv/), FFmpeg, and Node 22+ for the UI.

```bash
make setup          # uv sync --all-extras
make test-fast      # unit + fixture tests, offline, about a minute
make lint           # ruff check + ruff format --check
make test           # full suite including slow synthetic end-to-end tests (about 10 minutes)
make demo           # offline synthetic end-to-end run with plots in data/runs/demo/
make web-install && make web-build   # frontend
```

All tests and the demo run with `AERODOPPLER_OFFLINE=true`. Tests that need the network are
marked `network` and skipped offline.

## Making changes

1. Open an issue first for anything larger than a small fix, so the approach can be agreed.
2. Create a branch from `main`.
3. Keep changes focused. Add or update tests with the code: unit tests in `tests/unit/`,
   pipeline tests in `tests/integration/`, API and CLI tests in `tests/e2e/`.
4. Run `make format`, `make lint` and `make test-fast`. Run `make test` and `make demo` when you
   touch detection, scoring, search, gates or the API.
5. Update the docs that describe the behaviour you changed: `README.md`, `docs/theory.md`,
   `docs/validation.md`, and `IMPLEMENTATION_REPORT.md` for results and known limitations.
6. Open a pull request using the template and paste the actual command output.

### Changes to detection, scoring or search

These change results, so they need evidence:

- Run the synthetic benchmark (`uv run aerodoppler benchmark --out data/runs/benchmark_synthetic`)
  before and after, and report Mode A and Mode B errors per case.
- If you have the Livermore data and the matching ADS-B days imported, run
  `uv run aerodoppler benchmark --livermore <dir> --datasets <ids>` as well.
- Keep the physics emission-time correct, and keep the shared recording clock across events:
  an event must never be shifted in time independently of the others.
- Never tune thresholds on the cases you then report as results without saying so.

## Code style

- Python 3.12, formatted and linted with Ruff (`make format`, `make lint`), line length 100.
- SI units throughout: metres, seconds, hertz, m/s; UTC for absolute times; WGS84 lat/lon at the
  API boundary, ECEF or a local ENU frame internally. pyproj transforms use `always_xy=True`.
- Scores are relative fit values, not probabilities; name and document them that way.
- Prefer clear, tested numpy/numba code over clever code. Fail with an explicit error or an
  abstention rather than a silent fallback.

## Commit messages

Use a short imperative summary line ("Add distinct-location margin gate"), then a blank line and
the why, if it is not obvious.

## Releases (maintainers)

Releases are cut by `.github/workflows/release.yml`; git tags are the source of truth.

- **Actions → Release → Run workflow**, choose `patch`, `minor` or `major`. The workflow runs
  lint and the full offline test suite, computes the next `vX.Y.Z` from the latest tag, and
  pushes only the tag (it never commits to `main`).
- Or push a tag yourself: `git tag -a v0.2.0 -m v0.2.0 && git push origin v0.2.0`.

The tag is stamped into `src/aerodoppler/__init__.py` at build time. The wheel (with the web UI
bundled) and the Docker image are smoke-tested by `scripts/smoke_release.sh` before the GitHub
Release and the GHCR image are published. Every pull request runs the same build as a dry run.
Test a local build with:

```bash
cp -r web/out src/aerodoppler/web_static && uv build --out-dir dist
scripts/smoke_release.sh wheel dist/aerodoppler-*.whl "" --demo
```

## Licence

AeroDoppler is released under the [MIT License](LICENSE). By contributing, you agree that your
contributions are licensed under the same terms. Data you add keeps its own licence, which must be
stated: ADSB.lol data is ODbL 1.0, and the Livermore recordings are CC BY 4.0.
