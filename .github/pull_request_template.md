## What this changes

<!-- One or two sentences. Link the issue if there is one: "Closes #123". -->

## Why

## How it was tested

<!-- Paste the commands you ran and their actual output. -->

- [ ] `make lint`
- [ ] `make test-fast`
- [ ] `make test` (slow end-to-end), if search, scoring, detection or the API changed
- [ ] `make demo`, if search or scoring changed (report the Mode A / Mode B errors)

## Effect on results

<!-- For changes to detection, scoring, search or gates: before/after numbers on the synthetic
benchmark or another benchmark, including any cases that got worse. Write "none" otherwise. -->

## Checklist

- [ ] No private recordings, derived spectrograms or private locations are included in code, tests, fixtures or docs.
- [ ] New data files are synthetic or carry a licence and provenance note.
- [ ] Docs (README, `docs/`, `IMPLEMENTATION_REPORT.md`) are updated where behaviour changed.
- [ ] No claim of accuracy or confidence without a measurement behind it.
