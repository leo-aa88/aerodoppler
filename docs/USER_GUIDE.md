# AeroDoppler user guide

AeroDoppler proposes where a stationary recording was made by matching the pitch changes of
propeller aircraft in the audio (Doppler *ridges*) against recorded flight tracks (ADS-B). It runs
entirely on your machine. Analyse only recordings you own or have the owner's permission to
analyse. Results are hypotheses with uncalibrated relative scores, not verified locations.

## 1. Start

```bash
make setup                          # Python environment (uv)
make web-install                    # UI dependencies (Node 22+), once
make serve                          # builds the UI if needed; open http://127.0.0.1:8000
```

`make serve` is the whole application: API, UI and the job scheduler. `make web-dev` (port 3000)
is only for working on the frontend and needs `make serve` running too.

To see it work before using your own data: `make demo` builds a SYNTHETIC recording, flights
and terrain and runs both search modes (about 3 minutes); its results open in the Results page.

## 2. Import a recording (Import page)

Choose the file (WAV, FLAC, MP3, M4A, MP4…; decoded locally with FFmpeg) and say **what you know
about when it was recorded**:

| choice | enter | what it allows |
|---|---|---|
| exact start time | the time of the file's first sample with `Z` (UTC) or an offset, and how many seconds it could be off | known-time and unknown-time searches |
| only the UTC day | the day in UTC (mind midnight: your local date may differ) | unknown-time search (needs two aircraft passes) |
| a range of days | earliest and latest day | each day is a separate ~3.5 GB archive and search |
| only "on or before" | e.g. a video's publication date | nothing yet: the actual day is needed |
| unknown | – | nothing yet |

A publication or upload date is only an upper bound: the recording can be older. Never type a
guessed start time — a wrong time silently matches the wrong aircraft.

Files longer than 2 hours are not rejected: the first 2 hours are analysed and *analyse the next
segment* adds the rest. Event times stay tied to the original file (the offset is recorded).

The upload returns at once; decoding runs as a background job (see §6).

## 3. Import the flight archive of that day

Flight tracks come from ADSB.lol, which publishes **one whole-world archive per UTC day
(≈ 3.5 GB)**. It is downloaded once and cached even if your area is small, because the archive
cannot be subset remotely; then only aircraft inside your area (+15 km) are kept.

* Days appear a few hours after they end (UTC). A day that has not ended, has not been published
  yet, or predates February 2023 gives a clear message; another day is never substituted.
* The **area** limits where a location can be found. If you chose it from prior knowledge (a
  video title, a guess), a result inside it is a *bounded* search, not blind discovery.
* The import is a background job with stage progress (download with throughput, verification,
  trace parsing, segmentation). You can leave or reload the page. Memory is bounded: the importer
  sizes its parallelism from what your machine (WSL/container limits included) allows; a whole
  continental-US day was imported at a 4.6 GB peak.

## 4. Mark aircraft passes (Spectrogram lab)

The first view of a recording computes its spectrogram (a short job). The overview strip shows
the whole recording; click to move the view; ← → pan, + − zoom.

* **Auto-detect** scans the whole recording (whatever the view) and reports what it found and
  every rejected candidate with the reason (too short, too faint, constant tone). Click *show* on
  a rejected candidate to inspect it.
* **Draw a ridge by hand** if you see a slanted line the detector missed: click points along it,
  give its harmonic number, save.
* Every event gets an **evidence label**:
  * *plausible aircraft Doppler* — a descending pitch transition of several percent;
  * *weak Doppler evidence* — usable, but limits the confidence grade to medium;
  * *tonal ridge only* — no measurable transition (hum, machinery, music, or an aircraft heard only
    on a plateau). It cannot locate the recording. If you have checked that it *is* an aircraft,
    *I checked: aircraft* lets a search use it; the grade then stays low.
* **zoom** centres any event with start, closest-approach and end markers.

## 5. Search

The Search page lists what is missing before it lets you run:

* *Unknown clock time*: the UTC day and **two** passes — their time separation replaces the clock.
* *Known start time*: the verified UTC start and its uncertainty; one pass can be enough, but a
  single pass never reaches a high grade.
* Flight data: only archives of the recording's day and kind (real/SYNTHETIC) are selected;
  mixing needs an explicit development override.
* *Effort*: Quick (400 m grid), Standard (200 m, as in all benchmarks), Thorough (100 m, ~4× the
  work). *Estimate the work* counts grid points and flight segments and shows the memory budget
  before anything runs.

## 6. Background jobs (Jobs page)

Imports, decoding, spectrograms, detection, searches and benchmarks are jobs. The navigation bar
shows `N running · M queued · K failed`.

* **Reload / close the browser**: nothing stops; the pages re-attach to running jobs.
* **Server restart**: jobs keep running in their own processes; the restarted server shows them.
* **Crash or out-of-memory kill**: the job becomes *interrupted* with the stage it was in (imports
  resume automatically once). *retry* runs it again; downloaded archive parts are reused.
* **Queued**: at most one import and one search run at a time, plus two light jobs, within one
  memory budget; a job waits (it is not rejected) until it fits.
* **Cancel**: a queued job stops at once; a running one is asked to stop and, if it does not,
  terminated after 20–40 s.
* **details** shows memory, CPU, timings, the result, the error with expandable technical details,
  and the worker log.

Configuration and recovery rules: [`docs/jobs.md`](jobs.md).

## 7. Reading a result

* The page starts with the **verdict**. *LOW — the coordinates are NOT a verified location* means
  exactly that; the failed checks are listed in plain words. *MEDIUM* is a plausible hypothesis.
  *HIGH* means every check passed — still an uncalibrated relative fit, and on single-pass real
  recordings high answers have been hundreds of metres off.
* Scores are relative: differences between candidates matter, absolute values do not. Nothing is a
  probability or a confidence interval. The *score plateau* radius shows how well-pinned an answer
  is, not an error bar.
* *Ambiguous* lists other locations the top answer does not clearly beat.
* The map shows the top hypotheses and the flight tracks they use; other aircraft, the score
  landscape and an OpenStreetMap basemap (which reveals the viewed area to the tile server) are
  optional layers.

## 8. Troubleshooting

| symptom | what to do |
|---|---|
| "API not reachable" | start `make serve`; the address it tried is shown |
| no events detected | open *rejected candidates*; lower min. duration or threshold; draw the ridge by hand |
| "event is a tonal ridge…" when searching | deselect it, or review it in the lab if you are sure |
| "no imported flight archive covers …" | import that UTC day first |
| "archive not published yet" | retry a few hours after the UTC day ended |
| job stays queued | Jobs page: is a scheduler running, which job holds the slot, is memory reserved? |
| WSL freezes or runs out of memory | lower `AERODOPPLER_JOBS_MEMORY_BUDGET_MB`; raise `memory=` in Windows' `.wslconfig` |
| job "interrupted" | read its error (stage) and log; *retry* |

## 9. Glossary

**ridge** a line of energy in the spectrogram; **Doppler transition** the pitch drop as an aircraft
passes; **CPA** closest point of approach; **plateau** the steady pitch before/after a pass;
**unknown-time / known-time** the two search modes; **bounded search** a search inside an area you
chose; **job** a background unit of work with its own process.
