# Background jobs: architecture, API and operations

Everything that can take more than a moment — flight-archive imports, audio decoding,
spectrograms, ridge detection, geolocation searches and benchmarks — runs as a **job**: a row in
the SQLite database plus a separate worker process. The web server only creates and reads job
rows, so it stays responsive while workers are busy, and a page reload or a server restart
neither stops nor duplicates work.

## Why SQLite (and not a message broker)

AeroDoppler is a local, single-machine tool (Linux, macOS, WSL2). SQLite already holds its
state; with WAL mode and `BEGIN IMMEDIATE` transactions it gives atomic claims and state
transitions without another service to install, secure and keep running. The queue sits behind
the `JobQueue` interface (`src/aerodoppler/jobs/queue.py`); the analysis code never talks to
SQLite directly, so a different backend could replace it.

## Moving parts

| piece | file | role |
|---|---|---|
| queue | `jobs/queue.py` | `jobs` and `job_events` tables, state machine, atomic claim/admission, events with sequence ids |
| job types | `jobs/handlers.py` | what each type runs, its resource class, memory reservation, retry and resume rules |
| scheduler | `jobs/scheduler.py` | reconciliation, cancellation escalation, admission, spawning workers |
| worker | `jobs/worker.py` | one process per job: lease heartbeat, budgets, exactly one final state |
| service | `jobs/service.py` | submission, inline (CLI/benchmark) execution, legacy migration |
| API | `api/jobs_routes.py` | `/api/jobs…` endpoints, SSE progress |

The scheduler runs as a thread inside `aerodoppler serve` (set `AERODOPPLER_JOBS_SCHEDULER_IN_API=false`
to disable) or on its own with `aerodoppler jobs scheduler`. A lease in the `scheduler_lock` table
ensures only one scheduler dispatches at a time.

## Life of a job

```
queued ──claim──▶ claimed ──start──▶ running ──▶ succeeded | abstained | failed | cancelled
   │cancel                                │ worker lost
   ▼                                      ▼
cancelled                    interrupted (or queued again: resume-safe types, attempts left)
```

1. **Submit.** `POST` endpoints validate the request first (structured errors naming the field and
   the fix), then insert a `queued` row and answer `202 Accepted` with
   `{job_id, type, status, created_at, status_url}`. An identical request while the first is
   queued or running returns the same job (idempotency key: e.g. provider + day + area for
   imports, the full request for searches). A job whose memory reservation exceeds the whole
   budget is refused immediately with the reason.
2. **Admission.** Every second the scheduler claims the oldest queued jobs that fit:
   per-class concurrency (`import` 1, `analysis` 1 — searches and benchmarks, `light` 2 — audio,
   spectrogram, detection) **and** one shared memory budget (sum of the reservations of all
   claimed/running jobs ≤ `AERODOPPLER_JOBS_MEMORY_BUDGET_MB`, default 70 % of the memory this
   process may use, cgroup/WSL limits included). A job that does not fit for lack of memory keeps
   its reservation, so younger small jobs cannot starve it. Claiming is one guarded `UPDATE`
   inside `BEGIN IMMEDIATE`: concurrent schedulers or submissions can never exceed the limits.
3. **Run.** The scheduler spawns `python -m aerodoppler.jobs.worker <id> <token>` in its own
   process session. The worker starts only if the random token of the claim still matches,
   renews a 30 s lease every 5 s (heartbeat thread), and records memory (PSS of its whole process
   tree), CPU time (counted from the job's start, children that already exited included) and
   free disk. It stops cooperatively when cancelled, when its CPU-time budget
   (`AERODOPPLER_JOBS_MAX_CPU_HOURS`) or the free-disk floor (`AERODOPPLER_JOBS_MIN_FREE_DISK_MB`)
   is exceeded, or when it uses more than 1.5× its memory reservation (a worker that does not
   stop within 10 s of a memory overrun ends its own process group). Imports and searches also
   size their internal parallelism from the reservation and throttle themselves earlier.
4. **Finish.** The worker writes exactly one terminal state, guarded by its token. Transient
   failures (network, rate limit, provider outage) are re-queued with exponential backoff
   (`AERODOPPLER_JOBS_RETRY_BASE_S`, default 30 s, doubling, max 30 min) up to the type's
   `max_attempts`; permanent errors (bad input, missing data) fail at once and are never retried.

## What survives what

| event | what happens |
|---|---|
| browser reload / tab closed | nothing: the job runs in its own process; the page rediscovers active jobs from the server |
| API server stopped or restarted | workers keep running; the next scheduler adopts them (same lease, same process identity) |
| worker crash, `kill -9`, OOM kill | lease expires / process gone → **reconciled**: resume-safe types (flight import, spectrogram) are re-queued while attempts remain; others become **interrupted** with the reason. Rows are never deleted and never keep a slot |
| machine restart | same as a crash, at the first scheduler tick |
| PID reused by another process | not mistaken for the worker: identity = PID + process creation time + the job id on the worker's command line |
| worker alive but hung (no heartbeat for 15 min) | terminated and interrupted |

A "running" row is never taken as proof of a running worker; only a live process with the
recorded identity and a fresh lease is.

Resume semantics per type: **flight_import** writes its dataset atomically (a `.partial`
directory renamed at the end) and resumes downloads from `.part` files after verifying cached parts
by SHA-256, so it is re-run safely. **geolocation_search** writes an `INCOMPLETE` marker into its
output directory until the result is complete; an interrupted search's partial files are never
served as a result, and it is retried explicitly. **ridge_detection** and **audio_ingest** are not
re-run automatically (they would create duplicate events or recordings).

## Cancellation

`POST /api/jobs/{id}/cancel` (or *cancel* on the Jobs page, or `aerodoppler jobs cancel <id>`):

* a **queued** job is cancelled at once; no worker ever starts;
* a **running** job is asked to stop (flag polled within ~1 s by imports/searches, at the next
  heartbeat otherwise); after 20 s the scheduler sends SIGTERM to the worker's process group,
  after 40 s SIGKILL, and the job is finalised as cancelled;
* only the job's own scratch space is removed; downloaded archive parts and datasets are kept.

## CLI and benchmarks (inline jobs)

`aerodoppler flights import`, `aerodoppler search` and `aerodoppler benchmark` run in the
terminal but as **inline jobs**: the same admission (they wait — and say so — while another job
holds the slot or the memory), the same lease and heartbeat, and a final state written in a
`finally` block (Ctrl-C, SIGTERM and exceptions all end in a terminal state). A `kill -9` of the
CLI is reconciled like a worker crash. Searches started inside a benchmark run as *child* jobs
under the benchmark's admission and memory reservation.

## API

| method | path | purpose |
|---|---|---|
| POST | `/api/jobs` | submit `{type, params, idempotency_key?}` (flight_import, geolocation_search, spectrogram_generation, ridge_detection, benchmark) |
| GET | `/api/jobs` | list; filters `type`, `status` (comma list or `active`), `recording_id`, `dataset_id`, `created_after`, `created_before`, `limit`; includes counts, capacity and machine limits |
| GET | `/api/jobs/{id}` | one job (`?debug=true` adds technical error details) |
| POST | `/api/jobs/{id}/cancel` | cancel |
| POST | `/api/jobs/{id}/retry` | re-queue a failed, interrupted or cancelled worker job (same id, history kept) |
| GET | `/api/jobs/{id}/events` | Server-Sent Events; event ids are a global sequence, so reconnecting with `Last-Event-ID` resumes after the last event received. `?format=json&after=<seq>` returns the same events for polling clients |
| GET | `/api/jobs/{id}/logs` | last lines of the worker log |

Endpoints that used to compute synchronously now queue jobs and answer `202` with the job:
`POST /api/recordings` (upload → `audio_ingest`), `POST /api/recordings/{id}/segments`,
`POST /api/recordings/{id}/events/detect` (→ `ridge_detection`; the result holds the event ids and
the detection diagnostics), `GET /api/recordings/{id}/spectrogram` on first access
(→ `spectrogram_generation`), `POST /api/flight-datasets/import`, `POST /api/searches`. Upload,
segment and detect accept `?wait_s=N` to wait up to N seconds and answer `201` with the finished
result (for scripts). `/api/searches/*` keeps its response shapes; `/api/imports` lists import
jobs.

## Migration from earlier versions

On first start the queue copies every row of the old `searches` and `imports` tables into `jobs`
(ids, timestamps, statuses, results kept; the old tables are left untouched). Rows that were
still "running" are checked against their recorded worker process: a live worker is adopted, a
dead one becomes `interrupted`. This is recorded in `schema_migrations` and runs once. The 17
stale "running" rows that blocked new searches in the reported incident end up `interrupted` and
hold no slot.

## Configuration

| variable | default | meaning |
|---|---|---|
| `AERODOPPLER_JOBS_MEMORY_BUDGET_MB` | 70 % of usable RAM | memory all running jobs together may reserve |
| `AERODOPPLER_JOBS_MAX_IMPORT` / `_ANALYSIS` / `_LIGHT` | 1 / 1 / 2 | concurrency per class |
| `AERODOPPLER_IMPORT_MEMORY_BUDGET_MB` | 40 % of the budget, max 4 GB | reservation of one import |
| `AERODOPPLER_SEARCH_MEMORY_BUDGET_MB` | 55 % of the budget, max 8 GB | reservation of one search/benchmark |
| `AERODOPPLER_JOBS_MAX_CPU_HOURS` | 12 | CPU time per job (all its processes) |
| `AERODOPPLER_JOBS_MIN_FREE_DISK_MB` | 2000 | jobs stop below this much free disk |
| `AERODOPPLER_JOBS_RETRY_BASE_S` | 30 | first automatic retry delay |
| `AERODOPPLER_JOBS_SCHEDULER_IN_API` | true | run the scheduler inside `aerodoppler serve` |

On WSL2 the Linux side sees only what Windows grants it (`.wslconfig`, `memory=`); the budget is
derived from what Linux reports, cgroup limits included.

## Diagnosing a stuck job

* **Queued for a long time**: the Jobs page shows the scheduler state (none running → start
  `aerodoppler serve` or `aerodoppler jobs scheduler`), the running job holding the class slot,
  and the reserved vs. total memory. A job waits while an older job's reservation is pending.
* **"stalled?"**: running but no heartbeat for 2 minutes. The worker is busy in native code or
  hung; it is terminated after 15 minutes without heartbeats.
* **interrupted**: the error says in which stage the worker disappeared; *retry* re-runs it
  (imports resume their download). The worker log (`/api/jobs/{id}/logs`, *details → show log*)
  usually shows the last thing it did.
* **Command line**: `aerodoppler jobs list --status active`, `aerodoppler jobs show <id>`.

## Limitations

* Single machine. The scheduler and workers must share the data directory and the SQLite file;
  SQLite on network filesystems is not supported.
* The memory reservation is an estimate per job type (measured for imports and searches); the
  1.5× enforcement and the imports'/searches' own throttling are the backstops. A native library
  allocating faster than one heartbeat (5 s) can still overshoot briefly.
* Long-running parsing and scoring are not checkpointed: an interrupted import re-parses from the
  cached archive (its download is resumed), and an interrupted search starts over on retry.
* Inline CLI jobs are never signalled by the scheduler; cancelling one from the Jobs page takes
  effect at its next heartbeat.
