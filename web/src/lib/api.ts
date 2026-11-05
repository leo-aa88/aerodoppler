// Typed client for the local AeroDoppler API. Same-origin by default (FastAPI serves the
// static export); NEXT_PUBLIC_API_BASE overrides for `next dev`.
// `next dev` (port 3000) talks to the local API on 8000 by default; the production build is
// served by that same API, so it uses relative URLs (same origin, any port).
export const API_BASE: string =
  process.env.NEXT_PUBLIC_API_BASE ?? (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "");

export interface Recording {
  id: string;
  sha256: string;
  original_filename: string;
  duration_s: number;
  sample_rate_hz: number;
  channels: number;
  start_time_utc: string | null;
  start_time_uncertainty_s: number | null;
  recording_date_utc: string | null;
  owner_permission_confirmed: boolean;
  created_at: string;
  synthetic: boolean;
  notes: string | null;
  source_duration_s: number | null;
  truncated: boolean;
  segment_start_s: number;
  parent_recording_id: string | null;
  source_start_time_utc: string | null;
  time_basis: "verified_start" | "date_only" | "date_range" | "upper_bound_only" | "unknown";
  date_earliest_utc: string | null;
  date_latest_utc: string | null;
}

/** A background job (see docs/jobs.md): imports, ingest, spectrograms, detection, searches, benchmarks. */
export interface Job {
  job_id: string;
  id: string;
  type: "flight_import" | "audio_ingest" | "spectrogram_generation" | "ridge_detection" | "geolocation_search" | "benchmark" | "diagnostic";
  status: "queued" | "claimed" | "running" | "succeeded" | "abstained" | "failed" | "cancelled" | "interrupted";
  stage: string | null;
  status_url: string;
  created_at: string;
  started_at: string | null;
  updated_at: string;
  finished_at: string | null;
  queue_wait_s: number | null;
  duration_s: number | null;
  params: Record<string, unknown>;
  recording_id: string | null;
  recording?: string | null;
  dataset_id: string | null;
  mode: "worker" | "inline" | "child";
  attempt: number;
  max_attempts: number;
  next_attempt_at: string | null;
  cancel_requested: boolean;
  progress: { stage?: string; stages?: Record<string, Record<string, unknown>> };
  resources: { pss_mb?: number; peak_pss_mb?: number; memory_reserved_mb?: number; cpu_s?: number; disk_free_gb?: number };
  memory_reserved_mb: number;
  resource_class: string;
  error: { code: string | null; message: string | null; details: Record<string, unknown> | null } | null;
  result: Record<string, unknown> | null;
  has_log: boolean;
  stalled: boolean;
  worker_alive: boolean;
  legacy: boolean;
  outcome?: "created" | "reused";
}

export interface JobsOverview {
  jobs: Job[];
  counts: { queued: number; running: number; failed: number; interrupted: number; by_status: Record<string, number> };
  capacity: {
    class_limits: Record<string, number>;
    memory_budget_mb: number;
    memory_reserved_mb: number;
    running_by_class: Record<string, number>;
    scheduler: { active: boolean; owner?: string; pid?: number; note?: string };
  };
  resources: {
    cpu_count: number;
    host_total_mb: number;
    host_available_mb: number;
    cgroup_limit_mb: number | null;
    effective_limit_mb: number;
    effective_available_mb: number;
    wsl: boolean;
    data_dir: string;
    disk_free_gb: number;
  };
}

export const JOB_ACTIVE = new Set(["queued", "claimed", "running"]);

/** Follow a job until it ends: Server-Sent Events (the browser resumes after a dropped
 * connection with Last-Event-ID), falling back to polling every 2 s. */
export function followJob(id: string, onUpdate: (j: Job) => void): () => void {
  let stopped = false;
  let es: EventSource | null = null;
  let timer: ReturnType<typeof setInterval> | null = null;
  const refresh = () =>
    api
      .job(id)
      .then((j) => {
        if (stopped) return;
        onUpdate(j);
        if (!JOB_ACTIVE.has(j.status)) stop();
      })
      .catch(() => undefined);
  const poll = () => {
    if (!timer) timer = setInterval(refresh, 2000);
  };
  const stop = () => {
    stopped = true;
    es?.close();
    if (timer) clearInterval(timer);
  };
  try {
    es = new EventSource(`${API_BASE}/api/jobs/${id}/events`);
    let errors = 0;
    es.onmessage = refresh;
    ["snapshot", "status", "progress", "cancel_requested", "done"].forEach((k) => es?.addEventListener(k, refresh));
    es.onerror = () => {
      errors += 1;
      if (errors > 3) {
        es?.close();
        poll();
      }
    };
  } catch {
    poll();
  }
  refresh();
  return stop;
}

/** Resolve when a job reaches a final state (used after submitting short jobs). */
export function waitJob(id: string, onUpdate?: (j: Job) => void): Promise<Job> {
  return new Promise((resolve) => {
    const stop = followJob(id, (j) => {
      onUpdate?.(j);
      if (!JOB_ACTIVE.has(j.status)) {
        stop();
        resolve(j);
      }
    });
  });
}

/** Error returned by the API: a stable code, one sentence, the field it concerns, the fix. */
export class ApiError extends Error {
  status: number;
  code: string;
  field?: string;
  action?: string;
  constructor(status: number, code: string, message: string, field?: string, action?: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.field = field;
    this.action = action;
  }
  toString(): string {
    return this.action ? `${this.message} → ${this.action}` : this.message;
  }
}

export function errText(e: unknown): string {
  if (e instanceof ApiError) return e.toString();
  if (e instanceof TypeError) return `the local API is not reachable (${e.message}); start it with \`make serve\``;
  return String(e instanceof Error ? e.message : e);
}

export interface RidgePoint {
  relative_t_s: number;
  frequency_hz: number;
  uncertainty_hz: number;
}

export interface DopplerEvent {
  id: string;
  recording_id: string;
  audio_start_s: number;
  audio_end_s: number;
  ridge_points: RidgePoint[];
  candidate_harmonic_numbers: number[];
  f0_prior: number | null;
  detection_method: string;
  detection_quality: number;
  annotated_by: string | null;
  ridge_harmonic: number;
  extra?: Record<string, unknown>;
  assessment?: EventAssessment | null;
}

export interface EventAssessment {
  label: "plausible_doppler" | "weak_doppler" | "tonal_only";
  reasons: string[];
  metrics: {
    duration_s?: number;
    n_points?: number;
    f_mean_hz?: number;
    f_range_hz?: number;
    excursion_pct?: number;
    descent_pct?: number;
    max_slope_hz_per_s?: number;
    steepest_point_s?: number;
    plateaus_reached?: boolean;
    monotonic?: boolean;
    harmonics_seen?: number;
    implied_radial_speed_mps?: number;
  };
  review?: { accepted: boolean; note: string | null; at: string } | null;
}

export interface SearchEstimate {
  datasets: string[];
  dataset_warnings: string[];
  area_km2: number;
  grid_points: number;
  segments_in_window_and_area: number;
  segments_cap: number;
  assignment_cap: number;
  workers: number;
  memory_budget_mb: number;
  within_caps: boolean;
  notes: string[];
  reference_run: string;
}

export interface RejectedCandidate {
  t_start_s: number;
  t_end_s: number;
  f_mean_hz: number;
  f_range_hz: number;
  mean_z: number;
  reasons: string[];
  reason_text: string;
}

export interface DetectionDiagnostics {
  scanned: { t_start_s: number; t_end_s: number; f_min_hz: number; f_max_hz: number; note: string };
  parameters: Record<string, number>;
  proposals: number;
  from_ridge_tracker: number;
  from_template_fit: number;
  evidence_labels: Record<string, number>;
  rejected_counts: Record<string, number>;
  rejected: RejectedCandidate[];
  suggestions: string[];
}

export interface SpectrogramData {
  recording_id: string;
  times: number[];
  freqs: number[];
  values: number[][];
  layer: string;
  params: Record<string, unknown>;
  pooled: number[];
  synthetic: boolean;
}

export interface FlightDataset {
  dataset_id: string;
  provider: string | null;
  date_utc: string | null;
  bbox: number[] | null;
  segments: number | null;
  aircraft: number | null;
  synthetic: boolean;
  license: string | null;
  storage_bytes: number | null;
}

export interface SearchStatus {
  id: string;
  recording_id: string;
  status: string;
  stage: string | null;
  created_at: string;
  updated_at: string;
  progress: { stages?: Record<string, Record<string, unknown>>; elapsed_s?: number; cpu_seconds?: number; rss_mb?: number };
  error: string | null;
  cancel_requested: boolean;
  cpu_seconds: number | null;
  peak_rss_mb: number | null;
  request: Record<string, unknown>;
}

export interface ScoreComponents {
  per_event: number[];
  coverage: number[];
  silence_penalty: number;
  drift_penalty: number;
  type_prior: number;
}

export interface Candidate {
  lat: number;
  lon: number;
  ground_elevation_m: number | null;
  flight_segment_ids: string[];
  inferred_start_utc: string;
  global_time_shift_s: number;
  source_fundamental_hz_per_event: number[];
  score_total: number;
  score_components: ScoreComponents;
  flags: string[];
  rank: number;
  nominal_grid_spacing_m: number;
  estimated_uncertainty_m: number | null;
  uncertainty_label: string;
  candidate_id: string;
  extra: { segments?: { segment_id: string; icao24: string; type: string | null; class: string; callsign: string | null }[]; silence_log?: Record<string, unknown>[]; stage?: string; tau_abs?: string[] };
}

export interface StageCounter {
  stage: string;
  candidates_in: number;
  candidates_out: number;
  note: string;
  elapsed_s: number;
}

export interface Gate {
  name: string;
  passed: boolean;
  detail: string;
}

export interface CandidatesResponse {
  run_id: string;
  status: string;
  candidates: Candidate[];
  stage_counters: StageCounter[];
  abstention: { abstained: boolean; reason: string | null; detail: string | null } | null;
  warnings: string[];
  score_label: string;
  synthetic: boolean;
  confidence_grade?: string;
  gates?: Gate[];
  alternatives?: Alternative[];
}

export interface Alternative {
  lat: number;
  lon: number;
  score_total: number;
  gap_z: number;
  distance_from_top1_m: number;
  icao24: string[];
}

export interface CurveEvent {
  event_id: string;
  segment_id: string;
  f0_hz: number;
  alpha_per_s: number;
  score: number;
  coverage: number;
  t_rel_s: number[];
  doppler_factor: number[];
  slant_range_m: number[];
  predicted_hz_by_harmonic: Record<string, number[]>;
  observed_ridge: { t_rel_s: number[]; frequency_hz: number[]; harmonic: number };
  delayed_cpa_utc: string;
  delayed_cpa_rel_s: number;
}

export interface Curves {
  candidate_id: string;
  t0_abs: number;
  events: CurveEvent[];
}

export interface Track {
  segment_id: string;
  icao24: string;
  type: string | null;
  class: string;
  callsign: string | null;
  used: boolean;
  lat: number[];
  lon: number[];
  alt_hae_m: number[];
  t: number[];
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body: { detail?: unknown } | null = null;
    try {
      body = await res.json();
    } catch {
      /* not JSON */
    }
    const d = body?.detail;
    if (d && typeof d === "object" && "message" in (d as Record<string, unknown>)) {
      const o = d as { code?: string; message: string; field?: string; action?: string };
      throw new ApiError(res.status, o.code ?? "error", o.message, o.field, o.action);
    }
    throw new ApiError(res.status, "error", typeof d === "string" ? d : res.statusText || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => fetch(`${API_BASE}/health`).then((r) => j<{ status: string; version: string; offline: boolean; authorization_notice: string; host: string }>(r)),
  recordings: () => fetch(`${API_BASE}/api/recordings`).then((r) => j<{ recordings: Recording[] }>(r)),
  recording: (id: string) => fetch(`${API_BASE}/api/recordings/${id}`).then((r) => j<{ recording: Recording; events: DopplerEvent[]; searches: SearchStatus[] }>(r)),
  /** Upload; decoding runs as an audio_ingest job (the returned Job). */
  upload: (form: FormData) => fetch(`${API_BASE}/api/recordings`, { method: "POST", body: form }).then((r) => j<Job>(r)),
  purge: (id: string) => fetch(`${API_BASE}/api/recordings/${id}`, { method: "DELETE" }).then((r) => j<{ recording_id: string; removed: string[] }>(r)),
  spectrogram: (id: string, q: Record<string, string | number>, onJob?: (j: Job) => void) => {
    const qs = new URLSearchParams(Object.entries(q).map(([k, v]) => [k, String(v)]));
    // the first view of a recording queues its spectrogram job (202): follow it, then ask again
    const get = (): Promise<SpectrogramData> =>
      fetch(`${API_BASE}/api/recordings/${id}/spectrogram?${qs}`).then(async (r) => {
        if (r.status === 202) {
          const job = (await r.json()) as Job;
          onJob?.(job);
          const done = await waitJob(job.job_id, onJob);
          if (done.status !== "succeeded") throw new ApiError(500, done.error?.code ?? "job_failed", done.error?.message ?? `spectrogram job ${done.status}`);
          return get();
        }
        return j<SpectrogramData>(r);
      });
    return get();
  },
  events: (id: string) => fetch(`${API_BASE}/api/recordings/${id}/events`).then((r) => j<{ events: DopplerEvent[] }>(r)),
  addEvent: (id: string, body: { points: [number, number][]; harmonic: number; uncertainty_hz: number; annotated_by: string }) =>
    fetch(`${API_BASE}/api/recordings/${id}/events`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => j<{ events: DopplerEvent[] }>(r)),
  detect: (id: string, body: { threshold_z: number; min_duration_s: number; save: boolean }) =>
    fetch(`${API_BASE}/api/recordings/${id}/events/detect`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => j<Job>(r)),
  reviewEvent: (id: string, ev: string, accepted: boolean, note: string | null) =>
    fetch(`${API_BASE}/api/recordings/${id}/events/${ev}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ accepted, note }) }).then((r) =>
      j<{ events: DopplerEvent[] }>(r),
    ),
  estimateSearch: (body: Record<string, unknown>) =>
    fetch(`${API_BASE}/api/searches/estimate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => j<SearchEstimate>(r)),
  deleteEvent: (id: string, ev: string) => fetch(`${API_BASE}/api/recordings/${id}/events/${ev}`, { method: "DELETE" }).then((r) => j<{ deleted: string }>(r)),
  datasets: () => fetch(`${API_BASE}/api/flight-datasets`).then((r) => j<{ datasets: FlightDataset[] }>(r)),
  importFlights: (body: Record<string, unknown>) =>
    fetch(`${API_BASE}/api/flight-datasets/import`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) =>
      j<Omit<Job, "outcome"> & { outcome: "created" | "reused" | "exists"; dataset_id: string; job: Job | null }>(r),
    ),
  jobs: (q: Record<string, string> = {}) => fetch(`${API_BASE}/api/jobs?${new URLSearchParams(q)}`).then((r) => j<JobsOverview>(r)),
  job: (id: string, debug = false) => fetch(`${API_BASE}/api/jobs/${id}${debug ? "?debug=true" : ""}`).then((r) => j<Job>(r)),
  jobLogs: (id: string) => fetch(`${API_BASE}/api/jobs/${id}/logs?lines=300`).then((r) => j<{ lines: string[] }>(r)),
  cancelJob: (id: string) => fetch(`${API_BASE}/api/jobs/${id}/cancel`, { method: "POST" }).then((r) => j<Job>(r)),
  retryJob: (id: string) => fetch(`${API_BASE}/api/jobs/${id}/retry`, { method: "POST" }).then((r) => j<Job>(r)),
  createSegment: (id: string, start_s: number, duration_s: number | null) =>
    fetch(`${API_BASE}/api/recordings/${id}/segments`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start_s, duration_s }) }).then((r) =>
      j<Job>(r),
    ),
  createSearch: (body: Record<string, unknown>) => fetch(`${API_BASE}/api/searches`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => j<SearchStatus>(r)),
  searches: (rec?: string) => fetch(`${API_BASE}/api/searches${rec ? `?recording_id=${rec}` : ""}`).then((r) => j<{ searches: SearchStatus[] }>(r)),
  search: (id: string) => fetch(`${API_BASE}/api/searches/${id}`).then((r) => j<SearchStatus>(r)),
  cancel: (id: string) => fetch(`${API_BASE}/api/searches/${id}/cancel`, { method: "POST" }).then((r) => j<SearchStatus>(r)),
  candidates: (id: string) => fetch(`${API_BASE}/api/searches/${id}/candidates`).then((r) => j<CandidatesResponse>(r)),
  curves: (id: string, cand: string) => fetch(`${API_BASE}/api/searches/${id}/candidates/${cand}/curves`).then((r) => j<Curves>(r)),
  tracks: (id: string) => fetch(`${API_BASE}/api/searches/${id}/tracks`).then((r) => j<Track[]>(r)),
  heatmap: (id: string) => fetch(`${API_BASE}/api/searches/${id}/heatmap`).then((r) => j<{ points: [number, number, number, string][] }>(r)),
  exportUrl: (id: string, format: string) => `${API_BASE}/api/searches/${id}/export?format=${format}`,
  audit: () => fetch(`${API_BASE}/api/audit`).then((r) => j<{ entries: Record<string, unknown>[] }>(r)),
  eventsStream: (id: string) => new EventSource(`${API_BASE}/api/searches/${id}/events`),
};

export function fmtNum(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(digits);
}
