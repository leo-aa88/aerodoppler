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
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => fetch(`${API_BASE}/health`).then((r) => j<{ status: string; version: string; offline: boolean; authorization_notice: string; host: string }>(r)),
  recordings: () => fetch(`${API_BASE}/api/recordings`).then((r) => j<{ recordings: Recording[] }>(r)),
  recording: (id: string) => fetch(`${API_BASE}/api/recordings/${id}`).then((r) => j<{ recording: Recording; events: DopplerEvent[]; searches: SearchStatus[] }>(r)),
  upload: (form: FormData) => fetch(`${API_BASE}/api/recordings`, { method: "POST", body: form }).then((r) => j<{ recording: Recording; warnings: string[] }>(r)),
  purge: (id: string) => fetch(`${API_BASE}/api/recordings/${id}`, { method: "DELETE" }).then((r) => j<{ recording_id: string; removed: string[] }>(r)),
  spectrogram: (id: string, q: Record<string, string | number>) => {
    const qs = new URLSearchParams(Object.entries(q).map(([k, v]) => [k, String(v)]));
    return fetch(`${API_BASE}/api/recordings/${id}/spectrogram?${qs}`).then((r) => j<SpectrogramData>(r));
  },
  events: (id: string) => fetch(`${API_BASE}/api/recordings/${id}/events`).then((r) => j<{ events: DopplerEvent[] }>(r)),
  addEvent: (id: string, body: { points: [number, number][]; harmonic: number; uncertainty_hz: number; annotated_by: string }) =>
    fetch(`${API_BASE}/api/recordings/${id}/events`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => j<{ events: DopplerEvent[] }>(r)),
  detect: (id: string, body: { threshold_z: number; min_duration_s: number; save: boolean }) =>
    fetch(`${API_BASE}/api/recordings/${id}/events/detect`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => j<{ events: DopplerEvent[] }>(r)),
  deleteEvent: (id: string, ev: string) => fetch(`${API_BASE}/api/recordings/${id}/events/${ev}`, { method: "DELETE" }).then((r) => j<{ deleted: string }>(r)),
  datasets: () => fetch(`${API_BASE}/api/flight-datasets`).then((r) => j<{ datasets: FlightDataset[] }>(r)),
  importFlights: (body: Record<string, unknown>) => fetch(`${API_BASE}/api/flight-datasets/import`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => j<Record<string, unknown>>(r)),
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
