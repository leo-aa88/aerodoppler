"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { api, type DopplerEvent, type FlightDataset, type Recording, type SearchStatus } from "@/lib/api";

function SearchSetup() {
  const params = useSearchParams();
  const rec = params.get("rec") ?? "";
  const preEvents = (params.get("events") ?? "").split(",").filter(Boolean);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [recording, setRecording] = useState<Recording | null>(null);
  const [events, setEvents] = useState<DopplerEvent[]>([]);
  const [datasets, setDatasets] = useState<FlightDataset[]>([]);
  const [searches, setSearches] = useState<SearchStatus[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<"unknown-time" | "known-time">("unknown-time");
  const [sel, setSel] = useState<string[]>(preEvents);
  const [bbox, setBbox] = useState("");
  const [date, setDate] = useState("");
  const [startUtc, setStartUtc] = useState("");
  const [unc, setUnc] = useState("60");
  const [dsSel, setDsSel] = useState<string[]>([]);
  const [auth, setAuth] = useState(false);
  const [largeScale, setLargeScale] = useState(false);
  const [cpuHours, setCpuHours] = useState("1");
  const [grid, setGrid] = useState("200");
  const [tol, setTol] = useState("20");
  const [sweep, setSweep] = useState("20");
  const [workers, setWorkers] = useState("0");
  const [silence, setSilence] = useState(true);
  const [groundElev, setGroundElev] = useState("");
  const [tempC, setTempC] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState<SearchStatus | null>(null);

  useEffect(() => {
    api.recordings().then((r) => setRecordings(r.recordings)).catch(() => undefined);
    api.datasets().then((d) => setDatasets(d.datasets)).catch(() => undefined);
  }, []);
  const refreshSearches = useCallback(() => {
    if (rec) api.searches(rec).then((s) => setSearches(s.searches)).catch(() => undefined);
  }, [rec]);
  useEffect(() => {
    if (!rec) return;
    api.recording(rec)
      .then((r) => {
        setRecording(r.recording);
        setEvents(r.events);
        setSearches(r.searches);
        if (r.recording.recording_date_utc) setDate(r.recording.recording_date_utc);
        if (r.recording.start_time_utc) {
          setStartUtc(r.recording.start_time_utc);
          setMode("known-time");
        }
        if (r.recording.start_time_uncertainty_s) setUnc(String(r.recording.start_time_uncertainty_s));
      })
      .catch((e) => setErr(String(e)));
  }, [rec]);

  useEffect(() => {
    if (!live || !["queued", "running"].includes(live.status)) return;
    const es = api.eventsStream(live.id);
    es.addEventListener("progress", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setLive((l) => (l ? { ...l, status: d.status, stage: d.stage, progress: d.progress, error: d.error } : l));
    });
    es.addEventListener("done", () => {
      es.close();
      refreshSearches();
      api.search(live.id).then(setLive).catch(() => undefined);
    });
    es.onerror = () => es.close();
    return () => es.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live?.id]);

  const useDatasetBbox = (d: FlightDataset) => {
    if (d.bbox) setBbox(d.bbox.map((v) => v.toFixed(4)).join(","));
    if (d.date_utc) setDate(d.date_utc);
  };

  const submit = async () => {
    setErr(null);
    const parts = bbox.split(",").map((s) => parseFloat(s.trim()));
    if (parts.length !== 4 || parts.some(Number.isNaN)) return setErr("bbox must be WEST,SOUTH,EAST,NORTH");
    if (!auth) return setErr("Confirm authorization first.");
    if (mode === "unknown-time" && sel.length < 2) return setErr("Mode A needs two events.");
    if (mode === "known-time" && sel.length < 1) return setErr("Mode B needs at least one event.");
    const q: Record<string, unknown> = { coarse_grid_m: +grid, event_separation_tolerance_s: +tol, alignment_sweep_s: +sweep, workers: +workers, silence_constraint: silence };
    if (groundElev) {
      q.ground_elevation_m = +groundElev;
      q.ground_elevation_sigma_m = 30;
    }
    if (tempC) q.temperature_c = +tempC;
    setBusy(true);
    try {
      const s = await api.createSearch({ mode, recording_id: rec, event_ids: sel, date_utc: mode === "unknown-time" ? date : null, start_utc: mode === "known-time" ? startUtc : null, start_uncertainty_s: mode === "known-time" ? +unc : null, bbox: { west: parts[0], south: parts[1], east: parts[2], north: parts[3] }, flight_dataset_ids: dsSel, operator_authorized: true, national_scale_confirmed: largeScale, max_cpu_hours: +cpuHours, quality_settings: q });
      setLive(s);
      refreshSearches();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!rec) {
    return (
      <div className="panel">
        <h2>Search</h2>
        <p>Select a recording:</p>
        <ul>
          {recordings.map((r) => (
            <li key={r.id}>
              <Link href={`/search/?rec=${r.id}`}>
                {r.synthetic ? "[SYNTHETIC] " : ""}
                {r.original_filename} ({r.id})
              </Link>
            </li>
          ))}
        </ul>
      </div>
    );
  }
  const stages = live?.progress?.stages ?? {};
  return (
    <div>
      {recording?.synthetic && <div className="banner synthetic">SYNTHETIC DEMO DATA</div>}
      {err && <div className="banner danger">{err}</div>}
      <div className="grid2">
        <div className="panel">
          <h2>Search configuration — {recording?.original_filename}</h2>
          <div className="row">
            <label>
              <input type="radio" checked={mode === "unknown-time"} onChange={() => setMode("unknown-time")} /> Mode A · unknown clock time (date known, ≥2 events, Δt lock)
            </label>
            <label>
              <input type="radio" checked={mode === "known-time"} onChange={() => setMode("known-time")} /> Mode B · known start time ± uncertainty (≥1 event)
            </label>
          </div>
          <h3 style={{ marginTop: 10 }}>Events</h3>
          {events.map((ev) => (
            <label key={ev.id} style={{ display: "block" }}>
              <input type="checkbox" checked={sel.includes(ev.id)} onChange={() => setSel((s) => (s.includes(ev.id) ? s.filter((x) => x !== ev.id) : [...s, ev.id]))} /> <code>{ev.id}</code> {ev.audio_start_s.toFixed(0)}–{ev.audio_end_s.toFixed(0)} s · f0≈{ev.f0_prior?.toFixed(1) ?? "?"} Hz · h={ev.ridge_harmonic} · {ev.detection_method}
            </label>
          ))}
          {!events.length && (
            <p className="muted">
              no events — <Link href={`/lab/?rec=${rec}`}>annotate or auto-detect in the spectrogram lab</Link>
            </p>
          )}
          <div className="row" style={{ marginTop: 10 }}>
            <div className="col" style={{ minWidth: 280 }}>
              <label>bbox WEST,SOUTH,EAST,NORTH (keep it small: &lt; 40 000 km² unless confirmed)</label>
              <input value={bbox} onChange={(e) => setBbox(e.target.value)} placeholder="-75.25,39.82,-74.75,40.18" />
            </div>
            {mode === "unknown-time" ? (
              <div className="col">
                <label>UTC date</label>
                <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
              </div>
            ) : (
              <>
                <div className="col">
                  <label>UTC start</label>
                  <input value={startUtc} onChange={(e) => setStartUtc(e.target.value)} placeholder="YYYY-MM-DDTHH:MM:SSZ" />
                </div>
                <div className="col">
                  <label>± uncertainty (s)</label>
                  <input type="number" value={unc} onChange={(e) => setUnc(e.target.value)} />
                </div>
              </>
            )}
          </div>
          <h3 style={{ marginTop: 10 }}>Flight datasets (none selected = all)</h3>
          {datasets.map((d) => (
            <div key={d.dataset_id} className="row small">
              <label>
                <input type="checkbox" checked={dsSel.includes(d.dataset_id)} onChange={() => setDsSel((s) => (s.includes(d.dataset_id) ? s.filter((x) => x !== d.dataset_id) : [...s, d.dataset_id]))} /> {d.synthetic && <span className="tag synthetic">SYNTHETIC</span>}
                <code>{d.dataset_id}</code> {d.date_utc} · {d.segments} segments
              </label>
              <button onClick={() => useDatasetBbox(d)}>use its bbox/date</button>
            </div>
          ))}
          <h3 style={{ marginTop: 10 }}>Resources & quality</h3>
          <div className="row">
            <div className="col">
              <label>coarse grid (m)</label>
              <input type="number" value={grid} onChange={(e) => setGrid(e.target.value)} />
            </div>
            <div className="col">
              <label>Δt tolerance (s)</label>
              <input type="number" value={tol} onChange={(e) => setTol(e.target.value)} />
            </div>
            <div className="col">
              <label>alignment sweep ± (s)</label>
              <input type="number" value={sweep} onChange={(e) => setSweep(e.target.value)} />
            </div>
            <div className="col">
              <label>workers (0=auto)</label>
              <input type="number" value={workers} onChange={(e) => setWorkers(e.target.value)} />
            </div>
            <div className="col">
              <label>CPU budget (h)</label>
              <input type="number" value={cpuHours} onChange={(e) => setCpuHours(e.target.value)} />
            </div>
            <div className="col">
              <label>air temperature °C (blank = 20 ± unknown)</label>
              <input type="number" value={tempC} onChange={(e) => setTempC(e.target.value)} />
            </div>
            <div className="col">
              <label>flat ground elevation m (only if no DEM)</label>
              <input type="number" value={groundElev} onChange={(e) => setGroundElev(e.target.value)} />
            </div>
            <label>
              <input type="checkbox" checked={silence} onChange={(e) => setSilence(e.target.checked)} /> soft silence term
            </label>
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <label>
              <input type="checkbox" checked={auth} onChange={(e) => setAuth(e.target.checked)} /> <b>I am authorized to analyse this recording and locate where it was made.</b>
            </label>
            <label>
              <input type="checkbox" checked={largeScale} onChange={(e) => setLargeScale(e.target.checked)} /> confirm large-scale (regional/national) search — slow and resource intensive
            </label>
            <button className="primary" disabled={busy || !auth} onClick={submit}>
              {busy ? "submitting…" : "Run search"}
            </button>
          </div>
        </div>
        <div className="panel">
          <h2>Job status</h2>
          {live ? (
            <div>
              <div className="kv">
                <div>run</div>
                <div>
                  <code>{live.id}</code> <Link href={`/results/?run=${live.id}`}>→ results</Link>
                </div>
                <div>status</div>
                <div className={`status ${live.status}`}>{live.status}</div>
                <div>stage</div>
                <div>{live.stage}</div>
                <div>elapsed</div>
                <div>{live.progress?.elapsed_s ?? 0} s · CPU {live.progress?.cpu_seconds ?? "–"} s · RSS {live.progress?.rss_mb ?? "–"} MB</div>
                {live.error && (
                  <>
                    <div>error</div>
                    <div className="status failed">{live.error}</div>
                  </>
                )}
              </div>
              <div className="row" style={{ marginTop: 8 }}>
                <button className="danger" disabled={!["queued", "running"].includes(live.status)} onClick={() => api.cancel(live.id).then(setLive).catch((e) => setErr(String(e)))}>
                  Cancel
                </button>
              </div>
              <h3 style={{ marginTop: 10 }}>Stages & candidate counts</h3>
              <table>
                <tbody>
                  {Object.entries(stages).map(([k, v]) => (
                    <tr key={k}>
                      <td>
                        <code>{k}</code>
                      </td>
                      <td className="small">{JSON.stringify(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">no job launched yet</p>
          )}
          <h3 style={{ marginTop: 14 }}>Previous searches for this recording</h3>
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>mode</th>
                <th>status</th>
                <th>created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {searches.map((s) => (
                <tr key={s.id}>
                  <td>
                    <code>{s.id}</code>
                  </td>
                  <td>{String(s.request.mode)}</td>
                  <td className={`status ${s.status}`}>{s.status}</td>
                  <td className="small">{s.created_at}</td>
                  <td>
                    <Link href={`/results/?run=${s.id}`}>results →</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={<div className="muted">loading…</div>}>
      <SearchSetup />
    </Suspense>
  );
}
