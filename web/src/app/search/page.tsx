"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api, errText, type DopplerEvent, type FlightDataset, type Recording, type SearchEstimate, type SearchStatus } from "@/lib/api";

type Mode = "unknown-time" | "known-time";

const PRESETS: Record<string, { label: string; grid: number; tol: number; sweep: number; note: string }> = {
  quick: { label: "Quick check", grid: 400, tol: 20, sweep: 20, note: "400 m grid: about 4× less work than Standard; good to see whether anything matches at all" },
  standard: { label: "Standard", grid: 200, tol: 20, sweep: 20, note: "200 m grid, the setting all benchmarks use" },
  thorough: { label: "Thorough", grid: 100, tol: 30, sweep: 30, note: "100 m grid: about 4× more work and memory than Standard" },
};

function Help({ children }: { children: React.ReactNode }) {
  return (
    <details className="help">
      <summary>?</summary>
      <div>{children}</div>
    </details>
  );
}

function bboxArea(b: number[]): number {
  const [w, s, e, n] = b;
  const lat = ((s + n) / 2) * (Math.PI / 180);
  return Math.abs(n - s) * 111.32 * Math.abs(e - w) * 111.32 * Math.cos(lat);
}

function SearchSetup() {
  const params = useSearchParams();
  const rec = params.get("rec") ?? "";
  const preEvents = (params.get("events") ?? "").split(",").filter(Boolean);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [recording, setRecording] = useState<Recording | null>(null);
  const [events, setEvents] = useState<DopplerEvent[]>([]);
  const [datasets, setDatasets] = useState<FlightDataset[]>([]);
  const [searches, setSearches] = useState<SearchStatus[]>([]);
  const [err, setErr] = useState<ApiError | string | null>(null);
  const [mode, setMode] = useState<Mode>("unknown-time");
  const [sel, setSel] = useState<string[]>(preEvents);
  const [bbox, setBbox] = useState("");
  const [date, setDate] = useState("");
  const [startUtc, setStartUtc] = useState("");
  const [unc, setUnc] = useState("");
  const [dsSel, setDsSel] = useState<string[]>([]);
  const [dsTouched, setDsTouched] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [devOverrides, setDevOverrides] = useState(false);
  const [preset, setPreset] = useState("standard");
  const [largeScale, setLargeScale] = useState(false);
  const [grid, setGrid] = useState("200");
  const [tol, setTol] = useState("20");
  const [sweep, setSweep] = useState("20");
  const [workers, setWorkers] = useState("0");
  const [silence, setSilence] = useState(true);
  const [groundElev, setGroundElev] = useState("");
  const [tempC, setTempC] = useState("");
  const [busy, setBusy] = useState(false);
  const [live, setLive] = useState<SearchStatus | null>(null);
  const [estimate, setEstimate] = useState<SearchEstimate | null>(null);

  useEffect(() => {
    api.recordings().then((r) => setRecordings(r.recordings)).catch(() => undefined);
    api.datasets().then((d) => setDatasets(d.datasets)).catch(() => undefined);
  }, []);
  const refreshSearches = useCallback(() => {
    if (rec) api.searches(rec).then((s) => setSearches(s.searches)).catch(() => undefined);
  }, [rec]);
  useEffect(() => {
    if (!rec) return;
    api
      .recording(rec)
      .then((r) => {
        setRecording(r.recording);
        setSearches(r.searches);
        if (r.recording.recording_date_utc) setDate(r.recording.recording_date_utc);
        // the recording's own verified metadata, never an example value
        if (r.recording.start_time_utc) {
          setStartUtc(r.recording.start_time_utc);
          setMode("known-time");
        }
        if (r.recording.start_time_uncertainty_s != null) setUnc(String(r.recording.start_time_uncertainty_s));
      })
      .catch((e) => setErr(errText(e)));
    api.events(rec).then((e) => setEvents(e.events)).catch(() => undefined);
  }, [rec]);

  const applyPreset = (k: string) => {
    setPreset(k);
    setGrid(String(PRESETS[k].grid));
    setTol(String(PRESETS[k].tol));
    setSweep(String(PRESETS[k].sweep));
  };

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

  // the UTC day(s) the recording can be on, for dataset compatibility
  const searchDay = mode === "known-time" ? (startUtc ? startUtc.slice(0, 10) : null) : date || null;
  const compatible = useMemo(
    () => datasets.filter((d) => recording && d.synthetic === recording.synthetic && searchDay !== null && d.date_utc === searchDay),
    [datasets, recording, searchDay],
  );
  // default selection: the date-compatible archives of the same kind (never "all")
  useEffect(() => {
    if (!dsTouched) setDsSel(compatible.map((d) => d.dataset_id));
  }, [compatible, dsTouched]);
  useEffect(() => {
    if (!bbox && compatible.length === 1 && compatible[0].bbox) setBbox(compatible[0].bbox.map((v) => v.toFixed(4)).join(","));
  }, [compatible, bbox]);

  const bboxParts = bbox.split(",").map((s) => parseFloat(s.trim()));
  const bboxOk = bboxParts.length === 4 && !bboxParts.some(Number.isNaN) && bboxParts[0] < bboxParts[2] && bboxParts[1] < bboxParts[3];
  const area = bboxOk ? bboxArea(bboxParts) : 0;
  const selEvents = events.filter((e) => sel.includes(e.id));
  const usable = (e: DopplerEvent) => !e.assessment || e.assessment.label !== "tonal_only" || !!e.assessment.review?.accepted;
  const startOk = /(Z|[+-]\d{2}:?\d{2})$/i.test(startUtc.trim()) && !Number.isNaN(Date.parse(startUtc));
  const selDs = datasets.filter((d) => dsSel.includes(d.dataset_id));
  const mixed = selDs.some((d) => recording && d.synthetic !== recording.synthetic);
  const wrongDate = selDs.some((d) => searchDay && d.date_utc !== searchDay);

  const checks: { ok: boolean; text: string }[] = recording
    ? [
        mode === "known-time"
          ? { ok: startOk, text: startOk ? `start time ${startUtc}` : "known-time mode needs the verified UTC start time (with Z or an offset)" }
          : { ok: !!date, text: date ? `UTC day ${date}` : "unknown-time mode needs the UTC day of the recording" },
        ...(mode === "known-time" ? [{ ok: unc !== "" && +unc >= 0, text: unc !== "" ? `start-time uncertainty ±${unc} s` : "enter how many seconds the start time could be off" }] : []),
        mode === "unknown-time"
          ? { ok: selEvents.length >= 2, text: selEvents.length >= 2 ? `${selEvents.length} events selected` : `select two events (two aircraft passes) — ${selEvents.length} selected` }
          : { ok: selEvents.length >= 1, text: selEvents.length >= 1 ? `${selEvents.length} event(s) selected` : "select at least one event" },
        {
          ok: selEvents.every(usable),
          text: selEvents.every(usable)
            ? "selected events show a Doppler transition (or were reviewed)"
            : `event(s) ${selEvents.filter((e) => !usable(e)).map((e) => e.id).join(", ")} are tonal ridges without a Doppler transition: deselect or review them in the lab`,
        },
        { ok: bboxOk, text: bboxOk ? `search area ${area.toFixed(0)} km²` : "enter the search area (WEST,SOUTH,EAST,NORTH)" },
        {
          ok: selDs.length > 0 && ((!mixed && !wrongDate) || devOverrides),
          text:
            selDs.length === 0
              ? `no flight archive selected${searchDay ? ` — import ${searchDay} first` : ""}`
              : mixed
                ? "SYNTHETIC and real data are mixed (only allowed with development overrides)"
                : wrongDate
                  ? `a selected archive is not from ${searchDay} (only allowed with development overrides)`
                  : `${selDs.length} flight archive(s) of ${searchDay}`,
        },
        { ok: area <= 40000 || largeScale, text: area > 40000 ? "area above 40,000 km²: confirm the large search below" : "area size acceptable" },
      ]
    : [];
  const ready = checks.length > 0 && checks.every((c) => c.ok);

  const body = () => {
    const q: Record<string, unknown> = { coarse_grid_m: +grid, event_separation_tolerance_s: +tol, alignment_sweep_s: +sweep, workers: +workers, silence_constraint: silence };
    if (groundElev) {
      q.ground_elevation_m = +groundElev;
      q.ground_elevation_sigma_m = 30;
    }
    if (tempC) q.temperature_c = +tempC;
    return {
      mode,
      recording_id: rec,
      event_ids: sel,
      date_utc: mode === "unknown-time" ? date : null,
      start_utc: mode === "known-time" ? startUtc : null,
      start_uncertainty_s: mode === "known-time" && unc !== "" ? +unc : null,
      bbox: { west: bboxParts[0], south: bboxParts[1], east: bboxParts[2], north: bboxParts[3] },
      flight_dataset_ids: dsSel,
      national_scale_confirmed: largeScale,
      allow_date_mismatch: devOverrides && wrongDate,
      allow_synthetic_mix: devOverrides && mixed,
      quality_settings: q,
    };
  };

  const runEstimate = async () => {
    setErr(null);
    try {
      setEstimate(await api.estimateSearch(body()));
    } catch (e) {
      setErr(e instanceof ApiError ? e : errText(e));
    }
  };

  const submit = async () => {
    setErr(null);
    setBusy(true);
    try {
      const s = await api.createSearch(body());
      setLive(s);
      refreshSearches();
    } catch (e) {
      setErr(e instanceof ApiError ? e : errText(e));
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
  const timeBasis = recording?.time_basis;
  return (
    <div>
      {recording?.synthetic && <div className="banner synthetic">SYNTHETIC DEMO DATA</div>}
      {err && (
        <div className="banner danger">
          {typeof err === "string" ? err : err.message}
          {typeof err !== "string" && err.action && <div className="small">→ {err.action}</div>}
        </div>
      )}
      <div className="grid2">
        <div className="panel">
          <h2>Search — {recording?.original_filename}</h2>
          <div className="radio-list">
            <label>
              <input type="radio" checked={mode === "unknown-time"} onChange={() => setMode("unknown-time")} /> <b>Unknown clock time</b>: the UTC day is known, the time of day is not. Needs two
              aircraft passes; their time separation replaces the clock.
            </label>
            <label>
              <input type="radio" checked={mode === "known-time"} onChange={() => setMode("known-time")} disabled={timeBasis !== "verified_start" && !startUtc} /> <b>Known start time</b>: the
              recording&apos;s UTC start is known (± seconds). One pass can be enough, but one pass alone never reaches a high confidence grade.
              {timeBasis !== "verified_start" && <span className="small muted"> (this recording has no verified start time)</span>}
            </label>
          </div>
          {(timeBasis === "upper_bound_only" || timeBasis === "date_range" || timeBasis === "unknown") && (
            <div className="banner warn small">
              This recording&apos;s day is {timeBasis === "unknown" ? "unknown" : timeBasis === "upper_bound_only" ? `only known to be on or before ${recording?.date_latest_utc}` : `between ${recording?.date_earliest_utc} and ${recording?.date_latest_utc}`}. A search compares it
              with one UTC day of flights at a time; choose a day only if you have evidence for it — a guessed day mostly produces confident-looking false matches.
            </div>
          )}
          <h3 style={{ marginTop: 10 }}>Events</h3>
          {events.map((ev, i) => {
            const a = ev.assessment;
            return (
              <label key={ev.id} className="event-pick">
                <input type="checkbox" checked={sel.includes(ev.id)} onChange={() => setSel((s) => (s.includes(ev.id) ? s.filter((x) => x !== ev.id) : [...s, ev.id]))} /> #{i + 1}{" "}
                {ev.audio_start_s.toFixed(0)}–{ev.audio_end_s.toFixed(0)} s · f0 ≈ {ev.f0_prior?.toFixed(1) ?? "?"} Hz{" "}
                {a && (
                  <span className={`tag ${a.label === "plausible_doppler" ? "ev-good" : a.label === "weak_doppler" ? "ev-weak" : "ev-bad"}`}>
                    {a.label === "plausible_doppler" ? "plausible aircraft Doppler" : a.label === "weak_doppler" ? "weak Doppler evidence" : "tonal ridge only"}
                    {a.review?.accepted ? " · reviewed" : ""}
                  </span>
                )}
              </label>
            );
          })}
          {!events.length && (
            <p className="muted">
              no events — <Link href={`/lab/?rec=${rec}`}>detect or draw them in the spectrogram lab</Link>
            </p>
          )}
          <div className="row" style={{ marginTop: 10 }}>
            {mode === "unknown-time" ? (
              <div className="col">
                <label>UTC day</label>
                <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
              </div>
            ) : (
              <>
                <div className="col" style={{ minWidth: 230 }}>
                  <label>
                    UTC start (Z or offset) <Help>The time of the first sample of this recording. Taken from the recording&apos;s metadata when it was imported with one.</Help>
                  </label>
                  <input value={startUtc} onChange={(e) => setStartUtc(e.target.value)} />
                </div>
                <div className="col">
                  <label>± seconds</label>
                  <input type="number" min={0} value={unc} onChange={(e) => setUnc(e.target.value)} />
                </div>
              </>
            )}
            <div className="col" style={{ minWidth: 260 }}>
              <label>
                search area WEST,SOUTH,EAST,NORTH{" "}
                <Help>
                  The location is searched only inside this area. If you chose it from prior knowledge (a title, a guess), a result inside it is a <b>bounded</b> search, not blind discovery.
                  Above 40,000 km² the search is slow and memory heavy.
                </Help>
              </label>
              <input value={bbox} onChange={(e) => setBbox(e.target.value)} />
            </div>
          </div>
          <h3 style={{ marginTop: 10 }}>Flight data</h3>
          {compatible.length === 0 && searchDay && (
            <div className="small err-text">
              No {recording?.synthetic ? "SYNTHETIC" : "real"} flight archive for {searchDay} is imported. <Link href="/">Import it on the Import page</Link>.
            </div>
          )}
          {datasets
            .filter((d) => devOverrides || compatible.includes(d))
            .map((d) => (
              <div key={d.dataset_id} className="row small">
                <label>
                  <input
                    type="checkbox"
                    checked={dsSel.includes(d.dataset_id)}
                    onChange={() => {
                      setDsTouched(true);
                      setDsSel((s) => (s.includes(d.dataset_id) ? s.filter((x) => x !== d.dataset_id) : [...s, d.dataset_id]));
                    }}
                  />{" "}
                  {d.synthetic && <span className="tag synthetic">SYNTHETIC</span>}
                  {d.date_utc} · {d.segments?.toLocaleString()} segments · <code>{d.dataset_id}</code>
                  {!compatible.includes(d) && <span className="tag ev-bad">not {searchDay ?? "the recording's day"} / other kind</span>}
                </label>
                {d.bbox && (
                  <button className="small-btn" onClick={() => setBbox((d.bbox as number[]).map((v) => v.toFixed(4)).join(","))}>
                    use its area
                  </button>
                )}
              </div>
            ))}
          <h3 style={{ marginTop: 10 }}>Effort</h3>
          <div className="row">
            {Object.entries(PRESETS).map(([k, p]) => (
              <label key={k} title={p.note}>
                <input type="radio" checked={preset === k} onChange={() => applyPreset(k)} /> {p.label}
              </label>
            ))}
            <span className="small muted">{PRESETS[preset]?.note}</span>
          </div>
          <label style={{ display: "block", marginTop: 6 }}>
            <input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} /> show advanced settings
          </label>
          {advanced && (
            <div className="row" style={{ marginTop: 6 }}>
              <div className="col">
                <label>
                  grid spacing (m) <Help>Distance between candidate recorder positions in the first pass. Halving it means ~4× the work. Final answers are refined to 25 m anyway.</Help>
                </label>
                <input type="number" min={25} value={grid} onChange={(e) => setGrid(e.target.value)} />
              </div>
              <div className="col">
                <label>
                  separation tolerance (s) <Help>Unknown-time mode: how much the two events&apos; time separation may differ from two aircraft passes&apos; separation. 20 s is the tested default.</Help>
                </label>
                <input type="number" min={1} value={tol} onChange={(e) => setTol(e.target.value)} />
              </div>
              <div className="col">
                <label>
                  clock sweep ± (s) <Help>How far the fine clock fit may move each candidate. Larger values are slower and more prone to chance matches.</Help>
                </label>
                <input type="number" min={1} value={sweep} onChange={(e) => setSweep(e.target.value)} />
              </div>
              <div className="col">
                <label>
                  workers <Help>Parallel scoring processes. 0 = as many as the memory budget allows on this machine (recommended).</Help>
                </label>
                <input type="number" min={0} value={workers} onChange={(e) => setWorkers(e.target.value)} />
              </div>
              <div className="col">
                <label>
                  air temperature °C <Help>Sets the speed of sound. Leave empty for 20 °C; the result reports how sensitive it is to ±5 m/s.</Help>
                </label>
                <input type="number" value={tempC} onChange={(e) => setTempC(e.target.value)} />
              </div>
              <div className="col">
                <label>
                  flat ground height (m) <Help>Only when no terrain model is available for the area; otherwise the search refuses rather than assume a height. Flagged in the result.</Help>
                </label>
                <input type="number" value={groundElev} onChange={(e) => setGroundElev(e.target.value)} />
              </div>
              <label>
                <input type="checkbox" checked={silence} onChange={(e) => setSilence(e.target.checked)} /> penalise loud aircraft that should have been heard but were not
              </label>
              <label>
                <input type="checkbox" checked={devOverrides} onChange={(e) => setDevOverrides(e.target.checked)} /> development overrides (allow other days / SYNTHETIC mixing)
              </label>
            </div>
          )}
          {area > 40000 && (
            <label style={{ display: "block", marginTop: 6 }}>
              <input type="checkbox" checked={largeScale} onChange={(e) => setLargeScale(e.target.checked)} /> I understand a {area.toFixed(0)} km² search is slow and memory heavy
            </label>
          )}
          <h3 style={{ marginTop: 12 }}>Before you run</h3>
          <ul className="checklist">
            {checks.map((c) => (
              <li key={c.text} className={c.ok ? "ok" : "bad"}>
                {c.ok ? "✓" : "✗"} {c.text}
              </li>
            ))}
          </ul>
          <div className="row" style={{ marginTop: 8 }}>
            <button onClick={runEstimate} disabled={!ready || busy}>
              Estimate the work
            </button>
            <button className="primary" disabled={!ready || busy} onClick={submit} title={ready ? "" : "complete the checklist first"}>
              {busy ? "submitting…" : "Run search"}
            </button>
          </div>
          {estimate && (
            <div className="banner ok small" style={{ marginTop: 8 }}>
              <b>Workload</b>: {estimate.area_km2.toLocaleString()} km² · {estimate.grid_points.toLocaleString()} grid points · {estimate.segments_in_window_and_area.toLocaleString()} flight segments in the
              time window and area (cap {estimate.segments_cap.toLocaleString()}) · {estimate.workers} workers · memory budget {(estimate.memory_budget_mb / 1000).toFixed(1)} GB.{" "}
              {estimate.within_caps ? "Within the caps." : "Above a cap — see below."}
              {estimate.notes.map((n) => (
                <div key={n}>• {n}</div>
              ))}
              {estimate.dataset_warnings.map((n) => (
                <div key={n}>• {n}</div>
              ))}
              <div className="muted">{estimate.reference_run} (a measured reference, not a prediction for this search)</div>
            </div>
          )}
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
                <div>
                  {live.progress?.elapsed_s ?? 0} s · CPU {live.progress?.cpu_seconds ?? "–"} s
                </div>
                {live.error && (
                  <>
                    <div>error</div>
                    <div className="status failed">{live.error}</div>
                  </>
                )}
              </div>
              <div className="row" style={{ marginTop: 8 }}>
                <button className="danger" disabled={!["queued", "running"].includes(live.status)} onClick={() => api.cancel(live.id).then(setLive).catch((e) => setErr(errText(e)))}>
                  Cancel
                </button>
                <Link href="/jobs/">all jobs →</Link>
              </div>
              <details style={{ marginTop: 10 }}>
                <summary className="small">stage details</summary>
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
              </details>
            </div>
          ) : (
            <p className="muted">no search started from this page yet — running searches of all recordings are on the <Link href="/jobs/">Jobs page</Link></p>
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
