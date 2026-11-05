"use client";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, errText, waitJob, type FlightDataset, type Job, type Recording } from "@/lib/api";

type TimeKind = "verified_start" | "date_only" | "date_range" | "upper_bound_only" | "unknown";

const TIME_LABEL: Record<TimeKind, string> = {
  verified_start: "exact start time known",
  date_only: "only the UTC day known",
  date_range: "day within a range",
  upper_bound_only: "only “on or before” a date",
  unknown: "time unknown",
};

const ANALYSIS_LIMIT_S = 7200;

/** Parse an ISO time that must carry Z or an explicit offset; returns the UTC instant. */
function parseZoned(v: string): { utc: string | null; error: string | null } {
  const s = v.trim();
  if (!s) return { utc: null, error: null };
  if (!/(Z|[+-]\d{2}:?\d{2})$/i.test(s)) return { utc: null, error: "add Z for UTC or an offset such as -05:00 — a local clock time alone is ambiguous" };
  const t = Date.parse(s);
  if (Number.isNaN(t)) return { utc: null, error: "not a valid time (use 2026-11-03T15:30:00Z)" };
  return { utc: new Date(t).toISOString().replace(".000Z", "Z"), error: null };
}

function Help({ children }: { children: React.ReactNode }) {
  return (
    <details className="help">
      <summary>?</summary>
      <div>{children}</div>
    </details>
  );
}

function fmtDur(s: number): string {
  return s >= 3600 ? `${(s / 3600).toFixed(2)} h` : s >= 60 ? `${(s / 60).toFixed(1)} min` : `${s.toFixed(1)} s`;
}

export default function ImportPage() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [datasets, setDatasets] = useState<FlightDataset[]>([]);
  const [imports, setImports] = useState<Job[]>([]);
  const [ingesting, setIngesting] = useState<Job | null>(null);
  const [eventsCount, setEventsCount] = useState<Record<string, number>>({});
  const [selected, setSelected] = useState<string>("");
  const [showAllRecs, setShowAllRecs] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // recording form
  const [file, setFile] = useState<File | null>(null);
  const [kind, setKind] = useState<TimeKind>("unknown");
  const [startUtc, setStartUtc] = useState("");
  const [unc, setUnc] = useState("");
  const [date, setDate] = useState("");
  const [dEarliest, setDEarliest] = useState("");
  const [dLatest, setDLatest] = useState("");
  const [segStart, setSegStart] = useState("");
  const [segLen, setSegLen] = useState("");
  const [synthetic, setSynthetic] = useState(false);
  const [notes, setNotes] = useState("");
  // flights form
  const [fDate, setFDate] = useState("");
  const [fBbox, setFBbox] = useState("");
  const [fConfirm, setFConfirm] = useState(false);
  const [fProvider, setFProvider] = useState("adsblol");
  const [fLocal, setFLocal] = useState("");

  const refresh = useCallback(() => {
    api
      .recordings()
      .then((r) => {
        setRecordings(r.recordings);
        setSelected((cur) => cur || r.recordings[0]?.id || "");
        r.recordings.slice(0, 20).forEach((rec) =>
          api
            .events(rec.id)
            .then((e) => setEventsCount((m) => ({ ...m, [rec.id]: e.events.length })))
            .catch(() => undefined),
        );
      })
      .catch((e) => setErr(errText(e)));
    api.datasets().then((d) => setDatasets(d.datasets)).catch((e) => setErr(errText(e)));
    api
      .jobs({ type: "flight_import", limit: "20" })
      .then((d) => setImports(d.jobs))
      .catch(() => undefined);
  }, []);
  useEffect(refresh, [refresh]);

  const activeImports = imports.filter((j) => ["queued", "claimed", "running"].includes(j.status));
  // poll while an import runs (the job lives in its own process; a reload simply re-attaches)
  useEffect(() => {
    if (!activeImports.length) return;
    const t = setInterval(() => {
      api.jobs({ type: "flight_import", limit: "20" }).then((d) => {
        const wasActive = activeImports.map((j) => j.id);
        setImports(d.jobs);
        if (d.jobs.some((j) => wasActive.includes(j.id) && j.status === "succeeded")) api.datasets().then((x) => setDatasets(x.datasets));
      });
    }, 2000);
    return () => clearInterval(t);
  }, [activeImports]);
  // an audio import still running from before a reload: re-attach to it
  useEffect(() => {
    api
      .jobs({ type: "audio_ingest", status: "active", limit: "5" })
      .then((d) => {
        if (d.jobs.length) follow(d.jobs[0]);
      })
      .catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const follow = async (job: Job) => {
    setIngesting(job);
    const done = await waitJob(job.job_id, setIngesting);
    setIngesting(null);
    if (done.status === "succeeded") {
      const rid = String(done.result?.recording_id ?? "");
      const warnings = (done.result?.warnings as string[] | undefined) ?? [];
      setMsg(`Imported recording ${rid}. ${warnings.join(" ")}`);
      if (rid) setSelected(rid);
      refresh();
    } else {
      setErr(done.error?.message ?? `the audio import ${done.status}`);
    }
  };

  const startParsed = useMemo(() => parseZoned(startUtc), [startUtc]);
  const rec = recordings.find((r) => r.id === selected) ?? null;
  const recDate = rec?.recording_date_utc ?? null;
  const dsForRec = rec ? datasets.filter((d) => d.date_utc === recDate && d.synthetic === rec.synthetic) : [];
  const importForRec = rec ? activeImports.find((j) => String(j.params.date_utc) === recDate) : undefined;

  // prefill the flight form from the selected recording's day
  useEffect(() => {
    if (recDate && !fDate) setFDate(recDate);
  }, [recDate, fDate]);

  const formProblem = (): string | null => {
    if (!file) return "choose a file";
    if (kind === "verified_start") {
      if (startParsed.error) return `start time: ${startParsed.error}`;
      if (!startParsed.utc) return "enter the start time";
      if (!unc) return "enter the start-time uncertainty in seconds";
    }
    if (kind === "date_only" && !date) return "enter the UTC day";
    if (kind === "date_range" && (!dEarliest || !dLatest)) return "enter both the earliest and the latest day";
    if (kind === "date_range" && dEarliest > dLatest) return "the earliest day is after the latest day";
    if (kind === "upper_bound_only" && !dLatest) return "enter the date it was recorded on or before";
    return null;
  };

  const upload = async () => {
    setErr(null);
    setMsg(null);
    const p = formProblem();
    if (p) return setErr(p);
    const fd = new FormData();
    fd.append("file", file as File);
    if (kind === "verified_start") {
      fd.append("start_time_utc", startParsed.utc as string);
      fd.append("start_time_uncertainty_s", unc);
    }
    if (kind === "date_only") fd.append("recording_date_utc", date);
    if (kind === "date_range") {
      fd.append("date_earliest_utc", dEarliest);
      fd.append("date_latest_utc", dLatest);
    }
    if (kind === "upper_bound_only") fd.append("date_latest_utc", dLatest);
    if (segStart) fd.append("segment_start_s", segStart);
    if (segLen) fd.append("segment_duration_s", segLen);
    fd.append("synthetic", synthetic ? "true" : "false");
    if (notes) fd.append("notes", notes);
    setBusy(true);
    try {
      const job = await api.upload(fd); // uploaded; decoding runs as a job
      setBusy(false);
      await follow(job);
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const nextSegment = async (r: Recording) => {
    setErr(null);
    setBusy(true);
    try {
      const start = r.segment_start_s + r.duration_s;
      const job = await api.createSegment(r.id, start, null);
      setBusy(false);
      await follow(job);
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const importFlights = async () => {
    setErr(null);
    setMsg(null);
    const parts = fBbox.split(",").map((s) => parseFloat(s.trim()));
    if (parts.length !== 4 || parts.some((p) => Number.isNaN(p))) return setErr("area must be WEST,SOUTH,EAST,NORTH in degrees, e.g. -74.15,40.65,-73.85,40.90");
    if (parts[0] >= parts[2] || parts[1] >= parts[3]) return setErr("area: WEST must be < EAST and SOUTH < NORTH");
    setBusy(true);
    try {
      const r = await api.importFlights({
        provider: fProvider,
        date_utc: fDate,
        bbox: { west: parts[0], south: parts[1], east: parts[2], north: parts[3] },
        confirm_download: fConfirm,
        local_path: fLocal || null,
      });
      if (r.outcome === "exists") setMsg(`Flights for ${fDate} in this area are already imported (${r.dataset_id}). Nothing to do.`);
      else if (r.outcome === "reused") setMsg(`An import of this day and area is already queued or running; showing its progress below (nothing new started).`);
      else setMsg(`Import started. It runs in the background: you can leave or reload this page; progress is on the Jobs page.`);
      refresh();
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const steps: { label: string; done: boolean; detail: React.ReactNode }[] = rec
    ? [
        { label: "Recording imported", done: true, detail: `${rec.original_filename} · ${fmtDur(rec.duration_s)}${rec.truncated ? " (segment)" : ""}` },
        {
          label: "When it was recorded",
          done: rec.time_basis === "verified_start" || rec.time_basis === "date_only",
          detail:
            rec.time_basis === "verified_start"
              ? `start ${rec.start_time_utc} ±${rec.start_time_uncertainty_s} s → known-time or unknown-time search possible`
              : rec.time_basis === "date_only"
                ? `UTC day ${rec.recording_date_utc} → unknown-time search (needs two aircraft passes)`
                : rec.time_basis === "date_range"
                  ? `day between ${rec.date_earliest_utc} and ${rec.date_latest_utc}: each day is a separate ~3.5 GB archive and search; narrow it down if you can`
                  : rec.time_basis === "upper_bound_only"
                    ? `only “on or before ${rec.date_latest_utc}”: flight matching needs the actual day`
                    : "unknown: flight matching needs at least the UTC day",
        },
        {
          label: "Flight data for that day",
          done: dsForRec.length > 0,
          detail: dsForRec.length ? (
            `${dsForRec.length} archive(s) imported: ${dsForRec.map((d) => d.dataset_id).join(", ")}`
          ) : importForRec ? (
            <>
              importing now ({importForRec.stage}) — <Link href="/jobs/">progress</Link>
            </>
          ) : recDate ? (
            `not imported yet: import ${recDate} below`
          ) : (
            "needs the day first"
          ),
        },
        {
          label: "Aircraft events marked",
          done: (eventsCount[rec.id] ?? 0) > 0,
          detail: (
            <>
              {eventsCount[rec.id] ?? 0} event(s) — <Link href={`/lab/?rec=${rec.id}`}>open the Spectrogram lab</Link> to detect or draw them
            </>
          ),
        },
        {
          label: "Search",
          done: false,
          detail: <Link href={`/search/?rec=${rec.id}`}>set up a search →</Link>,
        },
      ]
    : [];
  const nextStep = steps.find((s) => !s.done);

  return (
    <div>
      {err && <div className="banner danger">{err}</div>}
      {msg && <div className="banner ok">{msg}</div>}
      {ingesting && (
        <div className="banner ok">
          Importing audio ({ingesting.status}, {ingesting.stage}) — decoding runs in the background; you can leave this page. <Link href="/jobs/">Jobs</Link>
        </div>
      )}
      <div className="panel">
        <h2>Getting started</h2>
        {!recordings.length ? (
          <p>
            AeroDoppler compares the pitch changes of propeller aircraft in a recording (Doppler &ldquo;ridges&rdquo;) with recorded flight tracks (ADS-B) to propose where it was recorded.
            Steps: <b>1</b> import a recording and say what you know about its time, <b>2</b> import the flight archive of that UTC day, <b>3</b> mark aircraft passes in the Spectrogram lab,{" "}
            <b>4</b> run a search. To see it work first, run <code>make demo</code> (SYNTHETIC data). <Link href="/guide/">Read the guide</Link>.
          </p>
        ) : (
          <>
            <div className="row">
              <div className="col">
                <label>recording</label>
                <select value={selected} onChange={(e) => setSelected(e.target.value)}>
                  {recordings.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.synthetic ? "SYNTHETIC · " : ""}
                      {r.original_filename.slice(0, 60)} · {r.id}
                    </option>
                  ))}
                </select>
              </div>
              {nextStep && (
                <div className="next-step">
                  next: <b>{nextStep.label}</b>
                </div>
              )}
            </div>
            <ol className="steps">
              {steps.map((s) => (
                <li key={s.label} className={s.done ? "done" : s === nextStep ? "next" : ""}>
                  <span className="mark">{s.done ? "✓" : s === nextStep ? "→" : "·"}</span> <b>{s.label}</b> <span className="small muted">{s.detail}</span>
                </li>
              ))}
            </ol>
          </>
        )}
      </div>
      <div className="grid2">
        <div className="panel">
          <h2>1 · Import a recording</h2>
          <div className="col">
            <label>Audio or video file (WAV/FLAC/MP3/MP4…, decoded locally with FFmpeg; nothing is uploaded anywhere)</label>
            <input type="file" accept="audio/*,video/*,.wav,.flac,.mp3,.m4a,.mp4,.mov,.mkv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <h3 style={{ marginTop: 12 }}>
            What do you know about when it was recorded?{" "}
            <Help>
              Flight tracks are stored per UTC day, so the search needs at least the day. A <b>publication or upload date</b> (YouTube, social media) is only an upper bound: the recording can
              be older. Do not enter it as the recording date unless you know it was recorded live that day. Never guess a start time: a wrong time silently matches the wrong aircraft.
            </Help>
          </h3>
          <div className="radio-list">
            {(Object.keys(TIME_LABEL) as TimeKind[]).map((k) => (
              <label key={k}>
                <input type="radio" name="tk" checked={kind === k} onChange={() => setKind(k)} /> {TIME_LABEL[k]}
              </label>
            ))}
          </div>
          {kind === "verified_start" && (
            <div className="row" style={{ marginTop: 8 }}>
              <div className="col" style={{ minWidth: 280 }}>
                <label>
                  time of the file&apos;s first sample, with Z (UTC) or an offset <Help>Examples: 2026-11-03T15:30:00Z (UTC), 2026-11-03T10:30:00-05:00 (New York in November). The UTC equivalent is shown below.</Help>
                </label>
                <input value={startUtc} onChange={(e) => setStartUtc(e.target.value)} placeholder="2026-11-03T15:30:00Z" />
                <span className={`small ${startParsed.error ? "err-text" : "muted"}`}>{startParsed.error ?? (startParsed.utc ? `= ${startParsed.utc.replace("T", " ").replace("Z", " UTC")}` : " ")}</span>
              </div>
              <div className="col">
                <label>
                  uncertainty (± s) <Help>How far off the start time could be: ~2 s for a phone with network time, 30–120 s for a time read off a clock or a stream overlay.</Help>
                </label>
                <input type="number" min={0} value={unc} onChange={(e) => setUnc(e.target.value)} />
              </div>
            </div>
          )}
          {kind === "date_only" && (
            <div className="col" style={{ marginTop: 8 }}>
              <label>UTC day of the recording (careful near midnight: UTC may differ from your local date)</label>
              <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
            </div>
          )}
          {kind === "date_range" && (
            <div className="row" style={{ marginTop: 8 }}>
              <div className="col">
                <label>earliest possible UTC day</label>
                <input type="date" value={dEarliest} onChange={(e) => setDEarliest(e.target.value)} />
              </div>
              <div className="col">
                <label>latest possible UTC day</label>
                <input type="date" value={dLatest} onChange={(e) => setDLatest(e.target.value)} />
              </div>
            </div>
          )}
          {kind === "upper_bound_only" && (
            <div className="col" style={{ marginTop: 8 }}>
              <label>recorded on or before (e.g. the publication date)</label>
              <input type="date" value={dLatest} onChange={(e) => setDLatest(e.target.value)} />
              <span className="small muted">Stored as an upper bound only. A search needs the actual day; the importer will not assume it.</span>
            </div>
          )}
          <details style={{ marginTop: 8 }}>
            <summary className="small">long file? analyse a part of it</summary>
            <p className="small muted">
              Up to {ANALYSIS_LIMIT_S / 60} min are analysed per recording; longer files are not rejected: the first {ANALYSIS_LIMIT_S / 60} min are analysed and you can add the next
              segment afterwards. All times stay tied to the original file (offsets are recorded).
            </p>
            <div className="row">
              <div className="col">
                <label>start at (s into the file)</label>
                <input type="number" min={0} value={segStart} onChange={(e) => setSegStart(e.target.value)} placeholder="0" />
              </div>
              <div className="col">
                <label>length (s)</label>
                <input type="number" min={1} value={segLen} onChange={(e) => setSegLen(e.target.value)} placeholder={`up to ${ANALYSIS_LIMIT_S}`} />
              </div>
            </div>
          </details>
          <div className="col" style={{ marginTop: 8 }}>
            <label>notes (optional)</label>
            <input value={notes} onChange={(e) => setNotes(e.target.value)} />
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <label>
              <input type="checkbox" checked={synthetic} onChange={(e) => setSynthetic(e.target.checked)} /> this is SYNTHETIC test audio
            </label>
            <button className="primary" disabled={busy || !file || formProblem() !== null} onClick={upload} title={formProblem() ?? ""}>
              {busy ? "working…" : "Import recording"}
            </button>
          </div>
          {file && formProblem() && <div className="small err-text">{formProblem()}</div>}
        </div>
        <div className="panel">
          <h2>2 · Flight archive for the recording&apos;s day</h2>
          <p className="small">
            ADSB.lol publishes one <b>whole-world archive per UTC day (≈ 3.5 GB)</b>. It is downloaded once and cached, even for a small area, because it cannot be subset remotely; then only
            aircraft inside your area (+15 km) are kept. Days appear a few hours after they end (UTC); archives exist from February 2023. Licence ODbL 1.0.
          </p>
          <div className="row">
            <div className="col">
              <label>source</label>
              <select value={fProvider} onChange={(e) => setFProvider(e.target.value)}>
                <option value="adsblol">ADSB.lol daily archive</option>
                <option value="local">local trace files / tar</option>
              </select>
            </div>
            <div className="col">
              <label>UTC day</label>
              <input type="date" value={fDate} onChange={(e) => setFDate(e.target.value)} />
            </div>
            <div className="col" style={{ minWidth: 260 }}>
              <label>
                area WEST,SOUTH,EAST,NORTH (degrees){" "}
                <Help>
                  Only aircraft in this area are kept, so the search can only find locations inside it. Choosing the area from something you already know (a video title, a guess) is a{" "}
                  <b>geographic prior</b>: a result inside it is not a blind discovery. Larger areas cost more disk and search time.
                </Help>
              </label>
              <input value={fBbox} onChange={(e) => setFBbox(e.target.value)} placeholder="-74.15,40.65,-73.85,40.90" />
            </div>
            {fProvider === "local" && (
              <div className="col" style={{ minWidth: 260 }}>
                <label>local path (directory or tar parts)</label>
                <input value={fLocal} onChange={(e) => setFLocal(e.target.value)} />
              </div>
            )}
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            {fProvider === "adsblol" && (
              <label>
                <input type="checkbox" checked={fConfirm} onChange={(e) => setFConfirm(e.target.checked)} /> download the ≈ 3.5 GB day if it is not cached
              </label>
            )}
            <button className="primary" disabled={busy || !fDate || !fBbox} onClick={importFlights}>
              {busy ? "working…" : "Import flights"}
            </button>
          </div>
          {activeImports.map((j) => (
            <div key={j.id} className="banner ok small">
              Importing {String(j.params.date_utc)}: <b>{j.stage}</b>
              {(() => {
                const d = j.progress?.stages?.download as { bytes_done?: number; bytes_total?: number } | undefined;
                const p = j.progress?.stages?.parse as { files_parsed?: number; aircraft_kept?: number } | undefined;
                if (j.stage === "download" && d?.bytes_total) return ` ${((d.bytes_done ?? 0) / 1e9).toFixed(2)} / ${(d.bytes_total / 1e9).toFixed(2)} GB`;
                if (j.stage === "parse" && p) return ` ${p.files_parsed?.toLocaleString()} trace files read, ${p.aircraft_kept?.toLocaleString()} aircraft kept`;
                return "";
              })()}{" "}
              · <Link href="/jobs/">details / cancel</Link>
            </div>
          ))}
          {imports
            .filter((j) => j.status === "failed" || j.status === "interrupted")
            .slice(0, 1)
            .map((j) => (
              <div key={j.id} className="banner warn small">
                Last import of {String(j.params.date_utc)} {j.status}: {j.error?.message} <Link href="/jobs/">retry on the Jobs page</Link>
              </div>
            ))}
          <h3 style={{ marginTop: 14 }}>Imported flight data</h3>
          <table>
            <thead>
              <tr>
                <th>day</th>
                <th>area</th>
                <th>segments</th>
                <th>aircraft</th>
                <th>dataset</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.dataset_id}>
                  <td>
                    {d.synthetic && <span className="tag synthetic">SYNTHETIC</span>}
                    {d.date_utc}
                  </td>
                  <td className="small">{d.bbox?.map((v) => v.toFixed(2)).join(", ")}</td>
                  <td>{d.segments?.toLocaleString()}</td>
                  <td>{d.aircraft?.toLocaleString()}</td>
                  <td className="small">
                    <code>{d.dataset_id}</code>
                  </td>
                </tr>
              ))}
              {!datasets.length && (
                <tr>
                  <td colSpan={5} className="muted">
                    none yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      <div className="panel">
        <h2>
          Recordings ({recordings.length}){" "}
          {recordings.length > 25 && (
            <button className="small-btn" onClick={() => setShowAllRecs((v) => !v)}>
              {showAllRecs ? "show the 25 newest" : "show all"}
            </button>
          )}
        </h2>
        <table>
          <thead>
            <tr>
              <th>file</th>
              <th>analysed part</th>
              <th>time</th>
              <th>events</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(showAllRecs ? recordings : recordings.slice(0, 25)).map((r) => (
              <tr key={r.id} className={r.id === selected ? "selected" : ""}>
                <td>
                  {r.synthetic && <span className="tag synthetic">SYNTHETIC</span>}
                  {r.original_filename}
                  <div className="small muted">
                    <code>{r.id}</code> · {r.sample_rate_hz} Hz · sha256 <code>{r.sha256.slice(0, 10)}…</code>
                  </div>
                </td>
                <td className="small">
                  {fmtDur(r.duration_s)}
                  {(r.truncated || r.segment_start_s > 0) && (
                    <div>
                      {r.segment_start_s.toFixed(0)}–{(r.segment_start_s + r.duration_s).toFixed(0)} s of {r.source_duration_s?.toFixed(0) ?? "?"} s
                      {r.truncated && (
                        <div>
                          <button className="small-btn" disabled={busy} onClick={() => nextSegment(r)}>
                            analyse the next segment
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </td>
                <td className="small">
                  {TIME_LABEL[(r.time_basis ?? "unknown") as TimeKind]}
                  <div className="muted">
                    {r.start_time_utc
                      ? `${r.start_time_utc} ±${r.start_time_uncertainty_s ?? "?"} s`
                      : r.recording_date_utc ?? (r.date_latest_utc ? `${r.date_earliest_utc ?? "…"} – ${r.date_latest_utc}` : "")}
                  </div>
                </td>
                <td>{eventsCount[r.id] ?? "–"}</td>
                <td>
                  <Link href={`/lab/?rec=${r.id}`}>spectrogram lab →</Link>
                </td>
              </tr>
            ))}
            {!recordings.length && (
              <tr>
                <td colSpan={5} className="muted">
                  no recordings yet — import one above, or run <code>make demo</code> for a SYNTHETIC example
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
