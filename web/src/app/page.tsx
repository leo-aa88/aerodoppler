"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type FlightDataset, type Recording } from "@/lib/api";

export default function ImportPage() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [datasets, setDatasets] = useState<FlightDataset[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [date, setDate] = useState("");
  const [startUtc, setStartUtc] = useState("");
  const [unc, setUnc] = useState("");
  const [synthetic, setSynthetic] = useState(false);
  const [notes, setNotes] = useState("");
  // flights import
  const [fDate, setFDate] = useState("");
  const [fBbox, setFBbox] = useState("");
  const [fConfirm, setFConfirm] = useState(false);
  const [fProvider, setFProvider] = useState("adsblol");
  const [fLocal, setFLocal] = useState("");

  const refresh = () => {
    api.recordings().then((r) => setRecordings(r.recordings)).catch((e) => setErr(String(e)));
    api.datasets().then((d) => setDatasets(d.datasets)).catch((e) => setErr(String(e)));
  };
  useEffect(refresh, []);

  const upload = async () => {
    setErr(null);
    setMsg(null);
    if (!file) return setErr("choose a file");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("owner_permission_confirmed", "true");
    if (date) fd.append("recording_date_utc", date);
    if (startUtc) fd.append("start_time_utc", startUtc);
    if (unc) fd.append("start_time_uncertainty_s", unc);
    fd.append("synthetic", synthetic ? "true" : "false");
    if (notes) fd.append("notes", notes);
    setBusy(true);
    try {
      const r = await api.upload(fd);
      setMsg(`Ingested ${r.recording.original_filename} as ${r.recording.id} (${r.recording.duration_s.toFixed(1)} s, ${r.recording.sample_rate_hz} Hz, sha256 ${r.recording.sha256.slice(0, 12)}…). ${r.warnings.join(" ")}`);
      refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const importFlights = async () => {
    setErr(null);
    setMsg(null);
    const parts = fBbox.split(",").map((s) => parseFloat(s.trim()));
    if (parts.length !== 4 || parts.some((p) => Number.isNaN(p))) return setErr("bbox must be WEST,SOUTH,EAST,NORTH");
    setBusy(true);
    try {
      const r = await api.importFlights({ provider: fProvider, date_utc: fDate, bbox: { west: parts[0], south: parts[1], east: parts[2], north: parts[3] }, confirm_download: fConfirm, local_path: fLocal || null });
      setMsg(`Imported dataset ${String(r.dataset_id)}: ${JSON.stringify(r.summary)}`);
      refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      {err && <div className="banner danger">{err}</div>}
      {msg && <div className="banner ok">{msg}</div>}
      <div className="grid2">
        <div className="panel">
          <h2>1 · Import a recording</h2>
          <div className="col">
            <label>Audio or video file (WAV/FLAC/MP3/MP4… decoded locally with FFmpeg)</label>
            <input type="file" accept="audio/*,video/*,.wav,.flac,.mp3,.m4a,.mp4,.mov,.mkv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <div className="col">
              <label>UTC recording date (Mode A, unknown time)</label>
              <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
            </div>
            <div className="col">
              <label>UTC start time (Mode B), e.g. 2026-09-08T14:07:13Z</label>
              <input value={startUtc} onChange={(e) => setStartUtc(e.target.value)} placeholder="YYYY-MM-DDTHH:MM:SSZ" />
            </div>
            <div className="col">
              <label>start-time uncertainty (s)</label>
              <input type="number" value={unc} onChange={(e) => setUnc(e.target.value)} placeholder="e.g. 60" />
            </div>
          </div>
          <div className="col" style={{ marginTop: 8 }}>
            <label>notes</label>
            <input value={notes} onChange={(e) => setNotes(e.target.value)} />
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <label>
              <input type="checkbox" checked={synthetic} onChange={(e) => setSynthetic(e.target.checked)} /> label as SYNTHETIC DEMO data
            </label>
            <button className="primary" disabled={busy || !file} onClick={upload}>
              {busy ? "working…" : "Ingest"}
            </button>
          </div>
        </div>
        <div className="panel">
          <h2>2 · Flight archive</h2>
          <p className="muted small">
            ADSB.lol publishes one whole-sky archive per UTC day (≈3.3 GB). It is downloaded once, cached, and filtered to your bounding box (+15 km margin). Licence ODbL 1.0. You must confirm the download size explicitly.
          </p>
          <div className="row">
            <div className="col">
              <label>provider</label>
              <select value={fProvider} onChange={(e) => setFProvider(e.target.value)}>
                <option value="adsblol">adsblol (daily GitHub release)</option>
                <option value="local">local trace files / tar</option>
              </select>
            </div>
            <div className="col">
              <label>UTC date</label>
              <input type="date" value={fDate} onChange={(e) => setFDate(e.target.value)} />
            </div>
            <div className="col" style={{ minWidth: 260 }}>
              <label>bbox WEST,SOUTH,EAST,NORTH</label>
              <input value={fBbox} onChange={(e) => setFBbox(e.target.value)} placeholder="-81.8,28.6,-80.8,29.6" />
            </div>
            {fProvider === "local" && (
              <div className="col" style={{ minWidth: 260 }}>
                <label>local path (directory or tar parts)</label>
                <input value={fLocal} onChange={(e) => setFLocal(e.target.value)} />
              </div>
            )}
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <label>
              <input type="checkbox" checked={fConfirm} onChange={(e) => setFConfirm(e.target.checked)} /> I accept downloading the full daily archive (≈3.3 GB) if not cached
            </label>
            <button className="primary" disabled={busy || !fDate || !fBbox} onClick={importFlights}>
              {busy ? "working…" : "Import flights"}
            </button>
          </div>
          <h3 style={{ marginTop: 14 }}>Available datasets</h3>
          <table>
            <thead>
              <tr>
                <th>dataset</th>
                <th>date</th>
                <th>bbox</th>
                <th>segments</th>
                <th>aircraft</th>
                <th>licence</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.dataset_id}>
                  <td>
                    {d.synthetic && <span className="tag synthetic">SYNTHETIC</span>}
                    <code>{d.dataset_id}</code>
                  </td>
                  <td>{d.date_utc}</td>
                  <td className="small">{d.bbox?.map((v) => v.toFixed(2)).join(", ")}</td>
                  <td>{d.segments}</td>
                  <td>{d.aircraft}</td>
                  <td className="small">{d.license}</td>
                </tr>
              ))}
              {!datasets.length && (
                <tr>
                  <td colSpan={6} className="muted">
                    none yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      <div className="panel">
        <h2>Recordings</h2>
        <table>
          <thead>
            <tr>
              <th>id</th>
              <th>file</th>
              <th>duration</th>
              <th>rate</th>
              <th>date / start</th>
              <th>sha256</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {recordings.map((r) => (
              <tr key={r.id}>
                <td>
                  {r.synthetic && <span className="tag synthetic">SYNTHETIC DEMO</span>}
                  <code>{r.id}</code>
                </td>
                <td>{r.original_filename}</td>
                <td>{r.duration_s.toFixed(1)} s</td>
                <td>
                  {r.sample_rate_hz} Hz · {r.channels} ch
                </td>
                <td className="small">
                  {r.recording_date_utc ?? "–"} {r.start_time_utc ? `· ${r.start_time_utc} ±${r.start_time_uncertainty_s ?? "?"} s` : ""}
                </td>
                <td className="small">
                  <code>{r.sha256.slice(0, 16)}…</code>
                </td>
                <td>
                  <Link href={`/lab/?rec=${r.id}`}>spectrogram lab →</Link>
                </td>
              </tr>
            ))}
            {!recordings.length && (
              <tr>
                <td colSpan={7} className="muted">
                  no recordings yet — ingest one above, or run <code>make demo</code> for a SYNTHETIC example
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
