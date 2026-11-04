"use client";
import { useEffect, useState } from "react";
import { api, type FlightDataset, type Recording } from "@/lib/api";

export default function DataPage() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [datasets, setDatasets] = useState<FlightDataset[]>([]);
  const [audit, setAudit] = useState<Record<string, unknown>[]>([]);
  const [confirm, setConfirm] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const refresh = () => {
    api.recordings().then((r) => setRecordings(r.recordings)).catch((e) => setErr(String(e)));
    api.datasets().then((d) => setDatasets(d.datasets)).catch(() => undefined);
    api.audit().then((a) => setAudit(a.entries)).catch(() => undefined);
  };
  useEffect(refresh, []);
  const purge = async (id: string) => {
    try {
      const r = await api.purge(id);
      setMsg(`Purged ${id}: removed ${r.removed.length} directories (original, analysis copy, spectrogram, events, every run).`);
      setConfirm(null);
      refresh();
    } catch (e) {
      setErr(String(e));
    }
  };
  return (
    <div>
      <div className="panel">
        <h2>Privacy defaults</h2>
        <ul className="small">
          <li>The server binds to 127.0.0.1 only. No recording, spectrogram, location or telemetry is sent to any external service.</li>
          <li>Flight archives are downloaded from ADSB.lol (GitHub releases) and DEM tiles from USGS/Copernicus only when you request an import or a search area. Those requests reveal the date and area of interest to those providers. Use <code>AERODOPPLER_OFFLINE=true</code> to forbid all network access.</li>
          <li>Map basemap tiles are off by default and must be enabled per view.</li>
          <li>Every ingest, event, search and purge is written to the local audit log below.</li>
          <li>Purging a recording deletes the original file, the analysis copy, spectrograms, events and all run outputs. This cannot be undone.</li>
        </ul>
      </div>
      {err && <div className="banner danger">{err}</div>}
      {msg && <div className="banner ok">{msg}</div>}
      <div className="panel">
        <h2>Recordings</h2>
        <table>
          <thead>
            <tr>
              <th>id</th>
              <th>file</th>
              <th>sha256</th>
              <th>created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {recordings.map((r) => (
              <tr key={r.id}>
                <td>
                  {r.synthetic && <span className="tag synthetic">SYNTHETIC</span>}
                  <code>{r.id}</code>
                </td>
                <td>{r.original_filename}</td>
                <td className="small">
                  <code>{r.sha256}</code>
                </td>
                <td className="small">{r.created_at}</td>
                <td>
                  {confirm === r.id ? (
                    <span>
                      really delete everything for this recording?{" "}
                      <button className="danger" onClick={() => purge(r.id)}>
                        yes, purge
                      </button>{" "}
                      <button onClick={() => setConfirm(null)}>no</button>
                    </span>
                  ) : (
                    <button className="danger" onClick={() => setConfirm(r.id)}>
                      purge…
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="panel">
        <h2>Flight datasets on disk</h2>
        <table>
          <thead>
            <tr>
              <th>dataset</th>
              <th>provider</th>
              <th>date</th>
              <th>segments</th>
              <th>size</th>
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
                <td>{d.provider}</td>
                <td>{d.date_utc}</td>
                <td>{d.segments}</td>
                <td>{d.storage_bytes ? `${(d.storage_bytes / 1e6).toFixed(1)} MB` : "–"}</td>
                <td className="small">{d.license}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">Datasets are plain Parquet files under <code>data/normalized/</code>; remove a directory to delete a dataset. Raw archives are under <code>data/raw/</code>.</p>
      </div>
      <div className="panel">
        <h2>Audit log (latest {audit.length})</h2>
        <table>
          <thead>
            <tr>
              <th>time</th>
              <th>action</th>
              <th>subject</th>
              <th>detail</th>
            </tr>
          </thead>
          <tbody>
            {audit.map((a, i) => (
              <tr key={i}>
                <td className="small">{String(a.ts)}</td>
                <td>
                  <code>{String(a.action)}</code>
                </td>
                <td className="small">
                  <code>{String(a.subject ?? "")}</code>
                </td>
                <td className="small">{String(a.detail ?? "")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
