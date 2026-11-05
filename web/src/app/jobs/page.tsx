"use client";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, errText, type ImportJob, type JobsOverview, type SearchJobSummary } from "@/lib/api";

const ACTIVE = new Set(["queued", "running"]);

function gb(b: unknown): string {
  return typeof b === "number" ? `${(b / 1e9).toFixed(2)} GB` : "?";
}

function ago(iso: string): string {
  const s = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (s < 90) return `${s.toFixed(0)} s ago`;
  if (s < 5400) return `${(s / 60).toFixed(0)} min ago`;
  return `${(s / 3600).toFixed(1)} h ago`;
}

function Bar({ value, total }: { value: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (100 * value) / total) : 0;
  return (
    <div className="bar" title={`${pct.toFixed(1)} %`}>
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}

const STAGE_TEXT: Record<string, string> = {
  starting: "starting the worker",
  locate: "looking up the archive",
  download: "downloading the daily archive",
  verify: "verifying the archive checksum",
  parse: "reading and filtering aircraft traces",
  bucket: "grouping samples by aircraft",
  segment: "splitting tracks into flight segments",
  write: "writing the dataset",
  done: "finished",
};

function ImportProgress({ job }: { job: ImportJob }) {
  const st = job.progress?.stages ?? {};
  const dl = st.download as { bytes_done?: number; bytes_total?: number; part?: string } | undefined;
  const pa = st.parse as
    | { files_parsed?: number; aircraft_kept?: number; samples_kept?: number; tree_rss_mb?: number; memory_budget_mb?: number; throttled?: boolean; workers?: number }
    | undefined;
  const sg = (st.segment ?? st.bucket) as { bucket?: number; buckets?: number; parts_done?: number; parts_total?: number } | undefined;
  return (
    <div className="small">
      <div>
        <b>{STAGE_TEXT[job.stage ?? ""] ?? job.stage}</b>
        {job.stalled && <span className="tag warn-tag">no heartbeat for &gt; 3 min — stalled?</span>}
      </div>
      {dl && dl.bytes_total ? (
        <div>
          download {gb(dl.bytes_done)} / {gb(dl.bytes_total)} <Bar value={dl.bytes_done ?? 0} total={dl.bytes_total} />
        </div>
      ) : null}
      {pa && (
        <div>
          {pa.files_parsed?.toLocaleString()} of ~68,700 trace files read · {pa.aircraft_kept?.toLocaleString()} aircraft kept ·{" "}
          {pa.samples_kept?.toLocaleString()} samples · {pa.workers} parsers · memory {pa.tree_rss_mb?.toFixed(0)} / {pa.memory_budget_mb} MB budget
          {pa.throttled ? " (throttled to stay within budget)" : ""}
          <Bar value={pa.files_parsed ?? 0} total={68700} />
        </div>
      )}
      {sg && sg.buckets ? (
        <div>
          segmentation bucket {sg.bucket ?? 0} / {sg.buckets}
          <Bar value={sg.bucket ?? 0} total={sg.buckets} />
        </div>
      ) : null}
    </div>
  );
}

function SearchProgress({ job }: { job: SearchJobSummary }) {
  const st = job.progress?.stages ?? {};
  const last = job.stage ? (st[job.stage] as { done?: number; total?: number } | undefined) : undefined;
  return (
    <div className="small">
      <b>{job.stage}</b>
      {last?.total ? (
        <>
          {" "}
          {last.done?.toLocaleString()} / {last.total.toLocaleString()} <Bar value={last.done ?? 0} total={last.total} />
        </>
      ) : null}
      {job.progress?.elapsed_s != null && <span className="muted"> · {job.progress.elapsed_s.toFixed(0)} s elapsed</span>}
      {job.stalled && <span className="tag warn-tag">no heartbeat for &gt; 3 min — stalled?</span>}
    </div>
  );
}

export default function JobsPage() {
  const [data, setData] = useState<JobsOverview | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [log, setLog] = useState<{ id: string; lines: string[] } | null>(null);
  const [showDone, setShowDone] = useState(true);

  const refresh = useCallback(() => {
    api
      .jobs()
      .then((d) => {
        setData(d);
        setErr(null);
      })
      .catch((e) => setErr(errText(e)));
  }, []);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [refresh]);
  useEffect(() => {
    if (!log) return;
    const t = setInterval(() => api.importLog(log.id).then((r) => setLog({ id: log.id, lines: r.lines })).catch(() => undefined), 3000);
    return () => clearInterval(t);
  }, [log]);

  const act = async (f: () => Promise<unknown>) => {
    try {
      await f();
      refresh();
    } catch (e) {
      setErr(errText(e));
    }
  };

  const imports = (data?.imports ?? []).filter((j) => showDone || ACTIVE.has(j.status));
  const searches = (data?.searches ?? []).filter((j) => showDone || ACTIVE.has(j.status));
  const r = data?.resources;
  return (
    <div>
      <h1 className="page-title">Jobs</h1>
      <p className="muted">
        Every flight import and search on this machine, for all recordings. Jobs run in their own processes: reloading this page or restarting the server does not stop them. A job whose
        process disappeared (crash, out of memory, machine restart) is shown as <b>interrupted</b> and can be retried; downloaded archive parts are kept.
      </p>
      {err && <div className="banner danger">{err}</div>}
      {r && (
        <div className="panel">
          <h2>Machine</h2>
          <div className="kv">
            <div>memory usable by AeroDoppler</div>
            <div>
              {(r.effective_available_mb / 1000).toFixed(1)} GB free of {(r.effective_limit_mb / 1000).toFixed(1)} GB
              {r.cgroup_limit_mb ? ` (container limit ${(r.cgroup_limit_mb / 1000).toFixed(1)} GB)` : ""}
              {r.wsl ? " · WSL2: the Windows host may cap this further (.wslconfig memory=)" : ""}
            </div>
            <div>CPUs</div>
            <div>{r.cpu_count}</div>
            <div>free disk ({r.data_dir})</div>
            <div>{r.disk_free_gb} GB — each imported ADSB.lol day needs ≈ 3.5 GB of cache</div>
            <div>job slots</div>
            <div>
              searches {data?.counts.searches_active}/{data?.counts.searches_limit} · imports {data?.counts.imports_active}/{data?.counts.imports_limit}
            </div>
          </div>
        </div>
      )}
      <label>
        <input type="checkbox" checked={showDone} onChange={(e) => setShowDone(e.target.checked)} /> show finished jobs
      </label>
      <div className="panel">
        <h2>Flight-archive imports</h2>
        <table>
          <thead>
            <tr>
              <th>day / area</th>
              <th>status</th>
              <th>progress</th>
              <th>started</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {imports.map((j) => (
              <tr key={j.id}>
                <td>
                  <b>{j.request.date_utc}</b>
                  <div className="small muted">
                    {j.request.bbox.west.toFixed(2)}, {j.request.bbox.south.toFixed(2)} → {j.request.bbox.east.toFixed(2)}, {j.request.bbox.north.toFixed(2)}
                    {j.request.exclude_jets ? " · jets excluded" : ""}
                  </div>
                  <code className="small">{j.id}</code>
                </td>
                <td>
                  <span className={`status ${j.status}`}>{j.status}</span>
                  {j.error && <div className="small err-text">{j.error}</div>}
                </td>
                <td>
                  {ACTIVE.has(j.status) ? (
                    <ImportProgress job={j} />
                  ) : j.status === "succeeded" ? (
                    <div className="small">
                      dataset <code>{j.dataset_id}</code>
                      {j.progress?.summary ? ` · ${(j.progress.summary.segments as number)?.toLocaleString()} segments, ${(j.progress.summary.aircraft as number)?.toLocaleString()} aircraft` : ""}
                      {j.peak_rss_mb ? ` · peak memory ${(j.peak_rss_mb / 1000).toFixed(1)} GB` : ""}
                    </div>
                  ) : (
                    <span className="small muted">stopped during {j.stage}</span>
                  )}
                </td>
                <td className="small">{ago(j.created_at)}</td>
                <td>
                  <div className="row">
                    {ACTIVE.has(j.status) && (
                      <button className="danger" onClick={() => act(() => api.cancelImport(j.id))} disabled={j.cancel_requested}>
                        {j.cancel_requested ? "cancelling…" : "cancel"}
                      </button>
                    )}
                    {["failed", "interrupted", "cancelled"].includes(j.status) && <button onClick={() => act(() => api.retryImport(j.id))}>retry</button>}
                    <button onClick={() => api.importLog(j.id).then((x) => setLog({ id: j.id, lines: x.lines })).catch((e) => setErr(errText(e)))}>log</button>
                  </div>
                </td>
              </tr>
            ))}
            {!imports.length && (
              <tr>
                <td colSpan={5} className="muted">
                  no imports{showDone ? "" : " running"}
                </td>
              </tr>
            )}
          </tbody>
        </table>
        {log && (
          <div style={{ marginTop: 10 }}>
            <div className="row">
              <b>log of {log.id}</b>
              <button onClick={() => setLog(null)}>close</button>
            </div>
            <pre>{log.lines.join("\n") || "(empty)"}</pre>
          </div>
        )}
      </div>
      <div className="panel">
        <h2>Searches</h2>
        <table>
          <thead>
            <tr>
              <th>recording</th>
              <th>mode</th>
              <th>status</th>
              <th>progress</th>
              <th>started</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {searches.map((j) => (
              <tr key={j.id}>
                <td>
                  <div className="small">{j.recording ?? j.recording_id}</div>
                  <code className="small">{j.id}</code>
                </td>
                <td className="small">{j.mode}</td>
                <td>
                  <span className={`status ${j.status}`}>{j.status}</span>
                  {j.error && <div className="small err-text">{j.error}</div>}
                </td>
                <td>{ACTIVE.has(j.status) ? <SearchProgress job={j} /> : <span className="small muted">{j.stage}</span>}</td>
                <td className="small">{ago(j.created_at)}</td>
                <td>
                  <div className="row">
                    {ACTIVE.has(j.status) && <button className="danger" onClick={() => act(() => api.cancel(j.id))}>cancel</button>}
                    {["succeeded", "abstained"].includes(j.status) && <Link href={`/results/?run=${j.id}`}>results →</Link>}
                  </div>
                </td>
              </tr>
            ))}
            {!searches.length && (
              <tr>
                <td colSpan={6} className="muted">
                  no searches{showDone ? "" : " running"}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
