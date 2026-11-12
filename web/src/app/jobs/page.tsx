"use client";
import Link from "next/link";
import { Fragment, useCallback, useEffect, useState } from "react";
import { JOB_ACTIVE, api, errText, followJob, type Job, type JobsOverview } from "@/lib/api";

const TYPE_LABEL: Record<string, string> = {
  flight_import: "flight import",
  audio_ingest: "audio import",
  spectrogram_generation: "spectrogram",
  ridge_detection: "ridge detection",
  geolocation_search: "search",
  benchmark: "benchmark",
  diagnostic: "self-test",
};

const STAGE_TEXT: Record<string, string> = {
  queued: "waiting for a free slot / memory",
  claimed: "starting a worker",
  starting: "starting",
  locate: "discovering the archive",
  download: "downloading the daily archive",
  verify: "verifying the archive checksum",
  parse: "reading traces, filtering to the area",
  bucket: "grouping samples by aircraft",
  segment: "building flight segments",
  write: "writing the dataset",
  complete: "complete",
  decode: "decoding audio",
  stft: "computing the spectrogram",
  detect: "detecting ridges",
  waiting_to_retry: "waiting to retry",
};

const FILTERS: Record<string, string> = {
  active: "Active",
  queued: "Queued",
  "succeeded,abstained": "Completed",
  "failed,interrupted": "Failed / interrupted",
  cancelled: "Cancelled",
  "": "All",
};

function fmtS(s: number | null | undefined): string {
  if (s == null) return "–";
  if (s < 90) return `${s.toFixed(0)} s`;
  if (s < 5400) return `${(s / 60).toFixed(0)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

function ago(iso: string | null): string {
  return iso ? `${fmtS((Date.now() - Date.parse(iso)) / 1000)} ago` : "–";
}

function Bar({ value, total }: { value?: number; total?: number }) {
  if (!total) {
    return (
      <div className="bar indeterminate" title="total unknown — counters only">
        <div />
      </div>
    );
  }
  const pct = Math.min(100, (100 * (value ?? 0)) / total);
  return (
    <div className="bar" title={`${pct.toFixed(1)} %`}>
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}

function subject(j: Job): React.ReactNode {
  const p = j.params as Record<string, unknown>;
  if (j.type === "flight_import") {
    const b = p.bbox as { west: number; south: number; east: number; north: number } | undefined;
    return (
      <>
        <b>{String(p.date_utc)}</b>
        {b && (
          <div className="small muted">
            {b.west.toFixed(2)}, {b.south.toFixed(2)} → {b.east.toFixed(2)}, {b.north.toFixed(2)}
            {p.exclude_jets ? " · jets excluded" : ""}
          </div>
        )}
      </>
    );
  }
  if (j.type === "audio_ingest") return <span className="small">{String(p.original_filename ?? p.parent_recording_id ?? "")}</span>;
  if (j.type === "benchmark") return <span className="small">{String(p.kind ?? "")}</span>;
  return <span className="small">{j.recording ?? j.recording_id ?? ""}</span>;
}

function Progress({ j }: { j: Job }) {
  const st = j.progress?.stages ?? {};
  const cur = j.stage ?? "";
  const info = (st[cur] ?? {}) as Record<string, number | string | boolean | undefined>;
  const label = STAGE_TEXT[cur] ?? cur;
  let body: React.ReactNode = null;
  if (cur === "download") {
    body = (
      <>
        {((Number(info.bytes_done) || 0) / 1e9).toFixed(2)} / {((Number(info.bytes_total) || 0) / 1e9).toFixed(2)} GB
        {info.throughput_mb_s ? ` · ${info.throughput_mb_s} MB/s` : ""} · part {String(info.part ?? "")}
        <Bar value={Number(info.bytes_done)} total={Number(info.bytes_total) || undefined} />
      </>
    );
  } else if (cur === "parse") {
    body = (
      <>
        {Number(info.files_parsed ?? 0).toLocaleString()} trace files read · {Number(info.aircraft_kept ?? 0).toLocaleString()} aircraft in the area ·{" "}
        {Number(info.samples_kept ?? 0).toLocaleString()} positions · {String(info.workers ?? "?")} parsers · memory {Number(info.tree_rss_mb ?? 0).toFixed(0)} /{" "}
        {String(info.memory_budget_mb ?? "?")} MB{info.throttled ? " (throttled to stay within budget)" : ""}
        <Bar />
      </>
    );
  } else if (cur === "segment" || cur === "bucket") {
    const done = Number(info.bucket ?? info.parts_done ?? 0), tot = Number(info.buckets ?? info.parts_total ?? 0);
    body = (
      <>
        {done} / {tot} <Bar value={done} total={tot || undefined} />
      </>
    );
  } else if (typeof info.done === "number" && typeof info.total === "number") {
    body = (
      <>
        {Number(info.done).toLocaleString()} / {Number(info.total).toLocaleString()} candidates <Bar value={Number(info.done)} total={Number(info.total)} />
      </>
    );
  } else if (typeof info.step === "number") {
    body = <Bar value={Number(info.step)} total={Number(info.steps)} />;
  }
  return (
    <div className="small">
      <b>{label}</b>
      {j.status === "queued" && j.next_attempt_at && <span> · retry at {new Date(j.next_attempt_at).toLocaleTimeString()}</span>}
      {j.stalled && <span className="tag warn-tag">no heartbeat for &gt; 2 min — stalled?</span>}
      {body && <div>{body}</div>}
    </div>
  );
}

function Details({ j }: { j: Job }) {
  const [full, setFull] = useState<Job | null>(null);
  const [log, setLog] = useState<string[] | null>(null);
  useEffect(() => {
    api.job(j.id, true).then(setFull).catch(() => undefined);
    // live updates for the expanded job: SSE, with polling as fallback
    return followJob(j.id, (x) => api.job(x.id, true).then(setFull).catch(() => undefined));
  }, [j.id]);
  const x = full ?? j;
  return (
    <div className="job-details">
      <div className="kv small">
        <div>job</div>
        <div>
          <code>{x.id}</code> · {x.mode} · attempt {x.attempt}/{x.max_attempts} · class {x.resource_class} · reserves {(x.memory_reserved_mb / 1000).toFixed(1)} GB
        </div>
        <div>timing</div>
        <div>
          created {new Date(x.created_at).toLocaleString()} · waited {fmtS(x.queue_wait_s)} · ran {fmtS(x.duration_s)}
        </div>
        <div>resources</div>
        <div>
          memory {x.resources?.pss_mb?.toFixed(0) ?? "–"} MB (peak {x.resources?.peak_pss_mb?.toFixed(0) ?? "–"}) · CPU {x.resources?.cpu_s?.toFixed(0) ?? "–"} s · disk free{" "}
          {x.resources?.disk_free_gb ?? "–"} GB
        </div>
        {x.result && (
          <>
            <div>result</div>
            <div>
              <pre>{JSON.stringify(x.result, null, 1).slice(0, 2000)}</pre>
            </div>
          </>
        )}
        {x.error && (
          <>
            <div>error</div>
            <div>
              <div className="err-text">{x.error.message}</div>
              {x.error.details && (
                <details>
                  <summary className="small">technical details</summary>
                  <pre>{typeof x.error.details === "object" ? (String((x.error.details as Record<string, unknown>).traceback ?? "") || JSON.stringify(x.error.details, null, 1)) : String(x.error.details)}</pre>
                </details>
              )}
            </div>
          </>
        )}
      </div>
      <button className="small-btn" onClick={() => api.jobLogs(x.id).then((r) => setLog(r.lines)).catch(() => setLog(["(no log)"]))}>
        show log
      </button>
      {log && <pre>{log.join("\n") || "(empty)"}</pre>}
    </div>
  );
}

export default function JobsPage() {
  const [data, setData] = useState<JobsOverview | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState("active");
  const [type, setType] = useState("");
  const [open, setOpen] = useState<string | null>(null);

  const refresh = useCallback(() => {
    const q: Record<string, string> = { limit: "200" };
    if (filter) q.status = filter;
    if (type) q.type = type;
    api
      .jobs(q)
      .then((d) => {
        setData(d);
        setErr(null);
      })
      .catch((e) => setErr(errText(e)));
  }, [filter, type]);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [refresh]);

  const act = async (f: () => Promise<unknown>) => {
    try {
      await f();
      refresh();
    } catch (e) {
      setErr(errText(e));
    }
  };

  const r = data?.resources;
  const cap = data?.capacity;
  const c = data?.counts;
  return (
    <div>
      <h1 className="page-title">Jobs</h1>
      <p className="muted">
        Every import, analysis and search on this machine, for all recordings. Each job runs in its own process: reloading the page or restarting the server neither stops nor duplicates it.
        A job whose process disappeared (crash, out of memory, machine restart) is shown as <b>interrupted</b> and can be retried; downloaded archive parts are kept.
      </p>
      {err && <div className="banner danger">{err}</div>}
      {cap && !cap.scheduler.active && (
        <div className="banner warn">No job scheduler is running, so queued jobs wait. It runs inside <code>aerodoppler serve</code>; on a headless machine use <code>aerodoppler jobs scheduler</code>.</div>
      )}
      {c && cap && r && (
        <div className="panel">
          <div className="row">
            <div className="stat">
              <b>{c.running}</b> running
            </div>
            <div className="stat">
              <b>{c.queued}</b> queued
            </div>
            <div className="stat">
              <b>{c.failed + c.interrupted}</b> failed or interrupted
            </div>
            <div className="small muted">
              memory reserved by running jobs {(cap.memory_reserved_mb / 1000).toFixed(1)} / {(cap.memory_budget_mb / 1000).toFixed(1)} GB budget · slots{" "}
              {Object.entries(cap.class_limits)
                .map(([k, v]) => `${k} ${cap.running_by_class[k] ?? 0}/${v}`)
                .join(", ")}
              <br />
              machine: {(r.effective_available_mb / 1000).toFixed(1)} GB free of {(r.effective_limit_mb / 1000).toFixed(1)} GB usable
              {r.cgroup_limit_mb ? ` (container limit ${(r.cgroup_limit_mb / 1000).toFixed(1)} GB)` : ""}
              {r.wsl ? " · WSL2 (the Windows .wslconfig memory= setting caps this)" : ""} · {r.cpu_count} CPUs · {r.disk_free_gb} GB free disk
            </div>
          </div>
        </div>
      )}
      <div className="row" style={{ margin: "8px 0" }}>
        {Object.entries(FILTERS).map(([k, label]) => (
          <button key={k} className={filter === k ? "primary small-btn" : "small-btn"} onClick={() => setFilter(k)}>
            {label}
          </button>
        ))}
        <select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="">all types</option>
          {Object.entries(TYPE_LABEL).map(([k, v]) => (
            <option key={k} value={k}>
              {v}
            </option>
          ))}
        </select>
      </div>
      <div className="panel">
        <table>
          <thead>
            <tr>
              <th>job</th>
              <th>subject</th>
              <th>status</th>
              <th>progress</th>
              <th>resources</th>
              <th>started</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(data?.jobs ?? []).map((j) => (
              <Fragment key={j.id}>
                <tr className={open === j.id ? "selected" : ""}>
                  <td>
                    <b>{TYPE_LABEL[j.type] ?? j.type}</b>
                    <div className="small muted">
                      <code>{j.id}</code>
                      {j.mode !== "worker" ? ` · ${j.mode}` : ""}
                      {j.legacy ? " · from an older version" : ""}
                    </div>
                  </td>
                  <td>{subject(j)}</td>
                  <td>
                    <span className={`status ${j.status}`}>{j.status}</span>
                    {j.attempt > 1 && <div className="small muted">attempt {j.attempt}/{j.max_attempts}</div>}
                    {j.error?.message && <div className="small err-text">{j.error.message.slice(0, 220)}</div>}
                  </td>
                  <td>
                    {JOB_ACTIVE.has(j.status) ? (
                      <Progress j={j} />
                    ) : j.type === "flight_import" && j.result ? (
                      <span className="small">
                        dataset <code>{String(j.result.dataset_id)}</code>
                      </span>
                    ) : (
                      <span className="small muted">{STAGE_TEXT[j.stage ?? ""] ?? j.stage}</span>
                    )}
                  </td>
                  <td className="small">
                    {j.resources?.peak_pss_mb ? `${(j.resources.peak_pss_mb / 1000).toFixed(2)} GB peak` : "–"}
                    <div className="muted">CPU {fmtS(j.resources?.cpu_s)}</div>
                  </td>
                  <td className="small">
                    {ago(j.created_at)}
                    {j.queue_wait_s != null && j.queue_wait_s > 2 && <div className="muted">waited {fmtS(j.queue_wait_s)}</div>}
                  </td>
                  <td>
                    <div className="row tight">
                      {JOB_ACTIVE.has(j.status) && (
                        <button className="small-btn danger" disabled={j.cancel_requested} onClick={() => act(() => api.cancelJob(j.id))}>
                          {j.cancel_requested ? "cancelling…" : "cancel"}
                        </button>
                      )}
                      {["failed", "interrupted", "cancelled"].includes(j.status) && j.mode === "worker" && (
                        <button className="small-btn" onClick={() => act(() => api.retryJob(j.id))}>
                          retry
                        </button>
                      )}
                      {j.type === "geolocation_search" && ["succeeded", "abstained"].includes(j.status) && <Link href={`/results/?run=${j.id}`}>results →</Link>}
                      <button className="small-btn" onClick={() => setOpen(open === j.id ? null : j.id)}>
                        {open === j.id ? "hide" : "details"}
                      </button>
                    </div>
                  </td>
                </tr>
                {open === j.id && (
                  <tr key={`${j.id}-d`}>
                    <td colSpan={7}>
                      <Details j={j} />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {!data?.jobs.length && (
              <tr>
                <td colSpan={7} className="muted">
                  no jobs in this view
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
