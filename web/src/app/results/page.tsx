"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { MapView } from "@/components/MapView";
import { Plot } from "@/components/Plot";
import { api, fmtNum, type Candidate, type CandidatesResponse, type Curves, type SearchStatus, type SpectrogramData, type Track } from "@/lib/api";

function Results() {
  const params = useSearchParams();
  const run = params.get("run") ?? "";
  const [all, setAll] = useState<SearchStatus[]>([]);
  const [status, setStatus] = useState<SearchStatus | null>(null);
  const [res, setRes] = useState<CandidatesResponse | null>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [heat, setHeat] = useState<[number, number, number, string][]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [curves, setCurves] = useState<Curves | null>(null);
  const [spec, setSpec] = useState<SpectrogramData[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [showAllHeat, setShowAllHeat] = useState(false);

  useEffect(() => {
    api.searches().then((s) => setAll(s.searches)).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!run) return;
    api.search(run).then(setStatus).catch((e) => setErr(String(e)));
    api.candidates(run)
      .then((c) => {
        setRes(c);
        if (c.candidates.length) setSelected(c.candidates[0].candidate_id);
      })
      .catch((e) => setErr(String(e)));
    api.tracks(run).then(setTracks).catch(() => undefined);
    api.heatmap(run).then((h) => setHeat(h.points)).catch(() => undefined);
  }, [run]);
  useEffect(() => {
    if (!run || !selected || !status) return;
    api.curves(run, selected)
      .then(async (c) => {
        setCurves(c);
        const views: SpectrogramData[] = [];
        for (const ev of c.events) {
          const t0 = Math.max(0, Math.min(...ev.t_rel_s) - 10), t1 = Math.max(...ev.t_rel_s) + 10;
          const fs = Object.values(ev.predicted_hz_by_harmonic).flat();
          const f0 = Math.max(0, Math.min(...fs) * 0.8), f1 = Math.min(900, Math.max(...fs) * 1.1);
          views.push(await api.spectrogram(status.recording_id, { t0, t1, f0, f1, max_frames: 800, max_bins: 500 }));
        }
        setSpec(views);
      })
      .catch((e) => setErr(String(e)));
  }, [run, selected, status]);

  const cand: Candidate | undefined = res?.candidates.find((c) => c.candidate_id === selected);
  const heatShown = useMemo(() => (showAllHeat ? heat : heat.filter((h) => h[3] !== "coarse_all")), [heat, showAllHeat]);
  const bbox = (status?.request?.bbox as { west: number; south: number; east: number; north: number } | undefined) ?? null;

  if (!run) {
    return (
      <div className="panel">
        <h2>Results</h2>
        <table>
          <thead>
            <tr>
              <th>run</th>
              <th>recording</th>
              <th>mode</th>
              <th>status</th>
              <th>created</th>
            </tr>
          </thead>
          <tbody>
            {all.map((s) => (
              <tr key={s.id}>
                <td>
                  <Link href={`/results/?run=${s.id}`}>
                    <code>{s.id}</code>
                  </Link>
                </td>
                <td>
                  <code>{s.recording_id}</code>
                </td>
                <td>{String(s.request.mode)}</td>
                <td className={`status ${s.status}`}>{s.status}</td>
                <td className="small">{s.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  return (
    <div>
      {res?.synthetic && <div className="banner synthetic">SYNTHETIC DEMO — flights and audio were generated; the receiver position is known to the demo but hidden from the search</div>}
      {err && <div className="banner danger">{err}</div>}
      {status && !["succeeded", "abstained"].includes(status.status) && (
        <div className="banner warn">
          Job status: <span className={`status ${status.status}`}>{status.status}</span> ({status.stage}) {status.error ?? ""} — <Link href={`/search/?rec=${status.recording_id}`}>back to search</Link>
        </div>
      )}
      {res?.abstention?.abstained && (
        <div className="banner danger">
          <b>Abstained: {res.abstention.reason}.</b> {res.abstention.detail}
        </div>
      )}
      {res?.warnings?.length ? <div className="banner warn">{res.warnings.join(" · ")}</div> : null}
      {res && res.candidates.length > 0 && (
        <div className={`banner ${res.confidence_grade === "high" ? "ok" : res.confidence_grade === "medium" ? "warn" : "danger"}`}>
          <b>Confidence grade: {(res.confidence_grade ?? "low").toUpperCase()}</b> (graded verdict from the gates below, not a probability)
          {res.alternatives?.length ? (
            <div style={{ margin: "6px 0 0 0" }}>
              <b>Ambiguous:</b> the top-1 does not clearly beat {res.alternatives.length} other location{res.alternatives.length > 1 ? "s" : ""} (within 0.5 z, ≥ 1 km apart). Read the answer as one of:{" "}
              {res.alternatives.map((a, i) => (
                <span key={i}>
                  {i > 0 ? "; " : ""}
                  <code>{a.lat.toFixed(5)}, {a.lon.toFixed(5)}</code> ({(a.distance_from_top1_m / 1000).toFixed(1)} km away, −{a.gap_z.toFixed(2)} z, {a.icao24.join("+")})
                </span>
              ))}
            </div>
          ) : null}
          <ul className="small" style={{ margin: "6px 0 0 16px" }}>
            {(res.gates ?? []).map((g) => (
              <li key={g.name}>
                {g.passed ? "✓" : "✗"} <code>{g.name}</code> — {g.detail}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="panel">
        <h2>
          Run <code>{run}</code> · {res?.score_label} · <span className="muted small">{status?.cpu_seconds ? `CPU ${status.cpu_seconds}s · peak RSS ${status.peak_rss_mb} MB` : ""}</span>
        </h2>
        <div className="row">
          <a href={api.exportUrl(run, "json")} target="_blank" rel="noreferrer">
            <button>Export JSON</button>
          </a>
          <a href={api.exportUrl(run, "geojson")} target="_blank" rel="noreferrer">
            <button>Export GeoJSON</button>
          </a>
          <a href={api.exportUrl(run, "manifest")} target="_blank" rel="noreferrer">
            <button>Reproducibility manifest</button>
          </a>
          <a href={api.exportUrl(run, "zip")}>
            <button>Download run folder (zip)</button>
          </a>
          <label>
            <input type="checkbox" checked={showAllHeat} onChange={(e) => setShowAllHeat(e.target.checked)} /> show all coarse-stage points
          </label>
        </div>
      </div>
      <div className="grid2">
        <div className="panel">
          <h3>Map — distinct candidate maxima, flight tracks, score landscape</h3>
          <MapView candidates={res?.candidates ?? []} tracks={tracks} heat={heatShown} selected={selected} onSelect={setSelected} bbox={bbox ? [bbox.west, bbox.south, bbox.east, bbox.north] : null} />
        </div>
        <div className="panel">
          <h3>Ranked hypotheses</h3>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>lat, lon</th>
                <th>score</th>
                <th>per event</th>
                <th>coverage</th>
                <th>silence</th>
                <th>flights</th>
                <th>inferred start (UTC)</th>
                <th>f0 (Hz)</th>
                <th>grid</th>
                <th>landscape ±</th>
              </tr>
            </thead>
            <tbody>
              {res?.candidates.map((c) => (
                <tr key={c.candidate_id} className={`clickable ${c.candidate_id === selected ? "selected" : ""}`} onClick={() => setSelected(c.candidate_id)}>
                  <td>{c.rank}</td>
                  <td>
                    {c.lat.toFixed(5)}, {c.lon.toFixed(5)}
                    <br />
                    <span className="muted small">ground {fmtNum(c.ground_elevation_m, 0)} m</span>
                  </td>
                  <td>
                    <b>{c.score_total.toFixed(2)}</b>
                  </td>
                  <td>{c.score_components.per_event.map((v) => v.toFixed(2)).join(" / ")}</td>
                  <td>{c.score_components.coverage.map((v) => (v * 100).toFixed(0) + "%").join(" / ")}</td>
                  <td>{c.score_components.silence_penalty.toFixed(2)}</td>
                  <td className="small">
                    {c.extra.segments?.map((s) => (
                      <div key={s.segment_id}>
                        <span className={`tag ${s.class}`}>{s.class}</span>
                        {s.icao24} {s.type ?? "?"} {s.callsign ?? ""}
                      </div>
                    ))}
                  </td>
                  <td className="small">{c.inferred_start_utc}</td>
                  <td className="small">{c.source_fundamental_hz_per_event.map((f) => f.toFixed(2)).join(" / ")}</td>
                  <td className="small">{c.nominal_grid_spacing_m} m</td>
                  <td className="small">{c.estimated_uncertainty_m !== null ? `${c.estimated_uncertainty_m.toFixed(0)} m` : "–"}</td>
                </tr>
              ))}
              {!res?.candidates.length && (
                <tr>
                  <td colSpan={11} className="muted">
                    no candidates
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          {cand && (
            <div style={{ marginTop: 10 }}>
              <div className="muted small">{cand.uncertainty_label}: the radius of the region whose score stays within 0.15 z of the best; <b>not a calibrated confidence region</b>.</div>
              <div className="small" style={{ marginTop: 6 }}>
                flags: {cand.flags.map((f) => (
                  <span key={f} className="tag">
                    {f}
                  </span>
                ))}
              </div>
              {cand.extra.silence_log?.length ? (
                <details className="small" style={{ marginTop: 6 }}>
                  <summary>silence-constraint log ({cand.extra.silence_log.length} flights checked between the events)</summary>
                  <pre>{JSON.stringify(cand.extra.silence_log, null, 1)}</pre>
                </details>
              ) : null}
            </div>
          )}
        </div>
      </div>
      {curves && (
        <div className="panel">
          <h3>
            Observed vs predicted Doppler curves at <code>{curves.candidate_id}</code> (dashed = predicted harmonics at the fitted f0 with the shared clock origin; red dots = observed ridge)
          </h3>
          <div className="grid2">
            {curves.events.map((ev, i) => {
              const sp = spec[i];
              const data: Partial<Plotly.PlotData>[] = [];
              if (sp) data.push({ type: "heatmap", x: sp.times, y: sp.freqs, z: sp.freqs.map((_, fi) => sp.times.map((_, ti) => sp.values[ti][fi])), zmin: -2, zmax: 8, colorscale: "Magma", showscale: false, hoverinfo: "skip" });
              Object.entries(ev.predicted_hz_by_harmonic).forEach(([h, f]) => data.push({ type: "scatter", mode: "lines", x: ev.t_rel_s, y: f, line: { color: "#4cc9f0", width: 1, dash: "dash" }, name: `predicted h${h}` }));
              data.push({ type: "scatter", mode: "markers", x: ev.observed_ridge.t_rel_s, y: ev.observed_ridge.frequency_hz, marker: { color: "#ef476f", size: 3 }, name: `observed ridge (h=${ev.observed_ridge.harmonic})` });
              return (
                <div key={ev.event_id}>
                  <Plot data={data} layout={{ title: { text: `${ev.event_id} ← ${ev.segment_id.split(":").slice(1, 2).join("")} · f0=${ev.f0_hz.toFixed(2)} Hz · score ${ev.score.toFixed(2)} · coverage ${(ev.coverage * 100).toFixed(0)}% · delayed CPA ${ev.delayed_cpa_utc}` }, xaxis: { title: { text: "recording time (s)" } }, yaxis: { title: { text: "Hz" } }, height: 420, showlegend: false }} style={{ height: 420 }} />
                </div>
              );
            })}
          </div>
        </div>
      )}
      {res?.stage_counters.length ? (
        <div className="panel">
          <h3>Pruning stages</h3>
          <table>
            <thead>
              <tr>
                <th>stage</th>
                <th>in</th>
                <th>out</th>
                <th>note</th>
                <th>s</th>
              </tr>
            </thead>
            <tbody>
              {res.stage_counters.map((s, i) => (
                <tr key={i}>
                  <td>
                    <code>{s.stage}</code>
                  </td>
                  <td>{s.candidates_in}</td>
                  <td>{s.candidates_out}</td>
                  <td className="small">{s.note}</td>
                  <td>{s.elapsed_s}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<div className="muted">loading…</div>}>
      <Results />
    </Suspense>
  );
}
