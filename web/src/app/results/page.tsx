"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { MapView } from "@/components/MapView";
import { Plot } from "@/components/Plot";
import { api, errText, fmtNum, type Candidate, type CandidatesResponse, type Curves, type SearchStatus, type SpectrogramData, type Track } from "@/lib/api";

const GATE_TEXT: Record<string, string> = {
  core_matches: "every event fits clearly better than chance",
  clock_lock: "the events agree on one recording clock",
  margin: "no comparably good answer with different aircraft",
  margin_distinct_location: "no comparably good answer ≥ 1 km away",
  no_unheard_close_pass: "no loud aircraft passed close without being heard",
  radius: "the best-scoring region is compact",
  track_support: "the aircraft tracks really cover the matched passes",
  search_complete: "the search did not discard a comparable candidate",
  sensitivity_stable: "the answer survives speed-of-sound and height perturbations",
  event_shape: "the events show both Doppler plateaus",
  event_evidence: "every event shows a real Doppler transition",
};

function Verdict({ res }: { res: CandidatesResponse }) {
  const abst = res.abstention?.abstained;
  const grade = abst ? "abstain" : (res.confidence_grade ?? "low");
  const gates = [...(res.gates ?? [])].sort((a, b) => Number(a.passed) - Number(b.passed));
  const failed = gates.filter((g) => !g.passed);
  const cls = grade === "high" ? "ok" : grade === "medium" ? "warn" : "danger";
  const headline: Record<string, string> = {
    abstain: "No answer: the search abstained",
    low: "LOW — the coordinates below are NOT a verified location",
    medium: "MEDIUM — a plausible hypothesis, not an established location",
    high: "HIGH — every check passed (still an uncalibrated relative fit, not a probability)",
  };
  return (
    <div className={`banner verdict ${cls}`}>
      <h2>{headline[grade]}</h2>
      {abst && (
        <div>
          <b>{res.abstention?.reason}</b>: {res.abstention?.detail}
          {res.candidates.length > 0 && " The best candidates found are listed below for inspection only."}
        </div>
      )}
      {!abst && grade !== "high" && failed.length > 0 && (
        <div>
          Not established because: {failed.map((g) => GATE_TEXT[g.name] ?? g.name).join("; ")} — {failed.length === 1 ? "this check" : "these checks"} failed. Treat the ranked
          positions as hypotheses to verify, not as where the recording was made.
        </div>
      )}
      {res.alternatives?.length ? (
        <div style={{ margin: "6px 0 0 0" }}>
          <b>Ambiguous:</b> the top answer does not clearly beat {res.alternatives.length} other location{res.alternatives.length > 1 ? "s" : ""} (within 0.5 z, ≥ 1 km apart): read it as one of{" "}
          {res.alternatives.map((a, i) => (
            <span key={i}>
              {i > 0 ? "; " : ""}
              <code>
                {a.lat.toFixed(4)}, {a.lon.toFixed(4)}
              </code>{" "}
              ({(a.distance_from_top1_m / 1000).toFixed(1)} km away, −{a.gap_z.toFixed(2)} z, {a.icao24.join("+")})
            </span>
          ))}
        </div>
      ) : null}
      <details style={{ marginTop: 6 }}>
        <summary className="small">all checks ({gates.filter((g) => g.passed).length} of {gates.length} passed)</summary>
        <ul className="small" style={{ margin: "6px 0 0 16px" }}>
          {gates.map((g) => (
            <li key={g.name}>
              {g.passed ? "✓" : "✗"} <b>{GATE_TEXT[g.name] ?? g.name}</b> <code>{g.name}</code> — {g.detail}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function Context({ status, res }: { status: SearchStatus; res: CandidatesResponse | null }) {
  const r = status.request as {
    mode?: string;
    start_utc?: string | null;
    start_uncertainty_s?: number | null;
    date_utc?: string | null;
    bbox?: { west: number; south: number; east: number; north: number };
    flight_dataset_ids?: string[];
  };
  return (
    <div className="panel small">
      <div className="kv">
        <div>time assumption</div>
        <div>
          {r.mode === "known-time"
            ? `known start ${r.start_utc} ± ${r.start_uncertainty_s} s (taken as given — a wrong start time matches the wrong aircraft)`
            : `UTC day ${r.date_utc}, time of day unknown (fitted from the event separation)`}
        </div>
        <div>search scope</div>
        <div>
          {r.bbox
            ? `bounded to ${r.bbox.west.toFixed(3)}, ${r.bbox.south.toFixed(3)} → ${r.bbox.east.toFixed(3)}, ${r.bbox.north.toFixed(3)}: a location outside this area cannot be found, and if the area came from prior knowledge this is not a blind result`
            : "blind (no geographic prior)"}
        </div>
        <div>flight data</div>
        <div>{(r.flight_dataset_ids ?? []).join(", ") || "–"}</div>
        <div>scores</div>
        <div>{res?.score_label ?? "relative fit score (uncalibrated)"} — differences between candidates matter, absolute values do not</div>
      </div>
    </div>
  );
}

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
  const [zoomTo, setZoomTo] = useState<{ lat: number; lon: number; key: number } | null>(null);

  useEffect(() => {
    api.searches().then((s) => setAll(s.searches)).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!run) return;
    api.search(run).then(setStatus).catch((e) => setErr(errText(e)));
    api.candidates(run)
      .then((c) => {
        setRes(c);
        if (c.candidates.length) setSelected(c.candidates[0].candidate_id);
      })
      .catch((e) => setErr(errText(e)));
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
      .catch((e) => setErr(errText(e)));
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
          Job status: <span className={`status ${status.status}`}>{status.status}</span> ({status.stage}) {status.error ?? ""} — <Link href="/jobs/">jobs</Link> ·{" "}
          <Link href={`/search/?rec=${status.recording_id}`}>back to search</Link>
        </div>
      )}
      {res && (res.abstention?.abstained || res.candidates.length > 0) && (
        <Verdict res={res} />
      )}
      {status && <Context status={status} res={res} />}
      {res?.warnings?.length ? (
        <details className="banner warn small">
          <summary>{res.warnings.length} warning(s) from the search</summary>
          <ul>
            {res.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </details>
      ) : null}
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
          <MapView candidates={res?.candidates ?? []} tracks={tracks} heat={heatShown} selected={selected} onSelect={setSelected} bbox={bbox ? [bbox.west, bbox.south, bbox.east, bbox.north] : null} zoomTo={zoomTo} />
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
                <th title="radius of the region whose score stays within 0.15 z of the best — uncalibrated, not a confidence interval">score plateau (uncal.)</th>
              </tr>
            </thead>
            <tbody>
              {res?.candidates.map((c) => (
                <tr key={c.candidate_id} className={`clickable ${c.candidate_id === selected ? "selected" : ""}`} onClick={() => setSelected(c.candidate_id)}>
                  <td>
                    {c.rank}
                    <div>
                      <button
                        className="small-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelected(c.candidate_id);
                          setZoomTo({ lat: c.lat, lon: c.lon, key: Date.now() });
                        }}
                      >
                        zoom
                      </button>
                    </div>
                  </td>
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
                  <td className="small">{c.source_fundamental_hz_per_event.map((f) => (f == null ? "–" : f.toFixed(2))).join(" / ")}</td>
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
                  <div className="plot-title small" title={ev.segment_id}>
                    event {ev.event_id} ← aircraft {ev.segment_id.split(":").slice(1, 2).join("")} · f0 {ev.f0_hz.toFixed(2)} Hz · score {ev.score.toFixed(2)} · coverage{" "}
                    {(ev.coverage * 100).toFixed(0)} % · delayed closest approach {ev.delayed_cpa_utc}
                  </div>
                  <Plot data={data} layout={{ xaxis: { title: { text: "recording time (s)" } }, yaxis: { title: { text: "Hz" } }, height: 420, showlegend: false, margin: { l: 60, r: 20, t: 10, b: 50 } }} />
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
