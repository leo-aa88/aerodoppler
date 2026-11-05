"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Plot } from "@/components/Plot";
import { api, errText, fmtNum, type DetectionDiagnostics, type DopplerEvent, type EventAssessment, type Recording, type SpectrogramData } from "@/lib/api";

// Okabe–Ito colour-blind-safe palette; every ridge is drawn on a black halo so it stays
// visible on any colormap (the old 1.5 px cyan line vanished on blue/bright backgrounds).
const PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"];
const COLORMAPS = ["Magma", "Viridis", "Cividis", "Greys"];

const EVIDENCE: Record<EventAssessment["label"], { text: string; cls: string; help: string }> = {
  plausible_doppler: { text: "plausible aircraft Doppler", cls: "ev-good", help: "a descending frequency transition of several percent, as a passing aircraft produces" },
  weak_doppler: { text: "weak Doppler evidence", cls: "ev-weak", help: "some descent, but small or without both plateaus; usable, but it limits the confidence grade" },
  tonal_only: {
    text: "tonal ridge only",
    cls: "ev-bad",
    help: "a tonal line without a measurable Doppler transition. It may be a stationary source (hum, machinery, music) or an aircraft heard only on a plateau; either way it cannot locate the recording",
  },
};

function steepestTime(ev: DopplerEvent): number {
  const m = ev.assessment?.metrics?.steepest_point_s;
  if (typeof m === "number") return m;
  const pts = ev.ridge_points;
  return pts.length ? pts[Math.floor(pts.length / 2)].relative_t_s : 0.5 * (ev.audio_start_s + ev.audio_end_s);
}

function eventBox(ev: DopplerEvent, padFrac = 0.4) {
  const fs = ev.ridge_points.map((p) => p.frequency_hz);
  const lo = Math.min(...fs), hi = Math.max(...fs);
  const span = Math.max(ev.audio_end_s - ev.audio_start_s, 20);
  const fpad = Math.max(25, 0.25 * (hi - lo), 0.08 * hi);
  return {
    t0: Math.max(0, ev.audio_start_s - padFrac * span),
    t1: ev.audio_end_s + padFrac * span,
    f0: Math.max(0, Math.floor(lo - fpad)),
    f1: Math.ceil(hi + fpad),
  };
}

function Lab() {
  const params = useSearchParams();
  const rec = params.get("rec") ?? "";
  const [recording, setRecording] = useState<Recording | null>(null);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [events, setEvents] = useState<DopplerEvent[]>([]);
  const [spec, setSpec] = useState<SpectrogramData | null>(null);
  const [overview, setOverview] = useState<SpectrogramData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState({ t0: 0, t1: 600, f0: 20, f1: 500 });
  const [form, setForm] = useState({ t0: "0", t1: "600", f0: "20", f1: "500" });
  const [vmin, setVmin] = useState(-2);
  const [vmax, setVmax] = useState(8);
  const [cmap, setCmap] = useState("Magma");
  const [layer, setLayer] = useState("whitened");
  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<[number, number][]>([]);
  const [harmonic, setHarmonic] = useState(1);
  const [showHarmonics, setShowHarmonics] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);
  const [focus, setFocus] = useState<string | null>(null);
  const [thr, setThr] = useState(2.5);
  const [minDur, setMinDur] = useState(40);
  const [diag, setDiag] = useState<DetectionDiagnostics | null>(null);
  const [rejectedBox, setRejectedBox] = useState<{ t0: number; t1: number; f: number } | null>(null);
  const plotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.recordings().then((r) => setRecordings(r.recordings)).catch(() => undefined);
  }, []);

  const loadEvents = useCallback(() => {
    if (!rec) return Promise.resolve([] as DopplerEvent[]);
    return api
      .events(rec)
      .then((e) => {
        setEvents(e.events);
        return e.events;
      })
      .catch((e) => {
        setErr(errText(e));
        return [] as DopplerEvent[];
      });
  }, [rec]);

  const loadView = useCallback(
    (v: { t0: number; t1: number; f0: number; f1: number }) => {
      if (!rec) return;
      setView(v);
      setForm({ t0: v.t0.toFixed(0), t1: v.t1.toFixed(0), f0: v.f0.toFixed(0), f1: v.f1.toFixed(0) });
      setBusy(true);
      api
        .spectrogram(rec, { ...v, layer, max_frames: 1200, max_bins: 700 })
        .then((s) => {
          setSpec(s);
          setErr(null);
        })
        .catch((e) => setErr(errText(e)))
        .finally(() => setBusy(false));
    },
    [rec, layer],
  );

  useEffect(() => {
    if (!rec) return;
    api
      .recording(rec)
      .then((r) => {
        setRecording(r.recording);
        loadEvents();
        const dur = Math.ceil(r.recording.duration_s);
        loadView({ t0: 0, t1: Math.min(600, dur), f0: 20, f1: 500 });
        api
          .spectrogram(rec, { t0: 0, t1: dur, f0: 20, f1: 900, layer: "whitened", max_frames: 900, max_bins: 120 })
          .then(setOverview)
          .catch(() => undefined);
      })
      .catch((e) => setErr(errText(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rec]);

  const zoomTo = (ev: DopplerEvent) => {
    setFocus(ev.id);
    setSelected((s) => (s.includes(ev.id) ? s : [...s, ev.id].slice(-2)));
    loadView(eventBox(ev));
    plotRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const fitAll = () => recording && loadView({ t0: 0, t1: Math.ceil(recording.duration_s), f0: view.f0, f1: view.f1 });

  // keyboard: ← → pan by half a view, + − zoom time (ignored while typing in a field)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || !recording) return;
      const span = view.t1 - view.t0;
      const dur = recording.duration_s;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const d = (e.key === "ArrowRight" ? 0.5 : -0.5) * span;
        const t0 = Math.max(0, Math.min(dur - span, view.t0 + d));
        loadView({ ...view, t0, t1: t0 + span });
      } else if (e.key === "+" || e.key === "=" || e.key === "-") {
        const c = 0.5 * (view.t0 + view.t1);
        const ns = Math.max(10, Math.min(dur, e.key === "-" ? span * 2 : span / 2));
        const t0 = Math.max(0, c - ns / 2);
        loadView({ ...view, t0, t1: Math.min(dur, t0 + ns) });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view, recording, loadView]);

  const data = useMemo(() => {
    if (!spec) return [] as Partial<Plotly.PlotData>[];
    const z = spec.freqs.map((_, fi) => spec.times.map((_, ti) => spec.values[ti][fi]));
    const traces: Partial<Plotly.PlotData>[] = [
      {
        type: "heatmap",
        x: spec.times,
        y: spec.freqs,
        z,
        zmin: vmin,
        zmax: vmax,
        colorscale: cmap,
        colorbar: { title: { text: layer === "whitened" ? "level (robust z)" : "power (dB)" } },
        hovertemplate: "t=%{x:.2f} s  f=%{y:.2f} Hz  level=%{z:.2f}<extra></extra>",
      },
    ];
    const anyFocus = focus !== null || selected.length > 0;
    events.forEach((ev, i) => {
      const c = PALETTE[i % PALETTE.length];
      const pts = ev.ridge_points;
      const xs = pts.map((p) => p.relative_t_s), ys = pts.map((p) => p.frequency_hz);
      const hot = ev.id === focus || selected.includes(ev.id);
      const dim = anyFocus && !hot;
      const label = ev.assessment ? EVIDENCE[ev.assessment.label].text : "";
      traces.push({ type: "scatter", mode: "lines", x: xs, y: ys, line: { color: "#000", width: hot ? 9 : 7 }, opacity: dim ? 0.35 : 0.9, showlegend: false, hoverinfo: "skip" });
      traces.push({
        type: "scatter",
        mode: "lines",
        x: xs,
        y: ys,
        line: { color: c, width: hot ? 4 : 3 },
        opacity: dim ? 0.45 : 1,
        name: `${i + 1}: ${ev.audio_start_s.toFixed(0)}–${ev.audio_end_s.toFixed(0)} s · ${label}`,
        hovertemplate: `event ${i + 1} (${ev.id})<br>t=%{x:.1f} s f=%{y:.1f} Hz<br>${label}<extra></extra>`,
      });
      if (pts.length) {
        const tc = steepestTime(ev);
        let k = 0;
        for (let j = 0; j < pts.length; j++) if (Math.abs(pts[j].relative_t_s - tc) < Math.abs(pts[k].relative_t_s - tc)) k = j;
        traces.push({
          type: "scatter",
          mode: "markers",
          x: [xs[0], xs[k], xs[xs.length - 1]],
          y: [ys[0], ys[k], ys[ys.length - 1]],
          marker: { color: c, size: hot ? 13 : 10, symbol: ["triangle-right", "diamond", "square"], line: { color: "#000", width: 2 } },
          opacity: dim ? 0.45 : 1,
          showlegend: false,
          text: ["start", "steepest point ≈ closest approach", "end"],
          hovertemplate: `event ${i + 1}: %{text}<br>t=%{x:.1f} s f=%{y:.1f} Hz<extra></extra>`,
        });
      }
      if (showHarmonics && hot) {
        for (const h of [1, 2, 3, 4, 5, 6]) {
          if (h === ev.ridge_harmonic) continue;
          traces.push({ type: "scatter", mode: "lines", x: xs, y: pts.map((p) => (p.frequency_hz / ev.ridge_harmonic) * h), line: { color: c, width: 1.5, dash: "dot" }, name: `${ev.id} h${h}`, showlegend: false, hoverinfo: "name" });
        }
      }
    });
    if (rejectedBox) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: [rejectedBox.t0, rejectedBox.t1, rejectedBox.t1, rejectedBox.t0, rejectedBox.t0],
        y: [rejectedBox.f - 6, rejectedBox.f - 6, rejectedBox.f + 6, rejectedBox.f + 6, rejectedBox.f - 6],
        line: { color: "#fff", width: 2, dash: "dash" },
        name: "rejected candidate",
        hoverinfo: "name",
      });
    }
    if (draft.length)
      traces.push({ type: "scatter", mode: "lines+markers", x: draft.map((p) => p[0]), y: draft.map((p) => p[1]), marker: { color: "#ff4d6d", size: 9, line: { color: "#000", width: 2 } }, line: { color: "#ff4d6d", width: 3 }, name: "manual ridge (draft)" });
    return traces;
  }, [spec, events, draft, vmin, vmax, layer, cmap, showHarmonics, selected, focus, rejectedBox]);

  const layout = useMemo<Partial<Plotly.Layout>>(
    () => ({
      xaxis: { title: { text: "recording time (s)" }, range: [view.t0, view.t1] },
      yaxis: { title: { text: "frequency (Hz)" }, range: [view.f0, view.f1] },
      legend: { orientation: "h", x: 0, y: 1.02, xanchor: "left", yanchor: "bottom" },
      margin: { l: 60, r: 20, t: 36, b: 50 },
      height: 560,
    }),
    [view],
  );

  const overviewData = useMemo(() => {
    if (!overview) return [] as Partial<Plotly.PlotData>[];
    const z = overview.freqs.map((_, fi) => overview.times.map((_, ti) => overview.values[ti][fi]));
    const tr: Partial<Plotly.PlotData>[] = [{ type: "heatmap", x: overview.times, y: overview.freqs, z, zmin: -1, zmax: 6, colorscale: "Greys", reversescale: true, showscale: false, hoverinfo: "x" }];
    events.forEach((ev, i) => {
      tr.push({ type: "scatter", mode: "lines", x: [ev.audio_start_s, ev.audio_end_s], y: [880, 880], line: { color: PALETTE[i % PALETTE.length], width: 8 }, showlegend: false, hoverinfo: "text", text: `event ${i + 1}` });
    });
    tr.push({ type: "scatter", mode: "lines", x: [view.t0, view.t1, view.t1, view.t0, view.t0], y: [view.f0, view.f0, view.f1, view.f1, view.f0], line: { color: "#ff4d6d", width: 2 }, showlegend: false, hoverinfo: "skip" });
    return tr;
  }, [overview, events, view]);

  const onClick = (x: number, y: number) => {
    if (!drawing) return;
    setDraft((d) => [...d, [x, y] as [number, number]].sort((a, b) => a[0] - b[0]));
  };

  const onOverviewClick = (x: number) => {
    if (!recording) return;
    const span = view.t1 - view.t0;
    const t0 = Math.max(0, Math.min(recording.duration_s - span, x - span / 2));
    loadView({ ...view, t0, t1: t0 + span });
  };

  const saveDraft = async () => {
    if (draft.length < 3) return setErr("a manual event needs at least three points");
    setBusy(true);
    try {
      await api.addEvent(rec, { points: draft, harmonic, uncertainty_hz: 1.0, annotated_by: "operator" });
      setDraft([]);
      setDrawing(false);
      loadEvents();
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const detect = async () => {
    setBusy(true);
    setErr(null);
    setInfo(null);
    try {
      const r = await api.detect(rec, { threshold_z: thr, min_duration_s: minDur, save: true });
      setDiag(r.diagnostics);
      const evs = await loadEvents();
      if (r.events.length) {
        setInfo(`${r.events.length} candidate event(s) proposed — check each one's evidence label below before using it.`);
        const first = evs.find((e) => e.id === r.events[0].id);
        if (first) zoomTo(first);
      }
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const review = async (ev: DopplerEvent, accepted: boolean) => {
    try {
      await api.reviewEvent(rec, ev.id, accepted, null);
      loadEvents();
    } catch (e) {
      setErr(errText(e));
    }
  };

  const toggle = (id: string) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-2)));
  const sel = events.filter((e) => selected.includes(e.id));
  const deltaT = sel.length === 2 ? Math.abs(steepestTime(sel[1]) - steepestTime(sel[0])) : null;

  if (!rec) {
    return (
      <div className="panel">
        <h2>Spectrogram lab</h2>
        <p>Select a recording:</p>
        <ul>
          {recordings.map((r) => (
            <li key={r.id}>
              <Link href={`/lab/?rec=${r.id}`}>
                {r.synthetic ? "[SYNTHETIC] " : ""}
                {r.original_filename} ({r.id})
              </Link>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  const applyForm = () => loadView({ t0: +form.t0, t1: +form.t1, f0: +form.f0, f1: +form.f1 });

  return (
    <div>
      {recording?.synthetic && <div className="banner synthetic">SYNTHETIC DEMO DATA — generated audio and flights, not a real recording</div>}
      {err && <div className="banner danger">{err}</div>}
      {info && <div className="banner ok">{info}</div>}
      {recording && (recording.truncated || recording.segment_start_s > 0) && (
        <div className="banner warn small">
          This recording is the part {recording.segment_start_s.toFixed(0)}–{(recording.segment_start_s + recording.duration_s).toFixed(0)} s of a {recording.source_duration_s?.toFixed(0)} s
          file. Times below are relative to {recording.segment_start_s.toFixed(0)} s into the file.
        </div>
      )}
      <div className="panel" ref={plotRef}>
        <div className="row">
          {(["t0", "t1", "f0", "f1"] as const).map((k) => (
            <div className="col narrow" key={k}>
              <label>{{ t0: "view from (s)", t1: "view to (s)", f0: "freq. from (Hz)", f1: "freq. to (Hz)" }[k]}</label>
              <input type="number" value={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.value })} onKeyDown={(e) => e.key === "Enter" && applyForm()} />
            </div>
          ))}
          <button className="primary" onClick={applyForm} disabled={busy}>
            {busy ? "loading…" : "Show"}
          </button>
          <button onClick={fitAll} disabled={busy}>
            whole recording
          </button>
          <div className="col narrow">
            <label>frequency preset</label>
            <select onChange={(e) => e.target.value && loadView({ ...view, ...JSON.parse(e.target.value) })} value="">
              <option value="">…</option>
              <option value='{"f0":20,"f1":250}'>20–250 Hz (light-aircraft engines)</option>
              <option value='{"f0":20,"f1":500}'>20–500 Hz (engines + propellers)</option>
              <option value='{"f0":50,"f1":900}'>50–900 Hz (harmonics)</option>
            </select>
          </div>
          <div className="col narrow">
            <label>layer</label>
            <select
              value={layer}
              onChange={(e) => {
                setLayer(e.target.value);
                if (e.target.value === "power") {
                  setVmin(-40);
                  setVmax(20);
                } else {
                  setVmin(-2);
                  setVmax(8);
                }
              }}
            >
              <option value="whitened">whitened (noise-normalised)</option>
              <option value="power">power (dB)</option>
            </select>
          </div>
          <div className="col narrow">
            <label>colours</label>
            <select value={cmap} onChange={(e) => setCmap(e.target.value)}>
              {COLORMAPS.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </div>
          <div className="col narrow">
            <label>contrast min / max</label>
            <div className="row tight">
              <input type="number" value={vmin} onChange={(e) => setVmin(+e.target.value)} style={{ width: 64 }} />
              <input type="number" value={vmax} onChange={(e) => setVmax(+e.target.value)} style={{ width: 64 }} />
            </div>
          </div>
        </div>
        {overview && (
          <div className="overview" title="whole recording: click to move the view there">
            <Plot
              data={overviewData}
              layout={{ height: 110, margin: { l: 60, r: 20, t: 4, b: 22 }, xaxis: { range: [0, recording?.duration_s ?? 1], fixedrange: true }, yaxis: { fixedrange: true, showticklabels: false } }}
              onClick={(x) => onOverviewClick(x)}
              config={{ displayModeBar: false }}
            />
          </div>
        )}
        {recording && (
          <div className="plot-title" title={recording.original_filename}>
            {recording.synthetic ? "SYNTHETIC DEMO — " : ""}
            {recording.original_filename} · {layer === "whitened" ? "noise-normalised" : "power"} spectrogram
            <span className="muted small">
              {" "}
              · {recording.duration_s.toFixed(0)} s analysed · {spec ? `display pooled ${spec.pooled[0]}×${spec.pooled[1]}` : ""} · keys: ← → pan, + − zoom
            </span>
          </div>
        )}
        <Plot data={data} layout={layout} onClick={onClick} />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={() => setDrawing((d) => !d)} className={drawing ? "primary" : ""}>
            {drawing ? "drawing: click points along a ridge" : "Draw a ridge by hand"}
          </button>
          <div className="col narrow">
            <label>its harmonic number</label>
            <input type="number" min={1} max={12} value={harmonic} onChange={(e) => setHarmonic(+e.target.value)} />
          </div>
          <button onClick={() => setDraft((d) => d.slice(0, -1))} disabled={!draft.length}>
            undo point
          </button>
          <button onClick={() => setDraft([])} disabled={!draft.length}>
            clear
          </button>
          <button className="primary" onClick={saveDraft} disabled={draft.length < 3 || busy}>
            Save drawn event ({draft.length} points)
          </button>
          <span className="spacer" />
          <div className="col narrow">
            <label>detector threshold (z)</label>
            <input type="number" step={0.5} value={thr} onChange={(e) => setThr(+e.target.value)} />
          </div>
          <div className="col narrow">
            <label>min. duration (s)</label>
            <input type="number" value={minDur} onChange={(e) => setMinDur(+e.target.value)} />
          </div>
          <button onClick={detect} disabled={busy}>
            Auto-detect (whole recording)
          </button>
          <label>
            <input type="checkbox" checked={showHarmonics} onChange={(e) => setShowHarmonics(e.target.checked)} /> harmonic guides for selected
          </label>
        </div>
      </div>
      {diag && (
        <div className="panel">
          <h2>Last automatic detection</h2>
          <p className="small">
            Scanned the <b>whole recording</b> ({diag.scanned.t_start_s.toFixed(0)}–{diag.scanned.t_end_s.toFixed(0)} s, {diag.scanned.f_min_hz.toFixed(0)}–{diag.scanned.f_max_hz.toFixed(0)} Hz)
            regardless of the view, with threshold z = {diag.parameters.threshold_z} and minimum duration {diag.parameters.min_duration_s} s. Proposed <b>{diag.proposals}</b> event(s) (
            {diag.from_ridge_tracker} from the ridge tracker, {diag.from_template_fit} from the straight-pass template fit)
            {Object.keys(diag.evidence_labels).length ? `: ${Object.entries(diag.evidence_labels).map(([k, v]) => `${v} ${EVIDENCE[k as EventAssessment["label"]]?.text ?? k}`).join(", ")}` : ""}. Rejected:{" "}
            {Object.keys(diag.rejected_counts).length ? Object.entries(diag.rejected_counts).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(", ") : "none"}.
          </p>
          {diag.suggestions.length > 0 && (
            <ul className="small">
              {diag.suggestions.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          )}
          {diag.rejected.length > 0 && (
            <details>
              <summary className="small">rejected candidates ({diag.rejected.length} strongest shown)</summary>
              <table>
                <thead>
                  <tr>
                    <th>time (s)</th>
                    <th>frequency</th>
                    <th>level</th>
                    <th>why rejected</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {diag.rejected.slice(0, 25).map((r, i) => (
                    <tr key={i}>
                      <td>
                        {r.t_start_s.toFixed(1)}–{r.t_end_s.toFixed(1)}
                      </td>
                      <td>
                        {r.f_mean_hz.toFixed(1)} Hz (range {r.f_range_hz.toFixed(1)} Hz)
                      </td>
                      <td>{r.mean_z.toFixed(1)}</td>
                      <td className="small">{r.reason_text}</td>
                      <td>
                        <button
                          className="small-btn"
                          onClick={() => {
                            setRejectedBox({ t0: r.t_start_s, t1: r.t_end_s, f: r.f_mean_hz });
                            const span = Math.max(r.t_end_s - r.t_start_s, 20);
                            loadView({ t0: Math.max(0, r.t_start_s - 0.5 * span), t1: r.t_end_s + 0.5 * span, f0: Math.max(0, r.f_mean_hz - 60), f1: r.f_mean_hz + 60 });
                          }}
                        >
                          show
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )}
        </div>
      )}
      <div className="panel">
        <h2>Events ({events.length})</h2>
        <p className="small muted">
          A detected ridge is a <b>tonal ridge candidate</b>; only a descending Doppler transition makes it evidence of a passing aircraft. Select two events for their time separation (used by the
          unknown-time search). Click <b>zoom</b> to inspect an event.
        </p>
        <table>
          <thead>
            <tr>
              <th></th>
              <th>#</th>
              <th>time (s)</th>
              <th>evidence</th>
              <th>shape</th>
              <th>f0 guess</th>
              <th>source</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev, i) => {
              const a = ev.assessment;
              const m = a?.metrics ?? {};
              return (
                <tr key={ev.id} className={selected.includes(ev.id) || focus === ev.id ? "selected" : ""}>
                  <td>
                    <input type="checkbox" checked={selected.includes(ev.id)} onChange={() => toggle(ev.id)} />
                  </td>
                  <td>
                    <span className="swatch" style={{ background: PALETTE[i % PALETTE.length] }} /> {i + 1}
                    <div className="small muted">
                      <code>{ev.id}</code>
                    </div>
                  </td>
                  <td>
                    {ev.audio_start_s.toFixed(1)} – {ev.audio_end_s.toFixed(1)}
                    <div className="small muted">steepest {steepestTime(ev).toFixed(1)}</div>
                  </td>
                  <td style={{ maxWidth: 360 }}>
                    {a && (
                      <>
                        <span className={`tag ${EVIDENCE[a.label].cls}`} title={EVIDENCE[a.label].help}>
                          {EVIDENCE[a.label].text}
                        </span>
                        {a.review && <span className="tag">{a.review.accepted ? "you accepted it as an aircraft" : "you rejected it"}</span>}
                        {a.reasons.length > 0 && <div className="small muted">{a.reasons.join("; ")}</div>}
                      </>
                    )}
                  </td>
                  <td className="small">
                    change {fmtNum(m.descent_pct, 2)} % at the steepest point (total {fmtNum(m.excursion_pct, 2)} %)
                    <br />
                    {m.f_range_hz?.toFixed(1)} Hz range · {m.plateaus_reached ? "plateaus reached" : "plateaus not reached"} · {m.harmonics_seen} harmonic(s)
                  </td>
                  <td>{fmtNum(ev.f0_prior, 1)} Hz</td>
                  <td className="small">
                    {ev.detection_method}
                    <br />
                    quality {ev.detection_quality.toFixed(2)} · {ev.ridge_points.length} pts
                  </td>
                  <td>
                    <div className="row tight">
                      <button className="small-btn" onClick={() => zoomTo(ev)}>
                        zoom
                      </button>
                      {a?.label === "tonal_only" && !a.review?.accepted && (
                        <button className="small-btn" onClick={() => review(ev, true)} title="only if you have checked that this is an aircraft pass">
                          I checked: aircraft
                        </button>
                      )}
                      {a?.review?.accepted && (
                        <button className="small-btn" onClick={() => review(ev, false)}>
                          undo acceptance
                        </button>
                      )}
                      <button className="small-btn danger" onClick={() => api.deleteEvent(rec, ev.id).then(() => loadEvents()).catch((e) => setErr(errText(e)))}>
                        delete
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {!events.length && (
              <tr>
                <td colSpan={8} className="muted">
                  no events yet — run the automatic detection or draw a ridge by hand
                </td>
              </tr>
            )}
          </tbody>
        </table>
        {deltaT !== null && (
          <div className="banner ok" style={{ marginTop: 10 }}>
            Separation between the steepest points of the two selected events: <b>{deltaT.toFixed(1)} s</b> (the unknown-time search matches it against pairs of aircraft passes).
          </div>
        )}
        <p>
          <Link href={`/search/?rec=${rec}${selected.length ? `&events=${selected.join(",")}` : ""}`}>→ set up a search with {selected.length ? "the selected events" : "these events"}</Link>
        </p>
      </div>
    </div>
  );
}

export default function LabPage() {
  return (
    <Suspense fallback={<div className="muted">loading…</div>}>
      <Lab />
    </Suspense>
  );
}
