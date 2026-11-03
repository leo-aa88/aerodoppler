"use client";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { Plot } from "@/components/Plot";
import { api, fmtNum, type DopplerEvent, type Recording, type SpectrogramData } from "@/lib/api";

function steepestTime(ev: DopplerEvent): number {
  // time of steepest descending slope on a lightly smoothed ridge ≈ delayed CPA
  const pts = ev.ridge_points;
  if (pts.length < 5) return 0.5 * (ev.audio_start_s + ev.audio_end_s);
  const k = Math.max(3, Math.round(8 / Math.max(0.05, pts[1].relative_t_s - pts[0].relative_t_s)) | 1);
  const f = pts.map((_, i) => {
    const lo = Math.max(0, i - (k >> 1)), hi = Math.min(pts.length, i + (k >> 1) + 1);
    let s = 0;
    for (let j = lo; j < hi; j++) s += pts[j].frequency_hz;
    return s / (hi - lo);
  });
  let best = 0, bi = -1;
  for (let i = 1; i < pts.length - 1; i++) {
    const sl = (f[i + 1] - f[i - 1]) / (pts[i + 1].relative_t_s - pts[i - 1].relative_t_s);
    if (sl < best) {
      best = sl;
      bi = i;
    }
  }
  return bi >= 0 ? pts[bi].relative_t_s : 0.5 * (ev.audio_start_s + ev.audio_end_s);
}

function Lab() {
  const params = useSearchParams();
  const rec = params.get("rec") ?? "";
  const [recording, setRecording] = useState<Recording | null>(null);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [events, setEvents] = useState<DopplerEvent[]>([]);
  const [spec, setSpec] = useState<SpectrogramData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [t0, setT0] = useState(0);
  const [t1, setT1] = useState(600);
  const [f0, setF0] = useState(20);
  const [f1, setF1] = useState(500);
  const [vmin, setVmin] = useState(-2);
  const [vmax, setVmax] = useState(8);
  const [layer, setLayer] = useState("whitened");
  const [drawing, setDrawing] = useState(false);
  const [draft, setDraft] = useState<[number, number][]>([]);
  const [harmonic, setHarmonic] = useState(1);
  const [showHarmonics, setShowHarmonics] = useState(true);
  const [selected, setSelected] = useState<string[]>([]);
  const [thr, setThr] = useState(2.5);
  const [minDur, setMinDur] = useState(40);

  useEffect(() => {
    api.recordings().then((r) => setRecordings(r.recordings)).catch(() => undefined);
  }, []);

  const loadEvents = useCallback(() => {
    if (!rec) return;
    api.events(rec).then((e) => setEvents(e.events)).catch((e) => setErr(String(e)));
  }, [rec]);

  useEffect(() => {
    if (!rec) return;
    api.recording(rec)
      .then((r) => {
        setRecording(r.recording);
        setEvents(r.events);
        setT1(Math.min(600, Math.ceil(r.recording.duration_s)));
      })
      .catch((e) => setErr(String(e)));
  }, [rec]);

  const loadSpec = useCallback(() => {
    if (!rec) return;
    setBusy(true);
    api.spectrogram(rec, { t0, t1, f0, f1, layer, max_frames: 1200, max_bins: 700 })
      .then((s) => {
        setSpec(s);
        setErr(null);
      })
      .catch((e) => setErr(String(e)))
      .finally(() => setBusy(false));
  }, [rec, t0, t1, f0, f1, layer]);

  useEffect(() => {
    if (recording) loadSpec();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recording]);

  const data = useMemo(() => {
    if (!spec) return [] as Partial<Plotly.PlotData>[];
    const z = spec.freqs.map((_, fi) => spec.times.map((_, ti) => spec.values[ti][fi]));
    const traces: Partial<Plotly.PlotData>[] = [{ type: "heatmap", x: spec.times, y: spec.freqs, z, zmin: vmin, zmax: vmax, colorscale: "Magma", colorbar: { title: { text: layer === "whitened" ? "z (robust)" : "dB" } }, hovertemplate: "t=%{x:.2f}s f=%{y:.2f}Hz z=%{z:.2f}<extra></extra>" }];
    const colors = ["#4cc9f0", "#06d6a0", "#ffd166", "#ef476f", "#c77dff", "#ffffff"];
    events.forEach((ev, i) => {
      const c = colors[i % colors.length];
      const pts = ev.ridge_points;
      traces.push({ type: "scatter", mode: "lines", x: pts.map((p) => p.relative_t_s), y: pts.map((p) => p.frequency_hz), line: { color: c, width: 1.5 }, name: `${ev.id} (h=${ev.ridge_harmonic}, ${ev.detection_method})`, hoverinfo: "name" });
      if (showHarmonics && selected.includes(ev.id)) {
        for (const h of [1, 2, 3, 4, 5, 6]) {
          if (h === ev.ridge_harmonic) continue;
          traces.push({ type: "scatter", mode: "lines", x: pts.map((p) => p.relative_t_s), y: pts.map((p) => (p.frequency_hz / ev.ridge_harmonic) * h), line: { color: c, width: 0.8, dash: "dot" }, name: `${ev.id} h${h}`, showlegend: false, hoverinfo: "name" });
        }
      }
    });
    if (draft.length) traces.push({ type: "scatter", mode: "lines+markers", x: draft.map((p) => p[0]), y: draft.map((p) => p[1]), marker: { color: "#ff6b6b", size: 7 }, line: { color: "#ff6b6b" }, name: "draft manual ridge" });
    return traces;
  }, [spec, events, draft, vmin, vmax, layer, showHarmonics, selected]);

  const layout = useMemo<Partial<Plotly.Layout>>(() => ({ title: { text: recording ? `${recording.synthetic ? "SYNTHETIC DEMO — " : ""}${recording.original_filename} · ${layer} spectrogram` : "" }, xaxis: { title: { text: "recording time (s)" } }, yaxis: { title: { text: "frequency (Hz)" } }, legend: { orientation: "h", y: -0.2 }, height: 560 }), [recording, layer]);

  const onClick = (x: number, y: number) => {
    if (!drawing) return;
    setDraft((d) => [...d, [x, y] as [number, number]].sort((a, b) => a[0] - b[0]));
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
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const detect = async () => {
    setBusy(true);
    try {
      const r = await api.detect(rec, { threshold_z: thr, min_duration_s: minDur, save: true });
      if (!r.events.length) setErr("No stable tonal ridge detected (abstention). Adjust threshold or annotate manually.");
      loadEvents();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
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

  return (
    <div>
      {recording?.synthetic && <div className="banner synthetic">SYNTHETIC DEMO DATA — generated audio and flights, not a real recording</div>}
      {err && <div className="banner danger">{err}</div>}
      <div className="panel">
        <div className="row">
          <div className="col">
            <label>t from (s)</label>
            <input type="number" value={t0} onChange={(e) => setT0(+e.target.value)} />
          </div>
          <div className="col">
            <label>t to (s)</label>
            <input type="number" value={t1} onChange={(e) => setT1(+e.target.value)} />
          </div>
          <div className="col">
            <label>f from (Hz)</label>
            <input type="number" value={f0} onChange={(e) => setF0(+e.target.value)} />
          </div>
          <div className="col">
            <label>f to (Hz)</label>
            <input type="number" value={f1} onChange={(e) => setF1(+e.target.value)} />
          </div>
          <div className="col">
            <label>layer</label>
            <select value={layer} onChange={(e) => setLayer(e.target.value)}>
              <option value="whitened">whitened (z)</option>
              <option value="power">power (dB)</option>
            </select>
          </div>
          <div className="col">
            <label>contrast min</label>
            <input type="number" value={vmin} onChange={(e) => setVmin(+e.target.value)} />
          </div>
          <div className="col">
            <label>contrast max</label>
            <input type="number" value={vmax} onChange={(e) => setVmax(+e.target.value)} />
          </div>
          <button className="primary" onClick={loadSpec} disabled={busy}>
            {busy ? "loading…" : "Update view"}
          </button>
          <span className="muted small">
            {recording ? `${recording.duration_s.toFixed(0)} s · analysis 2 kHz · ${spec ? `view pooled ${spec.pooled[0]}×${spec.pooled[1]}` : ""}` : ""}
          </span>
        </div>
        <Plot data={data} layout={layout} onClick={onClick} />
        <div className="row" style={{ marginTop: 8 }}>
          <button onClick={() => setDrawing((d) => !d)} className={drawing ? "primary" : ""}>
            {drawing ? "drawing: click ridge points on the plot" : "Draw manual ridge"}
          </button>
          <div className="col">
            <label>harmonic of the traced ridge</label>
            <input type="number" min={1} max={12} value={harmonic} onChange={(e) => setHarmonic(+e.target.value)} />
          </div>
          <button onClick={() => setDraft((d) => d.slice(0, -1))} disabled={!draft.length}>
            undo point
          </button>
          <button onClick={() => setDraft([])} disabled={!draft.length}>
            clear
          </button>
          <button className="primary" onClick={saveDraft} disabled={draft.length < 3 || busy}>
            Save manual event ({draft.length} pts)
          </button>
          <span className="spacer" />
          <div className="col">
            <label>auto threshold z</label>
            <input type="number" step={0.5} value={thr} onChange={(e) => setThr(+e.target.value)} />
          </div>
          <div className="col">
            <label>min duration (s)</label>
            <input type="number" value={minDur} onChange={(e) => setMinDur(+e.target.value)} />
          </div>
          <button onClick={detect} disabled={busy}>
            Auto-detect ridges
          </button>
          <label>
            <input type="checkbox" checked={showHarmonics} onChange={(e) => setShowHarmonics(e.target.checked)} /> harmonic overlay for selected
          </label>
        </div>
      </div>
      <div className="panel">
        <h2>Events ({events.length}) — select two for the separation readout</h2>
        <table>
          <thead>
            <tr>
              <th></th>
              <th>id</th>
              <th>window (s)</th>
              <th>steepest point (s)</th>
              <th>f0 prior (Hz)</th>
              <th>ridge h</th>
              <th>harmonics seen</th>
              <th>method</th>
              <th>quality</th>
              <th>points</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev) => (
              <tr key={ev.id} className={selected.includes(ev.id) ? "selected" : ""}>
                <td>
                  <input type="checkbox" checked={selected.includes(ev.id)} onChange={() => toggle(ev.id)} />
                </td>
                <td>
                  <code>{ev.id}</code>
                </td>
                <td>
                  {ev.audio_start_s.toFixed(1)} – {ev.audio_end_s.toFixed(1)}
                </td>
                <td>{steepestTime(ev).toFixed(1)}</td>
                <td>{fmtNum(ev.f0_prior, 1)}</td>
                <td>{ev.ridge_harmonic}</td>
                <td>{ev.candidate_harmonic_numbers.join(",")}</td>
                <td>
                  {ev.detection_method} {ev.annotated_by ? `(${ev.annotated_by})` : ""}
                </td>
                <td>{ev.detection_quality.toFixed(2)}</td>
                <td>{ev.ridge_points.length}</td>
                <td>
                  <button
                    className="danger"
                    onClick={() => {
                      api.deleteEvent(rec, ev.id).then(loadEvents).catch((e) => setErr(String(e)));
                    }}
                  >
                    delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {deltaT !== null && (
          <div className="banner ok" style={{ marginTop: 10 }}>
            Event separation Δt = <b>{deltaT.toFixed(1)} s</b> between the steepest-descent points of the two selected events (used as the Mode A lock).
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
