"use client";
import { useEffect, useRef, useState } from "react";
import type { Candidate, Track } from "@/lib/api";

// MapLibre is imported dynamically (browser only). Basemap tiles are OFF by default: loading
// remote tiles would reveal the analysed area to the tile host. Without a basemap the map
// draws its own labelled latitude/longitude grid so positions and distances stay readable.
type MapLibre = typeof import("maplibre-gl");

export interface MapViewProps {
  candidates: Candidate[];
  tracks: Track[];
  heat: [number, number, number, string][];
  truth?: { lat: number; lon: number } | null;
  selected?: string | null;
  onSelect?: (candidateId: string) => void;
  bbox?: number[] | null;
  zoomTo?: { lat: number; lon: number; key: number } | null;
}

const BLANK_STYLE = {
  version: 8 as const,
  sources: {},
  layers: [{ id: "bg", type: "background" as const, paint: { "background-color": "#0b1118" } }],
};

const OSM_STYLE = {
  version: 8 as const,
  sources: { osm: { type: "raster" as const, tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, attribution: "© OpenStreetMap contributors" } },
  layers: [{ id: "osm", type: "raster" as const, source: "osm" }],
};

const LAYERS = ["heat", "tracks", "tracks-used", "cands", "cands-ring", "truth", "bbox", "grid"];

/** Graticule spaced to the viewed extent (0.01°…5°), as GeoJSON lines. */
function graticule(b: { w: number; s: number; e: number; n: number }) {
  const span = Math.max(b.e - b.w, b.n - b.s);
  const steps = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
  const step = steps.find((s) => span / s <= 8) ?? 10;
  const feats: GeoJSON.Feature[] = [];
  for (let x = Math.ceil(b.w / step) * step; x <= b.e; x += step)
    feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[x, b.s], [x, b.n]] } });
  for (let y = Math.ceil(b.s / step) * step; y <= b.n; y += step)
    feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[b.w, y], [b.e, y]] } });
  return { fc: { type: "FeatureCollection" as const, features: feats }, step };
}

export function MapView({ candidates, tracks, heat, truth, selected, onSelect, bbox, zoomTo }: MapViewProps) {
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<import("maplibre-gl").Map | null>(null);
  const libRef = useRef<MapLibre | null>(null);
  const [basemap, setBasemap] = useState(false);
  const [ready, setReady] = useState(false);
  const [showAllTracks, setShowAllTracks] = useState(false);
  const [showHeat, setShowHeat] = useState(true);
  const [showBbox, setShowBbox] = useState(true);
  const [labels, setLabels] = useState<{ x: number; y: number; text: string }[]>([]);
  const [hover, setHover] = useState<string | null>(null);
  const handlersAttached = useRef(false);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  useEffect(() => {
    let cancelled = false;
    import("maplibre-gl").then((lib) => {
      if (cancelled || !el.current) return;
      libRef.current = lib;
      // MapLibre 6 loads its worker as a separate module next to its own file, which the
      // bundler does not emit; serve the copy from public/ (see the copy-map-worker script).
      lib.setWorkerUrl("/maplibre-gl-worker.mjs");
      const m = new lib.Map({ container: el.current, style: BLANK_STYLE, center: [0, 0], zoom: 2, attributionControl: false });
      m.addControl(new lib.NavigationControl(), "top-right");
      m.addControl(new lib.ScaleControl({ unit: "metric" }), "bottom-left");
      m.addControl(new lib.AttributionControl({ compact: true }));
      m.on("load", () => setReady(true));
      mapRef.current = m;
    });
    return () => {
      cancelled = true;
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const m = mapRef.current;
    if (!m || !ready) return;
    m.setStyle(basemap ? OSM_STYLE : BLANK_STYLE);
    m.once("styledata", () => draw(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [basemap, ready]);

  // HTML labels (rank numbers, grid values) instead of map glyphs: no font server needed
  const relabel = () => {
    const m = mapRef.current;
    if (!m) return;
    const out: { x: number; y: number; text: string }[] = [];
    candidates.forEach((c) => {
      const p = m.project([c.lon, c.lat]);
      out.push({ x: p.x, y: p.y, text: `#${c.rank}` });
    });
    if (!basemap) {
      const b = m.getBounds();
      const { fc, step } = graticule({ w: b.getWest(), s: b.getSouth(), e: b.getEast(), n: b.getNorth() });
      const digits = step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 1 : 0;
      fc.features.forEach((f) => {
        const c = (f.geometry as GeoJSON.LineString).coordinates;
        const vertical = c[0][0] === c[1][0];
        const pt = vertical ? m.project([c[0][0], b.getSouth()]) : m.project([b.getWest(), c[0][1]]);
        out.push({ x: vertical ? pt.x + 2 : 4, y: vertical ? pt.y - 14 : pt.y - 2, text: vertical ? `${c[0][0].toFixed(digits)}°` : `${c[0][1].toFixed(digits)}°` });
      });
    }
    setLabels(out);
  };

  const drawGrid = () => {
    const m = mapRef.current;
    if (!m || !m.isStyleLoaded()) return;
    if (m.getLayer("grid")) m.removeLayer("grid");
    if (m.getSource("grid")) m.removeSource("grid");
    if (!basemap) {
      const b = m.getBounds();
      const { fc } = graticule({ w: b.getWest(), s: b.getSouth(), e: b.getEast(), n: b.getNorth() });
      m.addSource("grid", { type: "geojson", data: fc });
      m.addLayer({ id: "grid", type: "line", source: "grid", paint: { "line-color": "#2b3a4a", "line-width": 1 } }, m.getLayer("heat") ? "heat" : undefined);
    }
    relabel();
  };

  useEffect(() => {
    const m = mapRef.current;
    if (!m || !ready) return;
    const h = () => drawGrid();
    m.on("moveend", h);
    m.on("move", relabel);
    return () => {
      m.off("moveend", h);
      m.off("move", relabel);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, basemap, candidates]);

  const draw = (fit: boolean) => {
    const m = mapRef.current;
    const lib = libRef.current;
    if (!m || !lib || !m.isStyleLoaded()) return;
    // remove every layer before any source (MapLibre refuses to remove a source in use)
    LAYERS.forEach((id) => {
      if (m.getLayer(id)) m.removeLayer(id);
    });
    LAYERS.forEach((id) => {
      if (m.getSource(id)) m.removeSource(id);
    });
    if (showBbox && bbox && bbox.length === 4) {
      const [w, s, e, n] = bbox;
      m.addSource("bbox", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[w, s], [e, s], [e, n], [w, n], [w, s]] } } });
      m.addLayer({ id: "bbox", type: "line", source: "bbox", paint: { "line-color": "#9aa7b4", "line-width": 1.5, "line-dasharray": [3, 2] } });
    }
    if (showHeat && heat.length) {
      const scores = heat.map((h) => h[2]);
      const lo = Math.min(...scores), hi = Math.max(...scores);
      m.addSource("heat", { type: "geojson", data: { type: "FeatureCollection", features: heat.map((h) => ({ type: "Feature", properties: { s: hi > lo ? (h[2] - lo) / (hi - lo) : 0.5 }, geometry: { type: "Point", coordinates: [h[1], h[0]] } })) } });
      m.addLayer({ id: "heat", type: "circle", source: "heat", paint: { "circle-radius": 3, "circle-opacity": 0.7, "circle-color": ["interpolate", ["linear"], ["get", "s"], 0, "#2c105c", 0.5, "#c43c4e", 1, "#fcffa4"] } });
    }
    const usedTracks = tracks.filter((t) => t.used);
    const otherTracks = tracks.filter((t) => !t.used);
    const fc = (ts: Track[]) => ({
      type: "FeatureCollection" as const,
      features: ts.map((t) => ({ type: "Feature" as const, properties: { label: `${t.icao24} ${t.type ?? "?"} ${t.callsign ?? ""}` }, geometry: { type: "LineString" as const, coordinates: t.lon.map((lo, i) => [lo, t.lat[i]]) } })),
    });
    if (showAllTracks && otherTracks.length) {
      m.addSource("tracks", { type: "geojson", data: fc(otherTracks) });
      m.addLayer({ id: "tracks", type: "line", source: "tracks", paint: { "line-color": "#5c7a99", "line-width": 1, "line-opacity": 0.45 } });
    }
    if (usedTracks.length) {
      m.addSource("tracks-used", { type: "geojson", data: fc(usedTracks) });
      m.addLayer({ id: "tracks-used", type: "line", source: "tracks-used", paint: { "line-color": "#06d6a0", "line-width": 3 } });
    }
    if (candidates.length) {
      m.addSource("cands", {
        type: "geojson",
        data: { type: "FeatureCollection", features: candidates.map((c) => ({ type: "Feature", properties: { id: c.candidate_id, rank: c.rank, sel: c.candidate_id === selected ? 1 : 0 }, geometry: { type: "Point", coordinates: [c.lon, c.lat] } })) },
      });
      m.addLayer({ id: "cands-ring", type: "circle", source: "cands", paint: { "circle-radius": ["case", ["==", ["get", "sel"], 1], 15, 11], "circle-color": "#000", "circle-opacity": 0.6 } });
      m.addLayer({
        id: "cands",
        type: "circle",
        source: "cands",
        paint: { "circle-radius": ["case", ["==", ["get", "sel"], 1], 11, 8], "circle-color": ["case", ["==", ["get", "rank"], 1], "#ef476f", "#f4a259"], "circle-stroke-color": "#fff", "circle-stroke-width": ["case", ["==", ["get", "sel"], 1], 3, 1.5] },
      });
    }
    // layer-bound handlers survive layer removal/re-adding by id: attach them once
    if (!handlersAttached.current) {
      handlersAttached.current = true;
      m.on("mousemove", "tracks-used", (e) => setHover(String(e.features?.[0]?.properties?.label ?? "")));
      m.on("mouseleave", "tracks-used", () => setHover(null));
      m.on("click", "cands", (e) => {
        const f = e.features?.[0];
        if (f && onSelectRef.current) onSelectRef.current(String(f.properties?.id));
      });
    }
    if (truth) {
      m.addSource("truth", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [truth.lon, truth.lat] } } });
      m.addLayer({ id: "truth", type: "circle", source: "truth", paint: { "circle-radius": 7, "circle-color": "#ffffff", "circle-stroke-color": "#000", "circle-stroke-width": 2 } });
    }
    if (fit) {
      // frame what matters: the candidates, the tracks they use and the search area — not
      // every background track of the day
      const pts: [number, number][] = [];
      candidates.forEach((c) => pts.push([c.lon, c.lat]));
      usedTracks.forEach((t) => t.lon.forEach((lo, i) => pts.push([lo, t.lat[i]])));
      if (bbox && bbox.length === 4) pts.push([bbox[0], bbox[1]], [bbox[2], bbox[3]]);
      if (pts.length) {
        const b = new lib.LngLatBounds(pts[0], pts[0]);
        pts.forEach((p) => b.extend(p));
        m.fitBounds(b, { padding: 40, duration: 0, maxZoom: 14 });
      }
    }
    drawGrid();
  };

  useEffect(() => {
    if (ready) draw(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidates, tracks, truth, bbox, ready]);
  useEffect(() => {
    if (ready) draw(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [heat, selected, showAllTracks, showHeat, showBbox]);
  useEffect(() => {
    const m = mapRef.current;
    if (m && zoomTo) m.flyTo({ center: [zoomTo.lon, zoomTo.lat], zoom: 14, duration: 600 });
  }, [zoomTo]);

  const nOther = tracks.filter((t) => !t.used).length;
  return (
    <div>
      <div className="row small" style={{ marginBottom: 6 }}>
        <label>
          <input type="checkbox" checked={showAllTracks} onChange={(e) => setShowAllTracks(e.target.checked)} /> other aircraft tracks ({nOther})
        </label>
        <label>
          <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> score landscape points
        </label>
        <label>
          <input type="checkbox" checked={showBbox} onChange={(e) => setShowBbox(e.target.checked)} /> search area
        </label>
        <label>
          <input type="checkbox" checked={basemap} onChange={(e) => setBasemap(e.target.checked)} /> OpenStreetMap basemap (contacts tile.openstreetmap.org and reveals the viewed area to it)
        </label>
      </div>
      <div className="map-wrap">
        <div ref={el} className="map" />
        <div className="map-labels">
          {labels.map((l, i) => (
            <span key={i} style={{ left: l.x, top: l.y }} className={l.text.startsWith("#") ? "cand-label" : "grid-label"}>
              {l.text}
            </span>
          ))}
        </div>
        {hover && <div className="map-hover">{hover}</div>}
      </div>
      <div className="muted small" style={{ marginTop: 4 }}>
        red = rank 1, orange = other distinct maxima (click to select) · green = flight tracks used by the shown hypotheses · dots = score landscape (bright = higher) · dashed = search area
      </div>
    </div>
  );
}
