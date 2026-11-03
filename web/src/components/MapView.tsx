"use client";
import { useEffect, useRef, useState } from "react";
import type { Candidate, Track } from "@/lib/api";

// MapLibre is imported dynamically (browser only). Basemap tiles are OFF by default: loading
// remote tiles would reveal the analysed area to the tile host. The operator can enable OSM
// tiles explicitly.
type MapLibre = typeof import("maplibre-gl");

export interface MapViewProps {
  candidates: Candidate[];
  tracks: Track[];
  heat: [number, number, number, string][];
  truth?: { lat: number; lon: number } | null;
  selected?: string | null;
  onSelect?: (candidateId: string) => void;
  bbox?: number[] | null;
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

export function MapView({ candidates, tracks, heat, truth, selected, onSelect, bbox }: MapViewProps) {
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<import("maplibre-gl").Map | null>(null);
  const libRef = useRef<MapLibre | null>(null);
  const [basemap, setBasemap] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    import("maplibre-gl").then((lib) => {
      if (cancelled || !el.current) return;
      libRef.current = lib;
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
    m.once("styledata", () => draw());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [basemap, ready]);

  const draw = () => {
    const m = mapRef.current;
    const lib = libRef.current;
    if (!m || !lib || !m.isStyleLoaded()) return;
    const rm = (id: string) => {
      if (m.getLayer(id)) m.removeLayer(id);
      if (m.getSource(id)) m.removeSource(id);
    };
    ["heat", "tracks", "tracks-used", "cands", "cand-labels", "truth", "bbox"].forEach(rm);
    if (bbox && bbox.length === 4) {
      const [w, s, e, n] = bbox;
      m.addSource("bbox", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[w, s], [e, s], [e, n], [w, n], [w, s]] } } });
      m.addLayer({ id: "bbox", type: "line", source: "bbox", paint: { "line-color": "#9aa7b4", "line-width": 1, "line-dasharray": [2, 2] } });
    }
    if (heat.length) {
      const scores = heat.map((h) => h[2]);
      const lo = Math.min(...scores), hi = Math.max(...scores);
      m.addSource("heat", { type: "geojson", data: { type: "FeatureCollection", features: heat.map((h) => ({ type: "Feature", properties: { s: hi > lo ? (h[2] - lo) / (hi - lo) : 0.5 }, geometry: { type: "Point", coordinates: [h[1], h[0]] } })) } });
      m.addLayer({ id: "heat", type: "circle", source: "heat", paint: { "circle-radius": 3, "circle-opacity": 0.75, "circle-color": ["interpolate", ["linear"], ["get", "s"], 0, "#2c105c", 0.5, "#c43c4e", 1, "#fcffa4"] } });
    }
    if (tracks.length) {
      const fc = (used: boolean) => ({ type: "FeatureCollection" as const, features: tracks.filter((t) => t.used === used).map((t) => ({ type: "Feature" as const, properties: { label: `${t.icao24} ${t.type ?? "?"} ${t.callsign ?? ""}` }, geometry: { type: "LineString" as const, coordinates: t.lon.map((lo, i) => [lo, t.lat[i]]) } })) });
      m.addSource("tracks", { type: "geojson", data: fc(false) });
      m.addLayer({ id: "tracks", type: "line", source: "tracks", paint: { "line-color": "#5c7a99", "line-width": 1.2, "line-opacity": 0.8 } });
      m.addSource("tracks-used", { type: "geojson", data: fc(true) });
      m.addLayer({ id: "tracks-used", type: "line", source: "tracks-used", paint: { "line-color": "#06d6a0", "line-width": 2.2 } });
    }
    if (candidates.length) {
      m.addSource("cands", { type: "geojson", data: { type: "FeatureCollection", features: candidates.map((c) => ({ type: "Feature", properties: { id: c.candidate_id, rank: c.rank, sel: c.candidate_id === selected ? 1 : 0 }, geometry: { type: "Point", coordinates: [c.lon, c.lat] } })) } });
      m.addLayer({ id: "cands", type: "circle", source: "cands", paint: { "circle-radius": ["case", ["==", ["get", "sel"], 1], 11, 8], "circle-color": ["case", ["==", ["get", "rank"], 1], "#ef476f", "#f4a259"], "circle-stroke-color": "#fff", "circle-stroke-width": ["case", ["==", ["get", "sel"], 1], 3, 1] } });
      m.addLayer({ id: "cand-labels", type: "symbol", source: "cands", layout: { "text-field": ["to-string", ["get", "rank"]], "text-size": 11, "text-font": ["Open Sans Regular", "Arial Unicode MS Regular"] }, paint: { "text-color": "#000" } });
      m.on("click", "cands", (e) => {
        const f = e.features?.[0];
        if (f && onSelect) onSelect(String(f.properties?.id));
      });
    }
    if (truth) {
      m.addSource("truth", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [truth.lon, truth.lat] } } });
      m.addLayer({ id: "truth", type: "circle", source: "truth", paint: { "circle-radius": 7, "circle-color": "#ffffff", "circle-stroke-color": "#000", "circle-stroke-width": 2 } });
    }
    // fit bounds
    const pts: [number, number][] = [];
    candidates.forEach((c) => pts.push([c.lon, c.lat]));
    tracks.forEach((t) => t.lon.forEach((lo, i) => pts.push([lo, t.lat[i]])));
    if (bbox && bbox.length === 4) pts.push([bbox[0], bbox[1]], [bbox[2], bbox[3]]);
    if (pts.length) {
      const b = new lib.LngLatBounds(pts[0], pts[0]);
      pts.forEach((p) => b.extend(p));
      m.fitBounds(b, { padding: 40, duration: 0, maxZoom: 14 });
    }
  };

  useEffect(() => {
    if (ready) draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidates, tracks, heat, truth, selected, bbox, ready]);

  return (
    <div>
      <div className="row" style={{ marginBottom: 6 }}>
        <label>
          <input type="checkbox" checked={basemap} onChange={(e) => setBasemap(e.target.checked)} /> show OpenStreetMap basemap (contacts tile.openstreetmap.org and reveals the viewed area to that server)
        </label>
        <span className="muted small">red = rank 1, orange = other distinct maxima, green = flight tracks used by the top hypotheses, grey = other tracks, dots = score landscape</span>
      </div>
      <div ref={el} className="map" />
    </div>
  );
}
