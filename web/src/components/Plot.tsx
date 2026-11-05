"use client";
import { useEffect, useRef } from "react";

// Plotly is loaded only in the browser (static export has no window at build time).
type PlotlyModule = typeof import("plotly.js-dist-min");
let plotlyPromise: Promise<PlotlyModule> | null = null;
function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) plotlyPromise = import("plotly.js-dist-min").then((m) => (m as unknown as { default?: PlotlyModule }).default ?? (m as unknown as PlotlyModule));
  return plotlyPromise;
}

export interface PlotProps {
  data: Partial<Plotly.PlotData>[];
  layout: Partial<Plotly.Layout>;
  onClick?: (x: number, y: number) => void;
  className?: string;
  style?: React.CSSProperties;
}

export function Plot({ data, layout, onClick, className, style }: PlotProps) {
  const ref = useRef<HTMLDivElement>(null);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let cancelled = false;
    loadPlotly().then((Plotly) => {
      if (cancelled || !ref.current) return;
      Plotly.react(el, data, { paper_bgcolor: "#171d24", plot_bgcolor: "#0b0f14", font: { color: "#e6edf3", size: 12 }, margin: { l: 60, r: 20, t: 40, b: 50 }, ...layout, xaxis: { automargin: true, ...layout.xaxis }, yaxis: { automargin: true, ...layout.yaxis } }, { responsive: true, displaylogo: false });
      const handler = (ev: Plotly.PlotMouseEvent) => {
        const p = ev.points?.[0];
        if (p && clickRef.current) clickRef.current(Number(p.x), Number(p.y));
      };
      (el as unknown as { on: (e: string, h: (ev: Plotly.PlotMouseEvent) => void) => void }).on("plotly_click", handler);
    });
    return () => {
      cancelled = true;
    };
  }, [data, layout]);
  // The container must be exactly as tall as the figure, otherwise the bottom of the figure
  // (axis titles) spills over whatever follows the plot.
  return <div ref={ref} className={className ?? "plot"} style={{ ...(layout.height ? { height: layout.height } : {}), ...style }} />;
}
