"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { API_BASE, api } from "@/lib/api";

const links = [
  ["/", "Import"],
  ["/lab/", "Spectrogram lab"],
  ["/search/", "Search"],
  ["/results/", "Results"],
  ["/jobs/", "Jobs"],
  ["/data/", "Data & privacy"],
  ["/guide/", "Guide"],
];

export function Nav() {
  const path = usePathname();
  const [health, setHealth] = useState<{ version: string; offline: boolean; host: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [active, setActive] = useState(0);
  useEffect(() => {
    api.health().then((h) => setHealth(h)).catch((e) => setErr(String(e)));
    const poll = () =>
      api
        .jobs()
        .then((j) => setActive(j.counts.searches_active + j.counts.imports_active))
        .catch(() => undefined);
    poll();
    const t = setInterval(poll, 5000);
    return () => clearInterval(t);
  }, []);
  return (
    <nav className="nav">
      <span className="brand">AeroDoppler</span>
      {links.map(([href, label]) => (
        <Link key={href} href={href} className={path === href || (href !== "/" && path?.startsWith(href)) ? "active" : ""}>
          {label}
          {href === "/jobs/" && active > 0 && <span className="count" title="jobs running">{active}</span>}
        </Link>
      ))}
      <span className="spacer" />
      <span className="muted small">
        {health ? `local API v${health.version} on ${health.host}${health.offline ? " · OFFLINE" : ""}` : err ? `API not reachable at ${API_BASE || (typeof window !== "undefined" ? window.location.origin : "")} (${err}) — start it with \`make serve\`` : "connecting…"}
      </span>
    </nav>
  );
}
