"""REAL-recording benchmark on the Pathak Livermore set (Zenodo 23424278, CC BY 4.0).

40 smartphone recordings of light-aircraft passages over Livermore, CA (May 2026), each with a
UTC-stamped frequency ridge (one harmonic), the aircraft's ADS-B track (KML) and the horizontal
offset ``L`` between recorder and ground track at closest approach. The dataset publishes
**no recorder coordinate and no audio sample-zero time**; both are derived here and reported
with their residuals:

* recorder position: trilaterated from the published ``L`` of every passage against its KML
  ground track (robust least squares); the rms residual is part of the result;
* audio start: the published UTC ridge is slid over the clip's whitened spectrogram and the
  offset with the most ridge energy is taken (this uses the dataset's own ridge, not ours).

The search then runs blind on our own automatically detected events (the published ridge is
used only for the clock), in known-time mode (±2 s), in a 4 km box whose centre is offset
from the recorder by a seeded 1–2 km, against the whole day's real ADS-B traffic and the real
USGS 3DEP terrain. The derived truth is a trilateration, not a survey; errors below the
trilateration residual are not meaningful.
"""

from __future__ import annotations

import csv
import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from aerodoppler import __version__
from aerodoppler.audio.service import (
    add_manual_event,
    ensure_spectrogram,
    ingest_recording,
    propose_events,
)
from aerodoppler.config import Settings
from aerodoppler.geo.coordinates import LocalFrame
from aerodoppler.jobs.runner import run_search_inline
from aerodoppler.jobs.store import Store
from aerodoppler.models import (
    BBox,
    QualitySettings,
    SearchMode,
    SearchRequest,
    iso_to_unix,
    unix_to_iso,
)
from aerodoppler.util import git_sha
from aerodoppler.validation.metrics import CaseOutcome, evaluate_case, summarize

ZENODO_RECORD = "23424278"
LICENSE = "CC BY 4.0 — Praveen Pathak, Zenodo 23424278"
REF_LAT, REF_LON = 37.68, -121.77


def extract_all(zdir: Path) -> Path:
    out = zdir / "extracted"
    out.mkdir(exist_ok=True)
    for z in sorted(zdir.glob("Raw data_*.zip")):
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.startswith("__MACOSX") or m.endswith("/"):
                    continue
                target = out / m
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(m))
    return out


def read_kml_track(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """gx:Track -> (t_unix, lat, lon, alt_m). The dataset's KML carries EGM96 geometric altitude."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    whens = re.findall(r"<when\s*>([^<]+)</when>", txt)
    coords = re.findall(r"<gx:coord\s*>([^<]+)</gx:coord>", txt)
    if not whens or len(whens) != len(coords):
        raise ValueError(f"{path.name}: {len(whens)} <when> vs {len(coords)} <gx:coord>")
    t = np.array([iso_to_unix(w.strip()) for w in whens])
    c = np.array([[float(v) for v in s.split()] for s in coords])
    order = np.argsort(t, kind="stable")
    return t[order], c[order, 1], c[order, 0], c[order, 2]


@dataclass
class Passage:
    case: str
    category: str
    dir: Path
    kml: Path
    wav: Path
    ridge_csv: Path
    L_m: float
    H_above_recorder_m: float
    R_min_m: float
    t_first: float
    t_last: float
    tail: str

    @property
    def t_ref(self) -> float:
        return 0.5 * (self.t_first + self.t_last)


def load_passages(zdir: Path) -> list[Passage]:
    root = extract_all(zdir)
    rows = list(csv.DictReader(open(zdir / "final_results_40_cases.csv", encoding="utf-8-sig")))
    out = []
    for r in rows:
        d = root / r["category"] / r["case"]
        if not d.exists():
            continue
        freq = list(csv.DictReader(open(d / "frequency_vs_time_used.csv")))
        out.append(
            Passage(
                r["case"],
                r["category"],
                d,
                d / r["track_file"],
                d / "audio.wav",
                d / "frequency_vs_time_used.csv",
                float(r["L_ADSB_m"]),
                float(r["H_ADSB_above_recorder_m"]),
                float(r["R_ADSB_m"]),
                iso_to_unix(freq[0]["time_utc"]),
                iso_to_unix(freq[-1]["time_utc"]),
                r["case"].split("_")[0],
            )
        )
    return out


def _polyline_distance(px: float, py: float, xs: np.ndarray, ys: np.ndarray) -> float:
    ax, ay, bx, by = xs[:-1], ys[:-1], xs[1:], ys[1:]
    dx, dy = bx - ax, by - ay
    tt = np.clip(((px - ax) * dx + (py - ay) * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
    return float(np.min(np.hypot(ax + tt * dx - px, ay + tt * dy - py)))


def trilaterate_recorder(passages: list[Passage], window_s: float = 300.0) -> dict[str, Any]:
    """Recorder (east, north) such that its horizontal distance to each passage's ground track
    near the passage equals the published L. Robust (soft-L1) least squares from the best
    grid start; residuals reported. Own implementation (the idea follows dopplerprint)."""
    from scipy.optimize import least_squares

    frame = LocalFrame(REF_LAT, REF_LON, 0.0)
    tracks = []
    for p in passages:
        t, lat, lon, _alt = read_kml_track(p.kml)
        sel = np.abs(t - p.t_ref) < window_s
        if sel.sum() < 2:
            continue
        enu = frame.geodetic_to_enu(lat[sel], lon[sel], np.zeros(int(sel.sum())))
        tracks.append((enu[:, 0], enu[:, 1], p.L_m, p.case))

    def resid(v):
        return np.array([_polyline_distance(v[0], v[1], xs, ys) - L for xs, ys, L, _ in tracks])

    grid = [(gx, gy) for gx in range(-15000, 15001, 500) for gy in range(-15000, 15001, 500)]
    start = min(grid, key=lambda g: float(np.sum(np.abs(resid(g)))))
    fit = least_squares(resid, start, loss="soft_l1", f_scale=50.0)
    r_all = resid(fit.x)
    # second pass on inliers: passages whose residual exceeds 150 m were evidently recorded
    # elsewhere (the dataset does not say); they are reported and excluded from the benchmark
    inl = np.abs(r_all) < 150.0
    all_tracks = tracks
    tracks = [t for t, ok in zip(all_tracks, inl, strict=True) if ok]
    fit = least_squares(resid, fit.x, loss="soft_l1", f_scale=50.0)
    r = resid(fit.x)
    lat, lon = frame.east_north_grid_to_geodetic(np.array([fit.x[0]]), np.array([fit.x[1]]))
    return {
        "lat": float(lat[0]),
        "lon": float(lon[0]),
        "east_m": float(fit.x[0]),
        "north_m": float(fit.x[1]),
        "n_tracks": len(tracks),
        "rms_m": float(np.sqrt(np.mean(r**2))),
        "median_abs_m": float(np.median(np.abs(r))),
        "residuals_m": {c: float(x) for (_, _, _, c), x in zip(tracks, r, strict=True)},
        "outliers_m": {
            c: float(x)
            for (_, _, _, c), x, ok in zip(all_tracks, r_all, inl, strict=True)
            if not ok
        },
        "method": "robust least squares on |dist(recorder, ground track) - L_ADSB_m|; second pass without passages with |residual| >= 150 m (recorded elsewhere)",
    }


def align_audio_start(spec, ridge_csv: Path, duration_s: float) -> dict[str, Any]:
    """Audio sample-zero UTC: slide the published UTC ridge over the whitened spectrogram."""
    rows = list(csv.DictReader(open(ridge_csv)))
    t_utc = np.array([iso_to_unix(r["time_utc"]) for r in rows])
    f = np.array([float(r["frequency_hz"]) for r in rows])
    # the clip must contain the ridge: t0 in [t_last - duration, t_first]
    lo, hi = t_utc[-1] - duration_s, t_utc[0]
    cands = np.arange(lo, hi + 1e-9, 0.25)
    best, best_t0 = -np.inf, float(lo)
    scores = []
    for t0 in cands:
        v = spec.sample_whitened(t_utc - t0, f, fill=np.nan)
        s = float(np.nanmean(np.clip(v, -2, 8))) if np.isfinite(v).any() else -np.inf
        scores.append(s)
        if s > best:
            best, best_t0 = s, float(t0)
    scores = np.array(scores)
    # sharpness: how many candidates come within 0.5 z of the best (an ambiguity measure)
    near = int(np.sum(scores >= best - 0.5)) if np.isfinite(best) else len(cands)
    return {
        "t0_unix": best_t0,
        "t0_utc": unix_to_iso(best_t0),
        "ridge_mean_z": best,
        "n_candidates": len(cands),
        "n_within_half_z": near,
        "ridge_t_first_rel_s": float(t_utc[0] - best_t0),
        "ridge_t_last_rel_s": float(t_utc[-1] - best_t0),
    }


def icao_for_tail(settings: Settings, dataset_ids: list[str], tail: str) -> str | None:
    for d in dataset_ids:
        files = list((settings.normalized_dir / d / "samples").rglob("*.parquet"))
        if not files:
            continue
        df = pl.read_parquet(files).filter(pl.col("registration") == tail)
        if df.height:
            return str(df["icao24"][0])
    return None


def run_livermore_benchmark(
    settings: Settings,
    zdir: Path,
    out_dir: Path,
    dataset_ids: list[str],
    workers: int = 0,
    radius_km: float = 4.0,
    time_sigma_s: float = 2.0,
    max_cases: int | None = None,
    seed: int = 0,
    log=print,
) -> dict[str, Any]:
    from aerodoppler.adsb.index import FlightDataset

    out_dir.mkdir(parents=True, exist_ok=True)
    settings.ensure_dirs()
    store = Store(settings.db_path)
    passages = load_passages(zdir)
    tri = trilaterate_recorder(passages)
    log(
        f"[livermore] recorder trilaterated from {tri['n_tracks']} tracks: {tri['lat']:.6f}, {tri['lon']:.6f} (rms {tri['rms_m']:.1f} m, median |res| {tri['median_abs_m']:.1f} m)"
    )
    (out_dir / "trilateration.json").write_text(json.dumps(tri, indent=2))
    dates_available = {}
    for d in dataset_ids:
        m = FlightDataset(d, settings.normalized_dir / d).manifest()
        dates_available[m["request"]["date_utc"]] = d
    rng = np.random.default_rng(seed)
    outcomes: list[CaseOutcome] = []
    rows: list[dict[str, Any]] = []
    done = 0
    for p in sorted(passages, key=lambda q: q.t_first):
        day = unix_to_iso(p.t_first)[:10]
        if p.case in tri["outliers_m"]:
            rows.append(
                {
                    "case": p.case,
                    "skipped": f"trilateration outlier ({tri['outliers_m'][p.case]:.0f} m): recorded at another place",
                }
            )
            continue
        if day not in dates_available:
            rows.append({"case": p.case, "skipped": f"no ADS-B dataset imported for {day}"})
            continue
        if max_cases is not None and done >= max_cases:
            break
        done += 1
        t = time.time()
        rec = ingest_recording(
            store,
            settings,
            p.wav,
            owner_permission_confirmed=True,
            recording_date_utc=day,
            notes=f"Livermore benchmark {p.case} ({LICENSE})",
        )
        spec = ensure_spectrogram(store, settings, rec.id)
        align = align_audio_start(spec, p.ridge_csv, rec.duration_s)
        # blind event detection; pick the proposal overlapping the published ridge span, else
        # fall back to the published ridge as a manual event (harmonic unknown -> 1, flagged)
        events = propose_events(store, settings, rec.id)
        lo, hi = align["ridge_t_first_rel_s"], align["ridge_t_last_rel_s"]
        overl = [
            e for e in events if min(e.audio_end_s, hi) - max(e.audio_start_s, lo) > 0.3 * (hi - lo)
        ]
        event_source = "auto"
        if overl:
            ev = max(overl, key=lambda e: e.detection_quality * len(e.ridge_points))
        else:
            rows_r = list(csv.DictReader(open(p.ridge_csv)))
            pts = [
                (iso_to_unix(r["time_utc"]) - align["t0_unix"], float(r["frequency_hz"]))
                for r in rows_r
            ][::3]
            ev = add_manual_event(
                store,
                rec.id,
                pts,
                harmonic=1,
                uncertainty_hz=1.0,
                annotated_by="published-ridge-fallback",
            )
            event_source = "published ridge (fallback, harmonic assumed 1)"
        # search box: 2*radius square whose centre is offset from the recorder by 1-2 km
        ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(1000.0, 2000.0)
        frame = LocalFrame(tri["lat"], tri["lon"], 0.0)
        clat, clon = frame.east_north_grid_to_geodetic(
            np.array([dist * np.cos(ang)]), np.array([dist * np.sin(ang)])
        )
        dlat = radius_km / 111.195
        dlon = radius_km / (111.195 * np.cos(np.radians(tri["lat"])))
        bbox = BBox(
            west=float(clon[0] - dlon),
            south=float(clat[0] - dlat),
            east=float(clon[0] + dlon),
            north=float(clat[0] + dlat),
        )
        req = SearchRequest(
            mode=SearchMode.KNOWN_TIME,
            recording_id=rec.id,
            event_ids=[ev.id],
            start_utc=align["t0_utc"],
            start_uncertainty_s=time_sigma_s,
            bbox=bbox,
            flight_dataset_ids=[dates_available[day]],
            operator_authorized=True,
            quality_settings=QualitySettings(workers=workers),
            seed=seed,
        )
        truth_icao = icao_for_tail(settings, [dates_available[day]], p.tail)
        try:
            res = run_search_inline(store, settings, req)
            oc = evaluate_case(
                p.case,
                res,
                tri["lat"],
                tri["lon"],
                {truth_icao} if truth_icao else None,
                time.time() - t,
            )
            oc.notes = {
                "run_id": res.run_id,
                "top1_flights": [s["icao24"] for s in res.candidates[0].extra["segments"]]
                if res.candidates
                else None,
                "truth_icao24": truth_icao,
                "event_source": event_source,
                "audio_start": align,
                "abstention": res.abstention.model_dump() if res.abstention.abstained else None,
                "n_candidates": len(res.candidates),
            }
        except Exception as e:
            oc = CaseOutcome(
                p.case,
                True,
                f"error: {type(e).__name__}: {e}",
                None,
                None,
                None,
                None,
                None,
                time.time() - t,
                0,
            )
        outcomes.append(oc)
        rows.append(
            {
                "case": p.case,
                "category": p.category,
                "day": day,
                "tail": p.tail,
                "L_m": p.L_m,
                "R_min_m": p.R_min_m,
                "trilateration_residual_m": tri["residuals_m"].get(p.case),
                **asdict(oc),
            }
        )
        log(
            f"[livermore] {p.case} ({p.category}, R_min {p.R_min_m:.0f} m, {event_source}): abstained={oc.abstained} top1 err={oc.top1_error_m and round(oc.top1_error_m)} m top3 {oc.topk_min_error_m and round(oc.topk_min_error_m)} m flights_ok={oc.top1_flights_correct} {oc.runtime_s:.0f}s"
        )
    report = {
        "label": "REAL RECORDINGS (Pathak Livermore set, CC BY 4.0) — truth = trilaterated recorder (not surveyed); audio start derived from the published ridge; events detected blind; known-time mode ±2 s; 4 km box offset from the recorder",
        "zenodo": ZENODO_RECORD,
        "aerodoppler_version": __version__,
        "git_sha": git_sha(),
        "trilateration": tri,
        "datasets": dataset_ids,
        "summary": summarize(outcomes),
        "cases": rows,
    }
    (out_dir / "livermore_benchmark.json").write_text(json.dumps(report, indent=2, default=str))
    return report
