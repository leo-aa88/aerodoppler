"""Performance baseline of the search on fixed workloads (opt-in, local; minutes to an hour).

Runs the existing benchmarks on small fixed subsets and collects, for every search they ran:
wall time, time per funnel stage, ultra-coarse screen throughput (candidates scored per second)
and peak memory of the whole process tree, plus the answer (error, abstention) so a faster
version can be checked for identical results.

    uv run python scripts/perf_baseline.py --out data/runs/perf/before --workloads W1,W2
    uv run python scripts/perf_baseline.py --out data/runs/perf/before --workloads W4 \\
        --livermore data/raw/zenodo_23424278 --datasets adsblol_2026-05-17_c432e6e8,adsblol_2026-05-16_c432e6e8

W1  SYNTHETIC cases base_snr10 and many_distractors (Mode A and B)
SYN the full SYNTHETIC benchmark (every case of ``aerodoppler benchmark``)
W2  semi-synthetic real-trajectory cases deland_s1 and n2920w8115_s1 (central Florida,
    dataset adsblol_2026-11-02_1e2f2f08, ~1,250 segments)
W3  semi-synthetic real-trajectory case in Manhattan (dataset adsblol_2026-11-04_262826ac,
    jets and helicopters included): the dense-traffic memory workload
R   a recorded search request replayed as is (``--request``: a request.json of an earlier run;
    for private recordings, never committed)
W4  the Livermore real-recording set (known time)

``--replay <perf_baseline.json>`` re-runs every search request recorded by an earlier baseline
(same recording, same events, same request) and compares the answers: the before/after check
for performance work. Rebuilding scenarios is not used for that because event detection on a
re-synthesised scene is not bit-reproducible (tiny float differences can flip a near-tie in
the event-pair choice).

Writes ``<out>/perf_baseline.json`` and prints one line per search.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

SCREEN = re.compile(r"(\d+) of (\d+) candidates .*? scored in ([\d.]+) s")


def _search_rows(
    settings, since: float, workload: str, job_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    from aerodoppler.jobs.service import queue_for

    q = queue_for(settings)
    if job_ids is not None:
        jobs = [j for j in (q.get(i) for i in job_ids) if j is not None]
    else:
        jobs = q.list(types=["geolocation_search"], created_after=since, limit=1000)
    rows = []
    for j in jobs:
        res = getattr(j, "result", None) or {}
        rdir = Path(res.get("result_dir") or "")
        stages: dict[str, float] = {}
        screen = None
        top1 = None
        if (rdir / "result.json").exists():
            r = json.loads((rdir / "result.json").read_text())
            for c in r.get("stage_counters", []):
                stages[c["stage"]] = c.get("elapsed_s")
                if c["stage"] == "ultra_coarse_screen":
                    m = SCREEN.search(c.get("note") or "")
                    n_in = c.get("candidates_in")
                    if m:
                        scored, total, secs = int(m[1]), int(m[2]), float(m[3])
                    else:  # exhaustive screen: everything scheduled was scored
                        scored, total, secs = n_in, n_in, c.get("elapsed_s") or 0.0
                    screen = {
                        "scored": scored,
                        "total": total,
                        "seconds": secs,
                        "per_s": round(scored / secs, 1) if secs else None,
                    }
            cands = r.get("candidates") or []
            if cands:
                top1 = {"lat": cands[0]["lat"], "lon": cands[0]["lon"]}
        res_ = j.resources or {}
        rows.append(
            {
                "workload": workload,
                "job_id": j.id,
                "status": j.status,
                "mode": (j.params or {}).get("mode"),
                "wall_s": round((j.finished_at or time.time()) - (j.started_at or j.created_at), 1),
                "peak_tree_pss_mb": res_.get("peak_pss_mb"),
                "cpu_s": res_.get("cpu_s"),
                "stages_s": stages,
                "screen": screen,
                "top1": top1,
                "abstained": (res.get("abstention") or {}).get("abstained"),
                "grade": res.get("confidence_grade"),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workloads", default="W1,W2")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--livermore", default=None)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--request", default=None, help="R: request.json of an earlier search")
    ap.add_argument("--replay", default=None, help="perf_baseline.json whose searches to re-run")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="replay: quality_settings override KEY=VALUE (repeatable), e.g. coarse_polish_span=0.5",
    )
    ap.add_argument("--only", default="", help="replay: comma-separated workloads, e.g. W1,W2,W4")
    a = ap.parse_args()

    from aerodoppler.config import get_settings
    from aerodoppler.validation import benchmark as B

    s = get_settings()
    s.ensure_dirs()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"workloads": {}, "searches": []}

    def log(m: str) -> None:
        print(m, file=sys.stderr, flush=True)

    if a.replay:
        sets = {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.set)}
        only = {w for w in a.only.split(",") if w}
        return _replay(s, Path(a.replay), out, a.workers, log, sets, only)
    for w in [x.strip() for x in a.workloads.split(",") if x.strip()]:
        t0 = time.time()
        if w == "W1":
            cases = [
                c for c in B.SYNTHETIC_CASES if c["name"] in ("base_snr10", "many_distractors")
            ]
            res = B.run_synthetic_benchmark(s, out / w, cases=cases, workers=a.workers, log=log)
        elif w == "SYN":
            res = B.run_synthetic_benchmark(s, out / w, workers=a.workers, log=log)
        elif w == "W2":
            res = B.run_real_trajectory_benchmark(
                s,
                "adsblol_2026-11-02_1e2f2f08",
                [("deland", 29.03, -81.25), ("n2920w8115", 29.2, -81.15)],
                out / w,
                workers=a.workers,
                seeds=(1,),
                log=log,
            )
        elif w == "W3":
            res = B.run_real_trajectory_benchmark(
                s,
                "adsblol_2026-11-04_262826ac",
                [("midtown_w", 40.765, -73.995)],
                out / w,
                workers=a.workers,
                seeds=(1,),
                log=log,
            )
        elif w == "R":
            from aerodoppler.jobs.runner import run_search_inline
            from aerodoppler.jobs.store import Store
            from aerodoppler.models import SearchRequest

            if not a.request:
                raise SystemExit("R needs --request <request.json>")
            req = SearchRequest.model_validate_json(Path(a.request).read_text())
            r = run_search_inline(Store(s.db_path), s, req)
            res = {"abstained": r.abstention.abstained, "grade": r.confidence_grade}
        elif w == "W4":
            from aerodoppler.validation.livermore import run_livermore_benchmark

            if not a.livermore:
                raise SystemExit("W4 needs --livermore <dir> --datasets <ids>")
            res = run_livermore_benchmark(
                s,
                Path(a.livermore),
                out / w,
                [d for d in a.datasets.split(",") if d],
                workers=a.workers,
                log=log,
            )
        else:
            raise SystemExit(f"unknown workload {w}")
        report["workloads"][w] = {
            "wall_s": round(time.time() - t0, 1),
            "summary": {k: v for k, v in (res or {}).items() if k != "cases"},
        }
        rows = _search_rows(s, t0, w)
        report["searches"] += rows
        for r in rows:
            sc = r["screen"] or {}
            log(
                f"[perf] {w} {r['job_id']} {r['mode']}: {r['wall_s']} s, peak {r['peak_tree_pss_mb']} MB, "
                f"screen {sc.get('scored')}/{sc.get('total')} at {sc.get('per_s')}/s, "
                f"{'abstained' if r['abstained'] else r['grade']}"
            )
    (out / "perf_baseline.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


def _replay(s, baseline: Path, out: Path, workers: int, log, sets=None, only=None) -> int:
    from aerodoppler.geo.coordinates import haversine_m
    from aerodoppler.jobs.runner import run_search_inline
    from aerodoppler.jobs.store import Store
    from aerodoppler.models import SearchRequest

    base = json.loads(baseline.read_text())
    report: dict[str, Any] = {"replay_of": str(baseline), "searches": []}
    report["overrides"] = sets or {}
    for b in base["searches"]:
        if only and b["workload"] not in only:
            continue
        req_path = s.runs_dir / b["job_id"] / "request.json"
        if not req_path.exists():
            log(f"[perf] {b['job_id']}: no request.json, skipped")
            continue
        req = SearchRequest.model_validate_json(req_path.read_text())
        upd = dict(sets or {})
        if workers:
            upd["workers"] = workers
        if upd:
            req = req.model_copy(
                update={"quality_settings": req.quality_settings.model_copy(update=upd)}
            )
        t0 = time.time()
        err = None
        job_ids: list[str] | None = None
        try:
            job_ids = [run_search_inline(Store(s.db_path), s, req).run_id]
        except Exception as e:  # a failure is a result too (e.g. the memory guard)
            err = f"{type(e).__name__}: {e}"
        # an identical earlier request that never ran is reused (idempotency), so look jobs up
        # by id, or by recent update when the search raised
        rows = _search_rows(s, t0, f"replay:{b['workload']}", job_ids)
        if not rows:
            from aerodoppler.jobs.service import queue_for

            recent = [
                j.id
                for j in queue_for(s).list(types=["geolocation_search"], limit=20)
                if j.updated_at >= t0
            ]
            rows = _search_rows(s, t0, f"replay:{b['workload']}", recent[:1])
        r = rows[-1] if rows else {"job_id": None}
        moved = None
        if r.get("top1") and b.get("top1"):
            moved = float(
                haversine_m(b["top1"]["lat"], b["top1"]["lon"], r["top1"]["lat"], r["top1"]["lon"])
            )
        same = (
            r.get("job_id") is not None
            and r.get("status") == b.get("status")
            and r.get("abstained") == b.get("abstained")
            and r.get("grade") == b.get("grade")
            and (moved is None or moved < 1.0)
        )
        row = {**r, "baseline_job": b["job_id"], "error": err, "top1_moved_m": moved, "same": same}
        report["searches"].append(row)
        sc, bsc = r.get("screen") or {}, b.get("screen") or {}
        log(
            f"[perf] {b['workload']} {b['job_id']} -> {r.get('job_id')}: "
            f"{b['wall_s']} -> {r.get('wall_s')} s, peak {b['peak_tree_pss_mb']} -> "
            f"{r.get('peak_tree_pss_mb')} MB, screen {bsc.get('per_s')} -> {sc.get('per_s')}/s, "
            f"{'SAME' if same else 'DIFFERENT'} (moved {moved}, {b.get('grade')} -> {r.get('grade')})"
            + (f" ERROR {err}" if err else "")
        )
        (out / "perf_replay.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
