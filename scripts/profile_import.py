"""Measure peak memory (whole process tree) and wall time of one ADS-B import.

Runs ``aerodoppler flights import`` in a child process, samples the RSS of the child and all
its descendants every 0.5 s, and kills the tree if it exceeds ``--kill-above-mb`` so a
measurement can never take the machine down. Offline: the archive must already be cached.

    uv run python scripts/profile_import.py --date 2026-11-03 --bbox -74.15,40.65,-73.85,40.9 \
        --out data/runs/profiling/manhattan_new.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def tree_rss_mb(p: psutil.Process) -> float:
    total = 0
    try:
        procs = [p, *p.children(recursive=True)]
    except psutil.Error:
        return 0.0
    for q in procs:
        try:
            total += q.memory_info().rss
        except psutil.Error:
            pass
    return total / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kill-above-mb", type=float, default=20000.0)
    ap.add_argument("--extra", default="", help="extra CLI args for `flights import`")
    a = ap.parse_args()
    cmd = [
        sys.executable,
        "-m",
        "aerodoppler.cli",
        "flights",
        "import",
        "--date",
        a.date,
        "--bbox",
        a.bbox,
        "--confirm-download",
        *a.extra.split(),
    ]
    env = {**os.environ, "AERODOPPLER_OFFLINE": "true"}
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    ps = psutil.Process(proc.pid)
    peak, samples, killed = 0.0, [], False
    while proc.poll() is None:
        r = tree_rss_mb(ps)
        peak = max(peak, r)
        samples.append((round(time.time() - t0, 1), round(r)))
        if r > a.kill_above_mb:
            os.killpg(proc.pid, signal.SIGKILL)
            killed = True
            break
        time.sleep(0.5)
    out_text = proc.communicate()[0] if not killed else ""
    res = {
        "cmd": " ".join(cmd[2:]),
        "returncode": proc.returncode,
        "killed_at_mb": a.kill_above_mb if killed else None,
        "wall_s": round(time.time() - t0, 1),
        "peak_tree_rss_mb": round(peak),
        "rss_timeline_mb": samples[:: max(1, len(samples) // 200)],
        "output_tail": (out_text or "")[-3000:],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rss_timeline_mb", "output_tail")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
