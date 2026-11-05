"""Opt-in real-workload check of the job system (needs a cached ADSB.lol day; ~10 min).

Drives a real `aerodoppler serve` (separate data directory, offline) through:

1. a whole-CONUS flight import (the workload that froze WSL before the bounded importer),
   a duplicate submission (must reuse the job) and a second import (must wait for the slot);
2. Jobs-page screenshots before and after a browser reload (same job, more progress);
3. stopping the API server mid-import, checking the worker survived, starting a new server
   (the job is still there, still progressing);
4. SIGKILL of the import worker during parsing (what an OOM kill looks like): the scheduler
   must reconcile it — imports are resume-safe, so it is requeued and completes;
5. machine memory sampled every second throughout.

    uv run python scripts/demo_job_resilience.py --date 2026-11-03 --out data/runs/jobs_demo

Writes ``<out>/report.json`` and screenshots; prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

CONUS = {"west": -125.0, "south": 24.0, "east": -66.0, "north": 50.0}
MANHATTAN = {"west": -74.15, "south": 40.65, "east": -73.85, "north": 40.9}


def http(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class Server:
    def __init__(self, port: int, env: dict, log: Path):
        self.port, self.env, self.log = port, env, log
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        fh = open(self.log, "ab")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "aerodoppler.cli", "serve", "--port", str(self.port)],
            env=self.env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
                return
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(
                signal.SIGTERM
            )  # the API process only; workers are their own sessions
            self.proc.wait(timeout=30)


def worker_pids(job_id: str) -> list[int]:
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = " ".join(p.info["cmdline"] or [])
        if "aerodoppler.jobs.worker" in cl and job_id in cl:
            out.append(p.info["pid"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-11-03")
    ap.add_argument("--out", default="data/runs/jobs_demo")
    ap.add_argument("--port", type=int, default=8013)
    a = ap.parse_args()
    root = Path.cwd()
    out = (root / a.out).resolve()
    data = out / "data"
    (data / "raw" / "adsblol").mkdir(parents=True, exist_ok=True)
    cached = root / "data" / "raw" / "adsblol" / a.date
    if not cached.exists():
        print(f"no cached archive at {cached}; import that day once first", file=sys.stderr)
        return 2
    link = data / "raw" / "adsblol" / a.date
    if not link.exists():
        link.symlink_to(cached)
    env = {**os.environ, "AERODOPPLER_DATA_DIR": str(data), "AERODOPPLER_OFFLINE": "true"}
    base = f"http://127.0.0.1:{a.port}"
    report: dict = {"steps": [], "memory_samples": []}

    def step(name: str, **kw) -> None:
        kw["t"] = round(time.time() - t0, 1)
        report["steps"].append({"step": name, **kw})
        print(
            f"[{kw['t']:7.1f}s] {name}: {json.dumps({k: v for k, v in kw.items() if k != 't'}, default=str)[:300]}"
        )

    stop_sampling = threading.Event()

    def sample() -> None:
        while not stop_sampling.wait(1.0):
            vm = psutil.virtual_memory()
            report["memory_samples"].append(
                [
                    round(time.time() - t0, 1),
                    round((vm.total - vm.available) / 1e6),
                    round(vm.available / 1e6),
                ]
            )

    t0 = time.time()
    threading.Thread(target=sample, daemon=True).start()
    server = Server(a.port, env, out / "server.log")
    server.start()
    step("server started", pid=server.proc.pid)
    body = {
        "provider": "adsblol",
        "date_utc": a.date,
        "bbox": CONUS,
        "confirm_download": True,
        "force": True,
    }
    code, r = http("POST", f"{base}/api/flight-datasets/import", body)
    job = r["job_id"]
    step("CONUS import submitted", http=code, job_id=job, outcome=r.get("outcome"))
    code, r2 = http("POST", f"{base}/api/flight-datasets/import", body)
    step(
        "duplicate submitted",
        http=code,
        same_job=r2.get("job_id") == job,
        outcome=r2.get("outcome"),
    )
    code, r3 = http("POST", f"{base}/api/flight-datasets/import", {**body, "bbox": MANHATTAN})
    second = r3["job_id"]
    step("second region submitted", http=code, job_id=second, status=r3.get("status"))

    from playwright.sync_api import sync_playwright

    def wait_stage(stage: str, timeout: float = 300) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, j = http("GET", f"{base}/api/jobs/{job}")
            if j.get("stage") == stage or j.get("status") not in ("queued", "claimed", "running"):
                return j
            time.sleep(0.5)
        return j

    j = wait_stage("parse")
    time.sleep(8)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 1000})
        pg.goto(f"{base}/jobs/", wait_until="networkidle")
        time.sleep(2)
        pg.screenshot(path=str(out / "jobs_before_reload.png"), full_page=True)
        before = pg.inner_text("body")
        pg.reload(wait_until="networkidle")
        time.sleep(4)
        pg.screenshot(path=str(out / "jobs_after_reload.png"), full_page=True)
        after = pg.inner_text("body")
        b.close()
    _, j = http("GET", f"{base}/api/jobs/{job}")
    step(
        "browser reloaded",
        job_visible_before=job in before,
        job_visible_after=job in after,
        status=j["status"],
        files_parsed=(j["progress"].get("stages", {}).get("parse") or {}).get("files_parsed"),
    )
    pids_before = worker_pids(job)
    server.stop()
    step("API server stopped", worker_pids=pids_before)
    time.sleep(5)
    alive = [pid for pid in pids_before if psutil.pid_exists(pid)]
    step("worker alive without a server", alive=alive)
    server = Server(a.port, env, out / "server.log")
    server.start()
    _, j = http("GET", f"{base}/api/jobs/{job}")
    step(
        "new server sees the job",
        status=j["status"],
        stage=j["stage"],
        files_parsed=(j["progress"].get("stages", {}).get("parse") or {}).get("files_parsed"),
    )
    time.sleep(5)
    pids = worker_pids(job)
    _, j = http("GET", f"{base}/api/jobs/{job}")
    if j["status"] == "running" and pids:
        os.kill(pids[0], signal.SIGKILL)
        step("worker SIGKILLed (simulated OOM)", pid=pids[0], stage=j["stage"])
    deadline = time.time() + 1800
    seen = set()
    while time.time() < deadline:
        _, j = http("GET", f"{base}/api/jobs/{job}")
        key = (j["status"], j["attempt"])
        if key not in seen:
            seen.add(key)
            step(
                "CONUS job state",
                status=j["status"],
                attempt=j["attempt"],
                stage=j["stage"],
                error=(j.get("error") or {}).get("message"),
            )
        if j["status"] in ("succeeded", "failed", "interrupted", "cancelled"):
            break
        time.sleep(2)
    _, s2 = http("GET", f"{base}/api/jobs/{second}")
    while s2["status"] in ("queued", "claimed", "running") and time.time() < deadline:
        time.sleep(2)
        _, s2 = http("GET", f"{base}/api/jobs/{second}")
    step("second import", status=s2["status"], queue_wait_s=s2.get("queue_wait_s"))
    _, ev = http("GET", f"{base}/api/jobs/{job}/events?format=json")
    report["conus_job"] = j
    report["conus_events"] = [
        (e["seq"], e["kind"], e["data"].get("status") or e["data"].get("stage"))
        for e in ev["events"]
        if e["kind"] != "progress" or e["data"].get("stage_changed")
    ]
    report["second_job"] = s2
    stop_sampling.set()
    server.stop()
    used = [m[1] for m in report["memory_samples"]]
    report["machine_memory_used_mb"] = {"min": min(used), "max": max(used)} if used else None
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps(
            {
                "conus": j["status"],
                "attempts": j["attempt"],
                "resources": j.get("resources"),
                "second": s2["status"],
                "machine_memory_used_mb": report["machine_memory_used_mb"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
