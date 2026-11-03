"""Take screenshots of the running application (SPEC §13: README screenshots must come from
the real app). Starts the local server on a free port (offline), opens the UI with Playwright
(headless Chromium) and saves PNGs under docs/img/.

Usage: uv run python scripts/screenshots.py [--data-dir data] [--out docs/img]
Requires a recording with a finished run (e.g. after `make demo`).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = dict(os.environ, AERODOPPLER_DATA_DIR=args.data_dir, AERODOPPLER_OFFLINE="true")
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "aerodoppler.api.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        recs = json.loads(urllib.request.urlopen(base + "/api/recordings").read())["recordings"]
        searches = json.loads(urllib.request.urlopen(base + "/api/searches").read())["searches"]
        done = [s for s in searches if s["status"] in ("succeeded", "abstained")]
        if not recs or not done:
            print("need at least one recording and one finished search (run `make demo` first)", file=sys.stderr)
            return 1
        rec = next((r for r in recs if any(s["recording_id"] == r["id"] for s in done)), recs[0])
        run = next(s for s in done if s["recording_id"] == rec["id"])
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1)
            shots = [
                ("/", "ui_import.png", 3000),
                (f"/lab/?rec={rec['id']}", "ui_spectrogram_lab.png", 9000),
                (f"/search/?rec={rec['id']}", "ui_search.png", 3000),
                (f"/results/?run={run['id']}", "ui_results.png", 12000),
                ("/data/", "ui_data_privacy.png", 3000),
            ]
            for path, name, wait in shots:
                pg.goto(base + path, wait_until="networkidle")
                pg.wait_for_timeout(wait)
                pg.screenshot(path=str(out / name), full_page=True)
                print("saved", out / name)
            b.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
