#!/usr/bin/env bash
# Release smoke test (used by .github/workflows/release.yml; runnable locally).
#
#   scripts/smoke_release.sh wheel <path/to/aerodoppler-*.whl> [expected-version] [--demo]
#       install the wheel into a fresh venv, check `aerodoppler --version`, start the server,
#       check /health and that / serves the bundled UI; --demo also runs the offline SYNTHETIC
#       end-to-end demo from the installed package (about 3 min).
#   scripts/smoke_release.sh url <base-url> [expected-version]
#       check /health and the UI of an already running server (e.g. the Docker image).
set -euo pipefail

mode="${1:?usage: smoke_release.sh wheel|url <target> [version] [--demo]}"
target="${2:?missing wheel path or base url}"
expect="${3:-}"
demo="${4:-}"

check_http() {
  local base="$1"
  for _ in $(seq 1 60); do
    curl -fsS "$base/health" >/dev/null 2>&1 && break
    sleep 1
  done
  local health
  health="$(curl -fsS "$base/health")"
  echo "health: $health"
  if [ -n "$expect" ]; then
    python3 -c 'import json,sys; v=json.loads(sys.argv[1])["version"]; sys.exit(0 if v==sys.argv[2] else f"version mismatch: /health reports {v}, expected {sys.argv[2]}")' "$health" "$expect"
  fi
  local page
  page="$(curl -fsS "$base/")"
  case "$page" in
    *"<html"*AeroDoppler*) echo "ui: bundled web UI is served at /" ;;
    *) echo "ui: / does not serve the web UI (got: ${page:0:200})" >&2; exit 1 ;;
  esac
  curl -fsS "$base/data/" | grep -q "Privacy defaults" || { echo "ui: /data/ page missing" >&2; exit 1; }
  echo "ui: /data/ page present"
}

case "$mode" in
  wheel)
    work="$(mktemp -d)"
    trap 'if [ -n "${server_pid:-}" ]; then kill "$server_pid" 2>/dev/null || true; fi; rm -rf "$work"' EXIT
    uv venv -q --python 3.12 "$work/venv"
    uv pip install -q --python "$work/venv/bin/python" "$target"
    bin="$work/venv/bin/aerodoppler"
    ver="$("$bin" --version)"
    echo "$ver"
    if [ -n "$expect" ] && [ "$ver" != "aerodoppler ${expect}" ]; then
      echo "version mismatch: expected 'aerodoppler ${expect}', got '$ver'" >&2; exit 1
    fi
    export AERODOPPLER_DATA_DIR="$work/data" AERODOPPLER_OFFLINE=true
    # run from an empty directory so the source checkout's web/out cannot mask a missing bundle
    (cd "$work" && exec "$bin" serve --host 127.0.0.1 --port 8765) >"$work/server.log" 2>&1 &
    server_pid=$!
    check_http "http://127.0.0.1:8765" || { cat "$work/server.log"; exit 1; }
    kill "$server_pid"; wait "$server_pid" 2>/dev/null || true
    if [ "$demo" = "--demo" ]; then
      (cd "$work" && "$bin" demo --out "$work/data/runs/demo" --workers 4 --skip-known-time) >"$work/demo.json" 2>"$work/demo.log" \
        || { tail -40 "$work/demo.log"; exit 1; }
      python3 - "$work/demo.json" <<'PY'
import json, sys
raw = open(sys.argv[1]).read()
s = json.loads(raw[raw.index("{"):])
a = s["runs"]["mode_a"]
print(f"demo: mode A abstained={a['abstained']} top-1 error {a.get('top1_error_m')} m (SYNTHETIC)")
if a["abstained"] or a.get("top1_error_m") is None or a["top1_error_m"] > 300:
    sys.exit("demo: synthetic Mode A did not recover the receiver within 300 m")
PY
    fi
    ;;
  url)
    check_http "$target"
    ;;
  *)
    echo "unknown mode: $mode" >&2; exit 2 ;;
esac
echo "smoke test OK"
