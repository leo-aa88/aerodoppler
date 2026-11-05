"""CLI behaviour: authorization gate, JSON outputs, local fixture import, error codes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from aerodoppler.cli import app

FIX = Path(__file__).parents[1] / "fixtures" / "adsblol" / "2026-11-02"


def _env(tmp_path: Path) -> dict[str, str]:
    return {"AERODOPPLER_DATA_DIR": str(tmp_path / "data"), "AERODOPPLER_OFFLINE": "true"}


def test_cli_ingest_without_authorization_flag_and_purge(tmp_path: Path, monkeypatch):
    for k, v in _env(tmp_path).items():
        monkeypatch.setenv(k, v)
    from aerodoppler.config import reset_settings_cache

    reset_settings_cache()
    wav = tmp_path / "x.wav"
    sf.write(
        str(wav), np.random.default_rng(0).standard_normal(8000 * 10).astype(np.float32) * 0.1, 8000
    )
    # --authorized is no longer required (still accepted as a no-op).
    r = CliRunner().invoke(
        app, ["audio", "ingest", str(wav), "--date", "2026-11-02", "--synthetic"]
    )
    assert r.exit_code == 0, r.output
    rec = json.loads(r.output[r.output.index("{") :])
    assert rec["owner_permission_confirmed"] is True and rec["duration_s"] == 10.0
    r = CliRunner().invoke(app, ["audio", "purge", rec["id"]])
    assert r.exit_code == 2  # needs --yes
    r = CliRunner().invoke(app, ["audio", "purge", rec["id"], "--yes"])
    assert (
        r.exit_code == 0
        and json.loads(r.output[r.output.index("{") :])["recording_id"] == rec["id"]
    )
    reset_settings_cache()


def test_cli_local_flight_import_and_listing(tmp_path: Path, monkeypatch):
    for k, v in _env(tmp_path).items():
        monkeypatch.setenv(k, v)
    from aerodoppler.config import reset_settings_cache

    reset_settings_cache()
    r = CliRunner().invoke(
        app,
        [
            "flights",
            "import",
            "--provider",
            "local",
            "--path",
            str(FIX),
            "--date",
            "2026-11-02",
            "--bbox",
            "-180,-90,180,90",
            "--workers",
            "1",
        ],
    )
    assert r.exit_code == 0, r.output
    out = json.loads(r.output[r.output.index("{") :])  # progress text may precede the JSON
    assert out["filter_counts"]["files"] == 6 and out["summary"]["segments"] >= 2
    r = CliRunner().invoke(app, ["flights", "list"])
    assert r.exit_code == 0 and out["dataset_id"] in r.output
    r = CliRunner().invoke(app, ["flights", "stats", out["dataset_id"]])
    assert r.exit_code == 0 and "coverage" in r.output
    r = CliRunner().invoke(app, ["flights", "stats", "nope"])
    assert r.exit_code == 1
    reset_settings_cache()


def test_cli_adsblol_offline_without_cache_fails_clearly(tmp_path: Path, monkeypatch):
    for k, v in _env(tmp_path).items():
        monkeypatch.setenv(k, v)
    from aerodoppler.config import reset_settings_cache

    reset_settings_cache()
    r = CliRunner().invoke(
        app,
        [
            "flights",
            "import",
            "--provider",
            "adsblol",
            "--date",
            "2020-01-01",
            "--bbox",
            "-1,-1,1,1",
            "--offline",
        ],
    )
    assert r.exit_code == 4
    reset_settings_cache()
