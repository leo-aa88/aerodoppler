"""AeroDoppler command-line interface (SPEC §11).

``--authorized`` is still accepted by ``audio ingest`` and ``search`` for compatibility but is
no longer required; analysing only recordings you own or may analyse is the operator's
responsibility (see docs/privacy.md).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from aerodoppler.config import get_settings
from aerodoppler.models import BBox

app = typer.Typer(
    help="AeroDoppler: consent-based aircraft-Doppler acoustic geolocation (local-first).",
    no_args_is_help=True,
)
flights_app = typer.Typer(
    help="Historical flight archives (ADSB.lol / local fixtures).", no_args_is_help=True
)
geo_app = typer.Typer(help="Elevation data.", no_args_is_help=True)
audio_app = typer.Typer(help="Recordings, spectrograms and Doppler events.", no_args_is_help=True)
results_app = typer.Typer(help="Search results.", no_args_is_help=True)
app.add_typer(flights_app, name="flights")
app.add_typer(geo_app, name="geo")
app.add_typer(audio_app, name="audio")
app.add_typer(results_app, name="results")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _version_callback(value: bool) -> None:
    if value:
        from aerodoppler import __version__

        typer.echo(f"aerodoppler {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    _setup_logging(verbose)
    get_settings().ensure_dirs()


# ---------------------------------------------------------------------------------------
# flights
# ---------------------------------------------------------------------------------------


@flights_app.command("import")
def flights_import(
    date: str = typer.Option(..., "--date", help="UTC date YYYY-MM-DD"),
    bbox: str = typer.Option(..., "--bbox", help="WEST,SOUTH,EAST,NORTH (degrees)"),
    provider: str = typer.Option("adsblol", "--provider", help="adsblol | local"),
    path: Path | None = typer.Option(
        None, "--path", help="local provider: directory of trace files or tar parts"
    ),
    margin_km: float | None = typer.Option(
        None, "--margin-km", help="audibility safety margin around bbox"
    ),
    storage_budget_gb: float | None = typer.Option(None, "--storage-budget-gb"),
    confirm_download: bool = typer.Option(
        False, "--confirm-download", help="accept the whole-day archive size"
    ),
    workers: int = typer.Option(
        0, "--workers", help="parallel parsers (0 = as many as the memory budget allows)"
    ),
    memory_budget_mb: float | None = typer.Option(
        None,
        "--memory-budget-mb",
        help="peak RAM for the whole import incl. parser processes (default: half of available, max 4096)",
    ),
    exclude_jets: bool = typer.Option(
        False, "--exclude-jets", help="drop aircraft classified as jets while parsing"
    ),
    gap_split_s: float | None = typer.Option(None, "--gap-split-s"),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Download/import one UTC day of flights for a bounding box and normalise to Parquet.

    Runs as an inline job: it waits for the import slot if another import is running, shows
    stage progress, and appears on the Jobs page like any other import."""
    s = get_settings()
    bb = BBox.parse(bbox)
    if provider not in ("adsblol", "local"):
        raise typer.BadParameter("provider must be adsblol or local")
    if provider == "local" and path is None:
        raise typer.BadParameter("--path is required for the local provider")
    from aerodoppler.jobs.queue import JobError
    from aerodoppler.jobs.service import JobRejected, run_inline

    params = {
        "provider": provider,
        "local_path": str(path) if path else None,
        "date_utc": date,
        "bbox": bb.model_dump(),
        "margin_km": s.adsb_bbox_margin_km if margin_km is None else margin_km,
        "storage_budget_gb": storage_budget_gb,
        "confirm_download": confirm_download,
        "workers": workers,
        "memory_budget_mb": memory_budget_mb,
        "exclude_jets": exclude_jets,
        "gap_split_s": s.adsb_gap_split_s if gap_split_s is None else gap_split_s,
    }
    if offline and not s.offline:
        from aerodoppler.config import Settings as _S

        s = _S(**{**s.model_dump(), "offline": True})
    try:
        job, _res = run_inline(s, "flight_import", params, log=lambda m: typer.secho(m, err=True))
    except JobError as e:
        typer.secho(f"import failed ({e.code}): {e}", fg=typer.colors.RED, err=True)
        code = {"download_not_confirmed": 3, "memory_budget": 5}.get(e.code, 4)
        raise typer.Exit(code=code) from e
    except JobRejected as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=5) from e
    from aerodoppler.adsb.index import FlightDataset

    ds = FlightDataset(job.result["dataset_id"], s.normalized_dir / job.result["dataset_id"])
    man = ds.manifest()
    typer.echo(
        json.dumps(
            {
                "dataset_id": ds.dataset_id,
                "summary": man["summary"],
                "filter_counts": {k: v for k, v in man["filter_counts"].items() if k != "errors"},
                "storage_bytes": man["storage_bytes"],
                "license": man["license"],
                "resources": man.get("resources"),
            },
            indent=2,
        )
    )


@flights_app.command("list")
def flights_list() -> None:
    from aerodoppler.adsb.index import list_datasets

    s = get_settings()
    rows = []
    for ds in list_datasets(s.normalized_dir):
        m = ds.manifest()
        rows.append(
            {
                "dataset_id": ds.dataset_id,
                "provider": m.get("provider", {}).get("provider"),
                "date": m.get("request", {}).get("date_utc"),
                "bbox": m.get("request", {}).get("bbox"),
                "segments": m.get("summary", {}).get("segments"),
                "synthetic": m.get("synthetic", False),
                "storage_bytes": m.get("storage_bytes"),
            }
        )
    typer.echo(json.dumps(rows, indent=2))


@flights_app.command("stats")
def flights_stats(dataset_id: str) -> None:
    from aerodoppler.adsb.index import FlightDataset, FlightIndex

    s = get_settings()
    ds = FlightDataset(dataset_id, s.normalized_dir / dataset_id)
    if not ds.exists():
        typer.secho("dataset not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(
        json.dumps(
            {"manifest": ds.manifest(), "coverage": FlightIndex([ds]).coverage_summary()},
            indent=2,
            default=str,
        )
    )


# ---------------------------------------------------------------------------------------
# geo
# ---------------------------------------------------------------------------------------


@geo_app.command("fetch-dem")
def geo_fetch_dem(
    bbox: str = typer.Option(..., "--bbox"),
    source: str = typer.Option("auto", "--source", help="auto | 3dep | copernicus | synthetic"),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Fetch (and cache) DEM tiles covering a bbox, or generate a labelled synthetic DEM."""
    from aerodoppler.geo.dem import DEMError, dem_cache_name, make_synthetic_dem, open_dem_for_bbox

    s = get_settings()
    bb = BBox.parse(bbox)
    try:
        if source == "synthetic":
            p = make_synthetic_dem(bb, s.dem_dir / dem_cache_name("synthetic", bb))
            typer.echo(json.dumps({"path": str(p), "synthetic": True}))
            return
        dem = open_dem_for_bbox(s.dem_dir, bb, prefer=source, offline=offline or s.offline)
    except DEMError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    typer.echo(json.dumps(dem.describe(), indent=2))


@geo_app.command("elevation")
def geo_elevation(lat: float, lon: float, offline: bool = typer.Option(False, "--offline")) -> None:
    from aerodoppler.geo.altitude import get_geoid
    from aerodoppler.geo.dem import DEMError, open_dem_for_bbox

    s = get_settings()
    bb = BBox(west=lon - 0.01, south=lat - 0.01, east=lon + 0.01, north=lat + 0.01)
    try:
        dem = open_dem_for_bbox(s.dem_dir, bb, offline=offline or s.offline)
        r = dem.elevation([lat], [lon])
    except DEMError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    typer.echo(
        json.dumps(
            {
                "elevation_m": float(r.elevation_m[0]),
                "sigma_m": float(r.sigma_m[0]),
                "datum": r.datum,
                "source": r.source,
                "flags": r.flags,
                "egm96_undulation_m": float(get_geoid().undulation_m(lat, lon)),
            }
        )
    )


# ---------------------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------------------


def _store():
    from aerodoppler.jobs.store import Store

    return Store(get_settings().db_path)


@audio_app.command("ingest")
def audio_ingest(
    path: Path = typer.Argument(..., exists=True, readable=True),
    authorized: bool = typer.Option(
        False, "--authorized", hidden=True
    ),  # no-op, kept for compatibility
    date: str | None = typer.Option(None, "--date", help="UTC recording date YYYY-MM-DD (Mode A)"),
    start_utc: str | None = typer.Option(
        None, "--start-utc", help="UTC start time, e.g. 2026-09-08T14:03:00Z (Mode B)"
    ),
    start_uncertainty_s: float | None = typer.Option(None, "--start-uncertainty-s"),
    synthetic: bool = typer.Option(False, "--synthetic", help="label as SYNTHETIC DEMO data"),
    notes: str | None = typer.Option(None, "--notes"),
) -> None:
    """Ingest a WAV/FLAC/MP3/video file: preserve the original, hash it, build the 2 kHz analysis copy."""
    from aerodoppler.audio.decode import AudioDecodeError
    from aerodoppler.audio.service import ingest_recording

    del authorized
    try:
        rec = ingest_recording(
            _store(),
            get_settings(),
            path,
            owner_permission_confirmed=True,
            recording_date_utc=date,
            start_time_utc=start_utc,
            start_time_uncertainty_s=start_uncertainty_s,
            synthetic=synthetic,
            notes=notes,
        )
    except AudioDecodeError as e:
        typer.secho(f"decode failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    typer.echo(rec.model_dump_json(indent=2))


@audio_app.command("spectrogram")
def audio_spectrogram(
    recording_id: str,
    window_s: float | None = typer.Option(None, "--window-s"),
    hop_s: float | None = typer.Option(None, "--hop-s"),
    fmax_hz: float | None = typer.Option(None, "--fmax-hz"),
    png: Path | None = typer.Option(
        None, "--png", help="also write a PNG of the whitened spectrogram"
    ),
) -> None:
    """Compute (or recompute) the multiresolution STFT and whitened representation."""
    from aerodoppler.audio.service import NotFound, ensure_spectrogram

    params = {
        k: v
        for k, v in {"window_s": window_s, "hop_s": hop_s, "fmax_hz": fmax_hz}.items()
        if v is not None
    }
    try:
        spec = ensure_spectrogram(
            _store(), get_settings(), recording_id, force=bool(params), **params
        )
    except NotFound as e:
        typer.secho("recording not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    out = {
        "recording_id": recording_id,
        "frames": int(len(spec.times)),
        "bins": int(len(spec.freqs)),
        "hop_s": spec.hop_s,
        "df_hz": spec.df_hz,
        "params": spec.params,
    }
    if png:
        from aerodoppler.validation.plots import plot_spectrogram

        plot_spectrogram(spec, png, title=f"{recording_id} whitened spectrogram")
        out["png"] = str(png)
    typer.echo(json.dumps(out, indent=2))


@audio_app.command("detect")
def audio_detect(
    recording_id: str,
    threshold_z: float = typer.Option(2.5, "--threshold-z"),
    min_duration_s: float = typer.Option(40.0, "--min-duration-s"),
    save: bool = typer.Option(True, "--save/--no-save", help="store proposals as editable events"),
) -> None:
    """Propose Doppler events automatically (Viterbi ridge tracking + harmonic grouping)."""
    from aerodoppler.audio.ridges import DetectorConfig
    from aerodoppler.audio.service import NotFound, propose_events

    try:
        events = propose_events(
            _store(),
            get_settings(),
            recording_id,
            DetectorConfig(threshold_z=threshold_z, min_duration_s=min_duration_s),
            save=save,
        )
    except NotFound as e:
        typer.secho("recording not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    if not events:
        typer.echo(
            json.dumps(
                {
                    "recording_id": recording_id,
                    "events": [],
                    "abstention": "no stable tonal ridge detected; annotate manually or check the spectrogram",
                },
                indent=2,
            )
        )
        return
    typer.echo(
        json.dumps(
            [
                {
                    "id": e.id,
                    "start_s": e.audio_start_s,
                    "end_s": e.audio_end_s,
                    "f0_prior_hz": e.f0_prior,
                    "harmonics": e.candidate_harmonic_numbers,
                    "ridge_harmonic": e.ridge_harmonic,
                    "quality": round(e.detection_quality, 3),
                    "n_points": len(e.ridge_points),
                }
                for e in events
            ],
            indent=2,
        )
    )


@audio_app.command("add-event")
def audio_add_event(
    recording_id: str,
    points: str = typer.Option(..., "--points", help="manual ridge points 't:f,t:f,...' (s:Hz)"),
    harmonic: int = typer.Option(1, "--harmonic"),
    uncertainty_hz: float = typer.Option(1.0, "--uncertainty-hz"),
) -> None:
    """Add a manually traced Doppler event."""
    from aerodoppler.audio.service import NotFound, add_manual_event

    pts = []
    for tok in points.split(","):
        t, f = tok.split(":")
        pts.append((float(t), float(f)))
    try:
        ev = add_manual_event(
            _store(), recording_id, pts, harmonic=harmonic, uncertainty_hz=uncertainty_hz
        )
    except NotFound as e:
        typer.secho("recording not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    typer.echo(ev.model_dump_json(indent=2))


@audio_app.command("events")
def audio_events(recording_id: str) -> None:
    evs = _store().list_events(recording_id)
    typer.echo(
        json.dumps(
            [
                {
                    "id": e.id,
                    "start_s": e.audio_start_s,
                    "end_s": e.audio_end_s,
                    "f0_prior_hz": e.f0_prior,
                    "method": e.detection_method,
                    "ridge_harmonic": e.ridge_harmonic,
                    "n_points": len(e.ridge_points),
                    "quality": e.detection_quality,
                }
                for e in evs
            ],
            indent=2,
        )
    )


@audio_app.command("list")
def audio_list() -> None:
    typer.echo(json.dumps([r.model_dump() for r in _store().list_recordings()], indent=2))


@audio_app.command("purge")
def audio_purge(
    recording_id: str, yes: bool = typer.Option(False, "--yes", help="confirm deletion")
) -> None:
    """Delete the original file, derived data, events and every run for this recording."""
    from aerodoppler.audio.service import NotFound, purge_recording

    if not yes:
        typer.secho(
            "refusing without --yes (this permanently deletes the recording and all derived data)",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        typer.echo(json.dumps(purge_recording(_store(), get_settings(), recording_id), indent=2))
    except NotFound as e:
        typer.secho("recording not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


# ---------------------------------------------------------------------------------------
# search / results / serve / demo
# ---------------------------------------------------------------------------------------


@app.command("search")
def search_cmd(
    recording: str = typer.Option(..., "--recording"),
    mode: str = typer.Option(..., "--mode", help="unknown-time (Mode A) | known-time (Mode B)"),
    events: str = typer.Option(..., "--events", help="comma-separated event ids"),
    bbox: str = typer.Option(..., "--bbox", help="WEST,SOUTH,EAST,NORTH"),
    date: str | None = typer.Option(None, "--date", help="UTC date for unknown-time mode"),
    start_utc: str | None = typer.Option(None, "--start-utc", help="UTC start for known-time mode"),
    start_uncertainty_s: float | None = typer.Option(None, "--start-uncertainty-s"),
    datasets: str | None = typer.Option(
        None, "--datasets", help="comma-separated dataset ids (default: all)"
    ),
    authorized: bool = typer.Option(
        False, "--authorized", hidden=True
    ),  # no-op, kept for compatibility
    national_scale_confirmed: bool = typer.Option(False, "--confirm-large-scale"),
    workers: int = typer.Option(0, "--workers"),
    grid_m: float = typer.Option(200.0, "--grid-m"),
    separation_tolerance_s: float = typer.Option(20.0, "--separation-tolerance-s"),
    no_silence: bool = typer.Option(False, "--no-silence", help="disable the soft silence term"),
    ground_elevation_m: float | None = typer.Option(
        None, "--ground-elevation", help="flat-ground fallback when no DEM is available"
    ),
    ground_elevation_sigma_m: float | None = typer.Option(None, "--ground-elevation-sigma"),
    temperature_c: float | None = typer.Option(None, "--temperature-c"),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Run a bounded Mode A / Mode B search in this process and print ranked hypotheses."""
    from aerodoppler.config import Settings, reset_settings_cache
    from aerodoppler.jobs.runner import run_search_inline
    from aerodoppler.models import QualitySettings, SearchMode, SearchRequest
    from aerodoppler.search.engine import SearchCancelled, SearchInputError

    del authorized
    s = get_settings()
    if offline and not s.offline:
        reset_settings_cache()
        s = Settings(**{**s.model_dump(), "offline": True})
    q = QualitySettings(
        coarse_grid_m=grid_m,
        event_separation_tolerance_s=separation_tolerance_s,
        silence_constraint=not no_silence,
        workers=workers,
        ground_elevation_m=ground_elevation_m,
        ground_elevation_sigma_m=ground_elevation_sigma_m,
        temperature_c=temperature_c,
    )
    try:
        req = SearchRequest(
            mode=SearchMode(mode),
            recording_id=recording,
            event_ids=[e for e in events.split(",") if e],
            date_utc=date,
            start_utc=start_utc,
            start_uncertainty_s=start_uncertainty_s,
            bbox=BBox.parse(bbox),
            flight_dataset_ids=[d for d in (datasets or "").split(",") if d],
            operator_authorized=True,
            national_scale_confirmed=national_scale_confirmed,
            quality_settings=q,
        )
    except ValueError as e:
        typer.secho(f"invalid request: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e

    def prog(stage: str, info: dict) -> None:
        typer.secho(
            f"[{info.get('elapsed_s', 0):7.1f}s] {stage}: "
            + json.dumps({k: v for k, v in info.items() if k not in ("elapsed_s",)}, default=str)[
                :160
            ],
            err=True,
        )

    try:
        res = run_search_inline(_store(), s, req, progress=prog)
    except SearchInputError as e:
        typer.secho(f"search input error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    except SearchCancelled as e:
        typer.secho("cancelled", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=130) from e
    typer.echo(_summarize_result(res))


def _summarize_result(res) -> str:
    out = {
        "run_id": res.run_id,
        "score_label": res.score_label,
        "confidence_grade": res.confidence_grade,
        "gates": [g.model_dump() for g in res.gates],
        "abstention": res.abstention.model_dump(),
        "warnings": res.warnings,
        "stage_counters": [c.model_dump() for c in res.stage_counters],
        "candidates": [
            {
                "rank": c.rank,
                "lat": round(c.lat, 6),
                "lon": round(c.lon, 6),
                "score_total": round(c.score_total, 3),
                "per_event": [round(x, 3) for x in c.score_components.per_event],
                "coverage": [round(x, 2) for x in c.score_components.coverage],
                "silence_penalty": round(c.score_components.silence_penalty, 3),
                "inferred_start_utc": c.inferred_start_utc,
                "f0_hz": [round(f, 2) for f in c.source_fundamental_hz_per_event],
                "flights": c.extra.get("segments"),
                "ground_elevation_m": c.ground_elevation_m,
                "grid_m": c.nominal_grid_spacing_m,
                "estimated_uncertainty_m": c.estimated_uncertainty_m,
                "uncertainty_label": c.uncertainty_label,
                "flags": c.flags,
            }
            for c in res.candidates
        ],
    }
    return json.dumps(out, indent=2, default=str)


def _search_job(run_id: str):
    from aerodoppler.jobs.service import queue_for

    j = queue_for(get_settings()).get(run_id)
    if j is None or j.type != "geolocation_search":
        typer.secho("search not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    return j


@results_app.command("export")
def results_export(
    run_id: str,
    format: str = typer.Option("json", "--format", help="json | geojson | manifest"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Export a run (reproducible JSON / GeoJSON / manifest)."""
    from aerodoppler.api.routes import result_to_geojson
    from aerodoppler.jobs.runner import INCOMPLETE
    from aerodoppler.models import SearchResult

    job = _search_job(run_id)
    d = Path(job.work_dir or "")
    if (d / INCOMPLETE).exists() or not (d / "result.json").exists():
        typer.secho(f"no valid result: the search is {job.status}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if format == "manifest":
        text = (d / "manifest.json").read_text()
    else:
        res = SearchResult.model_validate_json((d / "result.json").read_text())
        text = (
            res.model_dump_json(indent=2)
            if format == "json"
            else json.dumps(result_to_geojson(res), indent=2)
        )
    if out:
        out.write_text(text)
        typer.echo(str(out))
    else:
        typer.echo(text)


@results_app.command("show")
def results_show(run_id: str) -> None:
    from aerodoppler.jobs.runner import INCOMPLETE
    from aerodoppler.models import SearchResult

    job = _search_job(run_id)
    d = Path(job.work_dir or "")
    p = d / "result.json"
    if not p.exists() or (d / INCOMPLETE).exists():
        typer.echo(
            json.dumps(
                {
                    "status": job.status,
                    "stage": job.stage,
                    "error": job.error_message,
                    "progress": job.progress,
                },
                indent=2,
                default=str,
            )
        )
        return
    typer.echo(_summarize_result(SearchResult.model_validate_json(p.read_text())))


@results_app.command("list")
def results_list() -> None:
    from aerodoppler.jobs.service import queue_for

    jobs = queue_for(get_settings()).list(types=["geolocation_search"], limit=500)
    typer.echo(
        json.dumps(
            [
                {
                    "id": j.id,
                    "recording_id": j.recording_id,
                    "status": j.status,
                    "stage": j.stage,
                    "created_at": j.created_at,
                    "error": j.error_message,
                }
                for j in jobs
            ],
            indent=2,
            default=str,
        )
    )


jobs_app = typer.Typer(help="Inspect and control background jobs (imports, searches, …)")
app.add_typer(jobs_app, name="jobs")


@jobs_app.command("list")
def jobs_list(
    status: str | None = typer.Option(None, "--status", help="comma-separated, or 'active'"),
    type: str | None = typer.Option(None, "--type"),
    limit: int = typer.Option(30, "--limit"),
) -> None:
    """Jobs on this machine, newest first."""
    from aerodoppler.jobs.queue import ACTIVE
    from aerodoppler.jobs.service import queue_for

    q = queue_for(get_settings())
    statuses = list(ACTIVE) if status == "active" else (status.split(",") if status else None)
    for j in q.list(statuses=statuses, types=type.split(",") if type else None, limit=limit):
        err = f"  {j.error_message[:90]}" if j.error_message else ""
        typer.echo(
            f"{j.id}  {j.type:22} {j.status:12} {str(j.stage)[:24]:24} attempt {j.attempt}/{j.max_attempts}{err}"
        )


@jobs_app.command("show")
def jobs_show(job_id: str) -> None:
    from aerodoppler.api.jobs_routes import job_view
    from aerodoppler.jobs.service import queue_for

    j = queue_for(get_settings()).get(job_id)
    if j is None:
        typer.secho("job not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(job_view(j, debug=True), indent=2, default=str))


@jobs_app.command("cancel")
def jobs_cancel(job_id: str) -> None:
    """Cancel a job: queued jobs stop at once; running ones are asked to stop, then terminated."""
    from aerodoppler.jobs.service import queue_for

    j = queue_for(get_settings()).request_cancel(job_id)
    if j is None:
        typer.secho("job not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"{j.id}: {j.status}{' (cancellation requested)' if j.cancel_requested and j.status != 'cancelled' else ''}"
    )


@jobs_app.command("retry")
def jobs_retry(job_id: str) -> None:
    from aerodoppler.jobs.service import queue_for

    j = queue_for(get_settings()).requeue(job_id)
    if j is None:
        typer.secho(
            "only failed, interrupted or cancelled worker jobs can be retried",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"{j.id}: queued again")


@jobs_app.command("scheduler")
def jobs_scheduler() -> None:
    """Run the job scheduler without the web server (e.g. on a headless machine)."""
    from aerodoppler.jobs.scheduler import run_forever

    _setup_logging(False)
    run_forever(get_settings())


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the local API + UI server (localhost by default)."""
    from aerodoppler.api.app import run_server

    run_server(host, port, reload)


@app.command("demo")
def demo(
    out: Path = typer.Option(Path("data/runs/demo"), "--out"),
    seed: int = typer.Option(7, "--seed"),
    workers: int = typer.Option(0, "--workers"),
    snr_db: float = typer.Option(10.0, "--snr-db"),
    skip_known_time: bool = typer.Option(False, "--skip-known-time"),
) -> None:
    """Fully offline SYNTHETIC demo: build flights+audio, ingest, detect, search (Mode A and B), plot."""
    from aerodoppler.validation.demo import run_demo

    summary = run_demo(
        get_settings(),
        out,
        seed=seed,
        workers=workers,
        snr_db=snr_db,
        run_known_time=not skip_known_time,
        log=lambda m: typer.secho(m, err=True),
    )
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command("benchmark")
def benchmark(
    out: Path = typer.Option(Path("data/runs/benchmark"), "--out"),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="JSON manifest of owner-consented real recordings with surveyed coordinates",
    ),
    workers: int = typer.Option(0, "--workers"),
    cases: str | None = typer.Option(
        None, "--cases", help="comma-separated synthetic case names (default: all)"
    ),
    no_mode_b: bool = typer.Option(False, "--no-mode-b"),
    real_trajectories: str | None = typer.Option(
        None,
        "--real-trajectories",
        help="dataset id: semi-synthetic benchmark on REAL ADS-B tracks",
    ),
    observers: str | None = typer.Option(
        None, "--observers", help="name:lat:lon;name:lat:lon for --real-trajectories"
    ),
    seeds: str = typer.Option("1", "--seeds"),
    snr_db: float = typer.Option(10.0, "--snr-db"),
    livermore: Path | None = typer.Option(
        None,
        "--livermore",
        help="directory with the Zenodo 23424278 files: REAL-recording benchmark (needs --datasets of the imported days)",
    ),
    datasets: str | None = typer.Option(
        None, "--datasets", help="comma-separated dataset ids for --livermore"
    ),
    max_cases: int | None = typer.Option(None, "--max-cases"),
    oracle_events: bool = typer.Option(
        False,
        "--oracle-events",
        help="--real-trajectories: also run with manual events traced from the TRUE ridges (detector-vs-search diagnostic)",
    ),
    events_k: int = typer.Option(
        2,
        "--events-k",
        help="events per Mode A search in --real-trajectories (2 = pair; >2 = multi-event, experimental)",
    ),
    max_assignments: int = typer.Option(
        200_000,
        "--max-assignments",
        help="cap on Mode A pairs per search (consistency-prior selection when exceeded)",
    ),
) -> None:
    """Run the SYNTHETIC benchmark (default), the semi-synthetic real-trajectory benchmark, or a real-recording manifest benchmark."""
    import sys as _sys

    from aerodoppler.jobs.service import run_inline
    from aerodoppler.validation.benchmark import (
        SYNTHETIC_CASES,
        run_manifest_benchmark,
        run_real_trajectory_benchmark,
        run_synthetic_benchmark,
    )

    s = get_settings()
    kind = (
        "livermore"
        if livermore
        else "real_trajectories"
        if real_trajectories
        else "manifest"
        if manifest
        else "synthetic"
    )

    def _go(ctx) -> dict:
        def echo(m: str) -> None:
            typer.secho(m, err=True)
            ctx.check_cancel()  # Ctrl-C / SIGTERM / cancel from the Jobs page stop between steps

        if livermore is not None:
            from aerodoppler.validation.livermore import run_livermore_benchmark

            rep = run_livermore_benchmark(
                s,
                livermore,
                out,
                [d for d in (datasets or "").split(",") if d],
                workers=workers,
                max_cases=max_cases,
                log=echo,
            )
        elif real_trajectories:
            obs = [
                (p.split(":")[0], float(p.split(":")[1]), float(p.split(":")[2]))
                for p in (observers or "").split(";")
                if p
            ]
            rep = run_real_trajectory_benchmark(
                s,
                real_trajectories,
                obs,
                out,
                workers=workers,
                seeds=tuple(int(x) for x in seeds.split(",")),
                snr_db=snr_db,
                log=echo,
                max_assignments=max_assignments,
                events_k=events_k,
                oracle_events=oracle_events,
            )
        elif manifest is not None:
            rep = run_manifest_benchmark(s, manifest, out, workers=workers, log=echo)
        else:
            sel = (
                SYNTHETIC_CASES
                if not cases
                else [c for c in SYNTHETIC_CASES if c["name"] in set(cases.split(","))]
            )
            rep = run_synthetic_benchmark(
                s, out, cases=sel, workers=workers, mode_b=not no_mode_b, log=echo
            )
        return {
            "out": str(out),
            "summary": {k: v for k, v in rep.items() if k != "cases"},
            "_rep": rep,
        }

    # a benchmark is an inline job: same admission as queued work (it waits for the analysis
    # slot and memory), a lease and guaranteed finalisation; its searches run as child jobs
    _job, res = run_inline(
        s,
        "benchmark",
        {"kind": kind, "cli": _sys.argv[1:], "out": str(out)},
        fn=_go,
        log=lambda m: typer.secho(m, err=True),
    )
    rep = res.pop("_rep")
    typer.echo(json.dumps({k: v for k, v in rep.items() if k != "cases"}, indent=2, default=str))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
