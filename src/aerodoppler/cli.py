"""AeroDoppler command-line interface (SPEC §11).

Every command that touches a recording or produces a location requires ``--authorized``:
the operator confirms they own the recording or have the owner's permission to analyse it.
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

AUTH_TEXT = (
    "AUTHORIZATION REQUIRED: AeroDoppler analyses recordings only with the recording owner's "
    "permission. Pass --authorized to confirm you own this recording or have explicit permission "
    "to analyse it. Locations are hypotheses with uncalibrated relative scores, not facts."
)


def _require_auth(authorized: bool) -> None:
    if not authorized:
        typer.secho(AUTH_TEXT, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
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
    workers: int = typer.Option(0, "--workers", help="parallel parsers (0 = cpu count - 1)"),
    gap_split_s: float | None = typer.Option(None, "--gap-split-s"),
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Download/import one UTC day of flights for a bounding box and normalise to Parquet."""
    from aerodoppler.adsb.adsblol import AdsbLolProvider, ProviderUnavailable
    from aerodoppler.adsb.ingest import ImportRequest, StorageBudgetExceeded, import_flights
    from aerodoppler.adsb.providers import LocalTraceProvider

    s = get_settings()
    bb = BBox.parse(bbox)
    if provider == "adsblol":
        prov = AdsbLolProvider(offline=offline or s.offline)
    elif provider == "local":
        if path is None:
            raise typer.BadParameter("--path is required for the local provider")
        prov = LocalTraceProvider(path, source_label="local")
    else:
        raise typer.BadParameter("provider must be adsblol or local")
    import multiprocessing as mp

    req = ImportRequest(
        date_utc=date,
        bbox=bb,
        margin_km=s.adsb_bbox_margin_km if margin_km is None else margin_km,
        storage_budget_gb=storage_budget_gb,
        confirm_download=confirm_download,
        workers=workers or max(1, mp.cpu_count() - 1),
        gap_split_s=s.adsb_gap_split_s if gap_split_s is None else gap_split_s,
    )
    try:
        ds = import_flights(prov, req, s.raw_dir, s.normalized_dir)
    except StorageBudgetExceeded as e:
        typer.secho(str(e), fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=3) from e
    except ProviderUnavailable as e:
        typer.secho(f"provider unavailable: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4) from e
    man = ds.manifest()
    typer.echo(
        json.dumps(
            {
                "dataset_id": ds.dataset_id,
                "summary": man["summary"],
                "filter_counts": {k: v for k, v in man["filter_counts"].items() if k != "errors"},
                "storage_bytes": man["storage_bytes"],
                "license": man["license"],
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
