# Data sources

## ADSB.lol historical archive (primary flight provider)

* Docs: https://www.adsb.lol/docs/open-data/historical — licence ODbL 1.0.
* One GitHub release per UTC day and instance in `adsblol/globe_history_<YYYY>`:
  tag `v<YYYY.MM.DD>-planes-readsb-prod-0` (fallback `-staging-0`). Assets are tar split into
  `.tar.aa`, `.tar.ab`, … (≈3.3–3.5 GB per day in Nov 2026; verified 2026-11-03).
* Tar contents (verified on the real archive): `traces/<last two hex of icao>/trace_full_<icao>.json`
  (gzip-compressed readsb trace JSON), `heatmap/`, `acas/`, `README.txt`, `LICENSE-ODbL.txt`.
* readsb trace record (README-json.md "trace jsons"):
  `[dt, lat, lon, alt_baro_ft|"ground"|null, gs_kt, track, flags, vrate_fpm, aircraft_obj|null,
  position_type, alt_geom_ft, geom_rate, ias, roll]`; flags bit 1 stale, 2 new leg, 4 vertical
  rate geometric, 8 altitude geometric.
* The archive is only available as a whole-sky daily dump; `aerodoppler flights import` downloads
  it once (resumable, size-verified, SHA-256 recorded), caches it under `data/raw/adsblol/<date>/`
  and filters to the bbox (+15 km margin) into Parquet under `data/normalized/<dataset_id>/`.

Decision: OpenSky historical (Trino) access is **not** implemented — it requires an approved
research account and is not a plain download. The provider interface (`adsb/providers.py`)
allows adding it.

## Elevation

* **USGS 3DEP** via the National Map ImageServer `exportImage` (GeoTIFF for a bbox, NAVD88 metres,
  CONUS). Requests are tiled at ≤ 2000 px and cached under `data/dem/`.
* **Copernicus DEM GLO-30** public COGs on AWS (`copernicus-dem-30m` bucket), EGM2008 metres,
  1°×1° tiles, cached under `data/dem/copernicus/`.
* Local GeoTIFFs dropped into `data/dem/` are used first. A labelled **synthetic** DEM is used
  by the demo and tests. If nothing is available the search fails with an actionable error, or
  the operator supplies an explicit flat ground elevation with an uncertainty (flagged).

## Vertical datums

Aircraft geometric altitude is WGS84 ellipsoidal height; barometric altitude is pressure
altitude (ISA); DEMs are orthometric. AeroDoppler converts everything to ellipsoidal height using
the embedded 15′ EGM96 geoid (public domain NGA grid, 1.7 MB) and carries a per-datum
uncertainty (0 m geometric, ~1.5 m geoid model, ~60 m barometric, 80 m unknown). Mixed-datum and
baro-only segments are flagged in results.
