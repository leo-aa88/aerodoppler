# Privacy, consent and data handling

AeroDoppler is a **local-first research and forensics tool for recordings you own or are
permitted to analyse**. It can estimate where a stationary recording was made; that is exactly
why it must not be pointed at other people's recordings.

## What the software enforces

* **Authorization acknowledgement** — every CLI command that ingests a recording or runs a
  search needs `--authorized`; the API needs `owner_permission_confirmed=true` /
  `operator_authorized=true`; the UI has the corresponding checkboxes. Requests without it are
  rejected (HTTP 403 / exit code 2).
* **Localhost only** — `aerodoppler serve` binds to 127.0.0.1. Binding elsewhere logs a
  warning; put an authenticating reverse proxy in front if you must. The Docker compose file
  publishes the port on 127.0.0.1 only.
* **No telemetry, no uploads** — recordings, spectrograms, events and results never leave the
  machine. The only outbound requests are the ones you trigger: ADSB.lol archive downloads
  (GitHub), USGS 3DEP / Copernicus DEM tiles, and optional OpenStreetMap basemap tiles in the
  UI (off by default). Those requests disclose the date and area you are interested in to those
  providers. `AERODOPPLER_OFFLINE=true` forbids all network access.
* **Audit log** — ingest, events, searches, cancellations and purges are logged locally
  (`/api/audit`, Data & privacy page).
* **Purge** — deleting a recording removes the original file, the analysis copy, spectrograms,
  events and all run outputs (`aerodoppler audio purge <id> --yes`, `DELETE /api/recordings/{id}`).
* **Resource limits** — upload size, audio duration, concurrent jobs, bbox area (large-scale
  searches require explicit confirmation) and CPU budget are bounded.
* **Abstention** — when flight coverage, geometry, or ridge evidence is insufficient the search
  returns an explicit abstention reason instead of a location.

## What the software deliberately does not do

No bulk scraping of other people's recordings, no social-media integration, no identification
of recording owners, no covert tracking, no hosted analytics. Scores are **relative,
uncalibrated fit values**; the tool never claims an address or a probability.

## Licensing of flight data

ADSB.lol archives are ODbL 1.0 (attribution and share-alike for derived databases). The
normalised Parquet datasets under `data/normalized/` are derived databases — keep the
attribution in `manifest.json` if you redistribute them.
