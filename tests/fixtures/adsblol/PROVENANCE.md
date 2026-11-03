# ADSB.lol fixture provenance

Six real readsb `trace_full_*.json` files (gzip-compressed JSON, readsb trace format) copied
unmodified from the first 3 MB of
`https://github.com/adsblol/globe_history_2026/releases/download/v2026.11.02-planes-readsb-prod-0/v2026.11.02-planes-readsb-prod-0.tar.aa`
(retrieved 2026-11-03, `traces/04/` directory, UTC day 2026-11-02, pod planes-readsb-prod-0).

| file | type | points | notes |
|---|---|---|---|
| trace_full_a12704.json | C172 | 140 | piston GA |
| trace_full_a24504.json | C152 | 717 | 12 on-ground points |
| trace_full_a0e404.json | P28A | 803 | 17 on-ground points |
| trace_full_781004.json | A320 | 170 | jet, cruise |
| trace_full_a22904.json | T206 | 40 | all points on ground |
| trace_full_~26d904.json | (none) | 2 | non-ICAO (TIS-B/MLAT) address, no metadata |

License: Open Database License (ODbL) 1.0, © adsb.lol feeders — see `2026-11-02/ARCHIVE_README.txt`
(the archive's own README). These fixtures are used only to test the parser against the real
schema; the full archive is not redistributed here.
