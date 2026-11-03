"""ADSB.lol daily archive provider (SPEC §6, verified against upstream on 2026-11-03).

Facts verified against https://www.adsb.lol/docs/open-data/historical and the GitHub API:

* One GitHub release per UTC day per instance in ``adsblol/globe_history_<YYYY>``; tag
  ``v<YYYY.MM.DD>-planes-readsb-prod-0`` (fallback ``-staging-0``; ``-mlatonly-0`` is a
  separate MLAT-only dump).
* Assets are a tar split into ``.tar.aa``, ``.tar.ab``, … (≈3.3 GB/day in Nov 2026). The
  tar contains ``traces/<last two hex of icao>/trace_full_<icao>.json`` (gzip-compressed
  readsb trace JSON), ``heatmap/``, ``acas/``, ``README.txt`` and licence files.
* Licence: ODbL 1.0. Attribution: adsb.lol feeders.

The archive is only available as a whole-sky daily dump, so a bounded search still requires
downloading the full day once; the importer therefore shows the exact size, requires an
explicit storage budget / confirmation and caches the raw parts for re-use across bboxes.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from aerodoppler.adsb.providers import ArchiveAsset, ArchiveLocation, iter_tar_traces
from aerodoppler.util import sha256_file

LICENSE = "ODbL-1.0"
ATTRIBUTION = "Flight data © adsb.lol feeders, Open Database License (ODbL) 1.0, https://www.adsb.lol/docs/open-data/historical"
INSTANCES = ("planes-readsb-prod-0", "planes-readsb-staging-0")


class ProviderUnavailable(RuntimeError):
    pass


class AdsbLolProvider:
    name = "adsblol"

    def __init__(
        self, offline: bool = False, timeout_s: float = 60.0, github_token: str | None = None
    ):
        self.offline = offline
        self.timeout = timeout_s
        self.token = github_token

    # ---- discovery ---------------------------------------------------------------------
    @staticmethod
    def release_tag(date_utc: str, instance: str) -> str:
        d = dt.date.fromisoformat(date_utc)
        return f"v{d:%Y.%m.%d}-{instance}"

    @staticmethod
    def repo(date_utc: str) -> str:
        return f"adsblol/globe_history_{date_utc[:4]}"

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/vnd.github+json", "User-Agent": "aerodoppler/0.1"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def locate(self, date_utc: str, dest_dir: Path | None = None) -> ArchiveLocation:
        """Find release assets for a date (API), or fall back to local cached parts offline."""
        local = self._local_assets(dest_dir, date_utc) if dest_dir else []
        if self.offline:
            if not local:
                raise ProviderUnavailable(
                    f"offline mode and no cached archive parts for {date_utc} under {dest_dir}"
                )
            return ArchiveLocation(
                self.name,
                date_utc,
                local,
                LICENSE,
                ATTRIBUTION,
                ["offline: using cached parts, sizes not re-verified against GitHub"],
            )
        errors = []
        for inst in INSTANCES:
            tag = self.release_tag(date_utc, inst)
            url = f"https://api.github.com/repos/{self.repo(date_utc)}/releases/tags/{tag}"
            try:
                r = httpx.get(
                    url, headers=self._headers(), timeout=self.timeout, follow_redirects=True
                )
            except httpx.HTTPError as e:
                errors.append(f"{tag}: {e}")
                continue
            if r.status_code == 404:
                errors.append(f"{tag}: not found")
                continue
            if r.status_code == 403 and "rate limit" in r.text.lower():
                if local:
                    return ArchiveLocation(
                        self.name,
                        date_utc,
                        local,
                        LICENSE,
                        ATTRIBUTION,
                        ["GitHub API rate limited; using cached parts"],
                    )
                raise ProviderUnavailable(
                    "GitHub API rate limit reached; retry later or set a token"
                )
            r.raise_for_status()
            rel = r.json()
            assets = [
                ArchiveAsset(a["name"], a["browser_download_url"], int(a["size"]))
                for a in rel.get("assets", [])
                if ".tar" in a["name"]
            ]
            assets.sort(key=lambda a: a.name)
            if not assets:
                errors.append(f"{tag}: release has no tar assets")
                continue
            notes = [
                f"release {tag} published {rel.get('published_at')}",
                f"html_url {rel.get('html_url')}",
            ]
            if inst != INSTANCES[0]:
                notes.append("prod instance missing for this day; using staging")
            for a in assets:
                cached = next((c for c in local if c.name == a.name), None)
                if cached is not None:
                    a.local_path = cached.local_path
            return ArchiveLocation(self.name, date_utc, assets, LICENSE, ATTRIBUTION, notes)
        if local:
            return ArchiveLocation(
                self.name,
                date_utc,
                local,
                LICENSE,
                ATTRIBUTION,
                ["GitHub lookup failed; using cached parts"] + errors,
            )
        raise ProviderUnavailable(
            "no ADSB.lol release found for " + date_utc + ": " + "; ".join(errors)
        )

    @staticmethod
    def _local_assets(dest_dir: Path, date_utc: str) -> list[ArchiveAsset]:
        d = dest_dir / date_utc
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.tar*")):
            if p.suffix in {".part", ".tmp"}:
                continue
            out.append(ArchiveAsset(p.name, None, p.stat().st_size, p))
        return out

    # ---- download ----------------------------------------------------------------------
    def fetch(
        self, location: ArchiveLocation, dest_dir: Path, progress: bool = True, max_retries: int = 5
    ) -> ArchiveLocation:
        d = dest_dir / location.date_utc
        d.mkdir(parents=True, exist_ok=True)
        for a in location.assets:
            target = d / a.name
            a.local_path = target
            if (
                target.exists()
                and a.size_bytes is not None
                and target.stat().st_size == a.size_bytes
            ):
                continue
            if target.exists() and a.size_bytes is None:
                continue
            if self.offline or a.url is None:
                raise ProviderUnavailable(
                    f"{a.name} missing locally and downloads are disabled (offline)"
                )
            self._download_resumable(a.url, target, a.size_bytes, progress, max_retries)
            a.retrieved_at = dt.datetime.now(dt.UTC).isoformat()
        for a in location.assets:
            if a.local_path and a.sha256 is None:
                a.sha256 = sha256_file(a.local_path)
        meta = d / "retrieval.json"
        meta.write_text(json.dumps([a.__dict__ for a in location.assets], default=str, indent=2))
        return location

    def _download_resumable(
        self, url: str, target: Path, expected: int | None, progress: bool, max_retries: int
    ) -> None:
        from tqdm import tqdm

        tmp = target.with_suffix(target.suffix + ".part")
        attempt = 0
        while True:
            attempt += 1
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "aerodoppler/0.1"}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                with httpx.stream(
                    "GET", url, headers=headers, timeout=self.timeout, follow_redirects=True
                ) as r:
                    if r.status_code == 416:  # already complete
                        break
                    if have and r.status_code != 206:
                        have = 0  # server ignored the range: restart
                        tmp.unlink(missing_ok=True)
                    r.raise_for_status()
                    total = expected or (have + int(r.headers.get("Content-Length", "0")))
                    bar = tqdm(
                        total=total,
                        initial=have,
                        unit="B",
                        unit_scale=True,
                        desc=target.name,
                        disable=not progress,
                    )
                    with open(tmp, "ab" if have else "wb") as fh:
                        for chunk in r.iter_bytes(1 << 20):
                            fh.write(chunk)
                            bar.update(len(chunk))
                    bar.close()
                break
            except (httpx.HTTPError, OSError) as e:
                if attempt >= max_retries:
                    raise ProviderUnavailable(
                        f"download of {url} failed after {attempt} attempts: {e}"
                    ) from e
                time.sleep(min(30.0, 2.0**attempt))
        size = tmp.stat().st_size
        if expected is not None and size != expected:
            raise ProviderUnavailable(
                f"{target.name}: size {size} != expected {expected} (corrupt or truncated download)"
            )
        tmp.rename(target)

    # ---- parsing -----------------------------------------------------------------------
    def iter_traces(self, location: ArchiveLocation) -> Iterator[tuple[str, bytes]]:
        parts = [a.local_path for a in location.assets if a.local_path is not None]
        if not parts:
            raise ProviderUnavailable("no local archive parts; call fetch() first")
        yield from iter_tar_traces(sorted(parts))  # type: ignore[arg-type]

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "license": LICENSE,
            "attribution": ATTRIBUTION,
            "docs": "https://www.adsb.lol/docs/open-data/historical",
            "offline": self.offline,
        }


def read_trace_file(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return json.loads(data)
