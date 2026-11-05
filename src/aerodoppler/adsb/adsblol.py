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
import re
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


# First ADSB.lol release (repository globe_history_2023, "test" instances; verified 2026-11-05).
# There is no ADSB.lol repository for 2022 or earlier.
ARCHIVE_FIRST_DAY = dt.date(2023, 2, 16)
# The day D archive appeared at ~03:30 UTC on D+1 in Nov 2026 (2026-11-03 at 03:27 UTC on
# 11-04, 2026-11-04 at 03:28 UTC on 11-05); treat anything newer than ~8 h after the day's
# end as "possibly not yet published" rather than missing.
PUBLICATION_GRACE_H = 8.0


class ProviderUnavailable(RuntimeError):
    """Why an archive day cannot be used, with a machine-readable ``kind``.

    kinds: future_date, not_yet_published, before_archive_start, not_found, rate_limited,
    network, provider_error, offline_no_cache, corrupt_download, insufficient_disk, cancelled
    """

    def __init__(self, message: str, kind: str = "provider_error", **details: Any):
        super().__init__(message)
        self.kind = kind
        self.details = details


def availability_kind_for_missing(date_utc: str, now: dt.datetime | None = None) -> str:
    """Classify a day with no release: future, not yet published, before the archive, or
    genuinely missing. Never suggests a neighbouring day."""
    now = now or dt.datetime.now(dt.UTC)
    d = dt.date.fromisoformat(date_utc)
    if d < ARCHIVE_FIRST_DAY:
        return "before_archive_start"
    day_end = dt.datetime.combine(d + dt.timedelta(days=1), dt.time(), tzinfo=dt.UTC)
    if now < day_end:
        return "future_date"
    if (now - day_end).total_seconds() < PUBLICATION_GRACE_H * 3600.0:
        return "not_yet_published"
    return "not_found"


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
                    f"offline mode and no cached archive parts for {date_utc} under {dest_dir}",
                    kind="offline_no_cache",
                )
            return ArchiveLocation(
                self.name,
                date_utc,
                local,
                LICENSE,
                ATTRIBUTION,
                ["offline: using cached parts, sizes not re-verified against GitHub"],
            )
        kind0 = availability_kind_for_missing(date_utc)
        if kind0 in ("future_date", "before_archive_start") and not local:
            raise ProviderUnavailable(_missing_message(date_utc, kind0, None), kind=kind0)
        errors = []
        network_errors = 0
        for inst in INSTANCES:
            tag = self.release_tag(date_utc, inst)
            url = f"https://api.github.com/repos/{self.repo(date_utc)}/releases/tags/{tag}"
            try:
                r = httpx.get(
                    url, headers=self._headers(), timeout=self.timeout, follow_redirects=True
                )
            except httpx.HTTPError as e:
                errors.append(f"{tag}: {type(e).__name__}")
                network_errors += 1
                continue
            if r.status_code == 404:
                errors.append(f"{tag}: not found")
                continue
            if r.status_code in (403, 429) and (
                "rate limit" in r.text.lower() or r.headers.get("x-ratelimit-remaining") == "0"
            ):
                if local:
                    return ArchiveLocation(
                        self.name,
                        date_utc,
                        local,
                        LICENSE,
                        ATTRIBUTION,
                        ["GitHub API rate limited; using cached parts"],
                    )
                reset = r.headers.get("x-ratelimit-reset")
                when = (
                    dt.datetime.fromtimestamp(int(reset), dt.UTC).strftime("%H:%M UTC")
                    if reset and reset.isdigit()
                    else "later"
                )
                raise ProviderUnavailable(
                    f"GitHub API rate limit reached (60 requests/hour without a token); it resets at {when}. "
                    "Retry then, or set AERODOPPLER_GITHUB_TOKEN.",
                    kind="rate_limited",
                    retry_at=when,
                )
            if r.status_code >= 500:
                errors.append(f"{tag}: GitHub HTTP {r.status_code}")
                network_errors += 1
                continue
            if r.status_code >= 400:
                errors.append(f"{tag}: GitHub HTTP {r.status_code}")
                continue
            rel = r.json()
            assets = [
                ArchiveAsset(
                    a["name"],
                    a["browser_download_url"],
                    int(a["size"]),
                    expected_sha256=(a.get("digest") or "").removeprefix("sha256:") or None,
                )
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
        if network_errors and network_errors == len(errors):
            raise ProviderUnavailable(
                f"could not reach GitHub to look up the {date_utc} archive (network or GitHub "
                "outage); nothing was downloaded. Check the connection and retry.",
                kind="network",
                attempts=errors,
            )
        kind = availability_kind_for_missing(date_utc)
        latest = self.latest_published_day(date_utc[:4]) if kind == "not_yet_published" else None
        raise ProviderUnavailable(
            _missing_message(date_utc, kind, latest), kind=kind, latest_published=latest
        )

    def latest_published_day(self, year: str) -> str | None:
        """Newest prod/staging day in the year's repository (one API call), or None."""
        try:
            r = httpx.get(
                f"https://api.github.com/repos/adsblol/globe_history_{year}/releases?per_page=10",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return None
            for rel in r.json():
                tag = rel.get("tag_name", "")
                if any(tag.endswith(i) for i in INSTANCES):
                    return tag[1:11].replace(".", "-")
        except (httpx.HTTPError, ValueError):
            return None
        return None

    @staticmethod
    def _local_assets(dest_dir: Path, date_utc: str) -> list[ArchiveAsset]:
        d = dest_dir / date_utc
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.tar*")):
            # only complete parts: name.tar or name.tar.aa/.ab/…; never .part/.corrupt/sidecars
            if not re.search(r"\.tar(\.[a-z]{2})?$", p.name):
                continue
            out.append(ArchiveAsset(p.name, None, p.stat().st_size, p))
        return out

    # ---- download ----------------------------------------------------------------------
    def fetch(
        self,
        location: ArchiveLocation,
        dest_dir: Path,
        progress: bool = True,
        max_retries: int = 5,
        on_progress: Any = None,
        should_cancel: Any = None,
    ) -> ArchiveLocation:
        """Download missing parts (resuming ``.part`` files), then verify every part against
        the provider's SHA-256 digest when one is published (cached per size+mtime in
        ``<part>.verified.json`` so a 3.5 GB day is hashed once). A part that fails
        verification is renamed ``.corrupt`` (kept, not deleted) and downloaded again."""
        from aerodoppler.resources import disk_free_bytes

        d = dest_dir / location.date_utc
        d.mkdir(parents=True, exist_ok=True)
        missing = 0
        for a in location.assets:
            target = d / a.name
            complete = target.exists() and (
                a.size_bytes is None or target.stat().st_size == a.size_bytes
            )
            if not complete and a.size_bytes is not None:
                part = target.with_suffix(target.suffix + ".part")
                missing += a.size_bytes - (part.stat().st_size if part.exists() else 0)
        free = disk_free_bytes(d)
        if missing and free < missing + 2_000_000_000:
            raise ProviderUnavailable(
                f"not enough free disk under {d}: need {missing / 1e9:.2f} GB for the download "
                f"plus 2 GB working space, have {free / 1e9:.2f} GB",
                kind="insufficient_disk",
            )
        total = sum(a.size_bytes or 0 for a in location.assets) or None
        done_bytes = 0
        for a in location.assets:
            target = d / a.name
            a.local_path = target
            for attempt in (1, 2):
                if not (
                    target.exists()
                    and (a.size_bytes is None or target.stat().st_size == a.size_bytes)
                ):
                    if self.offline or a.url is None:
                        raise ProviderUnavailable(
                            f"{a.name} missing locally and downloads are disabled (offline)",
                            kind="offline_no_cache",
                        )
                    base = done_bytes

                    def _cb(
                        have: int,
                        _base: int = base,
                        _name: str = a.name,
                        _size: int | None = a.size_bytes,
                    ) -> None:
                        if on_progress:
                            on_progress(
                                "download",
                                {
                                    "part": _name,
                                    "bytes_done": _base + have,
                                    "bytes_total": total,
                                    "part_bytes_done": have,
                                    "part_bytes_total": _size,
                                },
                            )

                    self._download_resumable(
                        a.url, target, a.size_bytes, progress, max_retries, _cb, should_cancel
                    )
                    a.retrieved_at = dt.datetime.now(dt.UTC).isoformat()
                if on_progress:
                    on_progress("verify", {"part": a.name})
                a.sha256 = _cached_sha256(target)
                if a.expected_sha256 and a.sha256 != a.expected_sha256:
                    target.rename(target.with_suffix(target.suffix + ".corrupt"))
                    Path(str(target) + ".verified.json").unlink(missing_ok=True)
                    if attempt == 2 or self.offline:
                        raise ProviderUnavailable(
                            f"{a.name} failed SHA-256 verification twice (expected "
                            f"{a.expected_sha256[:12]}…, got {a.sha256[:12]}…); the bad copy was "
                            "kept as .corrupt",
                            kind="corrupt_download",
                        )
                    continue
                break
            done_bytes += a.size_bytes or 0
        meta = d / "retrieval.json"
        meta.write_text(json.dumps([a.__dict__ for a in location.assets], default=str, indent=2))
        return location

    def _download_resumable(
        self,
        url: str,
        target: Path,
        expected: int | None,
        progress: bool,
        max_retries: int,
        on_bytes: Any = None,
        should_cancel: Any = None,
    ) -> None:
        from tqdm import tqdm

        tmp = target.with_suffix(target.suffix + ".part")
        attempt = 0
        last_cb = 0.0
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
                            have += len(chunk)
                            bar.update(len(chunk))
                            now = time.time()
                            if now - last_cb >= 1.0:
                                last_cb = now
                                if on_bytes:
                                    on_bytes(have)
                                if should_cancel and should_cancel():
                                    raise ProviderUnavailable(
                                        "download cancelled; the partial file is kept and the "
                                        "next import resumes it",
                                        kind="cancelled",
                                    )
                    bar.close()
                break
            except (httpx.HTTPError, OSError) as e:
                if attempt >= max_retries:
                    raise ProviderUnavailable(
                        f"download of {target.name} failed after {attempt} attempts "
                        f"({type(e).__name__}); the partial file is kept and a retry resumes it",
                        kind="network",
                    ) from e
                time.sleep(min(30.0, 2.0**attempt))
        size = tmp.stat().st_size
        if expected is not None and size != expected:
            raise ProviderUnavailable(
                f"{target.name}: size {size} != expected {expected} (truncated download; retry resumes it)",
                kind="corrupt_download",
            )
        if on_bytes:
            on_bytes(size)
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


def _cached_sha256(path: Path) -> str:
    """SHA-256 of a file, cached next to it keyed by size and mtime."""
    side = Path(str(path) + ".verified.json")
    st = path.stat()
    if side.exists():
        try:
            c = json.loads(side.read_text())
            if c.get("size") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns:
                return str(c["sha256"])
        except (ValueError, KeyError):
            pass
    digest = sha256_file(path)
    side.write_text(json.dumps({"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}))
    return digest


def _missing_message(date_utc: str, kind: str, latest: str | None) -> str:
    if kind == "future_date":
        return (
            f"{date_utc} (UTC) has not ended yet, so no archive can exist. ADSB.lol publishes "
            "each UTC day a few hours after it ends."
        )
    if kind == "before_archive_start":
        return (
            f"ADSB.lol has no archive for {date_utc}: its daily archives start in February "
            f"2023 ({ARCHIVE_FIRST_DAY.isoformat()}). Flight trajectories for this date are not "
            "available from this provider, so a flight-matching search cannot run."
        )
    if kind == "not_yet_published":
        tail = f" The newest published day is {latest}." if latest else ""
        return (
            f"The ADSB.lol archive for {date_utc} is not published yet. Days usually appear "
            f"about 3–4 hours after the UTC day ends; retry later.{tail} No other date was "
            "substituted."
        )
    return (
        f"ADSB.lol has no prod or staging release for {date_utc} (the provider may have "
        "skipped this day). No other date was substituted."
    )
