"""Provider interface for historical flight archives (SPEC §6).

Two providers ship with v1: :class:`LocalTraceProvider` (offline fixtures / already extracted
archives, used by tests and the demo) and :class:`aerodoppler.adsb.adsblol.AdsbLolProvider`
(daily ADSB.lol GitHub releases). OpenSky historical Trino access is intentionally *not*
implemented: it requires an approved research account and is not a plain download.
"""

from __future__ import annotations

import gzip
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ArchiveAsset:
    name: str
    url: str | None
    size_bytes: int | None
    local_path: Path | None = None
    sha256: str | None = None
    retrieved_at: str | None = None
    expected_sha256: str | None = None  # digest published by the provider, when available


@dataclass
class ArchiveLocation:
    provider: str
    date_utc: str
    assets: list[ArchiveAsset]
    license: str
    attribution: str
    notes: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int | None:
        sizes = [a.size_bytes for a in self.assets]
        return None if any(s is None for s in sizes) else int(sum(sizes))  # type: ignore[arg-type]


class FlightProvider(Protocol):
    name: str

    def locate(self, date_utc: str) -> ArchiveLocation: ...

    def fetch(
        self, location: ArchiveLocation, dest_dir: Path, progress: bool = True
    ) -> ArchiveLocation: ...

    def iter_traces(self, location: ArchiveLocation) -> Iterator[tuple[str, bytes]]: ...

    def describe(self) -> dict[str, Any]: ...


def iter_tar_traces(parts: list[Path]) -> Iterator[tuple[str, bytes]]:
    """Stream ``traces/**/trace_full_*.json`` members from a (possibly split) tar."""
    reader = _ChainedReader(parts)
    with tarfile.open(fileobj=reader, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            base = name.rsplit("/", 1)[-1]
            if "/traces/" not in f"/{name}" or not base.startswith("trace_full_"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            yield name, fh.read()


class _ChainedReader:
    """Sequential read() over several files, as one stream (for split archives)."""

    def __init__(self, parts: list[Path]):
        self.parts = list(parts)
        self.idx = 0
        self.fh = open(self.parts[0], "rb") if self.parts else None

    def read(self, n: int = -1) -> bytes:
        if self.fh is None:
            return b""
        out = bytearray()
        while n < 0 or len(out) < n:
            chunk = self.fh.read(n - len(out) if n >= 0 else -1)
            if chunk:
                out.extend(chunk)
                if n < 0:
                    continue
            else:
                self.fh.close()
                self.idx += 1
                if self.idx >= len(self.parts):
                    self.fh = None
                    break
                self.fh = open(self.parts[self.idx], "rb")
        return bytes(out)

    def close(self) -> None:
        if self.fh:
            self.fh.close()


class LocalTraceProvider:
    """Reads trace files from a directory tree (``traces/xx/trace_full_*.json``) or tar parts."""

    name = "local"

    def __init__(self, path: Path, source_label: str = "local"):
        self.path = Path(path)
        self.source_label = source_label

    def locate(self, date_utc: str) -> ArchiveLocation:
        assets: list[ArchiveAsset] = []
        if self.path.is_dir():
            for p in sorted(self.path.rglob("trace_full_*.json*")):
                assets.append(ArchiveAsset(p.name, None, p.stat().st_size, p))
            for p in sorted(self.path.glob("*.tar*")):
                assets.append(ArchiveAsset(p.name, None, p.stat().st_size, p))
        elif self.path.exists():
            assets.append(ArchiveAsset(self.path.name, None, self.path.stat().st_size, self.path))
        return ArchiveLocation(
            self.name,
            date_utc,
            assets,
            license="as provided locally",
            attribution=f"local files under {self.path}",
            notes=["offline local import"],
        )

    def fetch(
        self, location: ArchiveLocation, dest_dir: Path, progress: bool = True
    ) -> ArchiveLocation:
        return location  # nothing to download

    def iter_traces(self, location: ArchiveLocation) -> Iterator[tuple[str, bytes]]:
        tars = [
            a.local_path for a in location.assets if a.local_path and ".tar" in a.local_path.name
        ]
        if tars:
            yield from iter_tar_traces(sorted(tars))  # type: ignore[arg-type]
        for a in location.assets:
            p = a.local_path
            if p is None or ".tar" in p.name:
                continue
            data = p.read_bytes()
            if p.suffix == ".gz" and data[:2] != b"\x1f\x8b":
                data = gzip.compress(data)
            yield str(p), data

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "path": str(self.path)}
