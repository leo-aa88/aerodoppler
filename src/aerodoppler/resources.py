"""Machine resource limits as the process actually sees them (host RAM, cgroup v1/v2 quota,
WSL), and budget-derived worker counts.

Heavy stages (ADS-B import, search) size their parallelism from these numbers instead of
``cpu_count()``, because on WSL2 or in a container the physical RAM is not what the process
may use, and a pool of N parsers each holding a Python + Polars runtime is what exhausted
memory during large imports.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

_CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V2_CUR = Path("/sys/fs/cgroup/memory.current")
_CGROUP_V1_MAX = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CGROUP_V1_CUR = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
_NO_LIMIT = 1 << 60  # cgroup v1 reports ~2^63 for "unlimited"


def _read_int(p: Path) -> int | None:
    try:
        s = p.read_text().strip()
    except OSError:
        return None
    if not s or s == "max":
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    return None if v >= _NO_LIMIT else v


def cgroup_memory() -> tuple[int | None, int | None]:
    """(limit_bytes, usage_bytes) of this process's memory cgroup, None when unlimited."""
    lim = _read_int(_CGROUP_V2_MAX)
    if lim is not None:
        return lim, _read_int(_CGROUP_V2_CUR)
    lim = _read_int(_CGROUP_V1_MAX)
    if lim is not None:
        return lim, _read_int(_CGROUP_V1_CUR)
    return None, None


def is_wsl() -> bool:
    return "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ


@dataclass
class ResourceSnapshot:
    cpu_count: int
    host_total_mb: float
    host_available_mb: float
    cgroup_limit_mb: float | None
    cgroup_usage_mb: float | None
    effective_limit_mb: float
    effective_available_mb: float
    wsl: bool

    def as_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def snapshot() -> ResourceSnapshot:
    vm = psutil.virtual_memory()
    lim, use = cgroup_memory()
    eff_limit = float(vm.total) if lim is None else float(min(vm.total, lim))
    eff_avail = float(vm.available)
    if lim is not None and use is not None:
        eff_avail = min(eff_avail, float(max(lim - use, 0)))
    mb = 1e6
    return ResourceSnapshot(
        cpu_count=os.cpu_count() or 1,
        host_total_mb=vm.total / mb,
        host_available_mb=vm.available / mb,
        cgroup_limit_mb=None if lim is None else lim / mb,
        cgroup_usage_mb=None if use is None else use / mb,
        effective_limit_mb=eff_limit / mb,
        effective_available_mb=eff_avail / mb,
        wsl=is_wsl(),
    )


def default_memory_budget_mb(fraction: float = 0.5, cap_mb: float = 4096.0) -> float:
    """A conservative per-job budget: ``fraction`` of what is available now, capped."""
    s = snapshot()
    return max(256.0, min(cap_mb, fraction * s.effective_available_mb))


def recommend_workers(
    budget_mb: float, per_worker_mb: float, base_mb: float = 400.0, cpu_cap: int | None = None
) -> int:
    """Workers that fit ``budget_mb`` after the parent's own ``base_mb``; at least 1, at most
    the CPU count minus one (and ``cpu_cap``)."""
    cpus = max(1, (os.cpu_count() or 2) - 1)
    if cpu_cap is not None:
        cpus = min(cpus, max(1, cpu_cap))
    fit = int((budget_mb - base_mb) // max(per_worker_mb, 1.0))
    return max(1, min(cpus, fit))


def tree_rss_mb(pid: int | None = None) -> float:
    """Resident memory of a process and all its descendants (pool workers included)."""
    try:
        p = psutil.Process(pid or os.getpid())
        procs = [p, *p.children(recursive=True)]
    except psutil.Error:
        return 0.0
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except psutil.Error:
            continue
    return total / 1e6


def tree_pss_mb(pid: int | None = None) -> float:
    """Proportional set size of a process tree: fork-shared pages are split between the
    processes that share them instead of being counted once per worker (RSS would count a
    forked pool's shared spectrogram N times). Falls back to RSS where PSS is unavailable."""
    try:
        p = psutil.Process(pid or os.getpid())
        procs = [p, *p.children(recursive=True)]
    except psutil.Error:
        return 0.0
    total = 0
    for q in procs:
        try:
            info = q.memory_full_info()
            total += getattr(info, "pss", info.rss)
        except psutil.Error:
            try:
                total += q.memory_info().rss
            except psutil.Error:
                continue
    return total / 1e6


def disk_free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free
