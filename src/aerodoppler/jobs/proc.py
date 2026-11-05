"""Process identity and heartbeats shared by search and import workers.

A job row stores the worker PID *and* that process's creation time; a row is considered live
only while a process with that PID exists, is not a zombie, and has the same creation time.
This survives PID reuse after a crash or reboot (a different process with the recorded PID is
not mistaken for the worker). Workers also write a heartbeat every few seconds; a live worker
without a recent heartbeat is reported as stalled, never killed automatically.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import psutil

STALL_AFTER_S = 180.0


def process_start_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return None


def job_process_alive(job: dict[str, Any]) -> bool:
    pid = job.get("pid")
    if not pid:
        return False
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        expected_start = job.get("pid_started_at")
        return expected_start is None or abs(proc.create_time() - expected_start) < 1.0
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        # A local process we cannot inspect is safer to treat as active than to duplicate.
        return True


def is_stalled(job: dict[str, Any], now: float | None = None) -> bool:
    hb = job.get("heartbeat_at")
    if job.get("status") != "running" or hb is None:
        return False
    return (now or time.time()) - float(hb) > STALL_AFTER_S


class Heartbeat:
    """Calls ``beat()`` every ``interval_s`` on a daemon thread until stopped."""

    def __init__(self, beat: Callable[[], None], interval_s: float = 5.0):
        self._beat = beat
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._beat()
            except Exception:  # a failed heartbeat must never kill the job
                pass

    def __enter__(self) -> Heartbeat:
        try:
            self._beat()
        except Exception:
            pass
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
