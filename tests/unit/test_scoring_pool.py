"""The search's fork-based scoring pool must stop even when the parent has a cooperative
SIGTERM handler (inline jobs install one) and the workers run extra threads (numba)."""

from __future__ import annotations

import multiprocessing as mp
import signal
import threading
import time

from aerodoppler.search.engine import _pool_worker_init


def _handler_in_worker(_x: int) -> bool:
    return signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def _idle_thread_then_report(_x: int) -> int:
    # a background thread like numba's: SIGTERM may be delivered to it, not the main thread
    threading.Thread(target=time.sleep, args=(3600,), daemon=True).start()
    return 1


def test_workers_reset_an_inherited_sigterm_handler():
    prev = signal.signal(signal.SIGTERM, lambda *_: None)  # cooperative: does not exit
    try:
        with mp.get_context("fork").Pool(2, initializer=_pool_worker_init) as pool:
            assert all(pool.map(_handler_in_worker, range(4)))
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_terminate_stops_threaded_workers_under_a_cooperative_parent_handler():
    prev = signal.signal(signal.SIGTERM, lambda *_: None)
    try:
        t0 = time.monotonic()
        pool = mp.get_context("fork").Pool(3, initializer=_pool_worker_init)
        assert sum(pool.map(_idle_thread_then_report, range(3), chunksize=1)) == 3
        pool.terminate()  # what `with Pool()` does on an exception or cancellation
        pool.join()
        assert time.monotonic() - t0 < 30
    finally:
        signal.signal(signal.SIGTERM, prev)
