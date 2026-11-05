"""Geolocation search execution inside the job system.

:func:`execute_search` runs the (unchanged) search engine for one job: memory guard against the
job's reservation, worker count derived from it, progress and cancellation through the job
context, and an ``INCOMPLETE`` marker in the output directory until the result is written, so a
crashed or cancelled search is never mistaken for a valid one.

:func:`run_search_inline` is the synchronous entry used by the CLI, the demo and benchmarks; it
goes through :func:`aerodoppler.jobs.service.run_inline` (admission, lease, guaranteed
finalisation) and returns the in-memory :class:`SearchResult`.

:func:`reconcile_stale_jobs` only serves the one-time migration of the pre-queue ``searches``
table.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aerodoppler.config import Settings
from aerodoppler.jobs.proc import job_process_alive, process_start_time
from aerodoppler.jobs.queue import JobCancelled, JobError
from aerodoppler.jobs.store import Store
from aerodoppler.models import SearchRequest, SearchResult

log = logging.getLogger(__name__)

_process_start_time = process_start_time
_job_process_alive = job_process_alive
INCOMPLETE = "INCOMPLETE"


class SearchResourceExceeded(RuntimeError):
    """Raised from the progress callback when a search exceeds its memory budget."""


def _job_errors():
    from aerodoppler.search.engine import SearchInputError

    class SearchInputJobError(JobError, SearchInputError):
        """A job error that callers of the pre-queue API can still catch as SearchInputError."""

    class SearchMemoryJobError(JobError, SearchResourceExceeded):
        pass

    return SearchInputJobError, SearchMemoryJobError


def execute_search(
    ctx: Any, req: SearchRequest, out_dir: Path
) -> tuple[SearchResult, dict[str, Any]]:
    from aerodoppler.resources import recommend_workers, tree_pss_mb
    from aerodoppler.search.engine import SearchCancelled, SearchEngine, SearchInputError

    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / INCOMPLETE
    marker.write_text(
        "this search has not finished: files in this directory are partial and are not a result\n"
    )
    (out_dir / "request.json").write_text(req.model_dump_json(indent=2))
    budget = float(ctx.memory_budget_mb or 4096.0)
    if not req.quality_settings.workers:
        # forked scorers share the spectrogram copy-on-write; ~250 MB private each measured on
        # the dense-traffic benchmark (3.4 GB peak for 6 workers + 2.0 GB parent)
        req = req.model_copy(
            update={
                "quality_settings": req.quality_settings.model_copy(
                    update={"workers": recommend_workers(budget, 250.0, base_mb=2000.0)}
                )
            }
        )
    mem = {"last": 0.0, "peak": 0.0}
    t0 = time.time()

    def _progress(stage: str, info: dict[str, Any]) -> None:
        now = time.time()
        if now - mem["last"] >= 2.0:
            mem["last"] = now
            used = tree_pss_mb()
            mem["peak"] = max(mem["peak"], used)
            if used > 1.5 * budget:
                raise SearchResourceExceeded(
                    f"the search used {used:.0f} MB (workers included), above 1.5x its "
                    f"{budget:.0f} MB budget, during '{stage}'. Use a smaller search area, a "
                    "coarser grid or a narrower time window, or raise "
                    "AERODOPPLER_SEARCH_MEMORY_BUDGET_MB."
                )
        ctx.progress(
            stage,
            {
                **info,
                "elapsed_s": round(now - t0, 1),
                "tree_pss_peak_mb": round(mem["peak"], 1),
                "memory_budget_mb": round(budget),
                "workers": req.quality_settings.workers,
            },
            min_interval_s=0.0,
        )

    engine = SearchEngine(
        ctx.store, ctx.settings, progress=_progress, cancel_check=ctx.should_cancel
    )
    try:
        result = engine.run(req, ctx.job.id, out_dir)
    except SearchCancelled as e:
        raise JobCancelled("cancelled by operator") from e
    except SearchResourceExceeded as e:
        (out_dir / "error.json").write_text(json.dumps({"error": str(e), "type": "memory_budget"}))
        raise _job_errors()[1](str(e), code="memory_budget") from e
    except SearchInputError as e:
        (out_dir / "error.json").write_text(json.dumps({"error": str(e), "type": "input"}))
        raise _job_errors()[0](str(e), code="input_error") from e
    marker.unlink(missing_ok=True)
    return result, {
        "status": "abstained" if result.abstention.abstained else "succeeded",
        "abstention": result.abstention.model_dump(),
        "candidates": len(result.candidates),
        "confidence_grade": result.confidence_grade,
        "result_dir": str(out_dir),
        "peak_pss_mb": round(mem["peak"], 1),
    }


def run_search_inline(
    store: Store, settings: Settings, req: SearchRequest, progress=None
) -> SearchResult:
    """Run a search synchronously as an inline (or child) job and return its result."""
    from aerodoppler.jobs.service import run_inline

    class _Out:
        def __init__(self, result: SearchResult, job_result: dict[str, Any]):
            self.result, self.job_result = result, job_result

    def fn(ctx: Any) -> _Out:
        if progress is not None:
            inner = ctx._report

            def both(stage: str, info: dict[str, Any]) -> None:
                inner(stage, info)
                progress(stage, info)

            ctx._report = both
        return _Out(*execute_search(ctx, req, ctx.work_dir))

    _job, out = run_inline(settings, "geolocation_search", req.model_dump(mode="json"), fn=fn)
    return out.result


# ---- legacy (pre-queue) searches table: used once by the migration --------------------------


def reconcile_stale_jobs(store: Store, queued_grace_s: float = 60.0) -> list[str]:
    """Mark legacy ``searches`` rows whose worker no longer exists as interrupted."""
    reconciled: list[str] = []
    now = time.time()
    for job in store.list_active_searches():
        stale = job["status"] == "running" and not _job_process_alive(job)
        if job["status"] == "queued":
            updated = datetime.fromisoformat(job["updated_at"].replace("Z", "+00:00"))
            stale = now - updated.astimezone(UTC).timestamp() > queued_grace_s
        if stale and store.finish_if_active(
            job["id"],
            status="interrupted",
            stage="worker_lost",
            error=(
                f"the search worker stopped without finishing (crash, kill, out of memory or "
                f"machine restart) during stage '{job.get('stage')}'; partial outputs are kept. "
                "Run the search again to retry."
            ),
        ):
            store.audit("search.reconciled", job["id"], {"previous_status": job["status"]})
            reconciled.append(job["id"])
    return reconciled
