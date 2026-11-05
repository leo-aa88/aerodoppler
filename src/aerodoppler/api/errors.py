"""Structured API errors.

Every error the API returns has the shape ``{"detail": {"code", "message", "field"?,
"action"?, "errors"?, ...}}``: a stable machine code, one human sentence, the request field
it concerns, and what to do about it. Request-validation failures are rewritten the same way,
without Pydantic's internal wording or documentation URLs, and unexpected exceptions never
expose stack traces or signed download URLs.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

_URL = re.compile(r"https?://\S+")


def scrub(text: str) -> str:
    """Remove URLs (signed GitHub asset URLs carry tokens) from user-facing text."""
    return _URL.sub("<url>", text)


def api_error(
    status: int,
    code: str,
    message: str,
    *,
    field: str | None = None,
    action: str | None = None,
    **extra: Any,
) -> HTTPException:
    detail: dict[str, Any] = {"code": code, "message": scrub(message)}
    if field:
        detail["field"] = field
    if action:
        detail["action"] = action
    detail.update(extra)
    return HTTPException(status_code=status, detail=detail)


_FIELD_LABELS = {
    "start_utc": "UTC start time",
    "start_uncertainty_s": "start-time uncertainty",
    "date_utc": "UTC date",
    "event_ids": "events",
    "bbox": "search area",
    "flight_dataset_ids": "flight datasets",
}


def _friendly(err: dict[str, Any]) -> dict[str, str]:
    loc = [str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path")]
    field = ".".join(loc)
    msg = str(err.get("msg", "invalid value"))
    msg = msg.removeprefix("Value error, ").removeprefix("Assertion failed, ")
    if err.get("type") == "missing":
        msg = "is required"
    elif err.get("type") == "string_pattern_mismatch":
        msg = "has the wrong format"
    label = _FIELD_LABELS.get(loc[0], field) if loc else "request"
    return {"field": field or "request", "message": f"{label} {msg}" if loc else msg}


async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [_friendly(e) for e in exc.errors()]
    message = "; ".join(e["message"] for e in errors[:4]) or "invalid request"
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_request",
                "message": scrub(message),
                "field": errors[0]["field"] if errors else None,
                "errors": errors,
            }
        },
    )
