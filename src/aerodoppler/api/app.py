"""FastAPI application factory. Binds to localhost by default and serves the built Next.js
export (see :func:`resolve_web_dir`) when present, so one process provides API + UI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aerodoppler import __version__
from aerodoppler.api.routes import router
from aerodoppler.config import Settings, get_settings
from aerodoppler.jobs.store import Store

log = logging.getLogger(__name__)
REPO_WEB_OUT = Path(__file__).resolve().parents[3] / "web" / "out"
PACKAGED_WEB = Path(__file__).resolve().parents[1] / "web_static"


def resolve_web_dir() -> Path | None:
    """Built UI location: $AERODOPPLER_WEB_DIR, else the copy bundled in the package (wheels,
    Docker), else ``web/out`` of a source checkout. None when no build is found."""
    env = os.environ.get("AERODOPPLER_WEB_DIR")
    candidates = ([Path(env)] if env else []) + [PACKAGED_WEB, REPO_WEB_OUT]
    for c in candidates:
        if (c / "index.html").is_file():
            return c
    return None


def create_app(settings: Settings | None = None, web_dir: Path | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    app = FastAPI(
        title="AeroDoppler",
        version=__version__,
        description="Consent-based aircraft-Doppler acoustic geolocation (local-first research tool).",
    )
    app.state.settings = settings
    app.state.store = Store(settings.db_path)
    app.include_router(router)

    @app.middleware("http")
    async def _no_external_leak_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    web = web_dir if web_dir is not None else resolve_web_dir()
    if web is not None and (web / "index.html").exists():
        app.mount("/_next", StaticFiles(directory=web / "_next"), name="next-static")

        # HEAD as well as GET: Next.js 16 prefetches pages with HEAD requests.
        @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
        async def _spa(path: str):
            # static export: /foo -> foo.html, /foo/ -> foo/index.html
            if path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "not found"})
            candidates = [
                web / path,
                web / f"{path}.html",
                web / path / "index.html",
                web / "index.html",
            ]
            for c in candidates:
                if c.is_file() and c.resolve().is_relative_to(web.resolve()):
                    return FileResponse(c)
            return FileResponse(web / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        async def _root():
            return JSONResponse(
                {
                    "name": "AeroDoppler",
                    "version": __version__,
                    "docs": "/docs",
                    "note": "frontend not built; run `make web-install && make web-build`",
                }
            )

    return app


def run_server(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    import uvicorn

    s = get_settings()
    host = host or s.host
    port = port or s.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "Binding to %s exposes private recordings/results to the network. Add authentication (reverse proxy) before doing this.",
            host,
        )
    uvicorn.run(
        "aerodoppler.api.app:create_app",
        host=host,
        port=port,
        factory=True,
        reload=reload,
        log_level="info",
    )
