"""FastAPI application.

Endpoints are deliberately synchronous (``def`` rather than ``async def``).
Building a composite is CPU-bound and does blocking network I/O, so FastAPI's
threadpool is the right place for it -- running it on the event loop would stall
every other request for the duration.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from atarra import __version__
from atarra.api.routers import imagery, system
from atarra.core.errors import AtarraError, ConfigError, ImageryError
from atarra.core.logging import configure_logging, get_logger

log = get_logger("api")


def create_app() -> FastAPI:
    """Build the application."""
    configure_logging()

    app = FastAPI(
        title="ATARRA",
        description=(
            "Aquatic invasive weed tracking from multispectral satellite telemetry. "
            "Discovers Sentinel-2 L2A scenes, computes spectral indices, and serves "
            "rendered composites for the remediation dashboard."
        ),
        version=__version__,
    )

    # The dashboard runs on a different origin during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(imagery.router)

    @app.exception_handler(ImageryError)
    async def imagery_error_handler(_: Request, exc: ImageryError) -> JSONResponse:
        # Upstream imagery problems are not our bug, and they are often transient
        # or fixable by the caller (a wider date window, a higher cloud limit).
        log.warning("imagery error: %s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc), "kind": "imagery"})

    @app.exception_handler(ConfigError)
    async def config_error_handler(_: Request, exc: ConfigError) -> JSONResponse:
        log.error("configuration error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc), "kind": "config"})

    @app.exception_handler(AtarraError)
    async def atarra_error_handler(_: Request, exc: AtarraError) -> JSONResponse:
        log.error("atarra error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc), "kind": "internal"})

    return app


app = create_app()
