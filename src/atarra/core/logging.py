"""Console logging.

Kept deliberately simple and dependency-free; the cache and pipeline modules log
their sizes and evictions through this so the "did something quietly eat my
disk?" question always has an answer in the terminal.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | int | None = None) -> None:
    """Install a single stdout handler on the root logger, once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level if level is not None else os.environ.get("ATARRA_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(resolved if isinstance(resolved, int) else str(resolved).upper())

    # GDAL/rasterio emit an INFO line on every remote open (the boto3 fallback
    # notice among them). At INFO that is hundreds of lines per mosaic and buries
    # our own logging. Their warnings and errors still come through.
    for noisy in ("rasterio", "rasterio._env", "rasterio.session", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    configure_logging()
    if not name.startswith("atarra"):
        name = f"atarra.{name}"
    return logging.getLogger(name)
