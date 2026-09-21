"""Satellite imagery discovery.

Two interchangeable backends sit behind one interface:

``stac``  Earth Search / AWS public COGs. No credentials, and Cloud-Optimized
          GeoTIFF layout means bands can be read with HTTP byte ranges rather
          than downloaded whole. This is the default.
``cdse``  ESA Copernicus Data Space Ecosystem via OAuth. Full ESA provenance, but
          requires an account and downloads whole products.

Both return the same :class:`~atarra.ingest.base.Scene` objects, so swapping the
provenance story does not touch a line of downstream code.
"""

from __future__ import annotations

from atarra.core.errors import ImageryError
from atarra.ingest.base import ImagerySource, Scene, select_best_per_period

__all__ = ["ImagerySource", "Scene", "select_best_per_period", "get_source"]


def get_source(name: str = "stac", **kwargs) -> ImagerySource:
    """Construct an imagery source by name."""
    key = name.strip().lower()
    if key == "stac":
        from atarra.ingest.stac import StacImagerySource

        return StacImagerySource(**kwargs)
    if key == "cdse":
        from atarra.ingest.cdse import CdseImagerySource

        return CdseImagerySource(**kwargs)
    raise ImageryError(f"unknown imagery source {name!r}; expected 'stac' or 'cdse'")
