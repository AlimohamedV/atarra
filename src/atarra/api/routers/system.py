"""Health, configuration, and cache introspection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from atarra import __version__
from atarra.core.config import get_bands, get_study_areas
from atarra.core.logging import get_logger
from atarra.core.settings import get_settings
from atarra.pipeline import get_cache
from atarra.preprocess.indices import INDEX_BANDS
from atarra.viz.render import legend

log = get_logger("api.system")

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    """Liveness plus the configuration the caller is actually running against.

    Reporting the reflectance convention here is intentional: it is the single
    setting most likely to silently invalidate every index if it drifts, so it is
    visible without reading a config file.
    """
    bands = get_bands()
    settings = get_settings()
    cache = get_cache()
    return {
        "status": "ok",
        "version": __version__,
        "imagery_source": settings.stac_url,
        "collection": settings.stac_collection,
        "data_dir": str(settings.data_dir),
        "reflectance": {
            "mode": bands.reflectance_mode,
            "scale": bands.reflectance_scale,
            "offset": bands.reflectance_offset,
            "max_negative_fraction": bands.max_negative_fraction,
        },
        "bands_8": bands.bands_8,
        "bands_rgb": bands.bands_rgb,
        "indices": sorted(INDEX_BANDS),
        "cache": cache.stats(),
    }


@router.get("/cache")
def cache_stats() -> dict:
    """Current cache occupancy against its configured ceiling."""
    return get_cache().stats()


@router.post("/cache/clear")
def cache_clear() -> dict:
    """Drop every cached composite. Useful when validating a pipeline change."""
    cache = get_cache()
    before = cache.size_bytes()
    cache.clear()
    log.info("cache cleared, freed %.1f MB", before / 1e6)
    return {"freed_bytes": before, "cache": cache.stats()}


@router.get("/areas")
def list_areas() -> dict:
    """Configured study zones, with their WGS84 footprints for mapping."""
    areas = get_study_areas()
    return {
        "count": len(areas),
        "areas": [
            {
                "key": area.key,
                "name": area.name,
                "role": area.role,
                "crs": area.crs,
                "bbox_wgs84": area.bbox.as_stac_query(),
                "geometry": area.bbox.as_geojson(),
            }
            for area in areas.values()
        ],
    }


@router.get("/legend/{index}")
def index_legend(index: str) -> dict:
    """Colour scale for an index, so the client's legend matches the server's render."""
    key = index.lower()
    if key not in INDEX_BANDS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown index {index!r}; known: {sorted(INDEX_BANDS)}",
        )
    return legend(key)
