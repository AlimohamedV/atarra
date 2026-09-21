"""Imagery discovery, composite summaries, and rendered overlays."""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response

from atarra.core.errors import ImageryError
from atarra.core.logging import get_logger
from atarra.pipeline import DEFAULT_PREVIEW_GSD, available_dates, load_composite
from atarra.preprocess.indices import INDEX_BANDS
from atarra.viz.render import apply_colormap, encode_png, render_true_color

log = get_logger("api.imagery")

router = APIRouter(prefix="/areas", tags=["imagery"])


def _parse_date(value: str | None, default: Date) -> Date:
    if not value:
        return default
    try:
        return Date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid date {value!r}; use YYYY-MM-DD") from exc


def _validate_index(index: str) -> str:
    key = index.lower()
    if key not in INDEX_BANDS:
        raise HTTPException(
            status_code=404, detail=f"unknown index {index!r}; known: {sorted(INDEX_BANDS)}"
        )
    return key


def _validate_gsd(gsd: float) -> float:
    if not (5.0 <= gsd <= 1000.0):
        raise HTTPException(status_code=400, detail="gsd must be between 5 and 1000 metres")
    return gsd


@router.get("/{area_key}/scenes")
def scenes(
    area_key: str,
    start: str | None = Query(None, description="ISO date, default 24 months ago"),
    end: str | None = Query(None, description="ISO date, default today"),
    max_cloud_cover: float | None = Query(None, ge=0, le=100),
    limit: int = Query(500, ge=1, le=2000),
) -> dict:
    """Which dates have usable imagery.

    This is what populates the dashboard's time slider -- the client is told
    exactly which dates carry data rather than making a user guess and wait for
    an empty render.
    """
    today = Date.today()
    window_start = _parse_date(start, today - timedelta(days=730))
    window_end = _parse_date(end, today)
    if window_start >= window_end:
        raise HTTPException(status_code=400, detail="start must be before end")

    try:
        items = available_dates(
            area_key, window_start, window_end, max_cloud_cover=max_cloud_cover, limit=limit
        )
    except ImageryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "area": area_key,
        "start": window_start.isoformat(),
        "end": window_end.isoformat(),
        "count": len(items),
        "dates": items,
    }


@router.get("/{area_key}/summary")
def summary(
    area_key: str,
    date: str | None = Query(None, description="ISO date of interest"),
    window_days: int = Query(3, ge=0, le=15, description="days either side to fill tile gaps"),
    gsd: float = Query(DEFAULT_PREVIEW_GSD, ge=5, le=1000),
    max_size: int = Query(1024, ge=64, le=8192),
    indices: str = Query("ndvi,ndwi,ndre,ndmi"),
) -> dict:
    """Statistics for a composite, plus the frame needed to overlay its render."""
    target = _parse_date(date, Date(2023, 8, 26))
    wanted = [i.strip().lower() for i in indices.split(",") if i.strip()]
    for name in wanted:
        _validate_index(name)
    _validate_gsd(gsd)

    composite = load_composite(
        area_key,
        target,
        window_days=window_days,
        gsd=gsd,
        max_size=max_size,
        indices=wanted,
    )
    payload = composite.summary()
    payload["image_corners"] = composite.grid.corners_wgs84()
    payload["available_indices"] = sorted(composite.indices)
    return payload


@router.get("/{area_key}/render.png")
def render(
    area_key: str,
    index: str = Query("ndvi", description="spectral index to render"),
    date: str | None = Query(None, description="ISO date of interest"),
    window_days: int = Query(3, ge=0, le=15),
    gsd: float = Query(DEFAULT_PREVIEW_GSD, ge=5, le=1000),
    max_size: int = Query(1024, ge=64, le=8192),
    vmin: float | None = Query(None),
    vmax: float | None = Query(None),
) -> Response:
    """Render one index as a transparent PNG overlay.

    Only one index per request on purpose. Each index needs its own bands fetched
    from the archive, and rendering four at once measured ~250 s cold against
    ~110 s for one -- so per-index requests plus the composite cache are both
    faster in practice and cheaper to cache.
    """
    key = _validate_index(index)
    target = _parse_date(date, Date(2023, 8, 26))
    _validate_gsd(gsd)

    composite = load_composite(
        area_key,
        target,
        window_days=window_days,
        gsd=gsd,
        max_size=max_size,
        indices=[key],
    )

    layer = composite.index(key)
    rgba = apply_colormap(layer, key, vmin=vmin, vmax=vmax)
    png = encode_png(rgba)

    log.info(
        "rendered %s %s %s: %dx%d, %.1f%% covered, %.0f kB",
        area_key,
        target,
        key,
        composite.grid.width,
        composite.grid.height,
        composite.coverage * 100,
        len(png) / 1024,
    )

    return Response(
        content=png,
        media_type="image/png",
        headers={
            # Derived artefacts are deterministic for a given set of parameters.
            "Cache-Control": "public, max-age=3600",
            "X-Atarra-Coverage": f"{composite.coverage:.4f}",
            "X-Atarra-Scenes": str(len(composite.scene_ids)),
            "X-Atarra-Grid": f"{composite.grid.width}x{composite.grid.height}",
        },
    )


@router.get("/{area_key}/truecolor.png")
def true_color(
    area_key: str,
    date: str | None = Query(None),
    window_days: int = Query(3, ge=0, le=15),
    gsd: float = Query(DEFAULT_PREVIEW_GSD, ge=5, le=1000),
    max_size: int = Query(1024, ge=64, le=8192),
) -> Response:
    """Natural-colour render, as a visual sanity check against the index layers."""
    target = _parse_date(date, Date(2023, 8, 26))
    _validate_gsd(gsd)

    composite = load_composite(
        area_key,
        target,
        window_days=window_days,
        gsd=gsd,
        max_size=max_size,
        bands=["B02", "B03", "B04", "B08"],
        indices=["ndvi"],
    )
    rgba = render_true_color(composite.stack)
    return Response(
        content=encode_png(rgba),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{area_key}/stats")
def index_stats(
    area_key: str,
    date: str | None = Query(None),
    window_days: int = Query(3, ge=0, le=15),
    gsd: float = Query(DEFAULT_PREVIEW_GSD, ge=5, le=1000),
) -> dict:
    """Histogram data for an index, for the dashboard's distribution panel."""
    key = "ndvi"
    target = _parse_date(date, Date(2023, 8, 26))
    composite = load_composite("".join(area_key), target, window_days=window_days, gsd=gsd, indices=[key])
    values = composite.index(key)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"area": area_key, "index": key, "count": 0, "histogram": []}

    counts, edges = np.histogram(finite, bins=40, range=(-1.0, 1.0))
    return {
        "area": area_key,
        "index": key,
        "date": target.isoformat(),
        "count": int(finite.size),
        "histogram": [
            {"bin_start": round(float(edges[i]), 3), "count": int(counts[i])}
            for i in range(len(counts))
        ],
    }
