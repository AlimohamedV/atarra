"""High-level orchestration: AOI + date -> a validated composite of indices.

This is the layer the API, CLI, and notebooks all sit on, so the awkward parts of
the data model live here in one place.

Two of those awkward parts are worth stating up front.

**A single acquisition date does not cover a study area.** Measured over Lake
Burullus, tile 36RTV covers 51 % of the AOI, 36RUV 34 %, 36STA 14 % and 36SUA
23 %. Worse, those tiles are not all acquired on the same day -- they sit on
different relative orbits -- so a single-date mosaic legitimately has holes. A
date window (default +/- 3 days) gathers the neighbouring overpasses and fills
them, with clearer scenes painting first.

**Reflectance conversion is validated, not assumed.** Every composite runs a
sanity check and reports how much of it came out non-physical; see
``configs/bands.yaml`` for why the archive's own metadata is not trusted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from atarra.core.cache import DiskCache
from atarra.core.config import BandsConfig, StudyArea, get_bands, get_study_area
from atarra.core.errors import ImageryError
from atarra.core.grids import Grid, grid_from_bbox
from atarra.core.logging import get_logger
from atarra.core.settings import get_settings
from rasterio.windows import Window
from atarra.ingest import ImagerySource, get_source
from atarra.ingest.base import Scene
from atarra.preprocess.indices import INDEX_BANDS, compute_indices
from atarra.preprocess.reader import BandStack, read_mosaic, validate_reflectance

log = get_logger("pipeline")

# Conservative default: at 30 m a full Burullus AOI is about 1.9k x 1.2k pixels,
# which is a couple of hundred KB of PNG and renders instantly in the browser.
DEFAULT_PREVIEW_GSD = 30.0


@dataclass
class Composite:
    """A validated stack of spectral indices for one AOI and date window."""

    area_key: str
    area_name: str
    target_date: Date
    window_days: int
    grid: Grid
    stack: BandStack
    indices: dict[str, np.ndarray]
    scene_ids: list[str]
    scene_details: list[dict]
    reflectance: dict
    from_cache: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.stack.coverage

    @property
    def bounds_wgs84(self) -> list[float]:
        return self.grid.bounds.as_stac_query()

    def index(self, name: str) -> np.ndarray:
        key = name.lower()
        if key not in self.indices:
            raise ImageryError(
                f"index {name!r} is not available for this composite; "
                f"available: {sorted(self.indices)}"
            )
        return self.indices[key]

    def summary(self) -> dict:
        """Compact description suitable for a JSON response."""
        index_stats = {}
        for name, values in self.indices.items():
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            index_stats[name] = {
                "min": round(float(finite.min()), 4),
                "median": round(float(np.median(finite)), 4),
                "max": round(float(finite.max()), 4),
                "p10": round(float(np.percentile(finite, 10)), 4),
                "p90": round(float(np.percentile(finite, 90)), 4),
            }
        return {
            "area": self.area_key,
            "area_name": self.area_name,
            "date": self.target_date.isoformat(),
            "window_days": self.window_days,
            "grid": {
                "crs": self.grid.crs.to_string(),
                "width": self.grid.width,
                "height": self.grid.height,
                "gsd": self.grid.resolution[0],
                "bounds_wgs84": [round(v, 6) for v in self.bounds_wgs84],
            },
            "coverage": round(self.coverage, 4),
            "scenes": self.scene_details,
            "indices": index_stats,
            "reflectance": self.reflectance,
            "warnings": self.warnings,
            "from_cache": self.from_cache,
        }


@lru_cache(maxsize=1)
def get_cache() -> DiskCache:
    """Process-wide capped cache for composite arrays."""
    settings = get_settings()
    settings.ensure_dirs()
    return DiskCache(settings.cache_dir, int(settings.max_cache_gb * 1e9), name="composite")


def choose_grid(
    area: StudyArea,
    *,
    gsd: float = DEFAULT_PREVIEW_GSD,
    max_size: int = 2048,
) -> Grid:
    """Pick a grid at the coarsest power-of-two multiple of ``gsd`` that fits.

    Coarsening rather than cropping keeps the whole AOI visible; a preview that
    silently clips the area of interest is worse than a slightly blurry one.
    """
    candidate = float(gsd)
    for _ in range(8):
        grid = grid_from_bbox(area.bbox, area.crs, candidate)
        if max(grid.width, grid.height) <= max_size:
            return grid
        candidate *= 2
    raise ImageryError(
        f"could not fit {area.key} into {max_size} px even at {candidate} m"
    )


def scenes_for_window(
    area: StudyArea,
    target_date: Date,
    *,
    window_days: int = 3,
    source: ImagerySource | None = None,
    bands: BandsConfig | None = None,
    max_cloud_cover: float | None = None,
    limit: int = 100,
) -> list[Scene]:
    """Every usable scene within ``+/- window_days`` of the target date."""
    cfg = bands or get_bands()
    src = source or get_source("stac")
    start = datetime.combine(target_date - timedelta(days=window_days), datetime.min.time())
    end = datetime.combine(target_date + timedelta(days=window_days), datetime.max.time())

    scenes = src.search(
        area.bbox,
        start,
        end,
        max_cloud_cover=max_cloud_cover,
        limit=limit,
    )
    required = cfg.bands_8 + [cfg.mask.name]
    usable = [s for s in scenes if s.has_bands(required)]
    dropped = len(scenes) - len(usable)
    if dropped:
        log.debug("dropped %d scene(s) missing required bands", dropped)

    if not usable:
        raise ImageryError(
            f"no usable scene for {area.key} between {start:%Y-%m-%d} and "
            f"{end:%Y-%m-%d}. Try widening --window-days or raising the cloud limit."
        )
    return sorted(usable, key=lambda s: (s.acquired, s.cloud_cover or 0.0))


def resolve_bands(
    bands: list[str] | None,
    index_names: list[str],
    cfg: BandsConfig,
) -> list[str]:
    """Decide which bands a request must pull from the archive.

    Narrowing to what the requested indices need is the whole point -- it roughly
    halves the number of remote reads for a single-index render. But it must
    never apply to an explicit band list: the true-colour render asks for blue
    and green, which no index consumes, so narrowing there would strip exactly
    the bands it needs and, worse, land its cache key on the plain NDVI key --
    returning a stack that cannot be rendered.
    """
    if bands is not None:
        return list(bands)

    needed: set[str] = set()
    for name in index_names:
        pair = INDEX_BANDS.get(name)
        if pair:
            needed.update(pair)
    narrowed = [b for b in cfg.bands_8 if b in needed]
    # An index name nothing recognises would otherwise yield an empty stack.
    return narrowed or list(cfg.bands_8)


def load_composite(
    area_key: str,
    target_date: Date,
    *,
    window_days: int = 3,
    gsd: float = DEFAULT_PREVIEW_GSD,
    max_size: int = 2048,
    bands: list[str] | None = None,
    indices: list[str] | None = None,
    source: ImagerySource | None = None,
    use_cache: bool = True,
    max_cloud_cover: float | None = None,
) -> Composite:
    """Build (or load) a validated index composite for an AOI and date window."""
    area = get_study_area(area_key)
    cfg = get_bands()
    band_names = list(bands or cfg.bands_8)
    index_names = [i.lower() for i in (indices or list(INDEX_BANDS))]

    band_names = resolve_bands(bands, index_names, cfg)

    scenes = scenes_for_window(
        area,
        target_date,
        window_days=window_days,
        source=source,
        bands=cfg,
        max_cloud_cover=max_cloud_cover,
    )
    grid = choose_grid(area, gsd=gsd, max_size=max_size)

    scene_signature = "|".join(sorted(s.id for s in scenes))
    cache = get_cache() if use_cache else None
    key_prefix = (
        f"composite/{area_key}/{target_date.isoformat()}/w{window_days}"
        f"/{grid.crs.to_string()}/{grid.resolution[0]:g}/{grid.width}x{grid.height}"
        f"/{'+'.join(band_names)}/{cfg.reflectance_mode}"
    )

    data_key = f"{key_prefix}/data"
    valid_key = f"{key_prefix}/valid"
    meta_key = f"{key_prefix}/meta"

    from_cache = False
    stack: BandStack | None = None

    if cache is not None:
        cached_data = cache.load(data_key)
        cached_valid = cache.load(valid_key)
        cached_meta = cache.load(meta_key)
        # The scene signature guards against a stale composite surviving an
        # archive that has since reprocessed or added scenes. It is stored as raw
        # bytes because the cache refuses pickled arrays, and comparing bytes
        # avoids ever having to unpickle untrusted content from disk.
        if (
            cached_data is not None
            and cached_valid is not None
            and cached_meta is not None
            and cached_meta.tobytes().decode("utf-8", "replace") == scene_signature
        ):
            stack = BandStack(
                data=cached_data,
                valid=cached_valid.astype(bool),
                grid=grid,
                band_names=band_names,
                window=Window(0, 0, grid.width, grid.height),
                scene_ids=[s.id for s in scenes],
            )
            from_cache = True
            log.info("composite cache HIT for %s %s", area_key, target_date)

    if stack is None:
        log.info(
            "building composite for %s %s from %d scene(s) at %g m",
            area_key,
            target_date,
            len(scenes),
            grid.resolution[0],
        )
        stack = read_mosaic(scenes, grid, band_names, bands_cfg=cfg)
        if cache is not None:
            cache.store(data_key, stack.data)
            cache.store(valid_key, stack.valid.astype(np.uint8))
            cache.store(meta_key, np.frombuffer(scene_signature.encode("utf-8"), dtype=np.uint8))

    warnings: list[str] = []
    reflectance = validate_reflectance(stack, cfg, label=f"{area_key} {target_date}")
    if reflectance.get("negative_fraction", 0) > cfg.max_negative_fraction:
        warnings.append(
            f"{reflectance['negative_fraction'] * 100:.1f}% of valid pixels converted to "
            "negative reflectance, which suggests the archive's convention has changed"
        )

    index_arrays = compute_indices(stack, index_names)
    missing = [i for i in index_names if i not in index_arrays]
    if missing:
        warnings.append(
            f"index(es) {missing} unavailable: their bands are not in this stack"
        )

    if stack.coverage < 0.5:
        warnings.append(
            f"only {stack.coverage * 100:.0f}% of the area was covered; "
            "widen --window-days or choose a different date"
        )

    scene_details = [
        {
            "id": s.id,
            "acquired": s.acquired.date().isoformat(),
            "platform": s.platform,
            "cloud_cover": s.cloud_cover,
            "tile": s.grid_code,
        }
        for s in scenes
    ]

    return Composite(
        area_key=area_key,
        area_name=area.name,
        target_date=target_date,
        window_days=window_days,
        grid=grid,
        stack=stack,
        indices=index_arrays,
        scene_ids=[s.id for s in scenes],
        scene_details=scene_details,
        reflectance=reflectance,
        from_cache=from_cache,
        warnings=warnings,
    )


def available_dates(
    area_key: str,
    start: Date,
    end: Date,
    *,
    source: ImagerySource | None = None,
    max_cloud_cover: float | None = None,
    limit: int = 500,
) -> list[dict]:
    """Which dates in a range have usable imagery, and how clear they were.

    Drives the dashboard's time slider: rather than making a user guess at dates,
    the client is told exactly which ones have data.
    """
    area = get_study_area(area_key)
    cfg = get_bands()
    src = source or get_source("stac")

    scenes = src.search(
        area.bbox,
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.max.time()),
        max_cloud_cover=max_cloud_cover,
        limit=limit,
    )
    required = cfg.bands_8 + [cfg.mask.name]

    by_date: dict[str, list[Scene]] = {}
    for scene in scenes:
        if not scene.has_bands(required):
            continue
        by_date.setdefault(scene.acquired.date().isoformat(), []).append(scene)

    return [
        {
            "date": day,
            "scenes": len(items),
            "tiles": sorted({s.grid_code for s in items if s.grid_code}),
            "best_cloud_cover": min(
                (s.cloud_cover for s in items if s.cloud_cover is not None), default=None
            ),
        }
        for day, items in sorted(by_date.items())
    ]
