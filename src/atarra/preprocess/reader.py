"""Windowed raster reads and mosaicking.

Two facts drive this module.

**Only a window is ever read.** Bands are Cloud-Optimized GeoTIFFs on public S3
with byte-range support, so reading a 512x512 window out of a 68 MB band file
transfers roughly half a megabyte. Nothing here downloads a whole scene.

**A single scene does not cover a study area.** Measured over Lake Burullus on one
acquisition date, tile 36RTV covered 51 % of the AOI, 36RUV 34 %, 36STA 14 % and
36SUA 23 % -- the overlap is real and no one tile suffices. So a date is read as a
*mosaic* of every scene available that day, painted clearest-first so that when
tiles overlap, the least-cloudy observation wins the contested pixels. Without
this, roughly half of every Burullus composite would silently be fill value.

Sentinel-2 tiles are rotated diamonds, so a scene's bounding box is a
conservative over-estimate of its data. Overlap and fill are therefore resolved by
*masking* (DN == 0, plus the SCL cloud/shadow classes) rather than by trusting
geometry alone.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from atarra.core.config import BandsConfig, get_bands
from atarra.core.errors import ImageryError
from atarra.core.grids import BBox, Grid
from atarra.core.logging import get_logger
from atarra.ingest.base import Scene

log = get_logger("preprocess.reader")


def _max_read_workers() -> int:
    """How many remote band reads to run at once.

    These reads are network-bound (a COG window costs a few range requests), so
    threads are the right tool -- the GIL is released inside GDAL during I/O. The
    cap keeps a full-AOI mosaic from opening dozens of sockets at once, which
    tends to trip connection limits before it saves time.
    """
    configured = os.environ.get("ATARRA_READ_WORKERS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    return max(2, min(12, (os.cpu_count() or 4) * 2))


@lru_cache(maxsize=1)
def _read_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=_max_read_workers(), thread_name_prefix="atarra-read")


@dataclass
class BandStack:
    """A reflectance stack on one canonical grid, with a validity mask."""

    data: np.ndarray  # (n_bands, H, W) float32, NaN where invalid
    valid: np.ndarray  # (H, W) bool
    grid: Grid
    band_names: list[str]
    window: Window
    scene_ids: list[str] = field(default_factory=list)
    scl: np.ndarray | None = None  # (H, W) uint8 scene classification layer

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def n_bands(self) -> int:
        return int(self.data.shape[0])

    @property
    def coverage(self) -> float:
        """Fraction of the requested window that received real observations."""
        return float(self.valid.mean()) if self.valid.size else 0.0

    def band(self, name: str) -> np.ndarray:
        """Return one band by name."""
        try:
            index = self.band_names.index(name)
        except ValueError as exc:
            raise ImageryError(
                f"band {name!r} not in stack; available: {self.band_names}"
            ) from exc
        return self.data[index]

    def percentile_stretch(self, band: str, low: float = 2.0, high: float = 98.0) -> np.ndarray:
        """Robustly rescale a band to 0..1 for display.

        Reflectance has a long tail (bright bare soil, specular water), so a
        min/max stretch renders almost everything mid-grey. Percentile clipping
        is what makes a composite actually readable.
        """
        values = self.band(band)
        finite = np.isfinite(values)
        if not finite.any():
            return np.zeros_like(values, dtype=np.float32)
        lo, hi = np.percentile(values[finite], [low, high])
        if hi <= lo:
            return np.zeros_like(values, dtype=np.float32)
        stretched = (values - lo) / (hi - lo)
        return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def reflectance_params(cfg: BandsConfig, scene: Scene) -> tuple[float, float]:
    """Resolve the DN -> reflectance conversion for a scene.

    The archive's item metadata and this project genuinely disagree about the
    offset (see configs/bands.yaml for the measurements). The config wins, and any
    disagreement is logged once per scene so the discrepancy stays visible rather
    than being quietly resolved in the archive's favour.
    """
    if (
        cfg.reflectance_mode == "dn_scale"
        and abs(scene.offset) > 1e-9
    ):
        log.debug(
            "scene %s advertises offset %.4g; using the validated %s convention "
            "(offset %.4g) instead",
            scene.id,
            scene.offset,
            cfg.reflectance_mode,
            cfg.reflectance_offset,
        )
    return cfg.reflectance_scale, cfg.reflectance_offset


def validate_reflectance(
    stack: "BandStack", cfg: BandsConfig | None = None, *, label: str = ""
) -> dict:
    """Sanity-check that a stack converted to physically plausible reflectance.

    Reflectance is bounded below by zero (barring the small negative values an
    applied BOA offset legitimately produces). A large negative fraction means the
    conversion convention no longer matches the archive, which would bias every
    index in the pipeline -- so it is worth an explicit warning rather than a
    silent wrong answer.
    """
    cfg = cfg or get_bands()
    values = stack.data[np.isfinite(stack.data)]
    if values.size == 0:
        return {"n": 0, "negative_fraction": 0.0, "min": None, "median": None, "max": None}

    negative_fraction = float((values < 0).mean())
    stats = {
        "n": int(values.size),
        "negative_fraction": negative_fraction,
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "coverage": stack.coverage,
    }

    if negative_fraction > cfg.max_negative_fraction:
        log.warning(
            "%sreflectance sanity check FAILED: %.1f%% of valid pixels are "
            "negative (limit %.1f%%), min=%.3f. The archive's reflectance "
            "convention may have changed -- re-run the check described in "
            "configs/bands.yaml before trusting any index.",
            f"{label} " if label else "",
            100 * negative_fraction,
            100 * cfg.max_negative_fraction,
            stats["min"],
        )
    if stats["max"] is not None and stats["max"] > 1.5:
        log.warning(
            "%sreflectance peak of %.3f exceeds 1.5, which is not physical",
            f"{label} " if label else "",
            stats["max"],
        )
    return stats


def _resampling(name: str) -> Resampling:
    try:
        return Resampling[name.lower()]
    except KeyError as exc:
        raise ImageryError(
            f"unsupported resampling {name!r}; expected one of "
            f"{[r.name for r in Resampling]}"
        ) from exc


def _read_resampled(
    href: str,
    grid: Grid,
    window: Window,
    *,
    resampling: Resampling,
    scale: float = 1.0,
    offset: float = 0.0,
    as_mask: bool = False,
) -> np.ndarray:
    """Read ``window`` of a remote band, resampled onto the target grid.

    ``out_shape`` does the resampling inside GDAL, so a 20 m band is never
    materialised at 20 m and then resampled in numpy -- it goes straight from
    remote bytes to target pixels.

    A warped VRT looks like the tidier way to express this (it can name the target
    grid up front, which should let GDAL pick a suitable overview level). It was
    measured here and came out ~12x *slower* than this approach, with output
    differing by up to 3153 DN, so it was reverted. Do not "simplify" this into a
    WarpedVRT without re-timing it.
    """
    height = int(round(window.height))
    width = int(round(window.width))
    target_transform = grid.transform @ Affine.translation(window.col_off, window.row_off)

    try:
        with rasterio.open(href) as src:
            if src.crs is None:
                raise ImageryError(f"{href} declares no CRS")
            left, bottom, right, top = array_bounds(height, width, target_transform)
            # Request exactly the source footprint that lands in this window.
            src_bounds = transform_bounds(grid.crs, src.crs, left, bottom, right, top, densify_pts=21)
            src_window = from_bounds(*src_bounds, transform=src.transform)
            data = src.read(
                1,
                window=src_window,
                out_shape=(height, width),
                resampling=resampling,
                boundless=True,
                fill_value=0,
            )
    except ImageryError:
        raise
    except Exception as exc:
        raise ImageryError(f"failed to read {href}: {exc}") from exc

    if as_mask:
        return data.astype(np.uint8, copy=False)
    return data.astype(np.float32) * np.float32(scale) + np.float32(offset)


def read_window(
    scene: Scene,
    grid: Grid,
    band_names: Sequence[str],
    window: Window,
    *,
    bands_cfg: BandsConfig | None = None,
    include_mask: bool = True,
) -> BandStack:
    """Read one scene into a single window of the target grid."""
    cfg = bands_cfg or get_bands()
    height, width = int(round(window.height)), int(round(window.width))
    scale, offset = reflectance_params(cfg, scene)

    # Each band is a separate remote file, so the reads are independent and are
    # dispatched together. Sequentially this dominates end-to-end runtime: a
    # full-AOI mosaic is over a hundred band reads.
    tasks: list[tuple[str, str, Resampling, bool, float, float]] = []
    for name in band_names:
        spec = cfg.spec(name)
        tasks.append((name, scene.asset(name), _resampling(spec.resample), False, scale, offset))

    wants_mask = include_mask and cfg.mask.name in scene.assets
    if wants_mask:
        tasks.append(
            (cfg.mask.name, scene.asset(cfg.mask.name), _resampling(cfg.mask.resample), True, 1.0, 0.0)
        )

    futures = [
        _read_pool().submit(
            _read_resampled,
            href,
            grid,
            window,
            resampling=resampling,
            scale=task_scale,
            offset=task_offset,
            as_mask=is_mask,
        )
        for _name, href, resampling, is_mask, task_scale, task_offset in tasks
    ]
    # Collect in submission order so channel order always matches `band_names`.
    read_results = [future.result() for future in futures]

    data = np.full((len(band_names), height, width), np.nan, dtype=np.float32)
    for i in range(len(band_names)):
        data[i] = read_results[i]

    scl = read_results[len(band_names)] if wants_mask else None
    invalid = np.zeros((height, width), dtype=bool)
    if scl is not None:
        for code in cfg.scl_invalid_classes:
            invalid |= scl == code

    # DN 0 is the archive's fill value, and stays distinguishable from real dark
    # water because real water reflectance never lands exactly on the fill code.
    fill = ~np.any(data != 0.0, axis=0)
    valid = ~(invalid | fill)
    data[:, ~valid] = np.nan

    return BandStack(
        data=data,
        valid=valid,
        grid=grid,
        band_names=list(band_names),
        window=window,
        scene_ids=[scene.id],
        scl=scl,
    )


def read_mosaic(
    scenes: Iterable[Scene],
    grid: Grid,
    band_names: Sequence[str],
    *,
    bands_cfg: BandsConfig | None = None,
    window: Window | None = None,
    include_mask: bool = True,
    target_coverage: float = 0.999,
) -> BandStack:
    """Mosaic every scene covering ``window`` of ``grid`` into one stack.

    Scenes are applied in ascending cloud cover, so clearer observations take
    precedence and later (cloudier) tiles only fill what is still missing.

    Painting stops early once ``target_coverage`` is reached. Because scenes are
    ordered clearest-first, any scene after that point could only overwrite pixels
    already supplied by a clearer observation -- so continuing would cost network
    reads for no change in the result. On a Burullus composite this typically ends
    the work in about half the scenes.
    """
    cfg = bands_cfg or get_bands()
    target = window or Window(0, 0, grid.width, grid.height)
    height, width = int(round(target.height)), int(round(target.width))

    data = np.full((len(band_names), height, width), np.nan, dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    scl_out: np.ndarray | None = None

    ordered = sorted(
        scenes,
        key=lambda s: s.cloud_cover if s.cloud_cover is not None else float("inf"),
    )
    if not ordered:
        raise ImageryError("read_mosaic received no scenes")

    window_bounds = _window_bounds(grid, target)
    used: list[str] = []
    contribution: dict[str, float] = {}

    for scene in ordered:
        if not scene.remote_readable:
            raise ImageryError(
                f"scene {scene.id} ({scene.source}) is not remotely readable. "
                "Download and resolve it first, or use the STAC source."
            )
        missing = [b for b in band_names if b not in scene.assets]
        if missing:
            log.debug("scene %s skipped: missing bands %s", scene.id, missing)
            continue

        scene_box = scene.bbox.to_crs(grid.crs)
        inter = _intersect(window_bounds, scene_box)
        if inter is None:
            continue

        sub = grid.window_for(inter)
        sub = _clip_to(sub, target, grid)
        if sub is None or sub.width <= 0 or sub.height <= 0:
            continue

        try:
            stack = read_window(
                scene, grid, band_names, sub, bands_cfg=cfg, include_mask=include_mask
            )
        except ImageryError as exc:
            # One unreadable tile must not lose the whole date; the mosaic just
            # ends up with a hole that `coverage` reports.
            log.warning("scene %s failed to read: %s", scene.id, exc)
            continue

        row0 = int(round(sub.row_off - target.row_off))
        col0 = int(round(sub.col_off - target.col_off))
        row1, col1 = row0 + int(round(sub.height)), col0 + int(round(sub.width))

        # Only fill pixels no clearer scene has claimed yet.
        hole = ~valid[row0:row1, col0:col1]
        fillable = hole & stack.valid
        if not fillable.any():
            continue

        for i in range(len(band_names)):
            data[i, row0:row1, col0:col1][fillable] = stack.data[i][fillable]
        valid[row0:row1, col0:col1] |= fillable

        if stack.scl is not None:
            if scl_out is None:
                scl_out = np.zeros((height, width), dtype=np.uint8)
            scl_out[row0:row1, col0:col1][fillable] = stack.scl[fillable]

        n_new = int(fillable.sum())
        used.append(scene.id)
        contribution[scene.id] = n_new / float(max(1, height * width))

        if valid.mean() >= target_coverage:
            remaining = len(ordered) - len(used)
            if remaining > 0:
                log.debug(
                    "coverage target %.3f reached after %d scene(s); skipping the "
                    "remaining %d",
                    target_coverage,
                    len(used),
                    remaining,
                )
            break

    if not used:
        raise ImageryError(
            f"no scene covered the requested window; {len(ordered)} candidate "
            "scene(s) were examined"
        )

    data[:, ~valid] = np.nan
    log.info(
        "mosaic from %d scene(s), coverage %.1f%%",
        len(used),
        100.0 * float(valid.mean()),
    )
    for scene_id, fraction in sorted(contribution.items(), key=lambda kv: -kv[1]):
        log.debug("   %s contributed %.1f%% of the window", scene_id, 100 * fraction)

    return BandStack(
        data=data,
        valid=valid,
        grid=grid,
        band_names=list(band_names),
        window=target,
        scene_ids=used,
        scl=scl_out,
    )


# --- geometry helpers --------------------------------------------------------
def _window_bounds(grid: Grid, window: Window) -> BBox:
    """CRS bounds of a window."""
    left, bottom, right, top = array_bounds(
        int(round(window.height)),
        int(round(window.width)),
        grid.transform @ Affine.translation(window.col_off, window.row_off),
    )
    return BBox(west=left, south=bottom, east=right, north=top, crs=grid.crs.to_string())


def _intersect(a: BBox, b: BBox) -> BBox | None:
    """Bounding-box intersection, or ``None`` when disjoint."""
    west = max(a.west, b.west)
    south = max(a.south, b.south)
    east = min(a.east, b.east)
    north = min(a.north, b.north)
    if west >= east or south >= north:
        return None
    return BBox(west=west, south=south, east=east, north=north, crs=a.crs)


def _clip_to(sub: Window, target: Window, grid: Grid) -> Window | None:
    """Clamp a sub-window so it lies inside ``target`` and the grid."""
    col_off = max(int(round(sub.col_off)), int(round(target.col_off)), 0)
    row_off = max(int(round(sub.row_off)), int(round(target.row_off)), 0)
    col_end = min(
        int(round(sub.col_off + sub.width)), int(round(target.col_off + target.width)), grid.width
    )
    row_end = min(
        int(round(sub.row_off + sub.height)), int(round(target.row_off + target.height)), grid.height
    )
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def mosaic_key(scene_date: date, grid: Grid, bands: Sequence[str], *, suffix: str = "") -> str:
    """A stable identity for a cached mosaic.

    Includes the grid geometry, because the same date at a different resolution or
    alignment is a genuinely different array and must not collide in the cache.
    """
    res_x, res_y = grid.resolution
    return (
        f"mosaic/{scene_date:%Y-%m-%d}/{grid.crs.to_string()}/{res_x:g}x{res_y:g}"
        f"/{grid.width}x{grid.height}/{'+'.join(bands)}{suffix}"
    )
