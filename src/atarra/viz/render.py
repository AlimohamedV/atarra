"""Server-side rendering of index composites and masks.

The dashboard needs to *see* the data, and shipping raw float arrays to the
browser is not an option (an 8-band 1024x1024 stack is 32 MB before encoding).
So the API renders PNGs here: RGBA, NaN-aware, with hand-rolled colour tables.

Colour tables rather than matplotlib on purpose. Matplotlib is installed, but it is
a heavy import for what is a 256-entry lookup, and it brings its own
figure-canvas state into a request handler. These tables are small, deterministic,
and trivially unit-testable.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from atarra.core.errors import ImageryError
from atarra.core.logging import get_logger

log = get_logger("viz.render")

# Anchor points for each index. Values are the physically meaningful ends of the
# scale, not the data's min/max: fixing them keeps the same colour meaning the
# same thing across dates, which is the entire point of a time slider.
INDEX_RANGES: dict[str, tuple[float, float]] = {
    "ndvi": (-0.2, 0.9),
    "ndwi": (-0.6, 0.8),
    "ndre": (-0.4, 0.7),
    "ndmi": (-0.5, 0.7),
}

_ANCHORS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    # brown -> yellow -> green: conventional vegetation ramp
    "ndvi": [
        (0.0, (165, 0, 38)),
        (0.25, (215, 48, 39)),
        (0.45, (254, 224, 139)),
        (0.65, (166, 217, 106)),
        (0.85, (0, 104, 55)),
        (1.0, (0, 60, 30)),
    ],
    # white -> deep blue: water
    "ndwi": [
        (0.0, (139, 69, 19)),
        (0.25, (240, 240, 220)),
        (0.5, (146, 197, 222)),
        (0.75, (40, 110, 190)),
        (1.0, (5, 35, 90)),
    ],
    # red-edge: magenta-ish so it is never confused with NDVI
    "ndre": [
        (0.0, (60, 20, 60)),
        (0.3, (150, 60, 130)),
        (0.55, (230, 140, 90)),
        (0.8, (250, 220, 120)),
        (1.0, (255, 255, 220)),
    ],
    # moisture: brown -> teal
    "ndmi": [
        (0.0, (140, 90, 50)),
        (0.35, (225, 210, 170)),
        (0.6, (120, 190, 180)),
        (0.85, (30, 120, 140)),
        (1.0, (5, 50, 80)),
    ],
}

# Fallback ramp for an index with no bespoke table.
_DEFAULT_ANCHORS = [
    (0.0, (0, 0, 60)),
    (0.5, (120, 120, 120)),
    (1.0, (255, 255, 255)),
]


@dataclass(frozen=True)
class RenderOptions:
    """Display controls for a render."""

    index: str
    vmin: float | None = None
    vmax: float | None = None
    stretch: bool = False
    alpha: int = 255
    transparent_invalid: bool = True


def _build_lut(anchors: list[tuple[float, tuple[int, int, int]]], size: int = 512) -> np.ndarray:
    """Interpolate anchor colours into a lookup table of shape (size, 3)."""
    stops = np.array([a for a, _ in anchors], dtype=np.float64)
    colors = np.array([c for _, c in anchors], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, size)
    lut = np.empty((size, 3), dtype=np.float64)
    for channel in range(3):
        lut[:, channel] = np.interp(xs, stops, colors[:, channel])
    return np.clip(lut, 0, 255).astype(np.uint8)


def apply_colormap(
    data: np.ndarray,
    index: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: int = 255,
    transparent_invalid: bool = True,
) -> np.ndarray:
    """Map a float index layer to an ``(H, W, 4)`` uint8 RGBA image."""
    if data.ndim != 2:
        raise ImageryError(f"expected a 2D index layer, got shape {data.shape}")

    default_lo, default_hi = INDEX_RANGES.get(index, (0.0, 1.0))
    lo = default_lo if vmin is None else float(vmin)
    hi = default_hi if vmax is None else float(vmax)
    if hi <= lo:
        raise ImageryError(f"vmin ({lo}) must be less than vmax ({hi})")

    anchors = _ANCHORS.get(index, _DEFAULT_ANCHORS)
    lut = _build_lut(anchors)

    valid = np.isfinite(data)
    normalized = np.zeros(data.shape, dtype=np.float64)
    np.clip((data - lo) / (hi - lo), 0.0, 1.0, out=normalized, where=valid)

    # Quantise into the LUT. `where=valid` keeps NaN pixels from producing an
    # out-of-range index.
    positions = np.zeros(data.shape, dtype=np.int64)
    np.multiply(normalized, len(lut) - 1, out=normalized, where=valid)
    positions[valid] = normalized[valid].astype(np.int64)

    rgba = np.zeros(data.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = lut[positions]
    rgba[..., 3] = np.where(valid, alpha, 0 if transparent_invalid else alpha).astype(np.uint8)
    return rgba


def render_true_color(stack, *, vmin: float = 0.0, vmax: float = 0.25, gamma: float = 1.0) -> np.ndarray:
    """Render an RGB image from a reflectance stack, if red/green/blue are present."""
    missing = [b for b in ("B04", "B03", "B02") if b not in stack.band_names]
    if missing:
        raise ImageryError(f"true-colour render needs {missing} in the stack")

    channels = []
    for band in ("B04", "B03", "B02"):
        values = stack.band(band)
        scaled = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        if gamma != 1.0:
            scaled = np.power(scaled, 1.0 / gamma)
        channels.append(scaled)

    rgb = np.stack(channels, axis=-1)
    valid = np.isfinite(rgb).all(axis=-1)
    # Reflectance has a narrow useful range; a mild gain avoids the washed-out
    # look that a straight linear stretch produces over water.
    rgb = np.clip(rgb * 1.35, 0.0, 1.0)

    # NaN cannot survive a cast to uint8 -- it yields undefined values. Replace it
    # before scaling so nodata pixels come out black (and fully transparent below)
    # rather than as arbitrary colour noise.
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    rgba = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    rgba[..., :3] = (rgb * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def render_mask(mask: np.ndarray, *, color=(255, 60, 60), alpha: int = 170) -> np.ndarray:
    """Render a boolean class mask as translucent RGBA."""
    mask = np.asarray(mask).astype(bool)
    rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    # Assign the scalar over the boolean selection. Broadcasting a full-frame
    # array here would raise: the left-hand side selects only the masked pixels,
    # so its shape is (n,) rather than the mask's (height, width).
    rgba[mask, 3] = np.uint8(alpha)
    return rgba


def encode_png(rgba: np.ndarray) -> bytes:
    """Encode an RGBA array as PNG bytes."""
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba, 0, 255).astype(np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ImageryError(f"expected an (H, W, 4) RGBA array, got {rgba.shape}")
    buffer = io.BytesIO()
    # The mode is inferred from the array: a uint8 (H, W, 4) array is RGBA. Passing
    # it explicitly is deprecated and is removed in Pillow 13.
    Image.fromarray(rgba).save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def save_png(path, rgba: np.ndarray) -> None:
    """Write an RGBA array to a PNG file."""
    data = encode_png(rgba)
    with open(path, "wb") as handle:
        handle.write(data)


def legend(index: str) -> dict:
    """Colour-scale metadata so the client can draw a legend consistently."""
    lo, hi = INDEX_RANGES.get(index, (0.0, 1.0))
    anchors = _ANCHORS.get(index, _DEFAULT_ANCHORS)
    return {
        "index": index,
        "vmin": lo,
        "vmax": hi,
        "stops": [
            {"value": round(lo + (hi - lo) * stop, 4), "color": "#%02x%02x%02x" % color}
            for stop, color in anchors
        ],
    }
