"""Turning grids into model tiles.

Training wants fixed-shape square tiles so batches stack without collation; a
ragged 256x91 edge tile either gets dropped or forces a batch of one. Inference
wants the opposite -- every pixel of the AOI must be predicted, so edges are
padded, predicted, then cropped back.

Both directions live here, and they are deliberately paired: a padding bug that
is not the exact inverse of the crop shifts every prediction along the edge by a
few pixels, which is invisible in an aggregate metric and obvious in a map.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from atarra.core.grids import Grid, TileWindow
from atarra.core.logging import get_logger

log = get_logger("preprocess.tiler")


def iter_tiles(grid: Grid, size: int, *, full_only: bool = True) -> Iterator[TileWindow]:
    """Yield tile windows for a grid (row-major, non-overlapping)."""
    yield from grid.tiles(size, full_only=full_only)


def extract_tile(array: np.ndarray, window: TileWindow) -> np.ndarray:
    """Cut a tile out of a ``(C, H, W)`` or ``(H, W)`` array."""
    if array.ndim == 2:
        return array[int(window.window.row_off) : int(window.window.row_off + window.window.height),
                     int(window.window.col_off) : int(window.window.col_off + window.window.width)]
    if array.ndim == 3:
        return array[
            :,
            int(window.window.row_off) : int(window.window.row_off + window.window.height),
            int(window.window.col_off) : int(window.window.col_off + window.window.width),
        ]
    raise ValueError(f"expected a 2D or 3D array, got shape {array.shape}")


def pad_to_tile(
    array: np.ndarray, size: int, *, fill: float = 0.0
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pad a tile up to ``size x size``.

    Returns the padded array and the ``(top, bottom, left, right)`` padding, which
    :func:`crop_from_padding` consumes to undo it exactly.
    """
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D array, got shape {array.shape}")

    height, width = array.shape[-2], array.shape[-1]
    if height > size or width > size:
        raise ValueError(
            f"tile {height}x{width} is larger than the target size {size}"
        )

    pad_h = size - height
    pad_w = size - width
    # Put the remainder on the bottom/right so the top-left origin stays put.
    top, left = 0, 0
    bottom, right = pad_h, pad_w

    if array.ndim == 2:
        padding = ((top, bottom), (left, right))
    else:
        padding = ((0, 0), (top, bottom), (left, right))

    padded = np.pad(array, padding, mode="constant", constant_values=fill)
    return padded, (top, bottom, left, right)


def crop_from_padding(array: np.ndarray, padding: tuple[int, int, int, int]) -> np.ndarray:
    """Undo :func:`pad_to_tile`."""
    top, bottom, left, right = padding
    height = array.shape[-2] - top - bottom
    width = array.shape[-1] - left - right
    if array.ndim == 2:
        return array[top : top + height, left : left + width]
    return array[..., top : top + height, left : left + width]


def tile_key(
    *,
    scene_date,
    grid: Grid,
    window: TileWindow,
    bands: Sequence[str] | None = None,
) -> str:
    """Stable cache identity for a tile.

    Encodes the grid geometry as well as the position: the same row/column at a
    different resolution is a different array, and a cache that confuses them
    produces predictions that are subtly misaligned with the imagery.
    """
    res_x, res_y = grid.resolution
    band_part = f"/{'+'.join(bands)}" if bands else ""
    return (
        f"tile/{scene_date:%Y-%m-%d}/{grid.crs.to_string()}/{res_x:g}x{res_y:g}"
        f"/s{window.size}/{window.name}{band_part}"
    )


def full_tile_count(grid: Grid, size: int) -> int:
    """How many complete tiles a grid yields."""
    return (grid.width // size) * (grid.height // size)
