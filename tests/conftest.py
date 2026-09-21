"""Shared fixtures.

Tests that touch the live archive are marked ``network`` and are deselected by
default, so the suite result never depends on a satellite catalogue being up.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from atarra.core.grids import BBox, Grid, grid_from_bbox
from atarra.preprocess.reader import BandStack


@pytest.fixture
def simple_bbox() -> BBox:
    """A deliberately small AOI so grids stay cheap in tests."""
    return BBox.from_sequence([30.80, 31.45, 30.85, 31.50])


@pytest.fixture
def simple_grid(simple_bbox: BBox) -> Grid:
    return grid_from_bbox(simple_bbox, "EPSG:32636", 10.0)


def make_stack(
    grid: Grid,
    band_names: list[str],
    *,
    fill: float = 0.2,
    seed: int = 0,
    valid_fraction: float = 1.0,
) -> BandStack:
    """Build a synthetic BandStack with plausible reflectance values."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, max(fill, 1e-6), size=(len(band_names), grid.height, grid.width)).astype(
        np.float32
    )
    valid = np.ones((grid.height, grid.width), dtype=bool)
    if valid_fraction < 1.0:
        cutoff = int(grid.height * (1.0 - valid_fraction))
        valid[:cutoff, :] = False
        data[:, ~valid] = np.nan
    return BandStack(
        data=data,
        valid=valid,
        grid=grid,
        band_names=list(band_names),
        window=_full_window(grid),
        scene_ids=["synthetic"],
    )


def _full_window(grid: Grid):
    from rasterio.windows import Window

    return Window(0, 0, grid.width, grid.height)


@pytest.fixture
def stack_builder():
    """Expose :func:`make_stack` so tests can build bespoke stacks."""
    return make_stack


@pytest.fixture
def synthetic_stack(simple_grid: Grid) -> BandStack:
    return make_stack(
        simple_grid, ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"], seed=7
    )


@pytest.fixture
def veg_water_stack(simple_grid: Grid):
    """A stack with a crisp vegetation/water split, for index assertions."""
    height, width = simple_grid.height, simple_grid.width
    # Open water: dark in NIR, and *higher* in red-edge than in NIR, because water
    # absorbs NIR more strongly than the shorter red-edge wavelengths. Setting the
    # two equal would give NDRE == 0, which is neither physical nor a useful test.
    nir = np.full((height, width), 0.025, dtype=np.float32)
    red = np.full((height, width), 0.04, dtype=np.float32)
    green = np.full((height, width), 0.05, dtype=np.float32)
    red_edge = np.full((height, width), 0.035, dtype=np.float32)
    swir = np.full((height, width), 0.04, dtype=np.float32)

    half = width // 2
    # Dense vegetation on the left: high NIR, low red, strong red-edge step.
    nir[:, :half] = 0.42
    red[:, :half] = 0.04
    green[:, :half] = 0.08
    red_edge[:, :half] = 0.18
    swir[:, :half] = 0.20

    data = np.stack([green, green, red, red_edge, nir, nir, swir, swir]).astype(np.float32)
    # Bands must follow the declared order.
    order = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"]
    reordered = np.stack(
        [
            np.full((height, width), 0.06, dtype=np.float32),  # B02 blue
            data[order.index("B03")],
            data[order.index("B04")],
            data[order.index("B05")],
            data[order.index("B08")],
            data[order.index("B8A")],
            data[order.index("B11")],
            data[order.index("B12")],
        ]
    )
    return BandStack(
        data=reordered,
        valid=np.ones((height, width), dtype=bool),
        grid=simple_grid,
        band_names=order,
        window=_full_window(simple_grid),
        scene_ids=["synthetic"],
    )


@pytest.fixture
def dates_2023() -> list[date]:
    return [date(2023, m, 15) for m in range(4, 11)]
