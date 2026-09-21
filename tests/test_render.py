"""Offline tests for the rendering layer.

The renderers are what a user actually looks at, and their failure modes are quiet
ones: an undefined cast can put arbitrary colours on screen at nodata pixels, and
an overlay whose corners do not match the grid it was drawn from will sit over the
wrong water without ever raising an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from atarra.core.errors import ImageryError
from atarra.viz.render import apply_colormap, encode_png, render_mask, render_true_color


class TestTrueColor:
    def test_returns_rgba_uint8(self, synthetic_stack):
        rgba = render_true_color(synthetic_stack)
        assert rgba.shape == (
            synthetic_stack.grid.height,
            synthetic_stack.grid.width,
            4,
        )
        assert rgba.dtype == np.uint8

    def test_missing_channels_are_reported(self, stack_builder, simple_grid):
        stack = stack_builder(simple_grid, ["B08"], fill=0.3)
        with pytest.raises(ImageryError, match="B04"):
            render_true_color(stack)

    def test_nodata_pixels_are_transparent_not_noise(self, stack_builder, simple_grid):
        """The regression: NaN cannot survive a cast to uint8.

        Nodata was reaching ``astype(np.uint8)`` unscaled, which numpy converts to
        an undefined value -- visible as random colour speckle at every tile edge.
        """
        stack = stack_builder(
            simple_grid, ["B02", "B03", "B04", "B08"], fill=0.3, valid_fraction=0.5
        )
        rgba = render_true_color(stack)

        invalid = ~stack.valid
        assert invalid.any(), "fixture should contain nodata"
        assert (rgba[..., 3][invalid] == 0).all(), "nodata must be fully transparent"
        assert (rgba[..., :3][invalid] == 0).all(), "nodata must not carry a colour"

        assert (rgba[..., 3][stack.valid] == 255).all()

    def test_no_runtime_warning_on_nodata(self, stack_builder, simple_grid, recwarn):
        """A silent ``invalid value encountered in cast`` must not reappear."""
        stack = stack_builder(
            simple_grid, ["B02", "B03", "B04", "B08"], fill=0.3, valid_fraction=0.5
        )
        render_true_color(stack)
        assert not [w for w in recwarn.list if "cast" in str(w.message)]


class TestColormap:
    def test_matches_legend_range(self):
        layer = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        rgba = apply_colormap(layer, "ndvi")
        assert rgba.shape == (8, 8, 4)
        assert rgba.dtype == np.uint8

    def test_nodata_is_transparent(self):
        layer = np.full((4, 4), np.nan, dtype=np.float32)
        rgba = apply_colormap(layer, "ndvi")
        assert (rgba[..., 3] == 0).all()

    def test_saturates_rather_than_wrapping(self):
        """Out-of-range values must clamp, not wrap around to the wrong colour."""
        low = apply_colormap(np.full((2, 2), -5.0, dtype=np.float32), "ndvi")
        floor = apply_colormap(np.full((2, 2), -1.0, dtype=np.float32), "ndvi")
        assert (low[..., :3] == floor[..., :3]).all()


class TestMask:
    def test_only_masked_pixels_are_painted(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 2] = True
        rgba = render_mask(mask)
        assert rgba[1, 2, 3] > 0
        assert (rgba[mask == False, 3] == 0).all()  # noqa: E712

    def test_encodes_to_a_valid_png(self):
        rgba = render_mask(np.ones((4, 4), dtype=bool))
        png = encode_png(rgba)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
