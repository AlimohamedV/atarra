"""Tests for spectral indices and tiling."""

from __future__ import annotations

import numpy as np
import pytest

from atarra.core.errors import ImageryError
from atarra.preprocess.indices import (
    INDEX_BANDS,
    compute_indices,
    ndmi,
    ndre,
    ndvi,
    ndwi,
    normalized_difference,
)
from atarra.preprocess.tiler import (
    crop_from_padding,
    extract_tile,
    full_tile_count,
    pad_to_tile,
)


class TestNormalizedDifference:
    def test_hand_computed_value(self):
        # (0.4 - 0.1) / (0.4 + 0.1) = 0.6
        result = normalized_difference(np.array([0.4]), np.array([0.1]))
        assert result[0] == pytest.approx(0.6)

    def test_is_bounded_for_non_negative_inputs(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, 5000).astype(np.float32)
        b = rng.uniform(0, 1, 5000).astype(np.float32)
        result = normalized_difference(a, b)
        finite = result[np.isfinite(result)]
        assert finite.min() >= -1.0 - 1e-6
        assert finite.max() <= 1.0 + 1e-6

    def test_degenerate_denominator_gives_nan_not_inf(self):
        """An `inf` survives a mean and a loss function; a NaN is honest."""
        result = normalized_difference(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_near_zero_denominator_is_rejected(self):
        """Opposite-signed bands on a dark pixel must not produce a huge value."""
        result = normalized_difference(np.array([0.0001]), np.array([-0.0001]), eps=1e-3)
        assert np.isnan(result[0])

    def test_propagates_nan_input(self):
        result = normalized_difference(np.array([np.nan]), np.array([0.1]))
        assert np.isnan(result[0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ImageryError, match="mismatched shapes"):
            normalized_difference(np.zeros((2, 2)), np.zeros((3, 3)))


class TestNamedIndices:
    def test_ndvi_matches_formula(self):
        assert ndvi(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(2 / 3)

    def test_ndwi_matches_formula(self):
        assert ndwi(np.array([0.3]), np.array([0.1]))[0] == pytest.approx(0.5)

    def test_ndre_matches_formula(self):
        assert ndre(np.array([0.4]), np.array([0.2]))[0] == pytest.approx(1 / 3)

    def test_ndmi_matches_formula(self):
        assert ndmi(np.array([0.4]), np.array([0.2]))[0] == pytest.approx(1 / 3)


class TestComputeIndices:
    def test_vegetation_and_water_separate_cleanly(self, veg_water_stack):
        """The whole project rests on this: reed and water must be distinguishable."""
        indices = compute_indices(veg_water_stack)
        ndvi_values = indices["ndvi"]
        half = ndvi_values.shape[1] // 2

        vegetation = np.nanmedian(ndvi_values[:, :half])
        water = np.nanmedian(ndvi_values[:, half:])

        assert vegetation > 0.7, f"vegetation NDVI should be high, got {vegetation}"
        assert water < 0.0, f"water NDVI should be low, got {water}"
        assert vegetation - water > 0.8

    def test_red_edge_separates_reed_from_open_water(self, veg_water_stack):
        """NDRE is the multispectral advantage: it must hold contrast NDVI does not."""
        indices = compute_indices(veg_water_stack)
        ndre_values = indices["ndre"]
        half = ndre_values.shape[1] // 2
        assert np.nanmedian(ndre_values[:, :half]) > 0.3
        assert np.nanmedian(ndre_values[:, half:]) < 0.0

    def test_ndwi_is_positive_over_water(self, veg_water_stack):
        ndwi_values = compute_indices(veg_water_stack)["ndwi"]
        half = ndwi_values.shape[1] // 2
        assert np.nanmedian(ndwi_values[:, half:]) > 0.2

    def test_all_indices_within_range(self, synthetic_stack):
        for name, values in compute_indices(synthetic_stack).items():
            finite = values[np.isfinite(values)]
            assert finite.min() >= -1.0001, f"{name} below -1"
            assert finite.max() <= 1.0001, f"{name} above 1"

    def test_skips_indices_whose_bands_are_absent(self, simple_grid, stack_builder):
        """This is what lets the 3-band RGB baseline reuse the same code path."""
        rgb_only = stack_builder(simple_grid, ["B02", "B03", "B04"], seed=1)
        available = compute_indices(rgb_only)
        # NDVI/NDRE/NDMI all need NIR, which an RGB stack does not have.
        assert "ndvi" not in available
        assert "ndre" not in available
        assert "ndmi" not in available
        # NDWI needs only green and NIR, so it is unavailable too.
        assert "ndwi" not in available
        assert available == {}

    def test_subset_selection(self, synthetic_stack):
        subset = compute_indices(synthetic_stack, ["ndvi"])
        assert set(subset) == {"ndvi"}

    def test_unknown_index_raises(self, synthetic_stack):
        with pytest.raises(ImageryError, match="unknown index"):
            compute_indices(synthetic_stack, ["ndwi_typo"])

    def test_index_bands_reference_declared_bands(self):
        declared = {"B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"}
        for name, (a, b) in INDEX_BANDS.items():
            assert a in declared, f"{name} uses undeclared band {a}"
            assert b in declared, f"{name} uses undeclared band {b}"


class TestTiler:
    def test_extract_tile_shape(self, simple_grid):
        array = np.zeros((3, simple_grid.height, simple_grid.width), dtype=np.float32)
        tile = next(iter(simple_grid.tiles(64)))
        assert extract_tile(array, tile).shape[0] == 3

    def test_pad_then_crop_is_identity(self):
        array = np.arange(3 * 50 * 30, dtype=np.float32).reshape(3, 50, 30)
        padded, padding = pad_to_tile(array, 64)
        assert padded.shape == (3, 64, 64)
        restored = crop_from_padding(padded, padding)
        assert np.array_equal(restored, array)

    def test_pad_2d(self):
        array = np.ones((10, 20), dtype=np.float32)
        padded, padding = pad_to_tile(array, 32)
        assert padded.shape == (32, 32)
        assert np.array_equal(crop_from_padding(padded, padding), array)

    def test_pad_places_content_at_origin(self):
        """Padding must not shift the content, or every edge prediction moves."""
        array = np.ones((8, 8), dtype=np.float32)
        padded, _ = pad_to_tile(array, 16)
        assert padded[0, 0] == 1.0
        assert padded[15, 15] == 0.0

    def test_pad_rejects_oversized_tile(self):
        with pytest.raises(ValueError, match="larger than the target"):
            pad_to_tile(np.zeros((100, 100)), 64)

    def test_full_tile_count(self, simple_grid):
        assert full_tile_count(simple_grid, 64) == (simple_grid.width // 64) * (
            simple_grid.height // 64
        )
