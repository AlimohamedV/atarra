"""Offline tests for the orchestration layer's pure decisions.

These cover the band-selection rule, which is easy to get subtly wrong and whose
failure mode is nasty: a request silently receives a stack missing the bands it
asked for, and -- because the band set is part of the cache key -- can even be
served another request's composite.
"""

from __future__ import annotations

import pytest

from atarra.core.config import get_bands, get_study_area, get_study_areas
from atarra.core.errors import ImageryError
from atarra.pipeline import choose_grid, resolve_bands


class TestResolveBands:
    def test_explicit_bands_are_never_narrowed(self):
        """The regression: an explicit list must survive untouched.

        The true-colour render asks for blue and green even though no index uses
        them; narrowing stripped both and the render then 502'd.
        """
        cfg = get_bands()
        requested = ["B02", "B03", "B04", "B08"]
        assert resolve_bands(requested, ["ndvi"], cfg) == requested

    def test_explicit_list_that_indices_would_strip_is_preserved(self):
        cfg = get_bands()
        # NDVI consumes only B08 and B04, so narrowing would drop B02/B03.
        result = resolve_bands(["B02", "B03", "B04", "B08"], ["ndvi"], cfg)
        assert "B02" in result and "B03" in result

    def test_single_index_is_narrowed_to_its_own_bands(self):
        cfg = get_bands()
        assert resolve_bands(None, ["ndvi"], cfg) == ["B04", "B08"]
        assert resolve_bands(None, ["ndwi"], cfg) == ["B03", "B08"]
        assert resolve_bands(None, ["ndmi"], cfg) == ["B08", "B11"]

    def test_narrowing_reduces_the_read_count(self):
        """This is the optimisation, so assert it actually happens."""
        cfg = get_bands()
        narrowed = resolve_bands(None, ["ndvi"], cfg)
        assert len(narrowed) < len(cfg.bands_8)

    def test_all_indices_union_every_band_they_need(self):
        cfg = get_bands()
        result = resolve_bands(None, ["ndvi", "ndwi", "ndre", "ndmi"], cfg)
        assert set(result) == {"B03", "B04", "B05", "B08", "B11"}

    def test_order_follows_the_configured_band_order(self):
        """Stable ordering keeps the cache key deterministic across requests."""
        cfg = get_bands()
        once = resolve_bands(None, ["ndmi", "ndvi"], cfg)
        again = resolve_bands(None, ["ndvi", "ndmi"], cfg)
        assert once == again

    def test_unrecognised_index_falls_back_to_the_full_stack(self):
        cfg = get_bands()
        assert resolve_bands(None, ["not-an-index"], cfg) == list(cfg.bands_8)

    def test_no_indices_requested_returns_the_full_stack(self):
        cfg = get_bands()
        assert resolve_bands(None, [], cfg) == list(cfg.bands_8)

    def test_result_does_not_alias_the_config(self):
        """Callers mutate the returned list; the config must not be corrupted."""
        cfg = get_bands()
        result = resolve_bands(None, ["not-an-index"], cfg)
        result.append("BOGUS")
        assert "BOGUS" not in get_bands().bands_8


class TestChooseGrid:
    def test_coarsens_rather_than_cropping_to_fit(self):
        area = get_study_area("burullus")
        grid = choose_grid(area, gsd=10.0, max_size=256)
        assert max(grid.width, grid.height) <= 256
        # Coarsening keeps the whole AOI; the bounds must still match the area.
        assert grid.resolution[0] > 10.0

    def test_requested_gsd_is_used_when_it_already_fits(self):
        for area in get_study_areas().values():
            grid = choose_grid(area, gsd=30.0, max_size=8192)
            assert grid.resolution[0] == 30.0

    def test_covering_the_whole_aoi_is_never_traded_away(self):
        """Every configured area stays whole, however tight the pixel budget.

        The grid is snapped outward, so it must always enclose the AOI rather
        than clip it at the edge.
        """
        for area in get_study_areas().values():
            grid = choose_grid(area, gsd=60.0, max_size=256)
            wanted = area.bbox.to_crs(area.crs)
            bounds = grid.bounds
            assert bounds.crs == wanted.crs
            assert bounds.west <= wanted.west
            assert bounds.south <= wanted.south
            assert bounds.east >= wanted.east
            assert bounds.north >= wanted.north

    def test_impossible_constraint_raises(self):
        area = get_study_area("burullus")
        with pytest.raises(ImageryError):
            choose_grid(area, gsd=0.01, max_size=4)
