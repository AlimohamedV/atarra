"""Tests for grid algebra, configuration, and the cache."""

from __future__ import annotations

import numpy as np
import pytest

from atarra.core.cache import DiskCache
from atarra.core.config import get_bands, get_study_areas, load_bands
from atarra.core.errors import ConfigError
from atarra.core.grids import BBox, grid_from_bbox, snap_bounds


class TestBBox:
    def test_rejects_degenerate(self):
        with pytest.raises(ValueError, match="degenerate"):
            BBox(west=1.0, east=1.0, south=0.0, north=1.0)

    def test_from_sequence(self):
        box = BBox.from_sequence([30.0, 31.0, 31.0, 32.0])
        assert (box.west, box.south, box.east, box.north) == (30.0, 31.0, 31.0, 32.0)

    def test_settlement_to_crs_takes_hull_of_all_corners(self):
        """Reprojecting only two corners would under-cover a rotated box."""
        box = BBox.from_sequence([30.55, 31.38, 31.15, 31.70])
        projected = box.to_crs("EPSG:32636")
        assert projected.crs == "EPSG:32636"
        back = projected.to_crs("EPSG:4326")
        assert back.west <= box.west + 1e-6
        assert back.east >= box.east - 1e-6

    def test_geojson_ring_is_closed(self):
        ring = BBox.from_sequence([30.0, 31.0, 31.0, 32.0]).as_geojson()["coordinates"][0]
        assert ring[0] == ring[-1]


class TestSnapping:
    def test_expands_outward_only(self):
        box = BBox.from_sequence([100.5, 200.5, 110.5, 210.5], crs="EPSG:32636")
        snapped = snap_bounds(box, 10.0)
        assert snapped.west <= box.west
        assert snapped.south <= box.south
        assert snapped.east >= box.east
        assert snapped.north >= box.north

    def test_lands_on_multiples(self):
        box = BBox.from_sequence([100.5, 200.5, 110.5, 210.5], crs="EPSG:32636")
        snapped = snap_bounds(box, 10.0)
        for value in (snapped.west, snapped.south, snapped.east, snapped.north):
            assert value % 10.0 == pytest.approx(0.0)


class TestGrid:
    def test_contains_its_aoi(self, simple_bbox):
        grid = grid_from_bbox(simple_bbox, "EPSG:32636", 10.0)
        assert grid.bounds.west <= simple_bbox.to_crs("EPSG:32636").west
        assert grid.bounds.east >= simple_bbox.to_crs("EPSG:32636").east

    def test_is_deterministic(self, simple_bbox):
        first = grid_from_bbox(simple_bbox, "EPSG:32636", 10.0)
        second = grid_from_bbox(simple_bbox, "EPSG:32636", 10.0)
        assert (first.width, first.height) == (second.width, second.height)
        assert first.transform == second.transform

    def test_affine_maps_pixels_to_crs(self, simple_grid):
        """The transform must round-trip: pixel -> CRS -> pixel."""
        inverse = ~simple_grid.transform
        for col, row in [(0, 0), (10, 5), (simple_grid.width - 1, simple_grid.height - 1)]:
            x, y = simple_grid.transform @ (col + 0.5, row + 0.5)
            back_col, back_row = inverse @ (x, y)
            assert back_col == pytest.approx(col + 0.5)
            assert back_row == pytest.approx(row + 0.5)

    def test_tiles_tile_the_grid_exactly(self, simple_grid):
        """Full tiles must be non-overlapping and cover the grid."""
        size = 64
        covered = np.zeros((simple_grid.height, simple_grid.width), dtype=int)
        count = 0
        for tile in simple_grid.tiles(size, full_only=True):
            window = tile.window
            covered[
                int(window.row_off) : int(window.row_off + window.height),
                int(window.col_off) : int(window.col_off + window.width),
            ] += 1
            count += 1
        assert count > 0
        assert covered.max() == 1, "tiles overlap"
        # With full_only, only the trimmed edges should be uncovered.
        assert covered[: (simple_grid.height // size) * size, : (simple_grid.width // size) * size].min() == 1

    def test_tile_count_matches_formula(self, simple_grid):
        size = 64
        expected = (simple_grid.width // size) * (simple_grid.height // size)
        assert len(list(simple_grid.tiles(size))) == expected

    def test_corners_are_a_closed_quad_in_wgs84(self, simple_grid):
        corners = simple_grid.corners_wgs84()
        assert len(corners) == 4
        for lon, lat in corners:
            assert 30.0 < lon < 32.0
            assert 31.0 < lat < 32.0
        # Corners must be given clockwise from top-left for a MapLibre image source.
        assert corners[0][1] > corners[3][1]

    def test_window_for_bbox(self, simple_grid, simple_bbox):
        window = simple_grid.window_for(simple_bbox)
        assert window.width > 0 and window.height > 0
        assert window.width <= simple_grid.width


class TestCache:
    def test_roundtrip(self, tmp_path):
        cache = DiskCache(tmp_path, max_bytes=10_000_000)
        array = np.arange(100, dtype=np.float32)
        cache.store("k", array)
        assert np.array_equal(cache.load("k"), array)

    def test_miss_returns_none(self, tmp_path):
        assert DiskCache(tmp_path, max_bytes=1000).load("absent") is None

    def test_get_or_compute_calls_once(self, tmp_path):
        cache = DiskCache(tmp_path, max_bytes=10_000_000)
        calls = []

        def compute():
            calls.append(1)
            return np.ones((4, 4), dtype=np.float32)

        cache.get_or_compute("k", compute)
        cache.get_or_compute("k", compute)
        assert len(calls) == 1

    def test_evicts_when_over_ceiling(self, tmp_path):
        # A 1 kB ceiling with 2 kB of entries must leave roughly one entry behind.
        cache = DiskCache(tmp_path, max_bytes=2048)
        cache.store("a", np.ones((64, 64), dtype=np.float32))
        cache.store("b", np.ones((64, 64), dtype=np.float32))
        cache.store("c", np.ones((64, 64), dtype=np.float32))
        assert cache.size_bytes() <= 2048
        assert cache.stats()["entries"] < 3

    def test_corrupt_entry_is_dropped_not_raised(self, tmp_path):
        cache = DiskCache(tmp_path, max_bytes=10_000_000)
        path = cache.store("k", np.ones(10, dtype=np.float32))
        path.write_bytes(b"not a numpy file")
        assert cache.load("k") is None
        assert not path.exists()


class TestConfig:
    def test_bands_load(self):
        bands = get_bands()
        assert bands.bands_8 == ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"]
        assert bands.bands_rgb == ["B04", "B03", "B02"]
        assert bands.mask.resample == "nearest"

    def test_reflectance_convention_is_the_validated_one(self):
        """The archive's item metadata disagrees with itself; config must be explicit."""
        bands = get_bands()
        assert bands.reflectance_mode == "dn_scale"
        assert bands.reflectance_offset == 0.0
        assert bands.reflectance_scale == pytest.approx(0.0001)

    def test_reflectance_helper(self):
        bands = get_bands()
        # A storage value of 3000 should be 0.30 reflectance.
        assert bands.reflectance(np.float32(3000)) == pytest.approx(0.3, abs=1e-6)

    def test_mask_must_be_nearest(self, tmp_path):
        """Interpolating class codes would invent classes that do not exist."""
        path = tmp_path / "bands.yaml"
        path.write_text(
            "bands:\n"
            "  B04: {asset: red, gsd: 10, resample: bilinear}\n"
            "  B03: {asset: green, gsd: 10, resample: bilinear}\n"
            "  B02: {asset: blue, gsd: 10, resample: bilinear}\n"
            "  B05: {asset: rededge1, gsd: 20, resample: bilinear}\n"
            "  B08: {asset: nir, gsd: 10, resample: bilinear}\n"
            "  B8A: {asset: nir08, gsd: 20, resample: bilinear}\n"
            "  B11: {asset: swir16, gsd: 20, resample: bilinear}\n"
            "  B12: {asset: swir22, gsd: 20, resample: bilinear}\n"
            "mask_band: {name: SCL, asset: scl, gsd: 20, resample: bilinear}\n"
            "bands_8: [B02, B03, B04, B05, B08, B8A, B11, B12]\n"
            "bands_10: [B02, B03, B04, B05, B08, B8A, B11, B12]\n"
            "bands_rgb: [B04, B03, B02]\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="nearest"):
            load_bands(path)

    def test_unknown_band_in_a_list_is_rejected(self, tmp_path):
        path = tmp_path / "bands.yaml"
        path.write_text(
            "bands:\n  B04: {asset: red, gsd: 10, resample: bilinear}\n"
            "mask_band: {name: SCL, asset: scl, gsd: 20, resample: nearest}\n"
            "bands_8: [B04, B99]\nbands_10: [B04]\nbands_rgb: [B04]\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="unknown bands"):
            load_bands(path)

    def test_study_areas_load(self):
        areas = get_study_areas()
        assert {"burullus", "manzala", "kafr_elsheikh_canal"} <= set(areas)
        assert areas["burullus"].role == "wetland"
        assert areas["kafr_elsheikh_canal"].role == "canal"
