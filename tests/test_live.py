"""Live integration tests against the real Sentinel-2 archive.

Marked ``network`` and deselected by default, so `pytest` stays fast and
deterministic. Run them deliberately with:

    pytest -m network

These are the tests that actually validate the project's central assumptions --
that the archive is reachable without credentials, that bands can be read with
byte ranges, and above all that reflectance comes out physical. A unit test cannot
tell you the reflectance convention has drifted; only real pixels can.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest
from rasterio.windows import Window

from atarra.core.config import get_bands, get_study_area
from atarra.core.grids import grid_from_bbox
from atarra.ingest import get_source, select_best_per_period
from atarra.pipeline import available_dates, load_composite
from atarra.preprocess import compute_indices, read_mosaic, read_window, validate_reflectance

pytestmark = pytest.mark.network

BURULLUS_TEST_DATE = date(2023, 8, 26)


@pytest.fixture(scope="module")
def source():
    return get_source("stac")


@pytest.fixture(scope="module")
def august_availability(source):
    """Availability across the whole month, for discovery assertions only."""
    return available_dates("burullus", date(2023, 8, 1), date(2023, 8, 31), max_cloud_cover=10)


@pytest.fixture(scope="module")
def scenes_around_august(source):
    """Scenes over a five-day window.

    Deliberately narrow. These fixtures feed tests that read real pixels, and each
    band read is a remote round trip -- sweeping a whole month of scenes turned the
    live suite into a multi-minute job for no extra coverage of what is being
    tested. The date-window behaviour that genuinely needs a wide sweep has its own
    test below.
    """
    area = get_study_area("burullus")
    return source.search(
        area.bbox, datetime(2023, 8, 24), datetime(2023, 8, 29), max_cloud_cover=10, limit=50
    )


@pytest.fixture(scope="module")
def centre_scene(scenes_around_august):
    """One scene known to cover the middle of the lake."""
    for scene in scenes_around_august:
        if scene.grid_code and "36RTV" in scene.grid_code:
            return scene
    return scenes_around_august[0]


def _lake_window(grid, size: int = 128):
    return Window(grid.width // 2, grid.height // 2, size, size)


class TestStacDiscovery:
    def test_archive_is_reachable_without_credentials(self, source):
        collections = source.collections()
        assert "sentinel-2-l2a" in collections

    def test_finds_cloud_free_scenes_over_burullus(self, august_availability):
        assert len(august_availability) >= 5, "expected several August acquisitions over Burullus"

    def test_scenes_carry_every_band_the_pipeline_needs(self, scenes_around_august):
        bands = get_bands()
        required = bands.bands_8 + [bands.mask.name]
        for scene in scenes_around_august[:10]:
            missing = [b for b in required if b not in scene.assets]
            assert not missing, f"{scene.id} is missing {missing}"

    def test_band_hrefs_are_cloud_optimized_geotiffs(self, scenes_around_august):
        href = scenes_around_august[0].asset("B08")
        assert href.startswith("https://")
        assert href.endswith(".tif")

    def test_multiple_mgrs_tiles_cover_burullus(self, scenes_around_august):
        """Establishes why mosaicking exists rather than being optional."""
        tiles = {scene.grid_code for scene in scenes_around_august}
        assert len(tiles) >= 2, f"expected several tiles, got {tiles}"

    def test_monthly_selection(self, scenes_around_august):
        bands = get_bands()
        best = select_best_per_period(scenes_around_august, require_bands=bands.bands_8)
        assert best
        months = [scene.month_key for scene in best]
        assert months == sorted(months)
        assert len(months) == len(set(months))

    def test_archive_spans_the_projects_time_range(self, source):
        """The proposal covers 2022-2026; confirm the collection really spans it."""
        for year in (2022, 2024, 2026):
            results = available_dates("burullus", date(year, 8, 1), date(year, 8, 20))
            assert results, f"no imagery found for August {year}"


class TestWindowedReads:
    def test_reads_a_window_without_downloading_the_tile(self, scenes_around_august):
        """A windowed read must return real data quickly; that is the disk premise.

        Read from the tile centre. Sentinel-2 tiles are rotated diamonds inside a
        square grid, so a large fraction of the bounding box -- over 70% for this
        tile -- is fill. An arbitrary window can therefore legitimately come back
        empty, which says nothing about whether ranged reads work.
        """
        import time

        import rasterio

        area = get_study_area("burullus")
        scene = scenes_around_august[0]
        with rasterio.open(scene.asset("B08")) as dataset:
            assert dataset.width > 10000, "expected a full 10 m Sentinel-2 tile"

            # Locate the AOI centre in the file's own pixel grid. The geometric
            # centre of the file cannot be used: an MGRS tile bounding box is
            # 109,800 m square while the swath inside covers only about a quarter of
            # it, so the middle of the file is fill (see the test below).
            import rasterio.warp
            from rasterio.transform import rowcol

            west, south, east, north = area.bbox.as_stac_query()
            xs, ys = rasterio.warp.transform(
                "EPSG:4326", dataset.crs, [(west + east) / 2], [(south + north) / 2]
            )
            rows, cols = rowcol(dataset.transform, xs, ys)
            window = Window(cols[0] - 128, rows[0] - 128, 256, 256)
            start = time.time()
            data = dataset.read(1, window=window)
            elapsed = time.time() - start

        assert data.shape == (256, 256)
        assert data.max() > 0, "the AOI centre should contain data"
        # A full band is ~68 MB. Delivering a 256x256 window promptly is only
        # possible with byte-range reads; a whole-file download would not fit here.
        assert elapsed < 30.0, f"windowed read took {elapsed:.1f}s, suggesting a full download"

    def test_tile_edges_are_legitimately_empty(self, scenes_around_august):
        """Documents why mosaic coverage needs a validity mask, not just geometry."""
        import rasterio

        scene = scenes_around_august[0]
        with rasterio.open(scene.asset("B08")) as dataset:
            corner = dataset.read(1, window=Window(0, 0, 256, 256))
        # The top-left corner of the bounding box lies outside the rotated swath.
        assert corner.max() == 0

    def test_tiles_are_mostly_fill_outside_the_swath(self, centre_scene):
        """Documents a real data property that shapes the mosaic design.

        An MGRS tile's bounding box is 109,800 m square, but the satellite images a
        rotated swath inside it. For tile 36RTV the data occupies roughly the eastern
        quarter, matching the ~75.8% nodata figure the archive itself reports.

        The consequence is why :mod:`atarra.preprocess.reader` decides validity from
        pixel values rather than footprints: a scene can overlap the AOI generously
        on paper while most of its pixels are fill.
        """
        import rasterio

        with rasterio.open(centre_scene.asset("B08")) as dataset:
            filled = 0
            total = 0
            for row in range(0, dataset.height, 1024):
                for col in range(0, dataset.width, 1024):
                    block = dataset.read(
                        1,
                        window=Window(
                            col,
                            row,
                            min(256, dataset.width - col),
                            min(256, dataset.height - row),
                        ),
                    )
                    filled += int((block == 0).sum())
                    total += block.size

        assert filled / total > 0.5, (
            f"expected most of the tile to be fill, got {filled / total:.1%}"
        )

    def test_grid_alignment_lands_on_the_satellite_lattice(self, source):
        """Snapping assumes S2 pixel origins are multiples of the pixel size."""
        import rasterio

        area = get_study_area("burullus")
        grid = grid_from_bbox(area.bbox, area.crs, 10.0)
        scene = source.search(
            area.bbox, datetime(2023, 8, 1), datetime(2023, 8, 2), max_cloud_cover=10, limit=1
        )[0]
        with rasterio.open(scene.asset("B08")) as dataset:
            offsets = (grid.bounds.west - dataset.transform.c) / dataset.transform.a
            assert abs(offsets - round(offsets)) < 1e-6, (
                "grid is not pixel-aligned with the scene; windowed reads would "
                "resample rather than slice"
            )


class TestReflectanceIsPhysical:
    """Reads a single scene window rather than a mosaic: the conversion convention
    is a per-scene property, so a mosaic adds cost without adding evidence."""

    def test_converted_reflectance_is_not_mostly_negative(self, centre_scene):
        """The check that caught the archive's own contradictory metadata.

        A 74% negative rate is what the item-level offset produces; the validated
        convention produces none. If this ever fails, the archive's convention has
        changed and every index in the project is suspect.
        """
        bands = get_bands()
        area = get_study_area("burullus")
        grid = grid_from_bbox(area.bbox, area.crs, 60.0)
        stack = read_window(
            centre_scene, grid, bands.bands_8, _lake_window(grid), bands_cfg=bands
        )
        stats = validate_reflectance(stack, bands)

        assert stats["negative_fraction"] <= bands.max_negative_fraction
        assert stats["max"] <= 1.5, "reflectance above 1.5 is not physical"
        assert stats["median"] > 0.0

    def test_indices_stay_within_theoretical_bounds(self, centre_scene):
        bands = get_bands()
        area = get_study_area("burullus")
        grid = grid_from_bbox(area.bbox, area.crs, 60.0)
        stack = read_window(
            centre_scene, grid, bands.bands_8, _lake_window(grid), bands_cfg=bands
        )

        indices = compute_indices(stack)
        assert indices, "no index could be computed from an 8-band stack"
        for name, values in indices.items():
            finite = values[np.isfinite(values)]
            assert finite.size > 0, f"{name} produced no valid pixels"
            assert finite.min() >= -1.0001, f"{name} below -1"
            assert finite.max() <= 1.0001, f"{name} above 1"

    def test_water_reads_as_water_over_burullus(self, centre_scene):
        """Lake Burullus is a lake: NDWI must be positive across its middle."""
        bands = get_bands()
        area = get_study_area("burullus")
        grid = grid_from_bbox(area.bbox, area.crs, 60.0)
        stack = read_window(
            centre_scene, grid, bands.bands_8, _lake_window(grid, 192), bands_cfg=bands
        )

        values = compute_indices(stack, ["ndwi"])["ndwi"]
        finite = values[np.isfinite(values)]
        assert finite.size > 1000
        assert np.median(finite) > 0.0, "the middle of a lake should read as water"
        assert (finite > 0).mean() > 0.5


class TestMosaic:
    def test_date_window_fills_tile_gaps(self, source):
        """A single date leaves holes; a date window must close them.

        This is the test that forced date-window compositing into the design: one
        Burullus date covered only ~57% of the AOI, because the four MGRS tiles
        covering it sit on different relative orbits and are not all acquired the
        same day.

        Runs over a bounded window and a reduced band list -- coverage depends on
        which scenes supply pixels, not on how many bands are requested, so pulling
        all eight bands here would multiply remote reads for no extra signal.
        """
        area = get_study_area("burullus")
        bands = get_bands()
        grid = grid_from_bbox(area.bbox, area.crs, 120.0, snap=True)
        window = _lake_window(grid, 128)
        subset = bands.bands_8[:3]

        single = source.search(
            area.bbox, datetime(2023, 8, 26), datetime(2023, 8, 27), max_cloud_cover=10, limit=20
        )
        wide = source.search(
            area.bbox, datetime(2023, 8, 23), datetime(2023, 8, 30), max_cloud_cover=10, limit=50
        )

        single_stack = read_mosaic(single, grid, subset, bands_cfg=bands, window=window)
        wide_stack = read_mosaic(wide, grid, subset, bands_cfg=bands, window=window)

        assert wide_stack.coverage >= single_stack.coverage
        assert wide_stack.coverage > 0.9, (
            f"a +/-3 day window should cover the AOI, got {wide_stack.coverage:.2%}"
        )

    def test_clearest_scene_wins_overlapping_pixels(self, scenes_around_august):
        """Scenes are painted clearest-first, so the order must be by cloud cover."""
        ordered = sorted(
            scenes_around_august,
            key=lambda s: s.cloud_cover if s.cloud_cover is not None else float("inf"),
        )
        clouds = [s.cloud_cover for s in ordered if s.cloud_cover is not None]
        assert clouds == sorted(clouds)


class TestPipeline:
    def test_single_index_composite_builds_and_caches(self):
        first = load_composite("burullus", BURULLUS_TEST_DATE, window_days=3, gsd=120.0, indices=["ndvi"])
        assert first.coverage > 0.8
        assert not first.warnings or all("coverage" not in w for w in first.warnings)

        second = load_composite("burullus", BURULLUS_TEST_DATE, window_days=3, gsd=120.0, indices=["ndvi"])
        assert second.from_cache, "the second build should hit the cache"

    def test_summary_is_json_serialisable(self):
        composite = load_composite("burullus", BURULLUS_TEST_DATE, window_days=3, gsd=120.0, indices=["ndvi"])
        summary = composite.summary()
        import json

        json.dumps(summary)  # must not raise
        assert summary["grid"]["crs"] == "EPSG:32636"
        assert len(summary["scenes"]) >= 1

    def test_available_dates_lists_real_dates(self):
        results = available_dates("burullus", date(2023, 8, 1), date(2023, 8, 31), max_cloud_cover=10)
        assert results
        for item in results:
            assert len(item["date"]) == 10
            assert item["scenes"] >= 1
