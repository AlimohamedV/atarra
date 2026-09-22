"""Tests for the on-disk tile store.

The store is what separates "a Colab session that fetches imagery" from "a Colab
session that trains", so the contract that matters is: what is written can be read
back unchanged, and the manifest never claims something the shards do not contain.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from atarra.core.errors import AtarraError
from atarra.core.grids import BBox, grid_from_bbox
from atarra.datasets.store import (
    REFLECTANCE_SCALE,
    TileStoreDataset,
    build_store,
    decode_reflectance,
    encode_reflectance,
    load_manifest,
)
from atarra.datasets.tile_dataset import geometric_split
from atarra.datasets.weak_labels import NUM_CLASSES
from atarra.pipeline import Composite
from atarra.preprocess.indices import compute_indices


def _composite(stack, grid, when: date = date(2024, 8, 20)) -> Composite:
    """A minimal Composite wrapping a synthetic stack."""
    return Composite(
        area_key="test_area",
        area_name="Test Area",
        target_date=when,
        window_days=3,
        grid=grid,
        stack=stack,
        indices=compute_indices(stack),
        scene_ids=["S2A_TEST"],
        scene_details=[{"id": "S2A_TEST", "cloud_cover": 1.0}],
        reflectance={"mode": "dn_scale", "negative_fraction": 0.0, "median": 0.2},
    )


@pytest.fixture
def patch_composite(monkeypatch):
    """Replace the archive fetch with a supplied composite.

    ``build_store`` imports ``load_composite`` inside the function body, so patching
    the attribute on the module is enough and no network is touched.
    """
    import atarra.pipeline as pipeline

    def install(*composites):
        sequence = list(composites)
        calls = {"count": 0}

        def fake(area_key, target_date, **kwargs):
            value = sequence[min(calls["count"], len(sequence) - 1)]
            calls["count"] += 1
            return value

        monkeypatch.setattr(pipeline, "load_composite", fake)
        return calls

    return install


class TestReflectanceCodec:
    def test_round_trips_values_on_the_stored_grid_exactly(self):
        """Sentinel-2 is DN * 1e-4, so this must be lossless, not merely close."""
        original = np.array([0.0, 0.0001, 0.0321, 0.5, 1.2345], dtype=np.float32)
        assert np.allclose(decode_reflectance(encode_reflectance(original)), original, atol=0)

    def test_quantises_to_the_nearest_step(self):
        # Deliberately not an exact .5 tie: float32 cannot represent one, and the
        # archive never produces one either, since DNs are integers scaled by 1e-4.
        original = np.array([0.12346, 0.99999], dtype=np.float32)
        restored = decode_reflectance(encode_reflectance(original))
        assert np.allclose(restored, [0.1235, 1.0], atol=1e-6)

    def test_non_finite_and_absurd_values_do_not_wrap_around(self):
        """NaN cast to an integer is undefined; unsigned wraparound would be a bug."""
        encoded = encode_reflectance(np.array([np.nan, np.inf, -np.inf, 1e6], dtype=np.float32))
        assert encoded.dtype == np.uint16
        assert int(encoded[0]) == 0 and int(encoded[1]) == 0 and int(encoded[2]) == 0
        assert int(encoded[3]) == np.iinfo(np.uint16).max

    def test_scale_matches_the_archive_convention(self):
        assert REFLECTANCE_SCALE == 10000.0


class TestBuildStore:
    def test_writes_a_manifest_that_matches_its_shards(
        self, tmp_path, patch_composite, veg_water_stack, simple_grid
    ):
        patch_composite(_composite(veg_water_stack, simple_grid))
        manifest = build_store(
            tmp_path / "store",
            "test_area",
            [date(2024, 8, 20)],
            gsd=10.0,
            tile_size=64,
            stride=32,
        )

        assert manifest["totals"]["shards"] == 1
        # The manifest on disk is what a later process reads, so trust the file.
        on_disk = load_manifest(tmp_path / "store")
        assert manifest["shards"][0]["tiles"] > 0
        assert on_disk["totals"]["tiles"] == manifest["shards"][0]["tiles"]
        assert on_disk["totals"]["tiles"] == manifest["totals"]["tiles"]
        assert on_disk["bands"] == [
            "B02",
            "B03",
            "B04",
            "B05",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]

        # Class tallies must be four long wherever a shard happens to contain none.
        for shard in manifest["shards"]:
            assert len(shard["class_counts"]) == NUM_CLASSES
        assert len(manifest["totals"]["class_counts"]) == NUM_CLASSES
        assert sum(manifest["totals"]["class_counts"]) <= manifest["totals"]["trainable_px"]

    def test_returns_tiles_that_can_be_read_back(
        self, tmp_path, patch_composite, veg_water_stack, simple_grid
    ):
        composite = _composite(veg_water_stack, simple_grid)
        patch_composite(composite)
        build_store(
            tmp_path / "store",
            "test_area",
            [date(2024, 8, 20)],
            gsd=10.0,
            tile_size=64,
            stride=32,
        )

        dataset = TileStoreDataset(tmp_path / "store")
        assert len(dataset) > 0
        item = dataset[0]
        assert item["image"].dtype == np.float32
        assert item["image"].shape == (8, 64, 64)
        assert item["mask"].dtype == np.int64
        assert item["mask"].shape == (64, 64)
        # Invalid pixels must arrive as the ignore index, never as class 0.
        assert set(np.unique(item["mask"])) <= {-1, 0, 1, 2, 3}

    def test_stored_pixels_survive_the_round_trip(
        self, tmp_path, patch_composite, synthetic_stack, simple_grid
    ):
        """Values must match the composite, not merely be plausible."""
        composite = _composite(synthetic_stack, simple_grid)
        patch_composite(composite)
        build_store(
            tmp_path / "store",
            "test_area",
            [date(2024, 8, 20)],
            gsd=10.0,
            tile_size=64,
            stride=64,
        )

        dataset = TileStoreDataset(tmp_path / "store")
        item = dataset[0]
        expected = synthetic_stack.band("B02")[:64, :64]
        assert np.allclose(item["image"][0], expected, atol=1e-4)

    def test_refuses_to_mix_two_different_grids(self, tmp_path, patch_composite, veg_water_stack, simple_grid):
        """Blocks are only comparable across shards if the grid is shared.

        Otherwise the same block key means different ground in different shards and
        the geometric split leaks between train and test.
        """
        other = grid_from_bbox(
            BBox.from_sequence([30.80, 31.45, 30.83, 31.48]), "EPSG:32636", 10.0
        )
        assert (other.width, other.height) != (simple_grid.width, simple_grid.height)
        patch_composite(
            _composite(veg_water_stack, simple_grid),
            _composite(veg_water_stack, other),
        )
        with pytest.raises(AtarraError, match="share one grid"):
            build_store(
                tmp_path / "store",
                "test_area",
                [date(2024, 8, 20), date(2024, 8, 25)],
                gsd=10.0,
                tile_size=64,
                stride=32,
            )

    def test_refuses_a_grid_that_was_coarsened_to_fit(self, tmp_path, patch_composite, veg_water_stack, simple_grid):
        """A coarsened grid would make the manifest lie about the stored scale.

        ``choose_grid`` doubles the resolution until the AOI fits ``max_size``. If
        that happens, tiles are stored at (say) 40 m while the manifest claims 10 m,
        and the model trains at the wrong scale with no error anywhere.
        """
        coarse = grid_from_bbox(simple_grid.bounds, "EPSG:32636", 40.0)
        patch_composite(_composite(veg_water_stack, coarse))
        with pytest.raises(AtarraError, match="Grid was coarsened"):
            build_store(
                tmp_path / "store",
                "test_area",
                [date(2024, 8, 20)],
                gsd=10.0,
                tile_size=64,
                stride=32,
            )

    def test_an_existing_store_is_not_silently_appended_to(
        self, tmp_path, patch_composite, veg_water_stack, simple_grid
    ):
        patch_composite(_composite(veg_water_stack, simple_grid))
        for rebuild in (False, True):
            build_store(
                tmp_path / "store",
                "test_area",
                [date(2024, 8, 20)],
                gsd=10.0,
                tile_size=64,
                stride=32,
                first=rebuild,
            )

        with pytest.raises(AtarraError, match="already holds a store"):
            build_store(
                tmp_path / "store",
                "test_area",
                [date(2024, 8, 20)],
                gsd=10.0,
                tile_size=64,
                stride=32,
            )

    def test_a_date_that_cannot_be_fetched_is_recorded_not_fatal(
        self, tmp_path, monkeypatch, veg_water_stack, simple_grid
    ):
        """One cloudy overpass must not throw away the dates that did work."""
        import atarra.pipeline as pipeline_module

        good = _composite(veg_water_stack, simple_grid)
        calls = {"n": 0}

        def flaky(area_key, target_date, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise AtarraError("no scenes with acceptable cloud cover")
            return good

        monkeypatch.setattr(pipeline_module, "load_composite", flaky)
        manifest = build_store(
            tmp_path / "store",
            "test_area",
            [date(2024, 8, 18), date(2024, 8, 20)],
            gsd=10.0,
            tile_size=64,
            stride=32,
        )

        assert len(manifest["shards"]) == 1
        assert len(manifest["skipped"]) == 1
        assert "cloud" in manifest["skipped"][0]["reason"]

    def test_no_usable_dates_raises_rather_than_writing_an_empty_store(
        self, tmp_path, monkeypatch
    ):
        import atarra.pipeline as pipeline_module

        def always_fails(area_key, target_date, **kwargs):
            raise AtarraError("nothing usable")

        monkeypatch.setattr(pipeline_module, "load_composite", always_fails)
        with pytest.raises(AtarraError, match="nothing was written"):
            build_store(
                tmp_path / "store",
                "test_area",
                [date(2024, 8, 20)],
                gsd=10.0,
                tile_size=64,
                stride=32,
            )


class TestStoreDataset:
    @pytest.fixture
    def store(self, tmp_path, patch_composite, veg_water_stack, simple_grid):
        patch_composite(_composite(veg_water_stack, simple_grid))
        build_store(
            tmp_path / "store",
            "test_area",
            [date(2024, 8, 20)],
            gsd=10.0,
            tile_size=64,
            stride=32,
        )
        return tmp_path / "store"

    def test_cuts_the_rgb_control_arm_from_one_download(self, store):
        """The 3-band baseline must not require a second fetch of the archive."""
        dataset = TileStoreDataset(store, band_names=["B04", "B03", "B02"])
        assert dataset.stored_bands == [
            "B02",
            "B03",
            "B04",
            "B05",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]
        assert dataset.band_names == ["B04", "B03", "B02"]
        image = dataset[0]["image"]
        assert image.shape[0] == 3

        # Channel order must follow the request, not the stored layout.
        full = TileStoreDataset(store)
        assert np.allclose(image[0], full[0]["image"][2])
        assert np.allclose(image[1], full[0]["image"][1])
        assert np.allclose(image[2], full[0]["image"][0])

    def test_a_band_that_was_not_stored_is_refused(self, store):
        with pytest.raises(AtarraError, match="Rebuild the store"):
            TileStoreDataset(store, band_names=["B01"])

    def test_geometric_split_applies_unchanged(self, store):
        """Duck-typing on `.records` and `.block` means no parallel implementation."""
        dataset = TileStoreDataset(store)
        splits = geometric_split(dataset)
        collected = sorted(splits["train"] + splits["val"] + splits["test"])
        assert collected == list(range(len(dataset))), "every tile lands in exactly one split"
        assert len(set(collected)) == len(collected)
        assert all(splits[name] for name in ("train", "val", "test"))

    def test_loss_weights_come_from_a_subset_when_asked(self, store):
        """Weighting from the whole store would leak the held-out class balance."""
        dataset = TileStoreDataset(store)
        whole = dataset.class_counts()
        first = dataset.class_counts(indices=[0])
        assert (first <= whole).all()
        assert first.sum() > 0

        weights = dataset.class_weights(indices=list(range(len(dataset))))
        assert weights.shape == (NUM_CLASSES,)
        assert np.isclose(weights.mean(), 1.0, atol=1e-5), "weights are normalised"

    def test_band_statistics_are_finite_and_ordered(self, store):
        stats = TileStoreDataset(store, band_names=["B04", "B03", "B02"]).band_statistics()
        assert stats["bands"] == ["B04", "B03", "B02"]
        assert len(stats["mean"]) == 3 and len(stats["std"]) == 3
        assert all(np.isfinite(v) for v in stats["mean"] + stats["std"])
        # A zero std would divide by zero inside the model's normalisation.
        assert all(v > 0 for v in stats["std"])

    def test_review_mask_is_persisted_as_the_annotation_worklist(
        self, store, veg_water_stack
    ):
        """The annotation queue must survive the session that built it.

        Re-deriving it later would mean re-fetching the imagery it came from. The mask
        is compared against a fresh labelling pass over the same window, because a
        plausible-looking mask that drifted from the rules would send annotators to the
        wrong pixels.
        """
        from atarra.datasets.weak_labels import weak_label

        dataset = TileStoreDataset(store)
        mask = dataset.review_at(0)
        assert mask is not None
        assert mask.dtype == np.bool_
        assert mask.shape == (64, 64)

        fresh = weak_label(compute_indices(veg_water_stack), valid=veg_water_stack.valid)
        record = dataset.records[0]
        assert record.row is not None and record.col is not None
        window = fresh["review"][record.row : record.row + 64, record.col : record.col + 64]
        assert np.array_equal(mask, window), "the stored mask must match the rule engine"

        # This fixture is a crisp water/vegetation split, so the rules are confident
        # throughout and the queue is legitimately empty. Report what was measured.
        assert dataset.review_fraction() == 0.0

    def test_a_tile_can_be_georeferenced_for_annotation(self, store):
        """A worklist that cannot be located on a map is unusable."""
        dataset = TileStoreDataset(store)
        corners = dataset.tile_corners_wgs84(0)
        assert len(corners) == 4
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        # Every corner must land inside the AOI the store was built from.
        assert 30.79 <= min(lons) and max(lons) <= 30.86
        assert 31.44 <= min(lats) and max(lats) <= 31.51
        # 64 px at 10 m is 640 m, which is a few thousandths of a degree here.
        assert 0.001 < (max(lons) - min(lons)) < 0.02

    def test_annotation_worklist_is_ranked_by_need(
        self, tmp_path, patch_composite, synthetic_stack, simple_grid
    ):
        """Effort must go where the rules are unsure, not where they are confident."""
        patch_composite(_composite(synthetic_stack, simple_grid))
        build_store(
            tmp_path / "noisy",
            "test_area",
            [date(2024, 8, 20)],
            gsd=10.0,
            tile_size=64,
            stride=32,
        )
        dataset = TileStoreDataset(tmp_path / "noisy")
        worklist = dataset.annotation_tiles(limit=5)

        assert worklist, "random spectra must leave pixels the rules cannot call"
        fractions = [item["review_fraction"] for item in worklist]
        assert fractions == sorted(fractions, reverse=True), "highest need first"
        assert all(len(item["corners_wgs84"]) == 4 for item in worklist)
        assert all(item["date"] == "2024-08-20" for item in worklist)

    def test_a_store_without_the_review_mask_still_trains(self, store):
        """Stores written before the mask existed must degrade, not break."""
        import os

        for entry in TileStoreDataset(store).manifest["shards"]:
            path = store / "shards" / entry["date"] / "review.npy"
            if path.exists():
                os.remove(path)

        dataset = TileStoreDataset(store)
        assert len(dataset) > 0
        assert dataset[0]["image"].shape[0] == 8
        assert dataset.review_at(0) is None

    def test_a_directory_without_a_manifest_is_reported_clearly(self, tmp_path):
        with pytest.raises(AtarraError, match="not a tile store"):
            TileStoreDataset(tmp_path)

    def test_a_future_format_version_is_refused(self, store):
        manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
        manifest["format_version"] = 99
        (store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(AtarraError, match="not supported"):
            TileStoreDataset(store)
