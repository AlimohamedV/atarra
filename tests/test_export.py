"""Tests for annotation pack export, and for scoring against hand labels.

The property that matters most here is the refusal: a scorer that quietly falls back to
the rule engine's own guesses would produce exactly the circular number a held-out
human-labelled set exists to replace.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from atarra.core.errors import AtarraError
from atarra.datasets.export import (
    MIN_REED_PIXELS_FOR_IOU,
    UNLABELLED,
    _assess,
    export_annotation_pack,
    load_annotations,
    load_pack,
    reserved_keys,
    score_annotation_pack,
)
from atarra.datasets.store import TileStoreDataset, build_store
from atarra.datasets.weak_labels import CLASS_NAMES, PHRAGMITES_CODE
from atarra.pipeline import Composite
from atarra.preprocess.indices import compute_indices

BANDS = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"]


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


def _install(monkeypatch, *composites):
    """Replace the archive fetch with supplied composites, in order."""
    import atarra.pipeline as pipeline_module

    sequence = list(composites)
    calls = {"n": 0}

    def fake(area_key, target_date, **kwargs):
        value = sequence[min(calls["n"], len(sequence) - 1)]
        calls["n"] += 1
        return value

    monkeypatch.setattr(pipeline_module, "load_composite", fake)


def _tiny_checkpoint(path, *, in_channels: int):
    """Write a real checkpoint without spending time training.

    Deliberately goes through the trainer's own saver: hand-building the dict would let
    the test pass against a format the trainer does not actually produce.
    """
    from atarra.models.segmentation import build_model
    from atarra.train.trainer import TrainConfig, _save_checkpoint

    model = build_model(in_channels=in_channels)
    _save_checkpoint(path, model, TrainConfig(), 1, {"mean_iou": 0.0}, None)
    return model


@pytest.fixture
def store(tmp_path, monkeypatch, synthetic_stack, store_grid):
    """A store whose tiles the rules are not confident about, so review is non-empty."""
    _install(monkeypatch, _composite(synthetic_stack, store_grid))
    build_store(
        tmp_path / "store",
        "test_area",
        [date(2024, 8, 20)],
        gsd=10.0,
        tile_size=64,
        stride=64,
    )
    return tmp_path / "store"


@pytest.fixture
def pack(tmp_path, store):
    return export_annotation_pack(store, tmp_path / "pack", limit=3)


class TestExport:
    def test_writes_chips_an_index_and_a_readme(self, tmp_path, pack):
        pack_dir = tmp_path / "pack"
        assert (pack_dir / "pack.json").exists()
        assert (pack_dir / "README.md").exists()
        assert (pack_dir / "labels").is_dir(), "the annotator needs somewhere to write"
        assert len(list((pack_dir / "chips").glob("*.tif"))) == len(pack["tiles"]) * 3

        index = json.loads((pack_dir / "annotation.geojson").read_text(encoding="utf-8"))
        assert index["type"] == "FeatureCollection"
        assert len(index["features"]) == len(pack["tiles"])
        for feature in index["features"]:
            ring = feature["geometry"]["coordinates"][0]
            assert ring[0] == ring[-1], "an unclosed ring is not valid GeoJSON"
            assert len(ring) == 5
            assert feature["properties"]["reed_px"] >= 0
            assert feature["properties"]["label_output"].startswith("labels/")

    def test_chips_are_georeferenced(self, tmp_path, pack):
        import rasterio

        pack_dir = tmp_path / "pack"
        for tile in pack["tiles"]:
            with rasterio.open(pack_dir / tile["image_chip"]) as source:
                assert source.crs is not None, "a chip without a CRS cannot be placed"
                assert source.width == source.height == 64
                assert source.count == len(BANDS)
                assert list(source.descriptions) == BANDS
                # The declared corners must be where the raster actually is.
                left, top = source.transform.c, source.transform.f
                assert left != 0 and top != 0, "an identity transform places the chip at 0,0"

    def test_chip_filenames_do_not_collide_across_dates(
        self, tmp_path, monkeypatch, synthetic_stack, store_grid
    ):
        """Regression test for shared tile keys between shards.

        Two dates used to produce identically named chips, so the second date silently
        overwrote the first -- and the annotator's label file would have been shared
        between two different tiles.
        """
        _install(monkeypatch, _composite(synthetic_stack, store_grid))
        store = tmp_path / "store"
        build_store(
            store,
            "test_area",
            [date(2024, 8, 20), date(2024, 8, 25)],
            gsd=10.0,
            tile_size=64,
            stride=64,
        )
        pack = export_annotation_pack(store, tmp_path / "pack", limit=4)

        stems = [tile["filename_stem"] for tile in pack["tiles"]]
        assert len(set(stems)) == len(stems), "two reserved tiles share a filename"
        assert len(set(tile["date"] for tile in pack["tiles"])) > 1, (
            "the fixture must span two dates to exercise the collision"
        )
        chips = sorted(p.name for p in (tmp_path / "pack" / "chips").glob("*_weak_labels.tif"))
        assert len(chips) == len(pack["tiles"])

    def test_refuses_to_overwrite_an_existing_pack(self, tmp_path, store):
        export_annotation_pack(store, tmp_path / "pack", limit=1)
        with pytest.raises(AtarraError, match="not empty"):
            export_annotation_pack(store, tmp_path / "pack", limit=1)
        # ...and can be told to replace it.
        export_annotation_pack(store, tmp_path / "pack", limit=1, overwrite=True)

    def test_refuses_an_unknown_strategy(self, tmp_path, store):
        with pytest.raises(AtarraError, match="unknown annotation strategy"):
            export_annotation_pack(store, tmp_path / "pack", strategy="vibes")

    def test_warns_when_too_few_reed_pixels_to_measure(self, tmp_path, store, caplog):
        """A test set without the target class cannot measure it."""
        pack = export_annotation_pack(store, tmp_path / "pack", limit=2)
        assert pack["reed_px_total"] >= 0
        assert pack["reed_px_sufficient"] == (
            pack["reed_px_total"] >= MIN_REED_PIXELS_FOR_IOU
        )
        if not pack["reed_px_sufficient"]:
            assert "reed IoU" in caplog.text

    def test_reed_strategy_selects_tiles_containing_the_class(self, tmp_path, store):
        pack = export_annotation_pack(
            store, tmp_path / "pack", limit=2, strategy="reed"
        )
        assert pack["strategy"] == "reed"
        assert all(tile["reed_px"] > 0 for tile in pack["tiles"])

    def test_reserved_keys_describe_the_pack(self, tmp_path, pack):
        keys = reserved_keys(tmp_path / "pack")
        assert keys == {tile["key"] for tile in pack["tiles"]}
        assert set(load_pack(tmp_path / "pack")["reserved_keys"]) == keys


class TestHoldout:
    def test_reserved_tiles_are_removed_from_the_dataset(self, tmp_path, pack, store):
        """A held-out set the model trained on is not held out."""
        every = TileStoreDataset(store)
        reserved = reserved_keys(tmp_path / "pack")
        assert reserved, "the pack must reserve something"

        held = TileStoreDataset(store, holdout_keys=reserved)
        assert len(held) == len(every) - len(reserved)
        assert {record.key for record in held.records}.isdisjoint(reserved)

    def test_excluding_an_empty_pack_is_refused(self, tmp_path, store):
        """Silently excluding nothing would make the holdout meaningless."""
        from atarra.train.run import train_from_store

        pytest.importorskip("torch")

        pack_dir = tmp_path / "empty_pack"
        pack_dir.mkdir()
        (pack_dir / "pack.json").write_text(
            json.dumps({"format_version": 1, "reserved_keys": []}), encoding="utf-8"
        )

        with pytest.raises(AtarraError, match="reserves no tiles"):
            train_from_store(store, epochs=1, exclude_pack=pack_dir)


def _report(**support_by_class) -> dict:
    """A minimal segmentation report carrying only per-class support."""
    return {
        "per_class": [
            {"class_name": name, "support_px": support_by_class.get(name, 0)}
            for name in CLASS_NAMES
        ]
    }


class TestAssessment:
    """Whether a score is a validation at all, decided before anyone quotes it.

    Every case here is a way to get a good-looking number from a model that has
    learned nothing.
    """

    def test_a_single_class_annotation_is_not_a_validation(self):
        """A model predicting that one class everywhere scores a perfect IoU."""
        result = _assess(_report(phragmites_australis=32768), {PHRAGMITES_CODE})
        assert result["assessable"] is False
        assert any("only 1 class" in reason for reason in result["blocking_reasons"])
        assert result["verdict"].startswith("NOT A VALIDATION")

    def test_too_few_reed_pixels_is_not_a_validation(self):
        report = _report(open_water=4000, crops_soil=4000, phragmites_australis=40)
        result = _assess(report, {0, 1})
        assert result["assessable"] is False
        assert any("reed pixel" in reason for reason in result["blocking_reasons"])
        assert result["reed_support_px"] == 40

    def test_a_degenerate_prediction_is_caught(self):
        """One predicted class across a multi-class annotation is not a segmentation."""
        report = _report(open_water=2000, phragmites_australis=2000)
        result = _assess(report, {0})
        assert result["assessable"] is False
        assert any("single class" in reason for reason in result["blocking_reasons"])

    def test_a_covered_annotation_is_assessable(self):
        report = _report(open_water=2000, crops_soil=1000, phragmites_australis=2048)
        result = _assess(report, {0, 1, 3})
        assert result["assessable"] is True
        assert result["blocking_reasons"] == []
        assert result["classes_absent"] == ["mixed_halophytes"]
        assert result["verdict"].startswith("Measured against 3 annotated class")


class TestLabels:
    def test_scoring_refuses_when_nothing_has_been_annotated(self, tmp_path, pack):
        """The central safety property: never fall back to the rule engine."""
        pytest.importorskip("torch")

        checkpoint = tmp_path / "best.pt"
        _tiny_checkpoint(checkpoint, in_channels=len(BANDS))

        with pytest.raises(AtarraError, match="nothing in .* annotated"):
            score_annotation_pack(tmp_path / "pack", checkpoint)

    def test_polygon_labels_are_reprojected_onto_the_chip(self, tmp_path, pack):
        """GeoJSON is WGS84; the chip is UTM. Rasterising without reprojecting is silent.

        A polygon covering half the tile must label about half its pixels. Without the
        reprojection the coordinates land nowhere near the chip and every pixel stays
        unlabelled, which the scorer reports as "nothing annotated".
        """
        pytest.importorskip("torch")

        pack_dir = tmp_path / "pack"
        tile = pack["tiles"][0]
        corners = tile["corners_wgs84"]  # TL, TR, BR, BL
        mid_top = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2]
        mid_bottom = [(corners[3][0] + corners[2][0]) / 2, (corners[3][1] + corners[2][1]) / 2]
        half = [corners[0], mid_top, mid_bottom, corners[3], corners[0]]

        (pack_dir / "labels" / f"{tile['filename_stem']}.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"class_code": PHRAGMITES_CODE},
                            "geometry": {"type": "Polygon", "coordinates": [half]},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        import rasterio

        with rasterio.open(pack_dir / tile["image_chip"]) as source:
            shape = (source.height, source.width)
            transform = source.transform
            crs = source.crs

        labels = load_annotations(
            pack_dir, tile["key"], shape=shape, transform=transform, crs=crs
        )
        assert labels is not None
        labelled = int((labels < UNLABELLED).sum())
        coverage = labelled / labels.size
        assert 0.4 < coverage < 0.6, f"expected about half the tile, got {coverage:.1%}"
        assert set(np.unique(labels[labels < UNLABELLED])) == {PHRAGMITES_CODE}

    def test_scoring_reports_against_the_human_labels(self, tmp_path, pack):
        pytest.importorskip("torch")

        pack_dir = tmp_path / "pack"
        tile = pack["tiles"][0]
        # Label everything as crops: the metric must reflect that, whatever the model says.
        label_raster = np.full((64, 64), CLASS_NAMES.index("crops_soil"), dtype=np.uint8)
        import rasterio

        with rasterio.open(pack_dir / tile["image_chip"]) as source:
            profile = {
                "driver": "GTiff",
                "height": 64,
                "width": 64,
                "count": 1,
                "dtype": "uint8",
                "crs": source.crs,
                "transform": source.transform,
            }
        with rasterio.open(
            pack_dir / "labels" / f"{tile['filename_stem']}.tif", "w", **profile
        ) as destination:
            destination.write(label_raster, 1)

        checkpoint = tmp_path / "best.pt"
        _tiny_checkpoint(checkpoint, in_channels=len(BANDS))

        result = score_annotation_pack(pack_dir, checkpoint)
        assert result["tiles_annotated"] == 1
        assert len(result["tiles_still_unlabelled"]) == len(pack["tiles"]) - 1
        crops = next(
            entry for entry in result["report"]["per_class"] if entry["class_name"] == "crops_soil"
        )
        assert crops["support_px"] == 64 * 64, "every pixel was labelled as crops"
        assert result["truth"].startswith("Scored against human annotations")
        assert (pack_dir / "score.json").exists()
        # One class annotated, so the tool must refuse to call this a validation --
        # otherwise a single-class label file reads as a passing grade.
        assert result["assessable"] is False
        assert result["assessment"]["classes_present"] == ["crops_soil"]

    def test_a_label_raster_on_the_wrong_grid_is_refused(self, tmp_path, pack):
        pytest.importorskip("torch")

        pack_dir = tmp_path / "pack"
        tile = pack["tiles"][0]
        import rasterio

        with rasterio.open(pack_dir / tile["image_chip"]) as source:
            profile = {
                "driver": "GTiff",
                "height": 32,
                "width": 32,
                "count": 1,
                "dtype": "uint8",
                "crs": source.crs,
                "transform": source.transform,
            }
        with rasterio.open(
            pack_dir / "labels" / f"{tile['filename_stem']}.tif", "w", **profile
        ) as destination:
            destination.write(np.zeros((32, 32), dtype=np.uint8), 1)

        with rasterio.open(pack_dir / tile["image_chip"]) as source:
            with pytest.raises(AtarraError, match="must be on the chip's grid"):
                load_annotations(
                    pack_dir,
                    tile["key"],
                    shape=(source.height, source.width),
                    transform=source.transform,
                    crs=source.crs,
                )

    def test_a_feature_without_a_class_is_refused(self, tmp_path, pack):
        """Silently treating a missing class as 0 would score it as open water."""
        pack_dir = tmp_path / "pack"
        tile = pack["tiles"][0]
        (pack_dir / "labels" / f"{tile['filename_stem']}.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [tile["corners_wgs84"] + [tile["corners_wgs84"][0]]],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        from affine import Affine

        with pytest.raises(AtarraError, match="no class"):
            # The transform is never reached: a feature with no class is refused before
            # anything is rasterised.
            load_annotations(
                pack_dir,
                tile["key"],
                shape=(64, 64),
                transform=Affine.identity(),
                crs="EPSG:32636",
            )
