"""Regression tests for the properties that make a reported score mean something.

Every test here corresponds to a way this pipeline could produce a convincing number
that measures nothing:

* test tiles sharing ground with training tiles, so the score is partly recall;
* a final report computed from weights the run itself discarded;
* annotated chips read on a scale four orders of magnitude off the training inputs;
* normalisation statistics drawn from imagery the model is scored on;
* augmented validation inputs, so the metric moves without the model moving;
* a holdout that covers one date of five, leaving the same reed bed in training;
* a target verdict nobody earned, printed from a file that does not say so;
* an initialisation seeded after construction, so no seed reproduces a run;
* an epoch whose final gradients were accumulated and then silently dropped.

The store-based tests share one small two-date store and one training run: they are
the slowest thing here and the properties are about that run, not about fresh ones.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from rasterio.windows import Window

from atarra.core.errors import AtarraError
from atarra.core.grids import BBox, grid_from_bbox
from atarra.datasets.export import (
    MIN_REED_PIXELS_FOR_IOU,
    _as_reflectance,
    export_annotation_pack,
    load_pack,
    score_annotation_pack,
)
from atarra.datasets.spatial import (
    SpatialSplit,
    assert_disjoint,
    intersecting_bounds,
    overlapping_pairs,
    split_bounds,
    tile_bounds,
)
from atarra.datasets.store import TileStoreDataset, build_store
from atarra.datasets.tile_dataset import geometric_split
from atarra.datasets.weak_labels import NUM_CLASSES, PHRAGMITES_CODE
from atarra.pipeline import Composite
from atarra.preprocess.indices import compute_indices
from atarra.preprocess.reader import BandStack

BANDS = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"]
TILE = 48


# --------------------------------------------------------------------------------------
# synthetic inputs
# --------------------------------------------------------------------------------------


def _integrity_grid():
    """A ~3 x 3 km AOI at 10 m: enough rows for three geographic strips, still cheap.

    The standard test grid is too small for a two-cut geographic split with a buffer --
    which is itself worth knowing, so the size is chosen here rather than reduced until
    the split stops being tested.
    """
    return grid_from_bbox(
        BBox.from_sequence([30.80, 31.45, 30.83, 31.48]), "EPSG:32636", 10.0
    )


def _structured_stack(grid, *, valid_fraction: float = 1.0):
    """Water on the right, dense reed on the left, both confidently labelled.

    Random reflectance would leave the rule engine unsure everywhere, so every pixel
    would be either review-flagged or untrainable and the run would have no supervision
    at all. The bands are chosen as in the physics: water is darker in NIR than in the
    red edge (setting them equal gives NDRE == 0, which is neither physical nor useful).
    """
    height, width = grid.height, grid.width
    half = width // 2
    green = np.full((height, width), 0.05, dtype=np.float32)
    red = np.full((height, width), 0.04, dtype=np.float32)
    red_edge = np.full((height, width), 0.035, dtype=np.float32)
    nir = np.full((height, width), 0.025, dtype=np.float32)
    swir = np.full((height, width), 0.04, dtype=np.float32)

    # Values chosen against `WeakLabelConfig`: NDVI 0.83, NDRE 0.55, NDMI 0.36,
    # NDWI -0.68. All four reed signals clear their thresholds, so the rule engine is
    # confident rather than deferring to review -- which matters because a store of
    # entirely ambiguous pixels trains on nothing and the run refuses to start.
    nir[:, :half] = 0.42
    green[:, :half] = 0.08
    red_edge[:, :half] = 0.12
    swir[:, :half] = 0.20

    valid = np.ones((height, width), dtype=bool)
    if valid_fraction < 1.0:
        # A rotated swath leaves an empty corner, as real imagery does.
        valid[int(height * (1.0 - valid_fraction)) :, :] = False

    data = np.stack(
        [np.full((height, width), 0.06, np.float32), green, red, red_edge, nir, nir, swir,
         swir]
    )
    data[:, ~valid] = np.nan
    return BandStack(
        data=data,
        valid=valid,
        grid=grid,
        band_names=list(BANDS),
        window=Window(0, 0, width, height),
        scene_ids=["synthetic"],
    )


def _composite(stack, grid, when: date = date(2024, 8, 20)) -> Composite:
    return Composite(
        area_key="integrity_area",
        area_name="Integrity Area",
        target_date=when,
        window_days=3,
        grid=grid,
        stack=stack,
        indices=compute_indices(stack),
        scene_ids=["S2A_TEST"],
        scene_details=[{"id": "S2A_TEST", "cloud_cover": 1.0}],
        reflectance={"mode": "dn_scale", "negative_fraction": 0.0, "median": 0.2},
    )


def _footprints(positions, size: int, *, dates: int = 1) -> np.ndarray:
    """Footprints of a regularly tiled AOI, repeated once per date, as a store holds them."""
    base = np.array(
        [(row, col, row + size, col + size) for row in positions for col in positions],
        dtype=np.int64,
    )
    if dates == 1:
        return base
    return np.concatenate([base] * dates, axis=0)


def _fingerprint(model) -> str:
    """A hash of the weights, to compare a model with a checkpoint file."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class _BatchLoader:
    """A DataLoader-shaped object over explicit batches, so tests control them exactly."""

    def __init__(self, batches):
        self.batches = list(batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def _batch(torch, *, size=2, channels=len(BANDS), pixels=32, supervised=True) -> dict:
    """One batch: half water, half reed, or nothing supervised at all."""
    rng = np.random.default_rng(0)
    image = torch.from_numpy(rng.random((size, channels, pixels, pixels), dtype=np.float32))
    mask = np.full((size, pixels, pixels), -1, dtype=np.int64)
    if supervised:
        half = pixels // 2
        mask[:, :half, :] = 0
        mask[:, half:, :] = PHRAGMITES_CODE
    return {"image": image, "mask": torch.from_numpy(mask)}


def _tiny_model(**kwargs):
    from atarra.models.segmentation import build_model

    return build_model(base_channels=8, depth=2, **kwargs)


# --------------------------------------------------------------------------------------
# 1. splits partition ground, not tile lists
# --------------------------------------------------------------------------------------


class TestGeographicSplit:
    """Overlapping tiles are the reason a random split is worthless in remote sensing."""

    def test_no_split_shares_a_pixel(self):
        positions = list(range(0, 1024, 128))  # 256 px tiles, stride 128: they overlap
        bounds = _footprints(positions, 256)
        splits = split_bounds(
            bounds, fractions=(0.70, 0.15, 0.15), seed=0, buffer_pixels=128
        )
        assert_disjoint(bounds, splits)  # raises if any pair shares a pixel

        names = list(splits)
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                shared = overlapping_pairs(
                    bounds[splits[names[first]]], bounds[splits[names[second]]]
                )
                assert shared == [], f"{names[first]}/{names[second]} share {shared[:3]}"

    def test_the_check_has_teeth(self):
        """The scheme this replaced must fail the same check, or it tests nothing.

        Whole ``(row // tile_size, col // tile_size)`` blocks assigned at random look
        spatial and are not: two tiles one block apart still share half their pixels.
        """
        positions = list(range(0, 1024, 128))
        bounds = _footprints(positions, 256)
        offender = None
        for seed in range(10):
            groups = _block_grouping(bounds, size=256, fractions=(0.70, 0.15, 0.15), seed=seed)
            for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
                if overlapping_pairs(bounds[groups[first]], bounds[groups[second]]):
                    offender = (seed, first, second)
                    break
            if offender:
                break
        assert offender, "the block split was expected to leak; the guard may be untested"

    def test_a_new_date_cannot_move_ground_between_splits(self):
        """Cuts follow footprints, so date 9 lands wherever date 1 does."""
        positions = list(range(0, 1024, 128))
        one = split_bounds(
            _footprints(positions, 256), fractions=(0.7, 0.15, 0.15), seed=0, buffer_pixels=128
        )
        many_bounds = _footprints(positions, 256, dates=5)
        many = split_bounds(
            many_bounds, fractions=(0.7, 0.15, 0.15), seed=0, buffer_pixels=128
        )

        base = _footprints(positions, 256)
        for name in one:
            expected = {tuple(row) for row in base[one[name]]}
            actual = {tuple(row) for row in many_bounds[many[name]]}
            assert actual == expected, f"{name} differs once dates are added"
            assert len(many[name]) == 5 * len(one[name])

    def test_splits_are_separated_by_the_buffer(self):
        positions = list(range(0, 1024, 128))
        bounds = _footprints(positions, 256)
        buffer = 128
        splits = split_bounds(
            bounds, fractions=(0.7, 0.15, 0.15), seed=1, buffer_pixels=buffer
        )
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            for a in bounds[splits[first]]:
                for b in bounds[splits[second]]:
                    row_gap = max(int(a[0]) - int(b[2]), int(b[0]) - int(a[2]), 0)
                    col_gap = max(int(a[1]) - int(b[3]), int(b[1]) - int(a[3]), 0)
                    assert max(row_gap, col_gap) >= buffer

    def test_a_store_too_small_to_split_is_refused(self):
        with pytest.raises(AtarraError, match="at least 3 distinct tile footprints"):
            split_bounds(
                _footprints([0], 256), fractions=(0.7, 0.15, 0.15), seed=0, buffer_pixels=0
            )

    def test_a_narrow_buffer_is_used_and_reported(self):
        """Disjointness is the invariant; the gap is a knob, so it narrows instead."""
        dataset = _StubDataset(rows=3, cols=2, size=64)
        splits = geometric_split(dataset, seed=0)
        assert isinstance(splits, SpatialSplit)
        assert splits.buffer_pixels < 32, "half a tile did not fit and must have narrowed"
        assert splits.buffer_pixels >= 0
        assert all(splits[name] for name in ("train", "val", "test"))
        assert_disjoint(tile_bounds(dataset), splits)

    def test_an_explicit_buffer_is_never_widened(self):
        dataset = _StubDataset(rows=12, cols=12, size=64)
        splits = geometric_split(dataset, seed=0, buffer_pixels=16)
        assert splits.buffer_pixels == 16


class _StubDataset:
    """The minimum a splitter needs: records and a tile size."""

    def __init__(self, *, rows: int, cols: int, size: int) -> None:
        from types import SimpleNamespace

        self.tile_size = size
        self.records = [
            SimpleNamespace(row=row * size, col=col * size, size=size)
            for row in range(rows)
            for col in range(cols)
        ]


def _block_grouping(bounds, *, size: int, fractions, seed: int) -> dict[str, list[int]]:
    """The superseded scheme, reproduced so the guard can be shown to catch it."""
    blocks: dict[tuple[int, int], list[int]] = {}
    for index, (top, left, _, _) in enumerate(bounds):
        blocks.setdefault((int(top) // size, int(left) // size), []).append(index)
    keys = sorted(blocks)
    order = np.random.default_rng(seed).permutation(len(keys))
    n_train = max(1, min(int(round(fractions[0] * len(keys))), len(keys) - 2))
    n_val = max(1, min(int(round(fractions[1] * len(keys))), len(keys) - n_train - 1))
    grouped = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    result: dict[str, list[int]] = {}
    for name, positions in grouped.items():
        indices: list[int] = []
        for position in positions:
            indices.extend(blocks[keys[int(position)]])
        result[name] = sorted(indices)
    return result


# --------------------------------------------------------------------------------------
# 2. the holdout covers ground, on every date
# --------------------------------------------------------------------------------------


class TestHoldoutGround:
    def test_a_reserved_footprint_is_matched_on_other_dates(self):
        positions = [0, 128, 256]
        bounds = _footprints(positions, 128, dates=3)
        reserved = bounds[[0]]  # one tile on the first date only
        excluded = intersecting_bounds(bounds, reserved, buffer_pixels=0)
        # The same ground on the other two dates, and nowhere else: 9 footprints per date,
        # of which exactly one matches the reserved position.
        assert excluded.tolist() == ([True] + [False] * 8) * 3

    def test_neighbours_within_the_buffer_are_excluded(self):
        bounds = np.array([[0, 0, 64, 64], [64, 0, 128, 64], [300, 0, 364, 64]])
        reserved = bounds[[0]]
        assert intersecting_bounds(bounds, reserved, buffer_pixels=16).tolist() == [
            True,
            True,
            False,
        ]

    def test_the_store_drops_the_same_ground_on_every_date(self, integrity_store):
        """A held-out reed bed is held out in September too."""
        every = TileStoreDataset(integrity_store)
        dates = {record.key.split("/")[0] for record in every.records}
        assert len(dates) > 1, "this store needs two dates for the property to mean anything"

        first_date = sorted(dates)[0]
        anchor = next(r for r in every.records if r.key.startswith(first_date))
        held = TileStoreDataset(integrity_store, holdout_keys=[anchor.key])

        surviving = {record.key for record in held.records}
        # The same position on the other date must be gone even though only one key
        # was named: `holdout_keys` is matched by footprint, not by string.
        same_place = [r.key for r in every.records if r.row == anchor.row and r.col == anchor.col]
        assert len(same_place) > 1
        assert not (set(same_place) & surviving)
        assert set(same_place) - {anchor.key} <= set(held.excluded_keys)

    def test_a_reserved_key_missing_from_the_store_is_refused(self, integrity_store):
        with pytest.raises(AtarraError, match="reserved tiles are missing"):
            TileStoreDataset(integrity_store, holdout_keys=["1999-01-01/c0/r0_c0"])

    def test_a_holdout_that_eats_the_store_names_that_cause(self, integrity_store):
        """Found on a real 6-tile smoke store: the pack took every tile with it.

        The failure then surfaced as "too few footprints to split", which points at the
        AOI rather than at the holdout and would send someone to widen an area that was
        never the problem.
        """
        every = TileStoreDataset(integrity_store)
        with pytest.raises(AtarraError, match="covers this whole store"):
            TileStoreDataset(
                integrity_store,
                holdout_keys=[every.records[0].key],
                holdout_buffer_pixels=100_000,
            )


# --------------------------------------------------------------------------------------
# 3. the report describes the checkpoint, not the last epoch
# --------------------------------------------------------------------------------------


class TestBestWeightsAreEvaluated:
    def test_the_final_report_is_the_saved_checkpoint(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")

        from atarra.train import trainer as trainer_module

        real_evaluate = trainer_module.evaluate
        # Script the selection signal so the curve is known exactly: the best epoch is
        # the first, and the last epoch is the worst. Whatever the final report says,
        # it must describe the epoch-1 weights the run kept.
        script = [0.60, 0.40, 0.30]
        seen: list[str] = []
        calls = {"n": 0}

        def scripted(model, loader, **kwargs):
            seen.append(_fingerprint(model))
            report = real_evaluate(model, loader, **kwargs)
            index = calls["n"]
            calls["n"] += 1
            if index < len(script):
                report["mean_iou"] = script[index]
            return report

        monkeypatch.setattr(trainer_module, "evaluate", scripted)

        model = _tiny_model(in_channels=len(BANDS))
        train_loader = _BatchLoader([_batch(torch), _batch(torch, size=1)])
        val_loader = _BatchLoader([_batch(torch)])
        run_dir = tmp_path / "run"
        config = trainer_module.TrainConfig(
            epochs=3, batch_size=2, seed=5, patience=9, output_dir=run_dir
        )

        summary = trainer_module.train(
            model,
            train_loader,
            val_loader,
            config=config,
            num_classes=NUM_CLASSES,
            device=torch.device("cpu"),
        )

        assert summary["epochs_run"] == 3
        assert summary["best_epoch"] == 1
        assert summary["evaluated_epoch"] == 1
        assert summary["best_val_mean_iou"] == 0.60
        assert summary["history"][-1]["val_mean_iou"] == 0.30

        # The weights the run rejected must differ from the ones it kept, or the test
        # would pass on a trainer that never restored anything.
        assert seen[2] != seen[0]

        saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor.cpu(), saved["model_state"][name].cpu()), name
        assert _fingerprint(model) == seen[0]

        # And the report is of *that* model: recomputing it agrees.
        recomputed = real_evaluate(
            model, val_loader, num_classes=NUM_CLASSES, device=torch.device("cpu")
        )
        assert summary["final_report"]["mean_iou"] == recomputed["mean_iou"]


# --------------------------------------------------------------------------------------
# 4. no supervision, no gradient
# --------------------------------------------------------------------------------------


class TestSupervisionGuards:
    def test_a_split_with_no_supervision_fails_loudly(self, tmp_path):
        torch = pytest.importorskip("torch")

        from atarra.train import trainer as trainer_module

        config = trainer_module.TrainConfig(epochs=2, output_dir=tmp_path / "run")
        with pytest.raises(AtarraError, match="no batch containing a single supervised pixel"):
            trainer_module.train(
                _tiny_model(in_channels=len(BANDS)),
                _BatchLoader([_batch(torch, supervised=False)]),
                _BatchLoader([_batch(torch, supervised=False)]),
                config=config,
                num_classes=NUM_CLASSES,
                device=torch.device("cpu"),
            )

    def test_an_unsupervised_batch_is_skipped_not_learned_from(self, tmp_path):
        """Cross-entropy over an empty label set is NaN; skipping must not end the epoch."""
        torch = pytest.importorskip("torch")

        from atarra.train import trainer as trainer_module

        config = trainer_module.TrainConfig(
            epochs=1, output_dir=tmp_path / "run", patience=1
        )
        summary = trainer_module.train(
            _tiny_model(in_channels=len(BANDS)),
            _BatchLoader([_batch(torch, supervised=False), _batch(torch)]),
            _BatchLoader([_batch(torch)]),
            config=config,
            num_classes=NUM_CLASSES,
            device=torch.device("cpu"),
        )
        entry = summary["history"][0]
        assert entry["train_batches_skipped"] == 1
        assert entry["train_batches"] == 1
        assert np.isfinite(entry["train_loss"])

    def test_a_non_finite_loss_stops_the_run(self, tmp_path):
        torch = pytest.importorskip("torch")

        from atarra.train import trainer as trainer_module

        poisoned = _batch(torch)
        poisoned["image"] = torch.full_like(poisoned["image"], float("inf"))
        config = trainer_module.TrainConfig(epochs=1, output_dir=tmp_path / "run")

        with pytest.raises(AtarraError, match="non-finite loss"):
            trainer_module.train(
                _tiny_model(in_channels=len(BANDS)),
                _BatchLoader([poisoned]),
                _BatchLoader([_batch(torch)]),
                config=config,
                num_classes=NUM_CLASSES,
                device=torch.device("cpu"),
            )


# --------------------------------------------------------------------------------------
# 5. gradient accumulation
# --------------------------------------------------------------------------------------


class TestGradientAccumulation:
    def test_a_partial_group_is_scaled_to_the_batches_it_has(self):
        """Each micro-batch divided by `accumulate_steps`; a short group must be rescaled.

        Otherwise the last few batches of every epoch move the weights by a fraction of
        the mean gradient, proportionally to how few tiles were left over.
        """
        torch = pytest.importorskip("torch")

        from atarra.train.trainer import TrainConfig, _apply_step

        def moved(*, group_size: int) -> float:
            model = _tiny_model(in_channels=3)
            for parameter in model.parameters():
                parameter.data.zero_()
                parameter.grad = torch.ones_like(parameter)
            start = next(model.parameters()).detach().clone()
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0, dampening=0)
            config = TrainConfig(accumulate_steps=4, grad_clip=0.0)
            _apply_step(optimizer, None, model, config, group_size)
            return float((start - next(model.parameters()).detach()).mean())

        assert moved(group_size=4) == pytest.approx(1.0, abs=1e-6)
        assert moved(group_size=1) == pytest.approx(4.0, abs=1e-6)
        assert moved(group_size=2) == pytest.approx(2.0, abs=1e-6)

    def test_the_last_group_of_an_epoch_is_not_dropped(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")

        from atarra.train import trainer as trainer_module

        steps: list[int] = []
        original = torch.optim.AdamW.step

        def counting(self, *args, **kwargs):
            steps.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(torch.optim.AdamW, "step", counting)

        config = trainer_module.TrainConfig(
            epochs=1, accumulate_steps=3, output_dir=tmp_path / "run", patience=1
        )
        summary = trainer_module.train(
            _tiny_model(in_channels=len(BANDS)),
            _BatchLoader([_batch(torch, size=1) for _ in range(4)]),
            _BatchLoader([_batch(torch)]),
            config=config,
            num_classes=NUM_CLASSES,
            device=torch.device("cpu"),
        )
        assert summary["history"][0]["train_batches"] == 4
        assert len(steps) == 2, "3+1 micro-batches must step twice, not once"


# --------------------------------------------------------------------------------------
# 6. statistics, augmentation and seeding
# --------------------------------------------------------------------------------------


class TestStatisticsStayInsideTheTrainingSplit:
    def test_held_out_imagery_cannot_move_the_statistics(self, integrity_store, monkeypatch):
        dataset = TileStoreDataset(integrity_store, band_names=["B04", "B08"])
        splits = geometric_split(dataset, seed=0)
        train_indices = splits["train"]
        held_out = set(splits["val"]) | set(splits["test"])
        assert held_out, "this test needs a non-empty held-out set"

        before = dataset.band_statistics(indices=train_indices)
        original = dataset._image_and_mask

        def poisoned(index):
            image, mask = original(index)
            if index in held_out:
                return image * 50.0 + 0.25, mask
            return image, mask

        monkeypatch.setattr(dataset, "_image_and_mask", poisoned)
        assert dataset.band_statistics(indices=train_indices) == before

        # Control: the same interference on a training tile *does* move them, so the
        # test can tell "unaffected" from "not looking".
        def poisoned_train(index):
            image, mask = original(index)
            if index in set(train_indices):
                return image * 50.0 + 0.25, mask
            return image, mask

        monkeypatch.setattr(dataset, "_image_and_mask", poisoned_train)
        assert dataset.band_statistics(indices=train_indices) != before

    def test_statistics_are_raw_and_ignore_invalid_pixels(self, integrity_store):
        """Augmentation and nodata must not reach the normalisation buffers."""
        dataset = TileStoreDataset(integrity_store, band_names=["B04", "B08"])
        splits = geometric_split(dataset, seed=0)
        stats = dataset.band_statistics(indices=splits["train"])
        assert stats["usable_pixels"] > 0
        assert "raw" in stats["source"]
        assert stats["bands"] == ["B04", "B08"]

    def test_an_empty_selection_is_refused_rather_than_averaged(self, integrity_store):
        with pytest.raises(AtarraError, match="no tiles available"):
            TileStoreDataset(integrity_store).band_statistics(indices=[])


class TestWorkerHandoff:
    """DataLoader workers are separate processes; a dataset that cannot cross must not."""

    def test_a_view_crosses_a_process_boundary_without_the_imagery(self, integrity_store):
        """Spawn pickles the dataset into each worker; the store must stay on disk."""
        torch = pytest.importorskip("torch")
        import pickle

        dataset = TileStoreDataset(integrity_store)
        payload = pickle.dumps(dataset.as_torch_dataset(indices=[0, 1], augment=False))
        shard_bytes = int(dataset._shards[0]["images"].nbytes)
        assert len(payload) < shard_bytes, "the tiles travelled with the payload"

        restored = pickle.loads(payload)
        assert restored.indices == [0, 1]
        assert restored.augment is False
        # Reopened on the far side, and reading the same tile as the parent.
        assert int(restored.dataset._shards[0]["images"].nbytes) == shard_bytes
        assert torch.equal(
            restored[1]["image"], torch.from_numpy(dataset[1]["image"])
        )
        assert torch.equal(
            restored[1]["mask"], torch.from_numpy(dataset[1]["mask"])
        )

    def test_a_dataloader_worker_produces_batches(self, integrity_store):
        torch = pytest.importorskip("torch")

        dataset = TileStoreDataset(integrity_store)
        splits = geometric_split(dataset, seed=0)
        loader = torch.utils.data.DataLoader(
            dataset.as_torch_dataset(indices=splits["test"][:4], augment=False),
            batch_size=2,
            num_workers=1,
        )
        batches = [batch["image"] for batch in loader]
        assert len(batches) == 2
        assert batches[0].shape[0] == 2


class TestValidationIsNotAugmented:
    def test_the_evaluation_view_returns_the_stored_tile_unchanged(self, integrity_store):
        torch = pytest.importorskip("torch")

        dataset = TileStoreDataset(integrity_store)
        splits = geometric_split(dataset, seed=0)
        view = dataset.as_torch_dataset(indices=splits["val"][:2], augment=False)
        for position, index in enumerate(splits["val"][:2]):
            stored = dataset[index]
            sample = view[position]
            assert torch.equal(sample["image"], torch.from_numpy(stored["image"]))
            assert torch.equal(sample["mask"], torch.from_numpy(stored["mask"]))

    def test_the_training_view_augments_and_advances_by_epoch(self, integrity_store):
        torch = pytest.importorskip("torch")

        dataset = TileStoreDataset(integrity_store)
        splits = geometric_split(dataset, seed=0)
        indices = splits["train"][:4]
        view = dataset.as_torch_dataset(indices=indices, augment=True)

        stored = [dataset[index]["image"] for index in indices]
        view.set_epoch(1)
        first = [view[position]["image"].numpy() for position in range(len(indices))]
        view.set_epoch(2)
        second = [view[position]["image"].numpy() for position in range(len(indices))]

        assert any(
            not np.allclose(original, augmented)
            for original, augmented in zip(stored, first)
        ), "the training view is not augmenting at all"
        assert any(
            not np.allclose(a, b) for a, b in zip(first, second)
        ), "a tile is augmented identically in every epoch"

    def test_the_epoch_reaches_the_loader_during_training(self, integrity_store):
        """The trainer must advance it; a fixed transform for 40 epochs is not augmentation."""
        from atarra.train.trainer import set_loader_epoch

        view = TileStoreDataset(integrity_store).as_torch_dataset(indices=[0])
        assert view.epoch == 0
        loader = type("_Loader", (), {"dataset": view})()
        set_loader_epoch(loader, 7)
        assert view.epoch == 7
        # A loader without the hook must be tolerated, not crash the run.
        set_loader_epoch(_BatchLoader([]), 3)


class TestSeeding:
    def test_the_same_seed_gives_the_same_initial_weights(self):
        pytest.importorskip("torch")

        from atarra.train.trainer import set_seed

        set_seed(11)
        first = _tiny_model(in_channels=len(BANDS))
        set_seed(11)
        second = _tiny_model(in_channels=len(BANDS))
        assert _fingerprint(first) == _fingerprint(second)

        # And the converse: a different seed must not, or the test proves nothing.
        set_seed(12)
        third = _tiny_model(in_channels=len(BANDS))
        assert _fingerprint(third) != _fingerprint(first)

    def test_two_runs_of_one_configuration_agree(self, integrity_store, tmp_path):
        """Seeded before construction, a run is a function of its configuration."""
        pytest.importorskip("torch")

        from atarra.train.run import train_from_store

        runs = tmp_path / "runs"
        first = train_from_store(
            integrity_store,
            epochs=2,
            batch_size=4,
            out_dir=runs,
            run_name="determinism_a",
            device="cpu",
            seed=3,
            bands=["B04", "B08"],
        )
        second = train_from_store(
            integrity_store,
            epochs=2,
            batch_size=4,
            out_dir=runs,
            run_name="determinism_b",
            device="cpu",
            seed=3,
            bands=["B04", "B08"],
        )
        assert first["test_report"]["mean_iou"] == pytest.approx(
            second["test_report"]["mean_iou"], abs=1e-9
        )
        assert first["band_statistics"]["mean"] == second["band_statistics"]["mean"]
        # Each checkpoint is ~30 MB, and pytest keeps the last few temp trees: leaving
        # them would add a few hundred MB to every run of this suite, and a full disk
        # turns an assertion failure into an unrelated write error.
        shutil.rmtree(runs)


# --------------------------------------------------------------------------------------
# 7. the reflectance contract between export and scoring
# --------------------------------------------------------------------------------------


class TestChipScale:
    def test_a_chip_holds_the_reflectance_the_model_was_trained_on(self, annotation_pack):
        import rasterio

        store = TileStoreDataset(annotation_pack.store)
        by_key = {record.key: index for index, record in enumerate(store.records)}
        pack = load_pack(annotation_pack.path)
        assert pack["imagery_units"].startswith("float32 reflectance")

        checked = 0
        for tile in pack["tiles"]:
            path = annotation_pack.path / tile["image_chip"]
            with rasterio.open(path) as source:
                stack = source.read()
                descriptions = list(source.descriptions)
            assert float(stack.max()) <= 1.0 + 1e-6, "a chip is not in reflectance"
            stored = store[by_key[tile["key"]]]["image"]
            order = [descriptions.index(name) for name in pack["imagery_bands"]]
            assert np.allclose(stack[order], stored, atol=1e-4)
            checked += 1
        assert checked, "the pack exported no tiles"

    def test_stored_integer_values_are_named_rather_than_silently_decoded(self):
        stored = np.array([[[0.0, 1234.0], [5678.0, 10000.0]]], dtype=np.float32)
        with pytest.raises(AtarraError, match="not reflectance"):
            _as_reflectance(stored, source="chip_0.tif")
        # The convention the pack defines must pass.
        assert _as_reflectance(np.array([[[0.2, 0.02]]], np.float32), source="chip") is not None

    def test_the_model_receives_the_chip_it_was_given(self, annotation_pack, monkeypatch):
        """The whole point: no second decode between the file and the first layer."""
        torch = pytest.importorskip("torch")
        import rasterio

        pack = load_pack(annotation_pack.path)
        recorder = _Recorder()
        monkeypatch.setattr(
            "atarra.models.segmentation.build_model", lambda **kwargs: recorder
        )
        monkeypatch.setattr("atarra.train.trainer.load_checkpoint", lambda *a, **k: {})

        result = score_annotation_pack(
            annotation_pack.path, annotation_pack.checkpoint, write=False
        )
        assert recorder.seen, "the model was never called"
        tile = pack["tiles"][0]
        with rasterio.open(annotation_pack.path / tile["image_chip"]) as source:
            chip = source.read()
            descriptions = list(source.descriptions)
        order = [descriptions.index(name) for name in pack["imagery_bands"]]
        assert np.allclose(recorder.seen[0][0], chip[order], atol=1e-5)
        assert result["band_order_verified"] is True


class _Recorder:
    """A stand-in segmentation model that records its input and emits a chosen mask."""

    def __init__(self, left_class: int = 0) -> None:
        self.left_class = left_class
        self.seen: list[np.ndarray] = []
        self.in_channels = len(BANDS)

    def __call__(self, tensor):
        import torch

        self.seen.append(tensor.detach().cpu().numpy().copy())
        logits = torch.zeros(
            tensor.shape[0], NUM_CLASSES, tensor.shape[2], tensor.shape[3]
        )
        logits[:, self.left_class, :, : tensor.shape[3] // 2] = 1.0
        return logits


# --------------------------------------------------------------------------------------
# 8. target verdicts
# --------------------------------------------------------------------------------------


class TestTargetVerdicts:
    def test_weak_label_metrics_do_not_claim_a_verdict(self, trained_run):
        """The training number compares the model with its own teacher."""
        targets = trained_run["targets"]
        assert targets["assessable"] is False
        assert targets["both_met"] is None
        assert targets["iou_met"] is None
        assert "not assessed" in targets["reason"]

    def test_meets_targets_still_answers_for_a_real_measurement(self):
        from atarra.train.metrics import meets_targets, unassessable_targets

        report = {"phragmites_iou": 0.9, "phragmites_f1": 0.9}
        assert meets_targets(report)["both_met"] is True
        assert meets_targets(report)["assessable"] is True
        assert unassessable_targets("nothing measured")["both_met"] is None

    def test_a_rejected_assessment_publishes_no_verdict(self, annotation_pack, monkeypatch):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(
            "atarra.models.segmentation.build_model", lambda **kwargs: _Recorder(left_class=0)
        )
        monkeypatch.setattr("atarra.train.trainer.load_checkpoint", lambda *a, **k: {})

        result = score_annotation_pack(
            annotation_pack.path, annotation_pack.checkpoint, write=False
        )
        assert result["assessable"] is False
        assert result["targets"]["both_met"] is None
        assert result["targets"]["reason"]

    def test_a_clean_annotation_produces_a_verdict(self, annotation_pack, monkeypatch):
        """The same pack, answered by a model that predicts both classes, is assessable."""
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(
            "atarra.models.segmentation.build_model",
            lambda **kwargs: _Recorder(left_class=PHRAGMITES_CODE),
        )
        monkeypatch.setattr("atarra.train.trainer.load_checkpoint", lambda *a, **k: {})

        result = score_annotation_pack(
            annotation_pack.path, annotation_pack.checkpoint, write=False
        )
        assert result["assessment"]["independence"]["status"] == "verified"
        assert result["assessable"] is True
        assert result["targets"]["both_met"] is True


class TestScoreRefusals:
    def test_scoring_on_trained_ground_is_refused(self, annotation_pack, monkeypatch):
        monkeypatch.setattr(
            "atarra.train.trainer.load_checkpoint", lambda *a, **k: {}
        )
        trained = _write_checkpoint(
            annotation_pack.path / "trained_on.pt",
            train_keys=[load_pack(annotation_pack.path)["tiles"][0]["key"]],
        )
        with pytest.raises(AtarraError, match="refusing to score"):
            score_annotation_pack(annotation_pack.path, trained, write=False)

    def test_a_band_order_mismatch_is_refused(self, annotation_pack, monkeypatch):
        """Eight channels in the wrong order is not a shape error anything can catch."""
        monkeypatch.setattr("atarra.train.trainer.load_checkpoint", lambda *a, **k: {})
        wrong = _write_checkpoint(
            annotation_pack.path / "wrong_bands.pt", bands=list(reversed(BANDS))
        )
        with pytest.raises(AtarraError, match="trained on bands"):
            score_annotation_pack(annotation_pack.path, wrong, write=False)

    def test_an_unverifiable_checkpoint_still_scores_but_says_so(
        self, annotation_pack, monkeypatch
    ):
        monkeypatch.setattr(
            "atarra.models.segmentation.build_model",
            lambda **kwargs: _Recorder(left_class=PHRAGMITES_CODE),
        )
        monkeypatch.setattr("atarra.train.trainer.load_checkpoint", lambda *a, **k: {})
        bare = _write_checkpoint(annotation_pack.path / "no_provenance.pt", provenance=None)

        result = score_annotation_pack(annotation_pack.path, bare, write=False)
        assert result["assessment"]["independence"]["status"] == "unverified"
        assert result["assessable"] is False
        assert result["targets"]["both_met"] is None


def _write_checkpoint(
    path: Path, *, bands=None, train_keys=None, provenance: dict | None | str = "default"
) -> Path:
    """A minimal checkpoint carrying the metadata the scorer reads before loading weights."""
    import torch

    if provenance == "default":
        provenance = {"train_keys": train_keys or [], "val_keys": [], "reserved_keys": []}
    torch.save(
        {
            "model_state": {},
            "in_channels": len(BANDS),
            "num_classes": NUM_CLASSES,
            "band_names": list(bands or BANDS),
            "provenance": provenance,
        },
        path,
    )
    return path


# --------------------------------------------------------------------------------------
# 9. the store's own bookkeeping
# --------------------------------------------------------------------------------------


class TestStoreBookkeeping:
    def test_a_review_fraction_is_a_fraction(self, integrity_store):
        dataset = TileStoreDataset(integrity_store)
        fraction = dataset.review_fraction()
        assert fraction is None or 0.0 <= fraction <= 1.0

    def test_a_filtered_view_does_not_borrow_whole_store_totals(self, integrity_store):
        every = TileStoreDataset(integrity_store)
        assert every.describe()["scope"] == "whole store"
        assert every.describe()["usable_px"] is not None

        anchor = every.records[0].key
        held = TileStoreDataset(integrity_store, holdout_keys=[anchor])
        described = held.describe()
        assert described["scope"] == "filtered view"
        assert described["usable_px"] is None
        assert described["review_fraction"] is None
        assert held.review_fraction() is None
        assert described["tiles"] < every.describe()["tiles"]

    def test_coverage_is_counted_once_per_date_not_once_per_tile(self, tmp_path):
        """The two bases must stay apart: mixing them is what inflated a fraction past 1."""
        import atarra.pipeline as pipeline
        from atarra.datasets.store import load_manifest

        grid = grid_from_bbox(
            BBox.from_sequence([30.80, 31.45, 30.81, 31.46]), "EPSG:32636", 10.0
        )
        stack = _structured_stack(grid, valid_fraction=0.5)
        composite = _composite(stack, grid)
        root = tmp_path / "overlapping"
        with mock.patch.object(pipeline, "load_composite", lambda *a, **k: composite):
            build_store(root, "test_area", [date(2024, 8, 20)], gsd=10.0, tile_size=48,
                        stride=24)

        manifest = load_manifest(root)
        totals = manifest["totals"]
        # Unique ground, matching what the swath actually covered.
        assert totals["usable_px"] == int(stack.valid.sum())
        # Per tile, so the same pixel is counted by every tile that covers it.
        assert sum(s["tile_usable_px"] for s in manifest["shards"]) > totals["usable_px"]
        # And the share is a share, which it is not if one term is per tile.
        assert totals["review_px"] <= totals["usable_px"]
        fraction = TileStoreDataset(root).review_fraction()
        assert fraction is None or 0.0 <= fraction <= 1.0

    def test_class_counts_follow_the_view(self, integrity_store):
        every = TileStoreDataset(integrity_store)
        held = TileStoreDataset(integrity_store, holdout_keys=[every.records[0].key])
        assert held.class_counts().sum() < every.class_counts().sum()
        assert held.class_counts().sum() == int(
            sum(held.class_counts(indices=range(len(held))).tolist())
        )


# --------------------------------------------------------------------------------------
# 10. checkpoints carry what a scorer needs
# --------------------------------------------------------------------------------------


class TestCheckpointProvenance:
    def test_the_checkpoint_records_bands_and_splits(self, trained_run, run_dir):
        torch = pytest.importorskip("torch")

        saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        assert saved["band_names"] == trained_run["model"]["bands"]
        provenance = saved["provenance"]
        assert provenance["train_keys"]
        assert provenance["test_keys"]
        assert provenance["split_buffer_pixels"] == trained_run["splits"]["buffer_pixels"]
        # Train, validation and test must be disjoint as *ground*, recorded as keys.
        assert not set(provenance["train_keys"]) & set(provenance["val_keys"])
        assert not set(provenance["train_keys"]) & set(provenance["test_keys"])

    def test_the_run_is_recorded_under_the_name_it_was_given(self, trained_run, run_dir):
        """A split loop that reuses this scope's variable silently renames the run.

        Found by running the real pipeline: a support-check loop assigned to `name`, so
        every metrics.json recorded the last split as the run's name while the files sat
        in the directory that was actually asked for.
        """
        assert trained_run["run"] == run_dir.name
        assert trained_run["checkpoint"] == str(run_dir / "best.pt")

    def test_the_run_records_the_split_it_actually_used(self, trained_run):
        splits = trained_run["splits"]
        assert splits["verified_disjoint"] is True
        assert splits["buffer_pixels"] >= 0
        for name in ("train", "val", "test"):
            assert splits[f"{name}_tiles"] > 0
            assert sum(splits["class_support"][name]) > 0

    def test_an_existing_run_directory_is_not_overwritten(
        self, integrity_store, trained_run, run_dir
    ):
        """Two runs in one directory are indistinguishable after the second overwrites it."""
        from atarra.train.run import train_from_store

        pytest.importorskip("torch")
        assert any(run_dir.iterdir()), "the shared run should have filled its directory"

        with pytest.raises(AtarraError, match="already holds a run"):
            # The occupied directory is the shared run's, so the guard is exercised at
            # no cost -- no second training run and no second 30 MB checkpoint.
            train_from_store(
                integrity_store,
                epochs=1,
                batch_size=4,
                out_dir=run_dir.parent,
                run_name=run_dir.name,
                device="cpu",
                bands=trained_run["model"]["bands"],
            )


# --------------------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def integrity_store(tmp_path_factory):
    """One small two-date store, shared: building it is the slow part of this module."""
    import atarra.pipeline as pipeline

    grid = _integrity_grid()
    stack = _structured_stack(grid)
    dates = [date(2024, 8, 20), date(2024, 9, 4)]
    composites = [_composite(stack, grid, when=when) for when in dates]

    calls = {"count": 0}

    def fake(area_key, target_date, **kwargs):
        composite = composites[min(calls["count"], len(composites) - 1)]
        calls["count"] += 1
        return composite

    root = tmp_path_factory.mktemp("integrity") / "store"
    with mock.patch.object(pipeline, "load_composite", fake):
        build_store(root, "integrity_area", dates, gsd=10.0, tile_size=TILE, stride=TILE)
    return root


@pytest.fixture(scope="module")
def trained_run(integrity_store, tmp_path_factory):
    """One real training run, reused by every test that reads a run's artifacts."""
    pytest.importorskip("torch")
    from atarra.train.run import train_from_store

    out = tmp_path_factory.mktemp("integrity_runs")
    metrics = train_from_store(
        integrity_store,
        epochs=2,
        batch_size=4,
        out_dir=out,
        run_name="shared",
        device="cpu",
        seed=3,
        bands=["B04", "B08"],
    )
    return metrics


@pytest.fixture(scope="module")
def run_dir(integrity_store, trained_run):
    return Path(trained_run["checkpoint"]).parent


@pytest.fixture(scope="module")
def annotation_pack(integrity_store, tmp_path_factory):
    """A pack with hand labels for both classes, and a checkpoint that may score it."""
    import rasterio

    root = tmp_path_factory.mktemp("integrity_pack") / "pack"
    try:
        pack = export_annotation_pack(
            integrity_store, root, limit=2, strategy="reed", overwrite=True
        )
    except AtarraError:
        # The reed strategy needs reed pixels; the fallback keeps the pack tests about
        # scoring rather than about what the rule engine happened to label.
        pack = export_annotation_pack(
            integrity_store, root, limit=2, strategy="random", overwrite=True
        )
    assert pack["tiles"], "the pack exported nothing to score"

    for tile in pack["tiles"]:
        chip_path = root / tile["image_chip"]
        with rasterio.open(chip_path) as source:
            profile = source.profile
            shape = (source.height, source.width)
        labels = np.full(shape, 255, dtype=np.uint8)
        half = shape[1] // 2
        labels[:, :half] = PHRAGMITES_CODE
        labels[:, half:] = 0
        assert half * shape[0] >= MIN_REED_PIXELS_FOR_IOU, "label enough reed to measure it"
        profile.update(dtype="uint8", count=1, compress="deflate")
        with rasterio.open(root / "labels" / f"{tile['filename_stem']}.tif", "w", **profile) as out:
            out.write(labels, 1)

    keys = [tile["key"] for tile in pack["tiles"]]
    checkpoint = _write_checkpoint(
        root / "scoring.pt",
        provenance={"train_keys": ["elsewhere/0"], "val_keys": ["elsewhere/1"], "reserved_keys": keys},
    )
    return _Pack(path=root, store=integrity_store, checkpoint=checkpoint)


class _Pack:
    """Where a pack lives and what may score it, so tests need not re-derive either."""

    def __init__(self, *, path: Path, store: Path, checkpoint: Path) -> None:
        self.path = path
        self.store = Path(store)
        self.checkpoint = checkpoint

