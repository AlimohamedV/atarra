"""Tests for metrics, weak-supervision labels, dataset splitting, and the model."""

from __future__ import annotations

import numpy as np
import pytest

from atarra.datasets.weak_labels import (
    CROPS_SOIL,
    OPEN_WATER,
    PHRAGMITES,
    WeakLabelConfig,
    label_statistics,
    weak_label,
)
from atarra.preprocess.indices import compute_indices
from atarra.train.metrics import (
    ConfusionAccumulator,
    confusion_matrix,
    f1_per_class,
    iou_per_class,
    mean_iou,
    meets_targets,
    pixel_accuracy,
    segmentation_report,
)


class TestConfusionMatrix:
    def test_perfect_prediction(self):
        target = np.array([0, 1, 2, 3])
        matrix = confusion_matrix(target, target)
        assert np.diag(matrix).tolist() == [1, 1, 1, 1]
        assert matrix.sum(axis=0).tolist() == [1, 1, 1, 1]

    def test_counts_off_diagonal(self):
        matrix = confusion_matrix(np.array([0, 0]), np.array([0, 1]))
        assert matrix[0, 0] == 1  # true 0, predicted 0
        assert matrix[1, 0] == 1  # true 1, predicted 0

    def test_rows_are_truth_columns_are_prediction(self):
        matrix = confusion_matrix(np.array([1]), np.array([0]))
        assert matrix[0, 1] == 1

    def test_ignore_index_is_excluded(self):
        matrix = confusion_matrix(np.array([0, 0]), np.array([0, -1]), ignore_index=-1)
        assert matrix.sum() == 1

    def test_out_of_range_values_are_ignored(self):
        matrix = confusion_matrix(np.array([0, 9]), np.array([0, 0]), num_classes=4)
        assert matrix.sum() == 1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            confusion_matrix(np.zeros((2, 2)), np.zeros((3, 3)))


class TestMetrics:
    def test_iou_of_perfect_prediction_is_one(self):
        matrix = confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 2]))
        ious = iou_per_class(matrix)
        # Class 3 never occurs, so its IoU is NaN and must not be counted as 0.
        assert np.isnan(ious[3])
        assert np.allclose(ious[:3], 1.0)

    def test_iou_hand_computed(self):
        # Class 0: one pixel correct, one mispredicted as class 1.
        # TP = 1, FP = 0 (nothing predicted as 0 that is not 0), FN = 1.
        # IoU = TP / (TP + FP + FN) = 1 / 2.
        matrix = np.zeros((4, 4), dtype=np.int64)
        matrix[0, 0] = 1
        matrix[0, 1] = 1
        assert iou_per_class(matrix)[0] == pytest.approx(0.5)

    def test_absent_class_is_nan_not_zero(self):
        """A tile with no water must not be scored as a water failure."""
        matrix = np.zeros((4, 4), dtype=np.int64)
        matrix[3, 3] = 10
        ious = iou_per_class(matrix)
        assert np.isnan(ious[0])
        assert ious[3] == pytest.approx(1.0)

    def test_mean_iou_excludes_absent_classes(self):
        matrix = np.zeros((4, 4), dtype=np.int64)
        matrix[3, 3] = 10
        assert mean_iou(matrix) == pytest.approx(1.0)

    def test_f1_hand_computed(self):
        matrix = np.zeros((4, 4), dtype=np.int64)
        matrix[3, 3] = 8
        matrix[0, 3] = 2  # true water predicted reed -> precision drops
        # precision 8/10, recall 1.0 -> F1 = 2*0.8*1/(1.8) = 0.888...
        assert f1_per_class(matrix)[3] == pytest.approx(2 * (8 / 10) * 1.0 / (1.0 + 8 / 10))

    def test_pixel_accuracy(self):
        matrix = np.array([[3, 1], [0, 0]], dtype=np.int64)
        assert pixel_accuracy(matrix) == pytest.approx(0.75)


class TestAccumulator:
    def test_accumulates_across_batches(self):
        """Per-batch averaging would over-weight small batches; counting does not."""
        accumulator = ConfusionAccumulator()
        accumulator.update(np.array([3, 3, 3]), np.array([3, 3, 3]))
        accumulator.update(np.array([0]), np.array([0]))
        assert accumulator.matrix.sum() == 4
        assert accumulator.matrix[3, 3] == 3

    def test_sum_of_batches_equals_one_pass(self):
        rng = np.random.default_rng(0)
        targets = rng.integers(0, 4, 1000)
        predictions = rng.integers(0, 4, 1000)

        one_pass = confusion_matrix(predictions, targets)
        accumulator = ConfusionAccumulator()
        for start in range(0, 1000, 128):
            accumulator.update(predictions[start : start + 128], targets[start : start + 128])
        assert np.array_equal(accumulator.matrix, one_pass)

    def test_reset(self):
        accumulator = ConfusionAccumulator()
        accumulator.update(np.array([1]), np.array([1]))
        accumulator.reset()
        assert accumulator.matrix.sum() == 0


class TestReport:
    def test_report_shape_and_targets(self):
        matrix = np.zeros((4, 4), dtype=np.int64)
        for code in range(4):
            matrix[code, code] = 100
        report = segmentation_report(matrix)
        assert report["mean_iou"] == pytest.approx(1.0)
        assert report["phragmites_iou"] == pytest.approx(1.0)
        assert len(report["per_class"]) == 4

        targets = meets_targets(report)
        assert targets["both_met"] is True

    def test_targets_not_met_when_reed_is_poor(self):
        matrix = np.zeros((4, 4), dtype=np.int64)
        for code in range(3):
            matrix[code, code] = 100
        matrix[0, 3] = 90  # reed badly misclassified
        matrix[3, 3] = 10
        assert meets_targets(segmentation_report(matrix))["both_met"] is False

    def test_empty_matrix_does_not_crash(self):
        report = segmentation_report(np.zeros((4, 4), dtype=np.int64))
        assert report["support_px"] == 0


class TestWeakLabels:
    def test_vegetation_and_water_are_separated(self, veg_water_stack):
        indices = compute_indices(veg_water_stack)
        result = weak_label(indices, valid=veg_water_stack.valid)
        labels = result["labels"]
        half = labels.shape[1] // 2

        water_half = labels[:, half:]
        assert (water_half == OPEN_WATER).mean() > 0.8, "water side should label as water"

    def test_reed_class_is_produced_for_a_reed_like_spectrum(self, simple_grid):
        """A textbook reed signature must land in the reed class, not a neighbour."""
        height, width = simple_grid.height, simple_grid.width
        indices = {
            "ndvi": np.full((height, width), 0.75, dtype=np.float32),
            "ndre": np.full((height, width), 0.45, dtype=np.float32),
            "ndmi": np.full((height, width), 0.25, dtype=np.float32),
            "ndwi": np.full((height, width), -0.30, dtype=np.float32),
        }
        result = weak_label(indices)
        assert (result["labels"] == PHRAGMITES).mean() > 0.9

    def test_dry_cropland_is_not_labelled_reed(self, simple_grid):
        """The confusion that matters most: vigorous crops must not become reed."""
        height, width = simple_grid.height, simple_grid.width
        indices = {
            "ndvi": np.full((height, width), 0.80, dtype=np.float32),
            "ndre": np.full((height, width), 0.08, dtype=np.float32),  # no red-edge step
            "ndmi": np.full((height, width), -0.15, dtype=np.float32),  # dry
            "ndwi": np.full((height, width), -0.45, dtype=np.float32),
        }
        result = weak_label(indices)
        assert (result["labels"] == PHRAGMITES).mean() < 0.05
        assert (result["labels"] == CROPS_SOIL).mean() > 0.5

    def test_ambiguous_pixels_go_to_review(self, simple_grid):
        """Abstaining is better than a confidently wrong label."""
        height, width = simple_grid.height, simple_grid.width
        indices = {
            "ndvi": np.full((height, width), 0.48, dtype=np.float32),
            "ndre": np.full((height, width), 0.15, dtype=np.float32),
            "ndmi": np.full((height, width), 0.0, dtype=np.float32),
            "ndwi": np.full((height, width), -0.10, dtype=np.float32),
        }
        result = weak_label(indices)
        assert result["review"].mean() > 0.5, "mid-range spectra should be sent to review"
        assert (result["labels"] != PHRAGMITES).all(), (
            "an ambiguous spectrum must never be confidently labelled as the target class"
        )

    def _uniform_indices(self, simple_grid, **values):
        height, width = simple_grid.height, simple_grid.width
        defaults = {"ndvi": 0.75, "ndre": 0.45, "ndmi": 0.25, "ndwi": -0.15}
        defaults.update(values)
        return {
            key: np.full((height, width), value, dtype=np.float32)
            for key, value in defaults.items()
        }

    def test_clear_reed_spectrum_is_confidently_labelled(self, simple_grid):
        """An unmistakable reed signature must reach the training set, not review."""
        indices = self._uniform_indices(simple_grid, ndvi=0.78, ndre=0.50, ndmi=0.30, ndwi=-0.12)
        result = weak_label(indices)
        assert (result["labels"] == PHRAGMITES).mean() > 0.9
        assert result["review"].mean() < 0.1
        # The reed score is the minimum of four terms and saturates at NDVI 0.85,
        # so 0.77 is a strong result here rather than a marginal one: it sits well
        # clear of the 0.60 review threshold.
        assert result["confidence"].mean() > 0.7

    def test_reed_that_also_looks_like_crop_goes_to_review(self, simple_grid):
        """The reed/crop confusion is the project's central risk; abstain on it.

        A very negative NDWI makes the pixel look like dry cropland to the crop
        rule, leaving only a narrow margin over the reed rule. Rather than pick a
        winner on a 0.07 margin, the rules send it for human verification.
        """
        indices = self._uniform_indices(simple_grid, ndvi=0.78, ndre=0.50, ndmi=0.30, ndwi=-0.35)
        result = weak_label(indices)
        assert (result["labels"] == PHRAGMITES).mean() > 0.9, "the rules should still point at reed"
        assert result["review"].mean() > 0.9, "but not claim confidence"

    def test_halophytes_are_always_sent_to_review(self, simple_grid):
        """The residual class cannot be confirmed from spectra, so it abstains."""
        height, width = simple_grid.height, simple_grid.width
        indices = {
            "ndvi": np.full((height, width), 0.30, dtype=np.float32),
            "ndre": np.full((height, width), 0.10, dtype=np.float32),
            "ndmi": np.full((height, width), 0.05, dtype=np.float32),
            "ndwi": np.full((height, width), 0.05, dtype=np.float32),
        }
        result = weak_label(indices)
        halophyte = result["labels"] == 2
        if halophyte.any():
            assert result["review"][halophyte].all()

    def test_every_class_survives_confidence_filtering(self, simple_grid):
        """No class may be annihilated by the ambiguity filter.

        Regression test for a real trap. REVIEW_THRESHOLD (0.60) is tuned for
        review-queue size, and the per-class scores saturate at different ceilings
        by design: the crop rule tops out at 0.597 and the halophyte rule at
        0.501. Reusing that threshold as a *loss* filter therefore dropped the
        entire cropland class by 0.003 of confidence -- silently turning a
        4-class problem into a 2-class one with no error anywhere. The filter is
        margin-based instead, so all four classes must stay trainable.
        """
        representatives = {
            PHRAGMITES: {"ndvi": 0.78, "ndre": 0.50, "ndmi": 0.30, "ndwi": -0.12},
            1: {"ndvi": 0.62, "ndre": 0.25, "ndmi": 0.08, "ndwi": -0.30},
            2: {"ndvi": 0.30, "ndre": 0.10, "ndmi": 0.05, "ndwi": 0.05},
            0: {"ndvi": 0.05, "ndre": 0.02, "ndmi": -0.10, "ndwi": 0.60},
        }
        for expected, values in representatives.items():
            result = weak_label(self._uniform_indices(simple_grid, **values))
            assert (result["labels"] == expected).mean() > 0.9, (
                f"expected class {expected} for {values}"
            )
            assert result["trainable"].mean() > 0.9, (
                f"class {expected} was filtered out of the training set; "
                "the ambiguity test must not be the review test"
            )

    def test_ambiguous_pixels_are_dropped_from_the_loss(self, simple_grid):
        """A reed/crop coin flip is uninformative, so it must not shape the loss."""
        indices = self._uniform_indices(
            simple_grid, ndvi=0.78, ndre=0.50, ndmi=0.30, ndwi=-0.35
        )
        result = weak_label(indices)
        assert (result["labels"] == PHRAGMITES).mean() > 0.9, "the rules still point at reed"
        assert result["ambiguous"].mean() > 0.9
        assert result["trainable"].mean() < 0.1, "but the loss must not learn from it"
        assert result["review"].mean() > 0.9, "it belongs in the annotation queue"

    def test_trainable_never_exceeds_usable(self, veg_water_stack):
        indices = compute_indices(veg_water_stack)
        result = weak_label(indices, valid=veg_water_stack.valid)
        assert not (result["trainable"] & ~result["usable"]).any()
        assert not (result["trainable"] & ~veg_water_stack.valid).any(), (
            "nodata must never be trainable"
        )

    def test_drop_ambiguous_can_be_disabled(self, simple_grid):
        from atarra.datasets.weak_labels import WeakLabelConfig

        indices = self._uniform_indices(
            simple_grid, ndvi=0.78, ndre=0.50, ndmi=0.30, ndwi=-0.35
        )
        result = weak_label(indices, config=WeakLabelConfig(drop_ambiguous=False))
        assert result["ambiguous"].mean() > 0.9, "the flag does not change the diagnosis"
        assert (result["trainable"] == result["usable"]).all()

    def test_statistics_account_for_dropped_pixels(self, simple_grid):
        indices = self._uniform_indices(
            simple_grid, ndvi=0.78, ndre=0.50, ndmi=0.30, ndwi=-0.35
        )
        stats = label_statistics(weak_label(indices))
        assert stats["trainable_px"] + stats["dropped_px"] == stats["usable_px"]
        assert stats["dropped_px"] > 0

    def test_invalid_mask_forces_ignore(self, simple_grid):
        height, width = simple_grid.height, simple_grid.width
        indices = {
            "ndvi": np.full((height, width), 0.75, dtype=np.float32),
            "ndre": np.full((height, width), 0.45, dtype=np.float32),
            "ndmi": np.full((height, width), 0.25, dtype=np.float32),
            "ndwi": np.full((height, width), -0.3, dtype=np.float32),
        }
        valid = np.zeros((height, width), dtype=bool)
        valid[: height // 2] = True
        result = weak_label(indices, valid=valid)
        assert (result["labels"][height // 2 :] == -1).all()
        assert (result["labels"][: height // 2] != -1).all()

    def test_missing_index_raises(self):
        from atarra.core.errors import AtarraError

        with pytest.raises(AtarraError, match="needs indices"):
            weak_label({"ndvi": np.zeros((2, 2), dtype=np.float32)})

    def test_statistics_sum_to_one(self, veg_water_stack):
        indices = compute_indices(veg_water_stack)
        result = weak_label(indices, valid=veg_water_stack.valid)
        stats = label_statistics(result)
        total = sum(item["fraction"] for item in stats["classes"])
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_deep_water_is_not_reed(self, simple_grid):
        """High NDWI must veto the reed class even with vegetation-like NDVI."""
        height, width = simple_grid.height, simple_grid.width
        indices = {
            "ndvi": np.full((height, width), 0.60, dtype=np.float32),
            "ndre": np.full((height, width), 0.35, dtype=np.float32),
            "ndmi": np.full((height, width), 0.20, dtype=np.float32),
            "ndwi": np.full((height, width), 0.55, dtype=np.float32),  # clearly water
        }
        result = weak_label(indices)
        assert (result["labels"] == PHRAGMITES).mean() < 0.05


class TestModel:
    """Model tests need torch, which is present as a CUDA build."""

    def test_forward_pass_8_band(self):
        torch = pytest.importorskip("torch")
        from atarra.models.segmentation import build_model

        model = build_model(in_channels=8, base_channels=8, depth=3)
        output = model(torch.zeros(2, 8, 64, 64))
        assert output.shape == (2, 4, 64, 64)

    def test_forward_pass_rgb_baseline(self):
        """The control arm must run through the identical code path."""
        torch = pytest.importorskip("torch")
        from atarra.models.segmentation import build_model

        model = build_model(variant="rgb", base_channels=8, depth=3)
        assert model.in_channels == 3
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 4, 64, 64)

    def test_odd_input_size_is_handled(self):
        torch = pytest.importorskip("torch")
        from atarra.models.segmentation import build_model

        model = build_model(in_channels=8, base_channels=8, depth=3)
        assert model(torch.zeros(1, 8, 65, 61)).shape == (1, 4, 65, 61)

    def test_wrong_channel_count_raises(self):
        torch = pytest.importorskip("torch")
        from atarra.core.errors import AtarraError
        from atarra.models.segmentation import build_model

        model = build_model(in_channels=8, base_channels=8, depth=3)
        with pytest.raises(AtarraError, match="expects 8 input channels"):
            model(torch.zeros(1, 3, 64, 64))

    def test_normalisation_buffers_are_registered(self):
        """Buffers, not plain attributes: they must survive a checkpoint round-trip."""
        torch = pytest.importorskip("torch")
        from atarra.models.segmentation import build_model

        model = build_model(in_channels=8, base_channels=8, depth=3, band_mean=[0.1] * 8, band_std=[0.05] * 8)
        assert "band_mean" in dict(model.named_buffers())
        assert "band_std" in dict(model.named_buffers())

    def test_the_rgb_control_arm_uses_the_supplied_statistics(self):
        """The arms must differ in their band set and in nothing else.

        Hard-coding the control arm's normalisation while the multispectral arm is
        normalised from its own training split makes the measured gain part band set
        and part invisible difference in preprocessing.
        """
        torch = pytest.importorskip("torch")
        from atarra.models.segmentation import build_model

        measured = build_model(
            variant="rgb", base_channels=8, depth=3, band_mean=[0.11, 0.09, 0.07],
            band_std=[0.04, 0.04, 0.03],
        )
        # The buffers are shaped (1, C, 1, 1) for broadcasting, so read them flat.
        assert measured.band_mean.reshape(-1).tolist() == pytest.approx([0.11, 0.09, 0.07])
        assert measured.band_std.reshape(-1).tolist() == pytest.approx([0.04, 0.04, 0.03])

        # With nothing supplied the published defaults still apply, so the preset
        # remains usable on its own.
        defaulted = build_model(variant="rgb", base_channels=8, depth=3)
        assert defaulted.band_mean.reshape(-1).tolist() == pytest.approx([0.12, 0.10, 0.08])
        assert defaulted.in_channels == 3

    def test_zero_std_is_rejected(self):
        pytest.importorskip("torch")
        from atarra.core.errors import AtarraError
        from atarra.models.segmentation import build_model

        with pytest.raises(AtarraError, match="strictly positive"):
            build_model(in_channels=4, base_channels=8, depth=3, band_std=[1, 1, 0, 1])

    def test_gradient_flows(self):
        torch = pytest.importorskip("torch")
        from atarra.models.segmentation import build_model

        model = build_model(in_channels=8, base_channels=8, depth=3)
        output = model(torch.rand(2, 8, 64, 64))
        output.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    @pytest.mark.gpu
    def test_runs_on_cuda_with_amp(self):
        """4.29 GB of VRAM is the target; a 64 px tile must fit easily."""
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from atarra.models.segmentation import build_model

        model = build_model(in_channels=8, base_channels=16, depth=3).cuda()
        with torch.amp.autocast("cuda", enabled=True):
            output = model(torch.rand(2, 8, 128, 128, device="cuda"))
        assert output.shape == (2, 4, 128, 128)


class TestBandStatistics:
    def test_computes_per_band_stats(self):
        from atarra.models.segmentation import compute_band_statistics

        tiles = [np.random.default_rng(i).uniform(0, 0.5, (4, 16, 16)).astype(np.float32) for i in range(3)]
        stats = compute_band_statistics(tiles)
        assert stats["n_bands"] == 4
        assert all(0.0 <= mean <= 0.5 for mean in stats["mean"])
        assert all(std > 0 for std in stats["std"])

    def test_floors_zero_std(self):
        """A constant band must not produce a division by zero in the model."""
        from atarra.models.segmentation import compute_band_statistics

        tiles = [np.full((2, 8, 8), 0.3, dtype=np.float32)]
        stats = compute_band_statistics(tiles)
        assert all(std >= 1e-3 for std in stats["std"])
