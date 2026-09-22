"""Segmentation metrics.

Everything derives from one confusion matrix rather than from per-batch averages.
That distinction is not pedantry: averaging per-batch IoU over-weights batches that
happen to contain few foreground pixels, so a model that predicts all-background on
most tiles and gets one small tile right can post a flattering mIoU. Accumulating
counts across the whole evaluation set and computing the metric once is the only
version of the number that means what a reader assumes it means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Class order is fixed pipeline-wide; imported so there is a single definition.
# Reed is last because it is the class the project exists to find, and the one whose
# per-class score gets quoted.
from atarra.datasets.weak_labels import (
    CLASS_NAMES,
    NUM_CLASSES,
    PHRAGMITES_CODE,
)


def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Count ``[true, predicted]`` pairs, ignoring ``ignore_index`` if given."""
    prediction = np.asarray(prediction).ravel()
    target = np.asarray(target).ravel()
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape; "
            f"got {prediction.shape} and {target.shape}"
        )

    if ignore_index is not None:
        keep = target != ignore_index
        prediction, target = prediction[keep], target[keep]

    valid = (
        (target >= 0)
        & (target < num_classes)
        & (prediction >= 0)
        & (prediction < num_classes)
    )
    indices = target[valid].astype(np.int64) * num_classes + prediction[valid].astype(np.int64)
    return np.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def iou_per_class(matrix: np.ndarray) -> np.ndarray:
    """Intersection over union per class. ``NaN`` where a class is absent.

    Absent classes are ``NaN`` rather than 0 so that they can be excluded from the
    mean instead of dragging it down -- a validation tile with no water in it
    should not count as a water failure.
    """
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    return np.divide(
        intersection, union, out=np.full_like(intersection, np.nan), where=union > 0
    )


def f1_per_class(matrix: np.ndarray) -> np.ndarray:
    """Per-class F1 (Dice) score."""
    intersection = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=1) + matrix.sum(axis=0)
    return np.divide(
        2.0 * intersection, denominator, out=np.full_like(intersection, np.nan), where=denominator > 0
    )


def mean_iou(matrix: np.ndarray, *, exclude_absent: bool = True) -> float:
    """Mean IoU across classes."""
    scores = iou_per_class(matrix)
    if exclude_absent:
        scores = scores[~np.isnan(scores)]
    return float(np.nanmean(scores)) if scores.size else float("nan")


def pixel_accuracy(matrix: np.ndarray) -> float:
    total = matrix.sum()
    return float(np.diag(matrix).sum() / total) if total else float("nan")


@dataclass
class ConfusionAccumulator:
    """Accumulate a confusion matrix across batches and tiles."""

    num_classes: int = NUM_CLASSES
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))

    def update(self, prediction: np.ndarray, target: np.ndarray, *, ignore_index: int | None = None) -> None:
        self.matrix += confusion_matrix(
            prediction, target, num_classes=self.num_classes, ignore_index=ignore_index
        )

    def report(self, class_names: list[str] | None = None) -> dict:
        return segmentation_report(self.matrix, class_names or CLASS_NAMES)

    def reset(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


def segmentation_report(
    matrix: np.ndarray, class_names: list[str] | None = None
) -> dict:
    """A full report, including the headline numbers the proposal commits to."""
    names = class_names or CLASS_NAMES
    ious = iou_per_class(matrix)
    f1s = f1_per_class(matrix)

    per_class = []
    for index, name in enumerate(names[: matrix.shape[0]]):
        support = int(matrix[index].sum())
        per_class.append(
            {
                "class_code": index,
                "class_name": name,
                "iou": None if np.isnan(ious[index]) else round(float(ious[index]), 4),
                "f1": None if np.isnan(f1s[index]) else round(float(f1s[index]), 4),
                "support_px": support,
            }
        )

    reed_iou = float(ious[PHRAGMITES_CODE]) if not np.isnan(ious[PHRAGMITES_CODE]) else None
    reed_f1 = float(f1s[PHRAGMITES_CODE]) if not np.isnan(f1s[PHRAGMITES_CODE]) else None

    return {
        "mean_iou": round(mean_iou(matrix), 4),
        "pixel_accuracy": round(pixel_accuracy(matrix), 4),
        "phragmites_iou": None if reed_iou is None else round(reed_iou, 4),
        "phragmites_f1": None if reed_f1 is None else round(reed_f1, 4),
        "per_class": per_class,
        "support_px": int(matrix.sum()),
        "confusion_matrix": matrix.tolist(),
    }


def meets_targets(report: dict, *, iou: float = 0.82, f1: float = 0.85) -> dict:
    """Check a report against the proposal's stated success criteria."""
    reed_iou = report.get("phragmites_iou")
    reed_f1 = report.get("phragmites_f1")
    return {
        "target_iou": iou,
        "target_f1": f1,
        "iou_met": reed_iou is not None and reed_iou >= iou,
        "f1_met": reed_f1 is not None and reed_f1 >= f1,
        "both_met": bool(
            reed_iou is not None
            and reed_f1 is not None
            and reed_iou >= iou
            and reed_f1 >= f1
        ),
    }
