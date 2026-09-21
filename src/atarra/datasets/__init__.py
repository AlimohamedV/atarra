"""Weak-supervision labelling and tile loading."""

from atarra.datasets.weak_labels import (
    WeakLabelConfig,
    label_statistics,
    weak_label,
)
from atarra.datasets.tile_dataset import CompositeTileDataset, geometric_split

__all__ = [
    "CompositeTileDataset",
    "WeakLabelConfig",
    "geometric_split",
    "label_statistics",
    "weak_label",
]
