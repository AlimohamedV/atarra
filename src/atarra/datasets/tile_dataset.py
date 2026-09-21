"""Tile datasets for segmentation training.

The single most important function here is :func:`geometric_split`, and the reason
is worth stating plainly.

Adjacent satellite tiles are heavily autocorrelated -- a reed bed straddling a tile
boundary appears in both halves, and two tiles 200 m apart share almost all of
their context. A random train/test split therefore puts near-duplicates of test
pixels into training, and the resulting mIoU measures memorisation, not
generalisation. It will look excellent and mean nothing.

Splitting by contiguous spatial block fixes this: whole regions go to one split, so
"did it learn to find reeds" is answered on ground the model has genuinely never
seen. Every number this project reports should come from a geometric split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger
from atarra.datasets.weak_labels import PHRAGMITES, WeakLabelConfig, weak_label

log = get_logger("datasets.tiles")

IGNORE_INDEX = -1


@dataclass
class TileRecord:
    """A single tile cut from a composite."""

    composite_index: int
    row: int
    col: int
    size: int
    block: tuple[int, int]

    @property
    def key(self) -> str:
        return f"c{self.composite_index}/r{self.row}_c{self.col}"


def _tile_grid(width: int, height: int, size: int, stride: int | None) -> Iterator[tuple[int, int]]:
    step = stride or size
    for row in range(0, height - size + 1, step):
        for col in range(0, width - size + 1, step):
            yield row, col


class CompositeTileDataset:
    """Tiles cut from index composites, labelled by the weak-supervision rules.

    Deliberately does not subclass ``torch.utils.data.Dataset`` at import time so
    that the labelling and splitting logic stays importable and testable without
    torch installed. :meth:`as_torch_dataset` wraps it when torch is available.
    """

    def __init__(
        self,
        composites: Sequence,
        *,
        band_names: Sequence[str] | None = None,
        tile_size: int = 256,
        stride: int | None = None,
        label_config: WeakLabelConfig | None = None,
        augment: bool = False,
        seed: int = 0,
        drop_empty: bool = False,
    ) -> None:
        if not composites:
            raise AtarraError("CompositeTileDataset needs at least one composite")

        self.composites = list(composites)
        self.tile_size = int(tile_size)
        self.stride = stride
        self.label_config = label_config or WeakLabelConfig()
        self.augment = augment
        self.seed = int(seed)

        first = self.composites[0]
        self.band_names = list(band_names or first.stack.band_names)

        # Precompute labels once; the rule engine is not free and __getitem__ is
        # called many times per epoch.
        self._labelled: list[dict] = []
        for composite in self.composites:
            indices = composite.indices
            result = weak_label(indices, config=self.label_config, valid=composite.stack.valid)
            self._labelled.append(result)

        self.records: list[TileRecord] = []
        for index, composite in enumerate(self.composites):
            grid = composite.grid
            for row, col in _tile_grid(grid.width, grid.height, self.tile_size, self.stride):
                if drop_empty:
                    labels = self._labelled[index]["labels"]
                    window = labels[row : row + self.tile_size, col : col + self.tile_size]
                    if not np.any(window == PHRAGMITES):
                        continue
                self.records.append(
                    TileRecord(
                        composite_index=index,
                        row=row,
                        col=col,
                        size=self.tile_size,
                        block=(row // self.tile_size, col // self.tile_size),
                    )
                )

        if not self.records:
            raise AtarraError(
                "no tiles were produced; the composites are smaller than the tile size"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        composite = self.composites[record.composite_index]
        labelled = self._labelled[record.composite_index]

        rows = slice(record.row, record.row + record.size)
        cols = slice(record.col, record.col + record.size)

        image = np.stack(
            [composite.stack.band(name)[rows, cols] for name in self.band_names], axis=0
        ).astype(np.float32)

        mask = labelled["labels"][rows, cols].astype(np.int64)
        confidence = labelled["confidence"][rows, cols].astype(np.float32)
        valid = labelled["usable"][rows, cols]

        # Invalid pixels must not contribute to the loss. -1 is the ignore index
        # the metrics and loss both understand.
        mask = np.where(valid, mask, IGNORE_INDEX)

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            image, mask = self._augment(image, mask, index)

        return {
            "image": image,
            "mask": mask,
            "confidence": confidence,
            "key": record.key,
            "block": record.block,
        }

    def _augment(self, image: np.ndarray, mask: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Random flips, right-angle rotations, and mild spectral jitter.

        Seeded per index so a given tile augments identically across epochs. Fully
        random augmentation would make the validation split non-reproducible, which
        quietly destroys the ability to compare two runs.
        """
        rng = np.random.default_rng(self.seed + index)

        if rng.random() < 0.5:
            image, mask = image[:, :, ::-1], mask[:, ::-1]
        if rng.random() < 0.5:
            image, mask = image[:, ::-1, :], mask[::-1, :]
        rotations = int(rng.integers(0, 4))
        if rotations:
            image = np.rot90(image, rotations, axes=(1, 2))
            mask = np.rot90(mask, rotations, axes=(0, 1))

        # Spectral jitter: a multiplicative per-band gain, simulating the
        # radiometric variation between overpasses and seasons. Applied to
        # reflectance only; the mask is untouched.
        if rng.random() < 0.5:
            gains = rng.normal(1.0, 0.05, size=(image.shape[0], 1, 1)).astype(np.float32)
            image = image * gains

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def index_statistics(self) -> dict:
        """Class balance across every tile, to set loss weights sensibly."""
        counts = np.zeros(4, dtype=np.int64)
        for labelled in self._labelled:
            labels, usable = labelled["labels"], labelled["usable"]
            for code in range(4):
                counts[code] += int(((labels == code) & usable).sum())
        total = int(counts.sum())
        return {
            "counts": counts.tolist(),
            "fractions": (counts / total).round(6).tolist() if total else [0.0] * 4,
            "total_px": total,
        }

    def class_weights(self, *, scheme: str = "inverse_frequency") -> np.ndarray:
        """Per-class loss weights.

        Inverse frequency by default. Reed beds are a small minority of any delta
        scene, so unweighted cross-entropy is minimised by predicting "not reed"
        everywhere -- which scores well on pixel accuracy and is useless.
        """
        counts = np.array(self.index_statistics()["counts"], dtype=np.float64)
        counts[counts == 0] = 1.0
        if scheme == "inverse_frequency":
            weights = counts.sum() / (counts * len(counts))
        elif scheme == "sqrt_inverse":
            weights = np.sqrt(counts.sum() / (counts * len(counts)))
        elif scheme == "uniform":
            weights = np.ones_like(counts)
        else:
            raise AtarraError(
                f"unknown class weighting scheme {scheme!r}; "
                "expected 'inverse_frequency', 'sqrt_inverse' or 'uniform'"
            )
        weights = np.clip(weights, 0.05, 50.0)
        return (weights / weights.mean()).astype(np.float32)

    def as_torch_dataset(self):
        """Wrap as a ``torch.utils.data.Dataset`` yielding tensors."""
        import torch

        dataset = self

        class _TorchDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(dataset)

            def __getitem__(self, index: int):
                sample = dataset[index]
                return {
                    "image": torch.from_numpy(sample["image"]),
                    "mask": torch.from_numpy(sample["mask"]),
                }

        return _TorchDataset()


def geometric_split(
    dataset: CompositeTileDataset,
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split tiles by contiguous spatial block, never at random.

    Whole tile-blocks are assigned to a split, so no test tile is adjacent to a
    training tile. Returns index lists keyed ``train`` / ``val`` / ``test``.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise AtarraError(f"fractions must be three values summing to 1, got {fractions}")

    blocks: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(dataset.records):
        blocks.setdefault(record.block, []).append(index)

    block_keys = sorted(blocks)
    if len(block_keys) < 3:
        raise AtarraError(
            f"only {len(block_keys)} spatial block(s) available; a geometric split needs "
            "at least 3. Use larger composites, or a smaller tile size, so the AOI "
            "covers multiple blocks."
        )

    # Deterministic shuffle: the same seed always yields the same partition, which
    # is what makes two experiments comparable.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(block_keys))

    n_train = int(round(fractions[0] * len(block_keys)))
    n_val = int(round(fractions[1] * len(block_keys)))
    # Guarantee every split is non-empty; with few blocks the rounding can starve
    # one otherwise.
    n_train = max(1, min(n_train, len(block_keys) - 2))
    n_val = max(1, min(n_val, len(block_keys) - n_train - 1))

    splits = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }

    result: dict[str, list[int]] = {}
    for name, positions in splits.items():
        indices: list[int] = []
        for position in positions:
            indices.extend(blocks[block_keys[int(position)]])
        result[name] = sorted(indices)

    log.info(
        "geometric split over %d block(s): train=%d val=%d test=%d tiles",
        len(block_keys),
        len(result["train"]),
        len(result["val"]),
        len(result["test"]),
    )
    return result
