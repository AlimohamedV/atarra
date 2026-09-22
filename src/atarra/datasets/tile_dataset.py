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


def augment_tile(
    image: np.ndarray, mask: np.ndarray, *, seed: int, index: int, epoch: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random flips, right-angle rotations, and mild spectral jitter.

    Training transforms vary by epoch and remain reproducible for a fixed seed.
    Validation and test views never call this function.

    Module-level rather than a method because the on-disk tile store needs exactly
    this transformation, and two implementations of "the same" augmentation drift.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, epoch]))

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


def summarise_counts(counts: np.ndarray) -> dict:
    """Turn a per-class pixel tally into the reported class-balance summary."""
    total = int(counts.sum())
    return {
        "counts": [int(c) for c in counts],
        "fractions": (counts / total).round(6).tolist() if total else [0.0] * len(counts),
        "total_px": total,
    }


def class_weights_from_counts(counts: np.ndarray, *, scheme: str) -> np.ndarray:
    """Per-class loss weights.

    Inverse frequency by default. Reed beds are a small minority of any delta
    scene, so unweighted cross-entropy is minimised by predicting "not reed"
    everywhere -- which scores well on pixel accuracy and is useless.
    """
    counts = np.asarray(counts, dtype=np.float64).copy()
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
        # `trainable`, not `usable`. The two differ where the rule engine was not
        # confident: those pixels are valid but uninformative, so they must not
        # shape the loss. See weak_labels.weak_label.
        valid = labelled["trainable"][rows, cols]

        # Excluded pixels must not contribute to the loss. -1 is the ignore index
        # the metrics and loss both understand.
        mask = np.where(valid, mask, IGNORE_INDEX)

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            image, mask = self._augment(image, mask, index)

        return {
            "image": image,
            "mask": mask,
            "confidence": confidence,
            # Carried through so the annotation queue can be exported without
            # re-running the rule engine over every composite.
            "review": labelled["review"][rows, cols],
            # Pixels the swath actually covered. Distinct from `mask != -1`, which
            # also excludes ambiguous pixels: ranking an annotation queue needs the
            # base rate, and without this the two cannot be told apart after the fact.
            "usable": labelled["usable"][rows, cols],
            # The rule engine's opinion *before* the loss mask discarded the ambiguous
            # pixels. `mask` cannot stand in for this: an ambiguous pixel is -1 there,
            # indistinguishable from nodata, and those are precisely the pixels an
            # annotator is being asked to adjudicate. Showing them as blank would hide
            # the guess they are supposed to correct.
            "rule_labels": labelled["labels"][rows, cols],
            "key": record.key,
            "block": record.block,
        }

    def _augment(self, image: np.ndarray, mask: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
        return augment_tile(image, mask, seed=self.seed, index=index)

    def pixel_totals(self) -> dict:
        """Unique-ground pixel tallies, counted once per composite rather than per tile.

        The rule engine already ran over each composite in full, so these come from
        that pass instead of from a union of overlapping tiles. Tiles overlap by
        design (stride < tile size), so summing over them would inflate both counts --
        and inflate them unevenly, since edge tiles cover less ground.
        """
        return {
            "usable_px": int(sum(np.count_nonzero(item["usable"]) for item in self._labelled)),
            "review_px": int(sum(np.count_nonzero(item["review"]) for item in self._labelled)),
        }

    def index_statistics(self) -> dict:
        """Class balance across every tile, to set loss weights sensibly."""
        counts = np.zeros(4, dtype=np.int64)
        for labelled in self._labelled:
            labels, usable = labelled["labels"], labelled["trainable"]
            for code in range(4):
                counts[code] += int(((labels == code) & usable).sum())
        return summarise_counts(counts)

    def class_weights(self, *, scheme: str = "inverse_frequency") -> np.ndarray:
        counts = np.array(self.index_statistics()["counts"], dtype=np.float64)
        return class_weights_from_counts(counts, scheme=scheme)

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
    dataset,
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
    buffer_pixels: int | None = None,
) -> SpatialSplit:
    """Partition complete footprints into contiguous regions, keeping dates together.

    Tiles crossing a boundary or its buffer are omitted from every split, and the
    result is verified disjoint before it is returned. The default gap is half a tile
    (``buffer_pixels=None``): a zero gap already prevents shared pixels, but tiles a
    few hundred metres apart share weather, water level and phenology, so a buffer is
    what keeps "held out" from meaning "held out except for its neighbours".

    The gap actually applied is on the returned ``SpatialSplit.buffer_pixels``, since
    it can be narrower than requested on a small store.
    """
    from atarra.datasets.spatial import (
        SpatialSplit,
        assert_disjoint,
        split_bounds,
        tile_bounds,
    )

    if buffer_pixels is None:
        size = getattr(dataset, "tile_size", None) or dataset.manifest.get("tile_size")
        if not size:
            raise AtarraError(
                "cannot choose a default split buffer without a tile size; pass "
                "buffer_pixels explicitly"
            )
        buffer_pixels = int(size) // 2
    if buffer_pixels < 0:
        raise AtarraError("the split buffer must be non-negative")

    bounds = tile_bounds(dataset)
    # Disjointness is the invariant and holds at any buffer >= 0; the buffer is the
    # weaker claim that no test tile sits *next to* a training tile. A small store may
    # not be able to afford the gap at all, and failing a long run over a quality knob
    # would be worse than using a smaller one -- so the ladder walks down to zero and
    # says so. It never widens beyond what the caller asked for.
    attempt = buffer_pixels
    ladder = [attempt]
    while attempt > 0:
        attempt //= 2
        if attempt not in ladder:
            ladder.append(attempt)

    result = None
    for attempt in ladder:
        try:
            result = split_bounds(
                bounds, fractions=fractions, seed=seed, buffer_pixels=attempt
            )
            break
        except AtarraError as error:
            last_error = error
    if result is None:
        raise last_error  # type: ignore[possibly-undefined]
    if attempt != buffer_pixels:
        log.warning(
            "a %d-pixel split gap leaves a region empty; using %d instead. Disjointness "
            "still holds, but the splits are less separated than requested.",
            buffer_pixels,
            attempt,
        )
    assert_disjoint(bounds, result)

    omitted = len(dataset.records) - sum(map(len, result.values()))
    log.info(
        "geographic split: train=%d val=%d test=%d tiles; %d boundary tile(s) omitted "
        "by a %d-pixel (%.0f m) gap",
        len(result["train"]),
        len(result["val"]),
        len(result["test"]),
        omitted,
        attempt,
        attempt * float(getattr(dataset, "gsd", 0) or _manifest_gsd(dataset)),
    )
    return result


def _manifest_gsd(dataset) -> float:
    """Ground sample distance for logging, or 0.0 when the dataset is not a store."""
    manifest = getattr(dataset, "manifest", None) or {}
    return float(manifest.get("gsd") or 0.0)
