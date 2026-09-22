"""Geographic partitions built from complete tile footprints, independent of date.

Tiles overlap each other -- that is what a stride smaller than the tile size buys --
so a split that assigns *tiles* to sides still leaves shared *pixels* on both sides
whenever two tiles that overlap land in different splits. A model trained on a pixel
and evaluated on a near-copy of it scores well without having learned anything, and
nothing in the loss curve reveals it.

So partitions here are defined on the ground, not on the tile list:

* a split is one or more contiguous strips along the AOI's long axis;
* a tile straddling a boundary, or sitting inside the buffer around it, belongs to
  **no** split and is discarded;
* reserved (annotation) regions are matched by footprint, on every date at once,
  because a held-out location is held out on the seventh of August too.

:func:`assert_disjoint` is the check that makes this verifiable rather than assumed,
and the training path runs it rather than trusting the geometry above.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from atarra.core.errors import AtarraError


class SpatialSplit(dict):
    """Index lists keyed ``train`` / ``val`` / ``test``, plus the gap that was used.

    A plain dict subclass so every existing ``splits["train"]`` keeps working, while
    the effective buffer -- which may be smaller than the one requested -- can travel
    into the run's metrics instead of living only in a log line.
    """

    def __init__(self, mapping: dict[str, list[int]], *, buffer_pixels: int) -> None:
        super().__init__(mapping)
        self.buffer_pixels = int(buffer_pixels)

# Cut positions are searched exhaustively only up to this many candidates per axis.
# Beyond it the candidates are thinned by quantile: a cut then lands within one tile
# width of the optimum, which is far below the precision the fractions deserve.
MAX_CUT_CANDIDATES = 96

# Broadcasting a whole split pair at once is fine for a store of a few hundred tiles
# and quadratic in memory beyond it, so pairs are compared in blocks of this height.
_COMPARE_BLOCK = 512


def tile_bounds(dataset) -> np.ndarray:
    """Half-open ``(top, left, bottom, right)`` pixel bounds per record, in grid pixels.

    Raises when a store predates stored offsets: without them the footprints are
    unknown, and every caller here exists precisely to reason about ground overlap.
    """
    default_size = getattr(dataset, "tile_size", None)
    if default_size is None:
        manifest = getattr(dataset, "manifest", None) or {}
        default_size = manifest.get("tile_size")
    if not default_size:
        raise AtarraError(
            "cannot determine the tile size, so tile footprints cannot be computed; "
            "pass a dataset exposing `tile_size` or a store manifest carrying one"
        )

    bounds = []
    for record in dataset.records:
        row = getattr(record, "row", None)
        col = getattr(record, "col", None)
        size = getattr(record, "size", None) or default_size
        if row is None or col is None:
            raise AtarraError(
                "tile footprints are missing; rebuild the store with offsets before "
                "splitting or reserving geographic regions"
            )
        bounds.append((int(row), int(col), int(row) + int(size), int(col) + int(size)))
    return np.asarray(bounds, dtype=np.int64).reshape(-1, 4)


def overlapping_pairs(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Every pair of footprints that share at least one pixel.

    Exact, not sampled. This is the property the split claims, so it is worth the
    quadratic comparison rather than a proxy like "blocks differ" -- which is what
    let overlapping train/test tiles through in the first place.
    """
    a = np.asarray(a, dtype=np.int64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.int64).reshape(-1, 4)
    if not len(a) or not len(b):
        return []

    pairs: list[tuple[int, int]] = []
    for start in range(0, len(a), _COMPARE_BLOCK):
        block = a[start : start + _COMPARE_BLOCK]
        # Half-open intervals: touching edges do not overlap.
        hits = (
            (block[:, None, 0] < b[None, :, 2])
            & (block[:, None, 2] > b[None, :, 0])
            & (block[:, None, 1] < b[None, :, 3])
            & (block[:, None, 3] > b[None, :, 1])
        )
        for row, col in np.argwhere(hits):
            pairs.append((start + int(row), int(col)))
    return pairs


def intersecting_bounds(
    bounds: np.ndarray, reserved: np.ndarray, *, buffer_pixels: int = 0
) -> np.ndarray:
    """Mark footprints overlapping any reserved region, or its surrounding gap."""
    if buffer_pixels < 0:
        raise AtarraError("the spatial buffer must be non-negative")
    bounds = np.asarray(bounds, dtype=np.int64).reshape(-1, 4)
    reserved = np.asarray(reserved, dtype=np.int64).reshape(-1, 4)
    overlaps = np.zeros(len(bounds), dtype=bool)
    # Padding the reserved box by the buffer turns "within `buffer` of the holdout"
    # into a plain intersection test against a grown rectangle.
    for top, left, bottom, right in np.unique(reserved, axis=0):
        overlaps |= (
            (bounds[:, 0] < bottom + buffer_pixels)
            & (bounds[:, 2] > top - buffer_pixels)
            & (bounds[:, 1] < right + buffer_pixels)
            & (bounds[:, 3] > left - buffer_pixels)
        )
    return overlaps


def _cut_candidates(ends: np.ndarray) -> np.ndarray:
    """Boundary positions worth trying, thinned by quantile past the search cap."""
    cuts = np.unique(ends)
    if len(cuts) <= MAX_CUT_CANDIDATES:
        return cuts
    positions = np.unique(np.linspace(0, len(cuts) - 1, MAX_CUT_CANDIDATES, dtype=int))
    return cuts[positions]


def split_bounds(
    bounds: np.ndarray,
    *,
    fractions: tuple[float, float, float],
    seed: int,
    buffer_pixels: int,
) -> SpatialSplit:
    """Partition footprints into three contiguous strips along the longest axis.

    Tiles crossing either cut, or inside ``buffer_pixels`` of it, are assigned to no
    split: they are dropped rather than leaked. Cuts depend only on unique footprints,
    so adding a date cannot move ground from one split to another, and the seed only
    decides which end becomes the training strip.
    """
    shares = np.asarray(fractions, dtype=float).reshape(-1)
    if (
        shares.shape != (3,)
        or not np.isfinite(shares).all()
        or np.any(shares <= 0)
        or not np.isclose(shares.sum(), 1.0)
    ):
        raise AtarraError("fractions must be three positive values summing to 1")
    if buffer_pixels < 0:
        raise AtarraError("the spatial buffer must be non-negative")
    if seed < 0:
        raise AtarraError("seed must be non-negative")

    bounds = np.asarray(bounds, dtype=np.int64).reshape(-1, 4)
    unique, inverse = np.unique(bounds, axis=0, return_inverse=True)
    # NumPy 2 changed the shape returned by `return_inverse` when `axis` is given;
    # flattening keeps this working on both the local 1.26 and Colab's 2.x.
    inverse = np.asarray(inverse).reshape(-1)[: len(bounds)]

    if len(unique) < 3:
        raise AtarraError(
            f"a geographic split needs at least 3 distinct tile footprints, and this "
            f"store has {len(unique)}. Use a larger area or smaller tiles."
        )
    spans = unique[:, 2:].max(axis=0) - unique[:, :2].min(axis=0)
    reverse = bool(np.random.default_rng(seed).integers(0, 2))

    # The long axis is tried first because strips across the short axis can leave a
    # region too narrow to hold a complete tile.
    for axis in np.argsort(-spans, kind="stable"):
        starts = unique[:, axis]
        ends = unique[:, axis + 2]
        if reverse:
            starts, ends = -ends, -starts
        cuts = _cut_candidates(ends)

        best_rank: tuple | None = None
        best_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for first_cut in cuts:
            first = ends <= first_cut
            if not first.any():
                continue
            after_first = starts >= first_cut + buffer_pixels
            if not after_first.any():
                continue
            for second_cut in cuts[cuts > first_cut + buffer_pixels]:
                second = after_first & (ends <= second_cut)
                if not second.any():
                    continue
                third = starts >= second_cut + buffer_pixels
                if not third.any():
                    continue
                counts = np.array([first.sum(), second.sum(), third.sum()], dtype=np.int64)
                discarded = len(unique) - int(counts.sum())
                # Fraction error first, then the least waste. Ties are broken by the
                # cut positions themselves so the result never depends on iteration
                # order -- two runs with one seed must agree exactly.
                rank = (
                    float(np.abs(counts / counts.sum() - shares).sum()),
                    discarded / len(unique),
                    discarded,
                    float(first_cut),
                    float(second_cut),
                )
                if best_rank is None or rank < best_rank:
                    best_rank, best_masks = rank, (first, second, third)

        if best_masks is not None:
            return SpatialSplit(
                {
                    name: np.flatnonzero(selected[inverse]).tolist()
                    for name, selected in zip(("train", "val", "test"), best_masks)
                },
                buffer_pixels=buffer_pixels,
            )

    raise AtarraError(
        f"this store cannot supply 3 non-empty geographic regions with a "
        f"{buffer_pixels}-pixel gap; use a larger area or smaller tiles, or lower "
        "--split-buffer if a tighter gap is defensible"
    )


def assert_disjoint(
    bounds: np.ndarray,
    groups: dict[str, Sequence[int]],
    *,
    label: str = "split",
) -> None:
    """Fail loudly if any two groups share a pixel.

    Called on every training run. The geometry above is meant to make overlap
    impossible, and a guard that never fires is the only way to know it did.
    """
    names = list(groups)
    bounds = np.asarray(bounds, dtype=np.int64).reshape(-1, 4)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            left = [index for index in groups[first] if index < len(bounds)]
            right = [index for index in groups[second] if index < len(bounds)]
            shared = overlapping_pairs(bounds[left], bounds[right])
            if shared:
                a, b = shared[0]
                raise AtarraError(
                    f"the {label} is not geographic: {len(shared)} tile pair(s) share "
                    f"pixels between {first!r} and {second!r}, e.g. "
                    f"{bounds[left[a]].tolist()} and {bounds[right[b]].tolist()}. "
                    "A model evaluated on ground it trained on reports a score that "
                    "measures memorisation."
                )
