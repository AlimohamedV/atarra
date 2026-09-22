"""On-disk tile store: build the dataset once, train from it many times.

Why this exists rather than training straight off composites.

Building a full training set means ranged reads of several gigabytes of Sentinel-2
scenes across a dozen overpasses. On Google Colab that cost cannot be paid inside the
same session that trains: free-tier sessions are capped and disconnect when idle, so a
fetch that dies at 90% takes the training run with it. Building writes shards to disk
(in the notebook, Drive) and training reads only those, which makes both steps
separately resumable and lets the expensive one happen once.

Why uint16 and not float16.

Sentinel-2 L2A arrives as integers with ``reflectance = DN * 1e-4``. Storing
reflectance as uint16 at 1e-4 scale therefore round-trips the archive *exactly*, at a
quarter of float32's size. float16 is smaller again, but it carries about three
significant decimal digits at magnitude 1.0 -- the same order as the red-edge
differences that separate reed from cropland in this project. Quantising the signal the
model is supposed to learn from, to save bytes we do not need, is a bad trade.

Shard layout::

    <root>/manifest.json
    <root>/shards/<date>/image.npy   uint16  [N, C, H, W]
    <root>/shards/<date>/mask.npy    int8    [N, H, W]   (-1 = ignore)
    <root>/shards/<date>/blocks.npy  int32   [N, 2]      (spatial block per tile)
    <root>/shards/<date>/offsets.npy int32   [N, 2]      (row, col of tile in grid)
    <root>/shards/<date>/review.npy  uint8   [N, H, W]   (1 = needs a human)
    <root>/shards/<date>/valid.npy   uint8   [N, H, W]   (1 = swath covered it)
    <root>/shards/<date>/keys.json   list[str]

The review mask is persisted because it is the annotation worklist. It records the
pixels the rule engine declined to be confident about, which is where human
adjudication buys the most accuracy per minute spent -- and re-deriving it after a
Colab session ends would mean re-fetching the imagery it came from.

The tile offsets matter for the same reason and are easy to overlook: a worklist that
says "these pixels need a human" without saying *where* them is unusable, because the
annotator cannot open them in QGIS. Row/col plus the manifest grid is what makes a
tile geographically addressable.

Tiles are memory-mapped, so a store larger than RAM is usable: only the pages actually
touched by a batch are read. In the Colab notebook the store is copied from Drive to
local disk first, because seek-heavy mmap over Drive's FUSE mount is slow.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger
from atarra.datasets.weak_labels import NUM_CLASSES, PHRAGMITES_CODE

log = get_logger("datasets.store")

FORMAT_VERSION = 1
STORED_DTYPE = np.uint16
REFLECTANCE_SCALE = 10000.0
MASK_DTYPE = np.int8
# Sentinel-2 reflectance stays inside [0, 1.5] for any real surface. This ceiling is
# the point at which uint16 at 1e-4 saturates, and nothing physical reaches it.
MAX_STORABLE_REFLECTANCE = np.iinfo(STORED_DTYPE).max / REFLECTANCE_SCALE


def encode_reflectance(values: np.ndarray) -> np.ndarray:
    """Quantise float reflectance to the stored integer form.

    Order matters. Non-finite values are cleared first, then the clip happens *before*
    the multiply. Scaling first and clipping after looks equivalent and is not: an
    infinity left in the array overflows the multiply, and the clip then has an
    infinity to catch -- which happens to saturate correctly today only because
    ``inf`` compares greater than the ceiling. A NaN is not so forgiving, and a NaN
    reaching ``astype`` is undefined behaviour rather than an error.
    """
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, MAX_STORABLE_REFLECTANCE)
    return np.rint(arr * REFLECTANCE_SCALE).astype(STORED_DTYPE)


def decode_reflectance(values: np.ndarray) -> np.ndarray:
    """Recover float reflectance from the stored integer form."""
    return np.asarray(values, dtype=np.float32) / REFLECTANCE_SCALE


@dataclass
class StoredTile:
    """One tile's location, carrying the spatial block the split depends on.

    Deliberately shaped like ``tile_dataset.TileRecord`` so ``geometric_split`` can
    partition a store without being modified or duplicated.
    """

    shard: int
    offset: int
    key: str
    block: tuple[int, int]
    row: int | None = None
    col: int | None = None


def _shard_dir(root: Path, date: str) -> Path:
    return Path(root) / "shards" / date


def write_shard(
    root: Path,
    date: str,
    *,
    images: np.ndarray,
    masks: np.ndarray,
    blocks: np.ndarray,
    keys: Sequence[str],
    num_classes: int,
    reviews: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    usable: np.ndarray | None = None,
    rule_labels: np.ndarray | None = None,
) -> dict:
    """Persist one date's tiles. Returns the manifest entry for the shard."""
    directory = _shard_dir(root, date)
    directory.mkdir(parents=True, exist_ok=True)

    np.save(directory / "image.npy", images)
    np.save(directory / "mask.npy", masks)
    np.save(directory / "blocks.npy", blocks)
    (directory / "keys.json").write_text(json.dumps(list(keys)), encoding="utf-8")
    if reviews is not None:
        np.save(directory / "review.npy", reviews.astype(np.uint8))
    if offsets is not None:
        np.save(directory / "offsets.npy", offsets.astype(np.int32))
    if usable is not None:
        np.save(directory / "valid.npy", usable.astype(np.uint8))
    if rule_labels is not None:
        np.save(directory / "rule.npy", rule_labels.astype(MASK_DTYPE))

    trainable = masks >= 0
    # `range` over a fixed class count rather than `masks.max()`: a shard whose tiles
    # happen to contain no reed, or no valid pixels at all, would otherwise report a
    # short counts list and silently misalign the per-class tallies in the manifest.
    counts = [
        int(((masks == code) & trainable).sum()) for code in range(num_classes)
    ]
    return {
        "date": date,
        "tiles": int(images.shape[0]),
        "class_counts": counts,
        "trainable_px": int(trainable.sum()),
        "usable_px": int(usable.sum()) if usable is not None else None,
        "review_px": int(reviews.sum()) if reviews is not None else None,
    }


def build_store(
    root: Path,
    area_key: str,
    dates: Sequence,
    *,
    gsd: float = 20.0,
    tile_size: int = 256,
    stride: int | None = None,
    max_size: int = 8192,
    window_days: int = 3,
    drop_empty: bool = False,
    source=None,
    max_cloud_cover: float | None = None,
    first: bool = False,
) -> dict:
    """Build (or add to) a tile store, one shard per date.

    Deliberately streams: one composite is materialised, tiled, written, and released
    before the next date is fetched. Accumulating every composite first would need
    ~7.6 GB at 10 m for this AOI, which is the failure this design exists to avoid.

    ``first`` truncates any existing store so a rebuild cannot silently mix two
    different configurations into one manifest.
    """
    from atarra.core.config import get_bands
    from atarra.datasets.tile_dataset import CompositeTileDataset
    from atarra.pipeline import load_composite

    root = Path(root)
    cfg = get_bands()

    if first and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    existing = root / "manifest.json"
    if existing.exists() and not first:
        raise AtarraError(
            f"{root} already holds a store; pass --rebuild to replace it, or choose "
            "another directory. Appending across configurations would produce a "
            "manifest that misdescribes half its shards."
        )

    manifest: dict = {
        "format_version": FORMAT_VERSION,
        "area": area_key,
        "gsd": float(gsd),
        "tile_size": int(tile_size),
        "stride": stride,
        "bands": list(cfg.bands_8),
        "dtype": np.dtype(STORED_DTYPE).name,
        "scale": REFLECTANCE_SCALE,
        "shards": [],
        "skipped": [],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    for date in dates:
        day = date.isoformat() if hasattr(date, "isoformat") else str(date)
        try:
            composite = load_composite(
                area_key,
                date,
                window_days=window_days,
                gsd=gsd,
                max_size=max_size,
                # Explicit bands: the store holds the full multispectral stack so that
                # the RGB control arm of the experiment can be cut at read time with
                # no second download.
                bands=list(cfg.bands_8),
                indices=["ndvi", "ndwi", "ndre", "ndmi"],
                source=source,
                max_cloud_cover=max_cloud_cover,
            )
        except AtarraError as error:
            log.warning("%s: skipped (%s)", day, error)
            manifest["skipped"].append({"date": day, "reason": str(error)})
            continue

        # `choose_grid` coarsens the grid until the AOI fits in `max_size`. If that
        # happened, the store would hold 40 m pixels while the manifest claimed 20 m
        # and the model would silently train at the wrong scale.
        actual_gsd = float(composite.grid.resolution[0])
        if abs(actual_gsd - float(gsd)) > 1e-6:
            raise AtarraError(
                f"{day}: grid resolution is {actual_gsd} m, not the requested {gsd} m. "
                f"Grid was coarsened to fit max_size={max_size}; raise max_size or use "
                "a larger gsd so the stored tiles match the manifest."
            )

        grid_meta = {
            "crs": composite.grid.crs.to_string(),
            "width": int(composite.grid.width),
            "height": int(composite.grid.height),
            "transform": [float(v) for v in composite.grid.transform[:6]],
        }
        if "grid" not in manifest:
            manifest["grid"] = grid_meta
        elif manifest["grid"] != grid_meta:
            # Spatial blocks are only comparable across shards if the grid is shared;
            # otherwise the same block key means different ground in different shards
            # and the geometric split leaks.
            raise AtarraError(
                f"{day}: grid {grid_meta} differs from the store's grid "
                f"{manifest['grid']}. All shards must share one grid for the spatial "
                "split to mean anything."
            )

        tiled = CompositeTileDataset(
            [composite],
            tile_size=tile_size,
            stride=stride,
            augment=False,
            drop_empty=drop_empty,
        )
        if not tiled.records:
            manifest["skipped"].append({"date": day, "reason": "no tiles"})
            continue

        items = [tiled[i] for i in range(len(tiled))]
        images = np.stack([encode_reflectance(item["image"]) for item in items], axis=0)
        masks = np.stack([item["mask"].astype(MASK_DTYPE) for item in items], axis=0)
        reviews = np.stack([item["review"] for item in items], axis=0)
        usable = np.stack([item["usable"] for item in items], axis=0)
        rule_labels = np.stack([item["rule_labels"] for item in items], axis=0)
        blocks = np.array([tiled.records[i].block for i in range(len(tiled))], dtype=np.int32)
        offsets = np.array(
            [(tiled.records[i].row, tiled.records[i].col) for i in range(len(tiled))],
            dtype=np.int32,
        )
        # The date is prefixed because `TileRecord.key` is `c{index}/r{row}_c{col}`,
        # and `index` is the position of the composite within one build call -- always
        # 0 here, since each date is tiled separately. Without the prefix every date
        # would produce a tile called `c0/r0_c0`: twelve shards sharing keys, so an
        # exported chip set would overwrite itself and a held-out key would silently
        # reserve the same position on every date.
        keys = [f"{day}/{tiled.records[i].key}" for i in range(len(tiled))]

        entry = write_shard(
            root,
            day,
            images=images,
            masks=masks,
            blocks=blocks,
            keys=keys,
            num_classes=NUM_CLASSES,
            reviews=reviews,
            offsets=offsets,
            usable=usable,
            rule_labels=rule_labels,
        )
        entry.update(
            {
                "coverage": round(float(composite.coverage), 4),
                "scenes": len(composite.scene_ids),
                "usable_px": int(composite.stack.valid.sum()),
            }
        )
        manifest["shards"].append(entry)

        log.info(
            "%s: %d tiles (%d scenes, %.1f%% coverage)",
            day,
            entry["tiles"],
            entry["scenes"],
            entry["coverage"] * 100,
        )

        # Release before the next date is fetched; this is the streaming guarantee.
        del composite, tiled, items, images, masks, reviews, usable, rule_labels

    if not manifest["shards"]:
        raise AtarraError(
            f"no usable imagery for {area_key} in the requested date range; "
            "nothing was written"
        )

    total_tiles = sum(s["tiles"] for s in manifest["shards"])
    merged = np.zeros(NUM_CLASSES, dtype=np.int64)
    for shard in manifest["shards"]:
        counts = shard["class_counts"]
        for code, value in enumerate(counts[: len(merged)]):
            merged[code] += value
    manifest["totals"] = {
        "tiles": total_tiles,
        "shards": len(manifest["shards"]),
        "class_counts": merged.tolist(),
        "trainable_px": int(sum(s["trainable_px"] for s in manifest["shards"])),
        "usable_px": int(sum(s["usable_px"] or 0 for s in manifest["shards"])),
        "review_px": int(sum(s["review_px"] or 0 for s in manifest["shards"])),
    }

    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info(
        "store written to %s: %d tiles across %d shard(s), classes %s",
        root,
        total_tiles,
        len(manifest["shards"]),
        merged.tolist(),
    )
    return manifest


def load_manifest(root: Path | str) -> dict:
    """Read and sanity-check a store's manifest."""
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise AtarraError(
            f"{root} is not a tile store (no manifest.json). Build one with "
            "`atarra dataset build`."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise AtarraError(
            f"store format {manifest.get('format_version')} is not supported "
            f"(this build reads {FORMAT_VERSION}); rebuild the store"
        )
    return manifest


class TileStoreDataset:
    """Memory-mapped view over a tile store, shaped like ``CompositeTileDataset``.

    Keeps ``records`` objects exposing ``.block`` so ``geometric_split`` applies
    unchanged, and returns the same item dict, so the trainer cannot tell the two
    dataset types apart.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        band_names: Sequence[str] | None = None,
        augment: bool = False,
        seed: int = 0,
        holdout_keys: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = load_manifest(self.root)
        self.augment = augment
        self.seed = int(seed)
        self.holdout_keys = frozenset(holdout_keys or ())

        stored = list(self.manifest["bands"])
        self.stored_bands = stored
        requested = list(band_names or stored)
        missing = [name for name in requested if name not in stored]
        if missing:
            raise AtarraError(
                f"store holds {stored}; cannot select {missing}. Rebuild the store to "
                "add bands -- reading them here would mean a second download."
            )
        self.band_names = requested
        self._channels = [stored.index(name) for name in requested]

        self._shards: list[dict] = []
        self.records: list[StoredTile] = []
        for shard_index, entry in enumerate(self.manifest["shards"]):
            directory = _shard_dir(self.root, entry["date"])
            images = np.load(directory / "image.npy", mmap_mode="r")
            masks = np.load(directory / "mask.npy", mmap_mode="r")
            blocks = np.load(directory / "blocks.npy")
            keys = json.loads((directory / "keys.json").read_text(encoding="utf-8"))
            # Absent in stores written before the review mask was persisted; such a
            # store still trains, it just cannot offer its annotation worklist.
            review_path = directory / "review.npy"
            reviews = np.load(review_path, mmap_mode="r") if review_path.exists() else None
            offset_path = directory / "offsets.npy"
            offsets = np.load(offset_path) if offset_path.exists() else None
            valid_path = directory / "valid.npy"
            valid = np.load(valid_path, mmap_mode="r") if valid_path.exists() else None
            rule_path = directory / "rule.npy"
            rules = np.load(rule_path, mmap_mode="r") if rule_path.exists() else None
            self._shards.append(
                {
                    "images": images,
                    "masks": masks,
                    "reviews": reviews,
                    "valid": valid,
                    "rules": rules,
                    "date": entry["date"],
                }
            )
            for offset in range(images.shape[0]):
                # Reserved tiles are dropped here rather than filtered later, so that
                # every downstream index and split covers only what may be trained on.
                if keys[offset] in self.holdout_keys:
                    continue
                self.records.append(
                    StoredTile(
                        shard=shard_index,
                        offset=offset,
                        key=keys[offset],
                        block=(int(blocks[offset, 0]), int(blocks[offset, 1])),
                        row=None if offsets is None else int(offsets[offset, 0]),
                        col=None if offsets is None else int(offsets[offset, 1]),
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        shard = self._shards[record.shard]

        image = decode_reflectance(shard["images"][record.offset])[self._channels]
        mask = np.asarray(shard["masks"][record.offset], dtype=np.int64)

        # Invalid pixels are stored as -1 already, but re-assert it so a store written
        # by an older build cannot leak unlabelled pixels into the loss.
        mask = np.where(mask < 0, -1, mask)

        if self.augment:
            from atarra.datasets.tile_dataset import augment_tile

            image, mask = augment_tile(image, mask, seed=self.seed, index=index)

        item = {
            "image": np.ascontiguousarray(image),
            "mask": np.ascontiguousarray(mask),
            "key": record.key,
            "block": record.block,
        }

        if not self.augment:
            # Annotation-support arrays, supplied only when nothing has been spatially
            # transformed. `augment_tile` flips and rotates the image and mask, and these
            # would stay put -- a desynced review mask would point an annotator at the
            # wrong ground, which is worse than not having one.
            valid = shard["valid"]
            if valid is None:
                raise AtarraError(
                    f"tile {record.key} has no stored validity mask, so it cannot be "
                    "exported for annotation. Rebuild the store. Training is unaffected."
                )
            item["usable"] = np.ascontiguousarray(valid[record.offset]).astype(bool)
            rules = shard["rules"]
            item["rule_labels"] = np.ascontiguousarray(
                rules[record.offset] if rules is not None else mask
            ).astype(np.int64)
            reviews = shard["reviews"]
            if reviews is not None:
                item["review"] = np.ascontiguousarray(reviews[record.offset]).astype(bool)
        return item

    def grid(self):
        """Reconstruct the common grid the shards were cut from."""
        from affine import Affine
        from rasterio.crs import CRS

        from atarra.core.grids import Grid

        meta = self.manifest["grid"]
        return Grid(
            crs=CRS.from_string(meta["crs"]),
            transform=Affine(*meta["transform"]),
            width=int(meta["width"]),
            height=int(meta["height"]),
        )

    def tile_transform(self, index: int):
        """Rasterio-style affine placing one tile on the ground.

        This is what turns the review mask from "some pixels need a human" into a
        window an annotator can actually open in QGIS. ``@`` rather than ``*`` for the
        composition: affine 3 deprecates ``*``, and it is removed in affine 4.
        """
        from affine import Affine

        record = self.records[index]
        if record.row is None or record.col is None:
            raise AtarraError(
                f"tile {record.key} has no stored offset, so it cannot be georeferenced; "
                "rebuild the store with a current version of `atarra dataset build`"
            )
        return self.grid().transform @ Affine.translation(record.col, record.row)

    def tile_corners_wgs84(self, index: int) -> list[list[float]]:
        """The tile's four corners as ``[lon, lat]``, for export or display."""
        from atarra.core.grids import Grid

        size = int(self.manifest["tile_size"])
        base = self.grid()
        return Grid(
            crs=base.crs,
            transform=self.tile_transform(index),
            width=size,
            height=size,
        ).corners_wgs84()

    def label_at(self, index: int) -> np.ndarray:
        """The stored class mask for one tile, ``-1`` where not usable.

        Reads the mask without decoding the imagery, which matters when ranking a whole
        store rather than inspecting one tile.
        """
        record = self.records[index]
        return np.asarray(self._shards[record.shard]["masks"][record.offset], dtype=np.int64)

    def rule_labels_at(self, index: int) -> np.ndarray:
        """The rule engine's ungated labels for one tile (``-1`` only where unusable).

        Falls back to the gated mask for stores written before this was persisted, in
        which case ambiguous pixels read as -1 and simply show as unlabelled.
        """
        record = self.records[index]
        rules = self._shards[record.shard]["rules"]
        if rules is None:
            return self.label_at(index)
        return np.asarray(rules[record.offset], dtype=np.int64)

    def usable_at(self, index: int) -> np.ndarray | None:
        """Which pixels the satellite swath actually covered, or None if not stored."""
        record = self.records[index]
        valid = self._shards[record.shard]["valid"]
        if valid is None:
            return None
        return np.asarray(valid[record.offset], dtype=bool)

    def review_at(self, index: int) -> np.ndarray | None:
        """The annotation worklist for one tile.

        True where the rule engine declined to be confident -- the pixels where a
        human judgement improves the training set most per minute spent. Returns
        ``None`` for a store written before the mask was persisted.
        """
        record = self.records[index]
        reviews = self._shards[record.shard]["reviews"]
        if reviews is None:
            return None
        return np.asarray(reviews[record.offset], dtype=bool)

    def review_fraction(self) -> float | None:
        """Share of stored pixels awaiting human adjudication, or None if unknown."""
        totals = self.manifest["totals"]
        total = totals.get("review_px")
        usable = totals.get("usable_px")
        if total is None or not usable:
            return None
        # Over usable pixels: review is a subset of usable, so the frame's empty
        # corners must not be in the denominator.
        return round(total / max(1, usable), 5)

    def class_counts(self, indices: Sequence[int] | None = None) -> np.ndarray:
        """Pixel tally per class, optionally restricted to a subset of tiles.

        Loss weights must come from the *training* split. Weighting them from the
        whole store leaks the validation and test class balance into a training-time
        decision, which is the subtle kind of leakage that makes a held-out score
        look better than the model is.
        """
        if indices is None:
            return np.array(self.manifest["totals"]["class_counts"], dtype=np.int64)

        counts = np.zeros(len(self.manifest["totals"]["class_counts"]), dtype=np.int64)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for position, index in enumerate(indices):
            record = self.records[index]
            by_shard.setdefault(record.shard, []).append((position, record.offset))

        for shard_index, pairs in by_shard.items():
            masks = self._shards[shard_index]["masks"]
            offsets = [offset for _, offset in pairs]
            # One fancy-index read per shard rather than a tile at a time; masks are
            # small (256x256 int8) so this stays cheap even across thousands of tiles.
            stacked = np.asarray(masks[offsets])
            trainable = stacked >= 0
            for code in range(len(counts)):
                counts[code] += int(((stacked == code) & trainable).sum())
        return counts

    def index_statistics(self, indices: Sequence[int] | None = None) -> dict:
        """Class balance across stored tiles, to set loss weights sensibly."""
        from atarra.datasets.tile_dataset import summarise_counts

        return summarise_counts(self.class_counts(indices))

    def class_weights(
        self, *, scheme: str = "inverse_frequency", indices: Sequence[int] | None = None
    ) -> np.ndarray:
        from atarra.datasets.tile_dataset import class_weights_from_counts

        counts = self.class_counts(indices)
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

    def band_statistics(self, sample_limit: int = 200) -> dict:
        """Per-band mean/std over a sample of tiles.

        Feeds the model's normalisation buffers. Sampling rather than sweeping every
        tile keeps this cheap on a store that may not fit in RAM.
        """
        if not len(self.records):
            raise AtarraError("store has no tiles")
        step = max(1, len(self.records) // max(1, sample_limit))
        stacked: list[np.ndarray] = []
        for index in range(0, len(self.records), step):
            stacked.append(self[index]["image"])
        block = np.concatenate([a.reshape(a.shape[0], -1) for a in stacked], axis=1)
        return {
            "mean": block.mean(axis=1).round(6).tolist(),
            "std": np.maximum(block.std(axis=1), 1e-4).round(6).tolist(),
            "bands": list(self.band_names),
            "sampled_tiles": len(stacked),
        }

    def dates(self) -> list[str]:
        return [entry["date"] for entry in self.manifest["shards"]]

    def annotation_ranking(
        self, *, strategy: str = "uncertainty", seed: int = 0
    ) -> list[tuple[float, int]]:
        """``(score, tile index)`` for every candidate tile, best first.

        Which tile is "best" depends on what the labels are for, so the strategy is
        explicit rather than implied:

        * ``uncertainty`` (default) - review density over *usable* ground. Finds what
          the rule engine cannot call, which is where a human decision improves the
          training set most per pixel. It is a deliberately **hard-case** selection:
          a test set built this way measures the model on the hardest pixels in the
          area, which is defensible but is not a representative sample.
        * ``reed`` - tiles richest in Phragmites pixels. A test set needs enough of the
          class it is measuring, and reed is only ~1.8% of this imagery, so an
          arbitrary selection can leave too few reed pixels for the per-class IoU to
          mean anything. One run of this project produced "reed IoU 0.0" from 36
          support pixels, which is a measurement of nothing.
        * ``random`` - a seeded sample, for a test set that is representative rather
          than adversarial.

        Exposed separately from :meth:`annotation_tiles` because the exporter needs the
        tile index to reach the pixels, while the CLI only needs the description.
        """
        key = strategy.strip().lower()
        if key not in {"uncertainty", "reed", "random"}:
            raise AtarraError(
                f"unknown annotation strategy {strategy!r}; expected 'uncertainty', "
                "'reed' or 'random'"
            )

        if key == "random":
            order = np.random.default_rng(seed).permutation(len(self.records))
            return [(float(len(order) - rank), int(index)) for rank, index in enumerate(order)]

        scored: list[tuple[float, int]] = []
        for index in range(len(self.records)):
            if key == "reed":
                labels = self.label_at(index)
                count = int((labels == PHRAGMITES_CODE).sum())
                if count > 0:
                    scored.append((float(count), index))
                continue

            mask = self.review_at(index)
            if mask is None:
                continue
            # Density over *usable* pixels. Dividing by the whole tile instead makes
            # this ratio a measure of how much of the bounding box the rotated swath
            # covers -- typically the dominant term -- so a tile that is 98% empty
            # ground ranks as maximally uncertain.
            usable = self.usable_at(index)
            if usable is None:
                raise AtarraError(
                    "this store predates the validity mask, so review density cannot be "
                    "computed without confusing empty swath corners for uncertainty. "
                    "Rebuild the store, or train with it as-is -- the ranking is only "
                    "needed for annotation."
                )
            labelled = int(usable.sum())
            if labelled == 0:
                continue
            density = float(mask[usable].mean())
            if density > 0:
                scored.append((density, index))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored

    def annotation_tiles(
        self, *, limit: int | None = None, strategy: str = "uncertainty", seed: int = 0
    ) -> list[dict]:
        """Tiles worth annotating first, described for display."""
        scored = self.annotation_ranking(strategy=strategy, seed=seed)
        if limit is not None:
            scored = scored[:limit]
        return [
            {
                "key": self.records[index].key,
                "date": self._shards[self.records[index].shard]["date"],
                "score": round(score, 5),
                "reed_px": int((self.label_at(index) == PHRAGMITES_CODE).sum()),
                "corners_wgs84": self.tile_corners_wgs84(index),
            }
            for score, index in scored
        ]

    def describe(self) -> dict:
        totals = self.manifest["totals"]
        return {
            "root": str(self.root),
            "area": self.manifest["area"],
            "gsd": self.manifest["gsd"],
            "tile_size": self.manifest["tile_size"],
            "stride": self.manifest.get("stride"),
            "stored_bands": self.stored_bands,
            "selected_bands": self.band_names,
            "shards": len(self.manifest["shards"]),
            "dates": self.dates(),
            "tiles": len(self.records),
            "class_counts": totals["class_counts"],
            "trainable_px": totals["trainable_px"],
            "usable_px": totals.get("usable_px"),
            "review_px": totals.get("review_px"),
            "review_fraction": self.review_fraction(),
        }


def iter_batches(dataset: TileStoreDataset, batch_size: int) -> Iterator[dict]:
    """Plain index batching, for callers that do not want a torch DataLoader."""
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        yield {
            "image": np.stack([item["image"] for item in items], axis=0),
            "mask": np.stack([item["mask"] for item in items], axis=0),
            "keys": [item["key"] for item in items],
        }
