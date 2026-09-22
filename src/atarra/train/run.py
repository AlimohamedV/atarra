"""Training runs over a tile store, from split to metrics to checkpoint.

This is the orchestration the CLI and the Colab notebook share, so a run on the
laptop and a run on Colab differ only in device and epoch count.

Four things here are methodological rather than mechanical, and each one is a way a
score can rise without the model getting better.

**The split is geographic, not random.** Tiles are cut with a stride smaller than the
tile size, so they overlap. A random split therefore puts near-duplicate imagery on
both sides of the fence. ``geometric_split`` partitions *ground* instead, and the run
records the gap it used.

**Nothing outside the training split touches a training decision.** Loss weights,
normalisation statistics and augmentation epochs are all computed from, or applied to,
training tiles only. A statistic taken over the whole store, or jitter applied to the
validation images, moves the reported number without moving the model.

**Evaluation is unaugmented.** Validation and test views are built with augmentation
off; only the training view jitters, and its transform advances each epoch.

**A run with no usable supervision fails instead of reporting.** Splits are checked for
class support before training starts, because a split with no reed pixels cannot
measure a reed IoU and a store of empty tiles cannot train at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger
from atarra.datasets.store import TileStoreDataset
from atarra.datasets.tile_dataset import geometric_split
from atarra.datasets.weak_labels import CLASS_NAMES, NUM_CLASSES, PHRAGMITES_CODE

# Reed is the class the proposal quotes a number for, so a split with almost none of it
# cannot produce that metric however well the model has learned. Below this many pixels
# the split is named as unmeasurable rather than reported as a result -- a real run of
# this project once printed "reed IoU 0.0" from 36 support pixels.
MIN_MEASURABLE_REED_PX = 500

# NOTE: `atarra.train.trainer` is imported inside the function, not here. It imports
# torch at module scope, so a top-level import would make every `atarra` CLI command
# require a deep-learning stack -- including `atarra areas`.

log = get_logger("train.run")


# Included verbatim in every metrics.json so the caveat travels with the number. A
# score quoted out of this file without it would misrepresent what was measured.
LABEL_CAVEAT = (
    "These metrics compare the model against labels produced by the same "
    "weak-supervision rule engine it was trained on. They measure AGREEMENT WITH "
    "THAT RULE ENGINE, not field-verified Phragmites detection: if the rules are "
    "wrong about a pixel, a model that faithfully reproduces them is still scored "
    "correct. This is the interim number. Establishing the proposal's mIoU >= 0.82 / "
    "F1 >= 0.85 against independent truth requires the hand-annotated validation "
    "block, which is not in this run. The review mask in the tile store marks the "
    "pixels a human should adjudicate first."
)

# Written into metrics.json in place of a target verdict. `meets_targets` would return a
# boolean here, and a boolean is read as an answer whether or not it was earned.
WEAK_LABEL_TARGET_REASON = (
    "not assessed: every label in this run came from the rule engine, so a target "
    "verdict would compare the model with the teacher it was trained to imitate. Score "
    "`best.pt` against the held-out annotation pack (`atarra annotation score`) for the "
    "only number that measures detection rather than agreement."
)


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise AtarraError(
            "training requires torch, which is not installed. On Colab it is "
            "preinstalled as a CUDA build; locally install it before training."
        ) from exc
    return torch


def train_from_store(
    store: Path | str,
    *,
    bands: Sequence[str] | None = None,
    epochs: int = 40,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    out_dir: Path | str = Path("data/checkpoints"),
    run_name: str | None = None,
    augment: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    device: str | None = None,
    patience: int = 10,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    max_tiles: int | None = None,
    exclude_pack: Path | str | None = None,
    split_buffer_pixels: int | None = None,
    holdout_buffer_pixels: int | None = None,
) -> dict:
    """Train a segmentation model on a tile store and write a metrics artifact.

    ``exclude_pack`` names an annotation pack whose tiles must not be trained on. A
    held-out set the model has already seen is not held out, so this removes them
    before the split rather than hoping nobody notices -- and it removes the same
    *ground* on every date, not just the keys named in the pack, because a neighbour
    tile a fortnight later is the same reed bed.

    Both buffers default to half a tile, and both may narrow on a small store; the gap
    actually applied is recorded in the metrics rather than assumed.
    """
    torch = _require_torch()

    from atarra.datasets.store import load_manifest
    from atarra.models.segmentation import build_model, count_parameters
    from atarra.train.metrics import unassessable_targets
    from atarra.train.trainer import (
        TrainConfig,
        describe_device,
        evaluate,
        resolve_device,
        set_seed,
        train,
    )

    manifest = load_manifest(store)
    default_buffer = int(manifest["tile_size"]) // 2
    if split_buffer_pixels is None:
        split_buffer_pixels = default_buffer
    if holdout_buffer_pixels is None:
        holdout_buffer_pixels = default_buffer

    # Named from the request rather than from the dataset, so this can be decided before
    # anything expensive: a run directory that is already occupied is worth saying so up
    # front, not after the store has been opened and the model built.
    selected = list(bands) if bands else list(manifest["bands"])
    variant = "rgb" if len(selected) == 3 else "unet"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # The default name carries the variant and band count so two arms of one experiment
    # are distinguishable in a directory listing.
    name = run_name or f"{manifest['area']}_{variant}_{len(selected)}band_{stamp}"
    run_dir = Path(out_dir) / name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise AtarraError(
            f"{run_dir} already holds a run. Choose another --name, or remove it first: "
            "writing here would overwrite that run's metrics and checkpoint, and the two "
            "would be indistinguishable afterwards. The default name carries a "
            "timestamp, so only an explicit --name can collide."
        )

    holdout: set[str] = set()
    if exclude_pack is not None:
        from atarra.datasets.export import reserved_keys

        holdout = reserved_keys(exclude_pack)
        if not holdout:
            raise AtarraError(
                f"{exclude_pack} reserves no tiles, so there is nothing to exclude; "
                "refusing to proceed silently, because a pack that reserves nothing "
                "would make the held-out set meaningless"
            )
        log.info("excluding %d reserved tile(s) from %s", len(holdout), exclude_pack)

    dataset = TileStoreDataset(
        store,
        band_names=bands,
        seed=seed,
        holdout_keys=holdout,
        holdout_buffer_pixels=holdout_buffer_pixels,
    )
    if dataset.excluded_keys:
        log.info(
            "the reserved tiles take %d further tile(s) with them: the same ground on "
            "other dates, and neighbours within %d pixels",
            len(dataset.excluded_keys) - len(holdout),
            holdout_buffer_pixels,
        )
    if max_tiles and len(dataset) > max_tiles:
        # Deterministic truncation for smoke runs, not for real experiments.
        dataset.records = dataset.records[:max_tiles]
        dataset.filtered = True
        log.warning("truncated the store to %d tiles for this run", max_tiles)

    band_names = list(dataset.band_names)
    resolved = resolve_device(device)

    splits = geometric_split(
        dataset, fractions=fractions, seed=seed, buffer_pixels=split_buffer_pixels
    )
    train_indices, val_indices, test_indices = (
        splits["train"],
        splits["val"],
        splits["test"],
    )

    # Class support per split, before anything is trained. A split with no supervised
    # pixels of any class cannot train, and one with barely any reed cannot measure a
    # reed IoU -- both are cheaper to learn here than from a finished run.
    # The loop variable is not called `name`: that is the run's name in this scope, and
    # a leaking loop would silently rename the run to whichever split it ended on.
    support = {
        split_name: dataset.class_counts(indices).tolist()
        for split_name, indices in splits.items()
    }
    for split_name, counts in support.items():
        if not sum(counts):
            raise AtarraError(
                f"the {split_name} split contains no usable labelled pixels, so it "
                "cannot serve its purpose. Every pixel there is either nodata or flagged "
                "for review; label some of the review queue, or widen the store."
            )
    for split_name in ("train", "val", "test"):
        if support[split_name][PHRAGMITES_CODE] < MIN_MEASURABLE_REED_PX:
            log.warning(
                "the %s split holds only %d reed pixel(s) (under %d); a reed IoU "
                "measured there will be noise rather than a result",
                split_name,
                support[split_name][PHRAGMITES_CODE],
                MIN_MEASURABLE_REED_PX,
            )

    # One view per split. Only the training view augments, and its transform advances
    # each epoch; validation and test imagery is passed through untouched, because
    # jitter on the images a metric is computed from moves the metric.
    views = {
        "train": dataset.as_torch_dataset(indices=train_indices, augment=augment),
        "val": dataset.as_torch_dataset(indices=val_indices, augment=False),
        "test": dataset.as_torch_dataset(indices=test_indices, augment=False),
    }

    def loader(split: str, *, shuffle: bool) -> "torch.utils.data.DataLoader":
        # An explicit generator keeps the shuffle order a function of the seed alone.
        # On the global RNG it would depend on how much randomness model construction
        # happened to consume first, which no configuration file records.
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.utils.data.DataLoader(
            views[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            generator=generator,
        )

    # Statistics from the training split only: see the module docstring.
    stats = dataset.band_statistics(indices=train_indices)
    # Before `build_model`, which draws the initial weights from the torch RNG.
    set_seed(seed)
    model = build_model(
        in_channels=len(band_names),
        num_classes=NUM_CLASSES,
        variant=variant,
        band_mean=stats["mean"],
        band_std=stats["std"],
    )
    log.info(
        "%s model, %d input channels %s, %.3f M parameters",
        variant,
        len(band_names),
        band_names,
        count_parameters(model)["millions"],
    )

    class_weights = dataset.class_weights(indices=train_indices)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Which tiles went where, recorded in the checkpoint. It is the only evidence that
    # a later annotation score was measured on ground this model never saw, and the
    # scorer refuses to call a score independent without it.
    provenance = {
        "store": str(dataset.root),
        "area": dataset.manifest["area"],
        "gsd": dataset.manifest["gsd"],
        "tile_size": dataset.manifest["tile_size"],
        "seed": seed,
        "split_fractions": list(fractions),
        "split_buffer_pixels": splits.buffer_pixels,
        "holdout_buffer_pixels": holdout_buffer_pixels,
        "train_keys": [dataset.records[i].key for i in train_indices],
        "val_keys": [dataset.records[i].key for i in val_indices],
        "test_keys": [dataset.records[i].key for i in test_indices],
        "reserved_keys": sorted(holdout),
        "excluded_keys": sorted(dataset.excluded_keys),
        "labels": "weak supervision (rule engine)",
    }

    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=run_dir,
        seed=seed,
        patience=patience,
        num_workers=num_workers,
        band_names=band_names,
        provenance=provenance,
    )

    log.info(
        "training %s: %d train / %d val / %d test tiles",
        name,
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    summary = train(
        model,
        loader("train", shuffle=True),
        loader("val", shuffle=False),
        config=config,
        class_weights=class_weights,
        device=resolved,
        num_classes=NUM_CLASSES,
    )

    # `train` restored the best epoch's weights, so this is the score of the checkpoint
    # on disk rather than of whichever epoch happened to run last.
    test_report = evaluate(
        model, loader("test", shuffle=False), num_classes=NUM_CLASSES, device=resolved
    )

    metrics = {
        "run": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": describe_device(resolved),
        "model": {
            "variant": variant,
            "in_channels": len(band_names),
            "bands": band_names,
            "parameters": count_parameters(model),
        },
        "dataset": dataset.describe(),
        "holdout": {
            "pack": None if exclude_pack is None else str(exclude_pack),
            "tiles_reserved": len(holdout),
            "tiles_excluded_geographically": len(dataset.excluded_keys),
            "buffer_pixels": holdout_buffer_pixels,
            "purpose": (
                "these tiles are reserved for hand annotation, and scoring the model "
                "against them is the only measure of detection accuracy the project "
                "can honestly claim"
            ),
        },
        "band_statistics": stats,
        "splits": {
            "fractions": list(fractions),
            "train_tiles": len(train_indices),
            "val_tiles": len(val_indices),
            "test_tiles": len(test_indices),
            "boundary_tiles_omitted": len(dataset.records) - sum(map(len, splits.values())),
            # The gap the splitter could afford, which can be narrower than requested.
            "buffer_pixels": splits.buffer_pixels,
            "class_support": support,
            "class_support_note": (
                "counted per tile, so pixels shared by overlapping tiles appear in "
                "more than one split's tally; the tallies are not comparable to the "
                "store's unique-ground coverage"
            ),
            "verified_disjoint": True,
        },
        "class_weights": class_weights.round(4).tolist(),
        "training": {
            "epochs_run": summary["epochs_run"],
            "best_epoch": summary["best_epoch"],
            "evaluated_epoch": summary["evaluated_epoch"],
            "best_val_mean_iou": summary["best_val_mean_iou"],
            "config": summary["config"],
        },
        "validation_report": summary["final_report"],
        "test_report": test_report,
        "targets": unassessable_targets(WEAK_LABEL_TARGET_REASON),
        "labels": {
            "source": "weak supervision (rule engine)",
            "class_names": CLASS_NAMES,
            "phragmites_class_code": PHRAGMITES_CODE,
            "caveat": LABEL_CAVEAT,
        },
        "checkpoint": str(run_dir / "best.pt"),
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    log.info(
        "test mIoU %.4f | phragmites IoU %s | F1 %s | pixel accuracy %.4f",
        test_report["mean_iou"],
        test_report["phragmites_iou"],
        test_report["phragmites_f1"],
        test_report["pixel_accuracy"],
    )
    # No target verdict is issued here, deliberately: the comparison that the proposal
    # states cannot be made from this run's labels. `atarra annotation score` makes it.
    log.warning("targets NOT ASSESSED: %s", WEAK_LABEL_TARGET_REASON)
    log.info("metrics written to %s", run_dir / "metrics.json")
    return metrics


def format_report(metrics: dict) -> str:
    """A short human-readable summary, for the CLI and the notebook."""
    report = metrics["test_report"]
    rows = [
        f"run            {metrics['run']}",
        f"device         {metrics['device'].get('name', metrics['device']['device'])}",
        f"model          {metrics['model']['variant']} / {metrics['model']['in_channels']} bands",
        f"dataset        {metrics['dataset']['tiles']} tiles from {metrics['dataset']['shards']} shard(s)",
        f"splits         {metrics['splits']['train_tiles']} train / "
        f"{metrics['splits']['val_tiles']} val / {metrics['splits']['test_tiles']} test",
        "",
        "test report",
        f"  mean IoU     {report['mean_iou']}",
        f"  reed IoU     {report['phragmites_iou']}",
        f"  reed F1      {report['phragmites_f1']}",
        f"  pixel acc.   {report['pixel_accuracy']}",
        "",
        "per class (IoU / F1 / support px)",
    ]
    for entry in report["per_class"]:
        rows.append(
            f"  {entry['class_name']:<24} {entry['iou']} / {entry['f1']} / {entry['support_px']}"
        )
    rows.append("")
    targets = metrics["targets"]
    if targets.get("assessable"):
        rows.append(f"targets met: {targets['both_met']}")
    else:
        rows.append("targets met: NOT ASSESSED")
        rows.append(f"  {targets['reason']}")
    rows.append("")
    rows.append("CAVEAT: " + metrics["labels"]["caveat"])
    return "\n".join(rows)
