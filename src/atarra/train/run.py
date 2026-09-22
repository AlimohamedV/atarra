"""Training runs over a tile store, from split to metrics to checkpoint.

This is the orchestration the CLI and the Colab notebook share, so a run on the
laptop and a run on Colab differ only in device and epoch count.

Two things here are methodological rather than mechanical.

**The split is spatial, not random.** ``geometric_split`` assigns whole tile blocks,
so no test tile is adjacent to a training tile. A random split over overlapping
satellite tiles puts near-duplicate imagery on both sides of the fence and inflates
the score.

**Loss weights come from the training split alone.** Reed is a small minority of any
delta scene, so unweighted cross-entropy is minimised by predicting "not reed"
everywhere. The correction is inverse-frequency weighting, but computing it over the
whole store would import the validation and test class balance into a training-time
decision. Weights are computed from training tiles only.
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
) -> dict:
    """Train a segmentation model on a tile store and write a metrics artifact.

    ``exclude_pack`` names an annotation pack whose tiles must not be trained on. A
    held-out set the model has already seen is not held out, so this removes them
    before the split rather than hoping nobody notices.
    """
    torch = _require_torch()

    from atarra.models.segmentation import build_model, count_parameters
    from atarra.train.metrics import meets_targets
    from atarra.train.trainer import (
        TrainConfig,
        describe_device,
        evaluate,
        resolve_device,
        train,
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
        store, band_names=bands, augment=augment, seed=seed, holdout_keys=holdout
    )
    if max_tiles and len(dataset) > max_tiles:
        # Deterministic truncation for smoke runs, not for real experiments.
        dataset.records = dataset.records[:max_tiles]
        log.warning("truncated the store to %d tiles for this run", max_tiles)

    band_names = list(dataset.band_names)
    variant = "rgb" if len(band_names) == 3 else "unet"
    resolved = resolve_device(device)

    splits = geometric_split(dataset, fractions=fractions, seed=seed)
    train_indices, val_indices, test_indices = (
        splits["train"],
        splits["val"],
        splits["test"],
    )
    for name, indices in splits.items():
        if not indices:
            raise AtarraError(f"the {name} split is empty; the store is too small to train on")

    torch_dataset = dataset.as_torch_dataset()

    def loader(indices: list[int], *, shuffle: bool) -> "torch.utils.data.DataLoader":
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(torch_dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    stats = dataset.band_statistics()
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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or f"{dataset.manifest['area']}_{variant}_{len(band_names)}band_{stamp}"
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=run_dir,
        seed=seed,
        patience=patience,
        num_workers=num_workers,
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
        loader(train_indices, shuffle=True),
        loader(val_indices, shuffle=False),
        config=config,
        class_weights=class_weights,
        device=resolved,
        num_classes=NUM_CLASSES,
    )

    # The trainer keeps the best-epoch weights rather than the last epoch's, so the
    # test score below is for the model that was actually selected.
    test_report = evaluate(
        model, loader(test_indices, shuffle=False), num_classes=NUM_CLASSES, device=resolved
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
            "tiles_excluded": len(holdout),
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
        },
        "class_weights": class_weights.round(4).tolist(),
        "training": {
            "epochs_run": summary["epochs_run"],
            "best_epoch": summary["best_epoch"],
            "best_val_mean_iou": summary["best_val_mean_iou"],
            "config": summary["config"],
        },
        "validation_report": summary["final_report"],
        "test_report": test_report,
        "targets": meets_targets(test_report),
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
    if not metrics["targets"]["both_met"]:
        log.warning(
            "proposal targets not met (mIoU >= %.2f, F1 >= %.2f); this is expected "
            "while the labels come from the rule engine",
            metrics["targets"]["target_iou"],
            metrics["targets"]["target_f1"],
        )
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
    rows.append(f"targets met: {metrics['targets']['both_met']}")
    rows.append("")
    rows.append("CAVEAT: " + metrics["labels"]["caveat"])
    return "\n".join(rows)
