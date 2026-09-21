"""Training loop.

Tuned for the hardware this was developed on: an RTX 2050 with 4.29 GB of VRAM.
That budget is the reason for most of the choices here -- mixed precision, a small
default batch, and 256 px tiles rather than 512. Gradient accumulation is provided
so an effective batch size can be reached without the memory to hold it.

Validation runs under ``torch.no_grad`` and accumulates one confusion matrix for
the whole split, so the reported mIoU is computed once over all pixels rather than
averaged per batch (see :mod:`atarra.train.metrics` for why that matters).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger
from atarra.train.metrics import ConfusionAccumulator, segmentation_report

log = get_logger("train.trainer")

IGNORE_INDEX = -1


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 40
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    accumulate_steps: int = 1
    num_workers: int = 0  # 0 keeps this safe on Windows, where workers fork poorly
    seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path("data/checkpoints"))
    patience: int = 10
    grad_clip: float = 1.0
    ignore_index: int = IGNORE_INDEX

    def as_dict(self) -> dict:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick a device, preferring CUDA when it is actually usable."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict:
    info = {"device": str(device), "cuda": torch.cuda.is_available()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            {
                "name": properties.name,
                "vram_gb": round(properties.total_memory / 1e9, 2),
                "capability": f"sm_{properties.major}{properties.minor}",
                "amp": True,
            }
        )
    return info


def build_loss(class_weights: np.ndarray | None, device: torch.device) -> nn.Module:
    """Cross-entropy with optional per-class weights and the ignore index.

    Weighting matters more here than in a balanced problem: reed beds occupy a
    small fraction of any delta scene, so the unweighted optimum is to predict
    "not reed" everywhere.
    """
    weights = None
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    *,
    num_classes: int = 4,
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> dict:
    """Evaluate a model, returning a full segmentation report."""
    device = device or resolve_device()
    model.eval()
    accumulator = ConfusionAccumulator(num_classes=num_classes)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).long()

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        accumulator.update(predictions, targets, ignore_index=IGNORE_INDEX)

    return accumulator.report()


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    config: TrainConfig | None = None,
    class_weights: np.ndarray | None = None,
    device: torch.device | None = None,
    num_classes: int = 4,
) -> dict:
    """Train a segmentation model and return the run history."""
    cfg = config or TrainConfig()
    device = device or resolve_device()
    _set_seed(cfg.seed)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("training on %s", describe_device(device))
    model = model.to(device)
    criterion = build_loss(class_weights, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    # 4 GB of VRAM makes mixed precision close to mandatory rather than an
    # optimisation: it roughly halves activation memory.
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    history: list[dict] = []
    best_iou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True).float()
            masks = batch["mask"].to(device, non_blocking=True).long()

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
                if cfg.accumulate_steps > 1:
                    loss = loss / cfg.accumulate_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.accumulate_steps == 0:
                if cfg.grad_clip:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach()) * (
                cfg.accumulate_steps if cfg.accumulate_steps > 1 else 1.0
            )
            batches += 1

        scheduler.step()
        train_loss = running_loss / max(1, batches)
        report = evaluate(model, val_loader, num_classes=num_classes, device=device)
        val_iou = report["mean_iou"]
        elapsed = time.time() - start

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_mean_iou": val_iou,
            "val_phragmites_iou": report["phragmites_iou"],
            "val_phragmites_f1": report["phragmites_f1"],
            "val_pixel_accuracy": report["pixel_accuracy"],
            "seconds": round(elapsed, 1),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
        }
        history.append(entry)
        log.info(
            "epoch %d/%d  loss %.4f  mIoU %.4f  reed IoU %s  (%.0fs)",
            epoch,
            cfg.epochs,
            train_loss,
            val_iou,
            report["phragmites_iou"],
            elapsed,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(cfg.output_dir / "best.pt", model, cfg, epoch, report, class_weights)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                log.info(
                    "early stopping at epoch %d: no improvement for %d epochs",
                    epoch,
                    cfg.patience,
                )
                break

    summary = {
        "config": cfg.as_dict(),
        "device": describe_device(device),
        "best_epoch": best_epoch,
        "best_val_mean_iou": best_iou,
        "epochs_run": len(history),
        "history": history,
    }

    (cfg.output_dir / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["final_report"] = evaluate(model, val_loader, num_classes=num_classes, device=device)

    log.info("best validation mIoU %.4f at epoch %d", best_iou, best_epoch)
    return summary


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: TrainConfig,
    epoch: int,
    report: dict,
    class_weights: np.ndarray | None,
) -> None:
    """Save weights plus everything needed to reproduce inference."""
    torch.save(
        {
            "model_state": model.state_dict(),
            # The normalisation buffers travel inside state_dict, so inference
            # cannot accidentally use different statistics from training.
            "config": cfg.as_dict(),
            "epoch": epoch,
            "report": report,
            "class_weights": None if class_weights is None else class_weights.tolist(),
            "in_channels": getattr(model, "in_channels", None),
            "num_classes": getattr(model, "num_classes", None),
        },
        path,
    )
    log.debug("saved checkpoint %s", path)


def load_checkpoint(path: Path, model: nn.Module, *, device: torch.device | None = None) -> dict:
    """Load weights into a model, returning the checkpoint metadata."""
    device = device or resolve_device()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError as exc:
        raise AtarraError(f"checkpoint not found: {path}") from exc
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return {k: v for k, v in checkpoint.items() if k != "model_state"}
