"""Multispectral semantic segmentation.

The network is a standard U-Net, with two decisions that are specific to this
problem.

**The input stem takes N channels directly.** No pretending a 3-channel
ImageNet-pretrained backbone can accept eight bands by tiling or by averaging the
extras. Averaging SWIR into a red channel destroys exactly the information the
project claims as its advantage, and the whole hypothesis rests on those bands
carrying signal. Training the same architecture with 8 channels and with 3 is also
what makes the "multispectral gain" claim a controlled experiment rather than an
assertion -- the only thing that changes is the band list.

**Band normalisation lives inside the model.** Reflectance statistics differ per
band and per season by an order of magnitude, so unnormalised inputs make training
unstable. Putting the normalisation in the module (as registered buffers) rather
than in the data loader means the statistics are saved in the checkpoint and
travel with it: inference cannot silently use different normalisation from
training, which is a classic way to ship a model that works in the notebook and
fails in the API.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atarra.core.errors import AtarraError

CLASS_NAMES = ["open_water", "crops_soil", "mixed_halophytes", "phragmites_australis"]
NUM_CLASSES = len(CLASS_NAMES)
PHRAGMITES_CODE = 3


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net for multispectral inputs.

    ``band_mean`` / ``band_std`` are registered as buffers, so they follow the model
    to whatever device it is moved to and are serialised in ``state_dict``.
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        band_mean: Sequence[float] | None = None,
        band_std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise AtarraError(f"in_channels must be positive, got {in_channels}")
        if depth < 2:
            raise AtarraError(f"depth must be at least 2, got {depth}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth

        mean = np.asarray(band_mean if band_mean is not None else [0.0] * in_channels, dtype=np.float32)
        std = np.asarray(band_std if band_std is not None else [1.0] * in_channels, dtype=np.float32)
        if mean.shape != (in_channels,) or std.shape != (in_channels,):
            raise AtarraError(
                f"band_mean/band_std must each have {in_channels} entries; "
                f"got {mean.shape} and {std.shape}"
            )
        if np.any(std <= 0):
            raise AtarraError("band_std must be strictly positive")
        self.register_buffer("band_mean", torch.from_numpy(mean).view(1, -1, 1, 1))
        self.register_buffer("band_std", torch.from_numpy(std).view(1, -1, 1, 1))

        # Encoder channels per level: base, 2*base, 4*base, ...
        channels = [base_channels * (2**level) for level in range(depth)]

        self.stem = DoubleConv(in_channels, channels[0])
        self.down_blocks = nn.ModuleList(
            [DoubleConv(channels[level], channels[level + 1]) for level in range(depth - 1)]
        )
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2, dropout=dropout)

        # Decoder: one block per encoder level, each consuming the upsampled
        # previous output concatenated with that level's skip. Channel arithmetic
        # is derived rather than hand-written -- an off-by-one here is a shape
        # mismatch that only appears at runtime.
        up_blocks: list[nn.Module] = []
        incoming = channels[-1] * 2  # bottleneck output width
        for level in reversed(range(depth)):
            up_blocks.append(DoubleConv(incoming + channels[level], channels[level]))
            incoming = channels[level]
        self.up_blocks = nn.ModuleList(up_blocks)

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise raw reflectance using the buffers."""
        return (x - self.band_mean) / self.band_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise AtarraError(f"expected a 4D (B, C, H, W) tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise AtarraError(
                f"model expects {self.in_channels} input channels, got {x.shape[1]}"
            )

        x = self.normalize(x)

        # Encoder. skips ends up holding one feature map per level, shallowest
        # first, which is the order the decoder consumes them in reverse.
        skips: list[torch.Tensor] = []
        features = self.stem(x)
        skips.append(features)

        for block in self.down_blocks:
            features = self.pool(features)
            features = block(features)
            skips.append(features)

        features = self.pool(features)
        features = self.bottleneck(features)

        for block in self.up_blocks:
            skip = skips.pop()
            # Resize to the skip's exact shape rather than by a factor of two.
            # Pooling floors, so on an odd-sized input (61 px wide -> 30 -> 15 -> 7)
            # the encoder and decoder resolutions drift apart and doubling cannot
            # recover them. The skip is the authoritative resolution at this level,
            # so interpolating to it is both correct and simpler than reconciling
            # after the fact.
            features = F.interpolate(
                features, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = block(torch.cat([features, skip], dim=1))

        return self.head(features)


def build_model(
    *,
    in_channels: int = 8,
    num_classes: int = NUM_CLASSES,
    base_channels: int = 32,
    depth: int = 4,
    band_mean: Sequence[float] | None = None,
    band_std: Sequence[float] | None = None,
    variant: str = "unet",
    **kwargs,
) -> nn.Module:
    """Construct a segmentation model.

    ``variant`` selects the architecture. ``rgb`` is a convenience that builds the
    3-channel baseline, which is the control arm of the multispectral experiment.
    """
    key = variant.strip().lower()
    if key in {"rgb", "rgb_baseline"}:
        red, green, blue = 0.12, 0.10, 0.08
        return UNet(
            in_channels=3,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            band_mean=[red, green, blue],
            band_std=[0.06, 0.05, 0.04],
        )
    if key != "unet":
        raise AtarraError(f"unknown model variant {variant!r}; expected 'unet' or 'rgb'")

    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        depth=depth,
        band_mean=band_mean,
        band_std=band_std,
        **kwargs,
    )


def count_parameters(model: nn.Module) -> dict:
    """Parameter counts, split into total and trainable."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "millions": round(total / 1e6, 3)}


def compute_band_statistics(
    tiles: Sequence[np.ndarray], mask: np.ndarray | None = None
) -> dict:
    """Per-band mean and standard deviation over a set of tiles.

    Used to populate the model's normalisation buffers from the training split
    only. Computing these over the whole dataset would leak validation statistics
    into training -- a subtle version of the same leakage that random splitting
    causes.
    """
    if not tiles:
        raise AtarraError("compute_band_statistics needs at least one tile")

    stacked = np.stack([np.asarray(t, dtype=np.float32) for t in tiles], axis=0)
    channels = stacked.shape[1]
    flat = stacked.transpose(1, 0, 2, 3).reshape(channels, -1)

    if mask is not None:
        flat = flat[:, np.asarray(mask, dtype=bool).ravel()]

    finite = np.isfinite(flat)
    means, stds = [], []
    for channel in range(channels):
        values = flat[channel][finite[channel]]
        if values.size == 0:
            means.append(0.0)
            stds.append(1.0)
            continue
        means.append(float(values.mean()))
        # A zero standard deviation would divide by zero in the model; floor it.
        stds.append(max(float(values.std()), 1e-3))

    return {"mean": means, "std": stds, "n_bands": channels, "n_pixels": int(flat.shape[1])}
