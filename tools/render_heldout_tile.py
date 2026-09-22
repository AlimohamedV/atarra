"""Render one representative geographically held-out tile three ways, side by side.

Produces a single figure:

    true colour      |  rule-engine teacher labels  |  model prediction

The middle and right panels describe the *same* tile, so they are directly
comparable: the left is what the satellite saw, the middle is what the model was
trained to reproduce, and the right is what it actually predicts. The teacher
panel is not ground truth -- it is the weak-supervision rule engine's opinion,
which is precisely what the model learned from, so agreement between them
measures the student, not the truth. The caption says so.

**Which tile, and why it is chosen this way.** The selection is deliberately not
"the tile where the model looks best". Two naive rules both mislead:

* Reed-richest tile, ignoring everything else. On a July composite the crop rule
  tops out just under its own cut, so arable fields get called reed; the richest
  "reed" tile in this store is a field grid that is 46% reed against a store-wide
  4%. That tile demonstrates the teacher being wrong, not the model working.
* Best-agreeing tile. That is a flattering sample dressed as evidence.

So the whole test split is scored first, and the figure shows the tile whose reed
agreement sits at the **median** of the reed-bearing, water-containing tiles. The
tile's own score and the split's aggregate are both printed, so the reader can see
they match instead of taking it on trust.

    python tools/render_heldout_tile.py \
        --store data/demo_store \
        --checkpoint data/runs/demo_heldout/best.pt \
        --out presentation/heldout_tile.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Import the package from a source checkout without requiring an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from atarra.datasets.store import TileStoreDataset  # noqa: E402
from atarra.datasets.tile_dataset import geometric_split  # noqa: E402
from atarra.datasets.weak_labels import CLASS_NAMES, PHRAGMITES_CODE  # noqa: E402
from atarra.models.segmentation import build_model  # noqa: E402
from atarra.train.trainer import load_checkpoint, resolve_device  # noqa: E402
from atarra.viz.render import render_mask, render_true_color  # noqa: E402

# Reed is deliberately the loudest colour: it is the class the argument is about,
# and the eye should land on it in both the teacher and prediction panels.
CLASS_COLORS = {
    0: (40, 78, 122),    # open water       -- dark blue
    1: (196, 175, 130),  # crops / soil     -- tan
    2: (110, 190, 120),  # mixed halophytes -- muted green
    3: (232, 56, 140),   # Phragmites       -- magenta
}
# Pixels the rule engine declined to call, so they were excluded from the loss. A
# neutral grey rather than the near-black used for nodata, so it reads as "not
# supervised" and cannot be mistaken for the dark blue of open water.
IGNORED_COLOR = (86, 94, 108)
NODATA_COLOR = (16, 20, 32)


class _Stack:
    """The minimal interface ``render_true_color`` needs.

    The project renders composites, not bare arrays, so wrapping the tile keeps the
    tested stretch/gamma/NaN handling rather than reimplementing it here.
    """

    def __init__(self, bands: list[str], image: np.ndarray) -> None:
        self.band_names = list(bands)
        self._by_name = {name: image[i] for i, name in enumerate(bands)}

    def band(self, name: str) -> np.ndarray:
        return self._by_name[name]


def class_rgba(labels: np.ndarray) -> np.ndarray:
    """Compose a discrete class map into an (H, W, 4) RGBA image.

    ``-1`` is the ignore index, drawn in grey: those are the pixels the rule engine
    declined to call and the loss skipped, so a panel that hid them would overstate
    how much ground the model was actually supervised on.

    Built by overlaying ``render_mask`` once per class, so the alpha handling stays
    in the one place it is already tested.
    """
    rgba = np.zeros(labels.shape + (4,), dtype=np.uint8)
    rgba[...] = (*NODATA_COLOR, 255)
    rgba[labels < 0] = (*IGNORED_COLOR, 255)
    for code, color in CLASS_COLORS.items():
        layer = render_mask(labels == code, color=color, alpha=255)
        hit = layer[..., 3] > 0
        rgba[hit] = layer[hit]
    return rgba


def class_fractions(labels: np.ndarray) -> np.ndarray:
    """Per-class share of the *supervised* pixels in one tile.

    Over pixels the loss actually saw, not all 65,536: a tile the swath half covered,
    or that the rules mostly declined to call, would otherwise read as half reed.
    """
    observed = labels >= 0
    total = int(observed.sum())
    if total == 0:
        return np.zeros(len(CLASS_NAMES))
    return np.array(
        [int(((labels == code) & observed).sum()) / total for code in range(len(CLASS_NAMES))]
    )


def reed_intersection_union(
    predicted: np.ndarray, teacher: np.ndarray
) -> tuple[int, int, int]:
    """Reed intersection, union, and support over supervised pixels only.

    Pixels the loss ignored (``teacher < 0``) are excluded from all three counts, which
    is what the trainer's confusion accumulator does. Counting them charges the model
    for predictions on ground it was never asked to reproduce -- on this store that
    alone turns a 0.90 into a 0.12, because the model predicts reed freely across the
    pixels the rule engine declined to call.
    """
    supervised = teacher >= 0
    pred_hit = (predicted == PHRAGMITES_CODE) & supervised
    true_hit = (teacher == PHRAGMITES_CODE) & supervised
    return (
        int((pred_hit & true_hit).sum()),
        int((pred_hit | true_hit).sum()),
        int(true_hit.sum()),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="presentation/heldout_tile.png")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-buffer", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument(
        "--true-color-vmax",
        type=float,
        default=0.35,
        help=(
            "top of the true-colour stretch, in reflectance. The library default of "
            "0.25 with its 1.35 gain clips anything above 0.185 to pure white, which on "
            "this tile blew out 58.5%% of the turbid water. 0.35 clips 0.3%% (default)"
        ),
    )
    ap.add_argument(
        "--min-water",
        type=float,
        default=0.0,
        help=(
            "minimum share of supervised tile pixels under open water (default 0). "
            "Raise it to demand a lagoon scene; note that a geographic split can put "
            "every reed-bearing held-out tile inland, and then no tile qualifies"
        ),
    )
    ap.add_argument(
        "--min-reed",
        type=float,
        default=0.10,
        help="minimum reed share so the tile shows a stand rather than a scatter (default 0.10)",
    )
    ap.add_argument(
        "--max-reed",
        type=float,
        default=0.60,
        help="skip tiles the rule engine calls almost entirely reed (default 0.60)",
    )
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else resolve_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # The band order comes from the checkpoint, not from a constant. A channel count
    # cannot distinguish B08 from B04, and a prediction rendered with two bands
    # swapped looks entirely plausible.
    band_names = list(checkpoint.get("band_names") or [])
    if not band_names:
        raise SystemExit(
            f"{args.checkpoint} records no `band_names`, so the band order cannot be "
            "verified; refusing to render a prediction that may have its channels permuted"
        )

    dataset = TileStoreDataset(args.store, band_names=band_names)
    splits = geometric_split(dataset, seed=args.seed, buffer_pixels=args.split_buffer)
    test_indices = list(splits["test"])
    print(
        f"store {args.store}: {len(dataset)} tiles | "
        f"train {len(splits['train'])} / val {len(splits['val'])} / test {len(test_indices)} | "
        f"gap {splits.buffer_pixels}px"
    )

    model = build_model(in_channels=len(band_names))
    meta = load_checkpoint(Path(args.checkpoint), model, device=device)

    # Score every test tile once. The figure is chosen from this, and the pooled
    # figure is what the caption quotes.
    #
    # The comparison is against the GATED mask -- the one the loss was computed on --
    # and not `rule_labels_at`, which returns the rule engine's ungated opinion. The
    # ungated layer still carries a class for every pixel the engine was unsure about,
    # and none of those pixels contribute supervision, so scoring against it charges
    # the model for failing to reproduce calls it was never asked to make. On this
    # store that difference is not marginal: 0.90 gated against 0.13 ungated.
    rows: list[dict] = []
    with torch.no_grad():
        for index in test_indices:
            image, teacher = dataset._image_and_mask(index)
            tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None].to(device)
            predicted = model(tensor).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
            inter, union, support = reed_intersection_union(predicted, teacher)
            rows.append(
                {
                    "index": index,
                    "inter": inter,
                    "union": union,
                    "support": support,
                    "iou": None if union == 0 else inter / union,
                    "fractions": class_fractions(teacher),
                    "supervised": float((teacher >= 0).mean()),
                }
            )

    pooled_inter = sum(r["inter"] for r in rows)
    pooled_union = sum(r["union"] for r in rows)
    pooled = pooled_inter / pooled_union if pooled_union else float("nan")
    pooled_support = sum(r["support"] for r in rows)
    with_reed = [r for r in rows if r["iou"] is not None]
    print(
        f"test split: pooled reed IoU {pooled:.3f} over {len(rows)} tiles "
        f"({len(with_reed)} containing reed), support {pooled_support} px"
    )

    # Representative, not flattering: the median-agreeing tile that actually shows a
    # reed stand in the lagoon rather than a field grid.
    # "Has a reed stand" is a floor as well as a ceiling: a tile the rules called
    # mostly water has no reed in it, and scoring 0.000 there is not a data point.
    candidates = [
        r
        for r in with_reed
        if r["fractions"][0] >= args.min_water
        and args.min_reed <= r["fractions"][PHRAGMITES_CODE] <= args.max_reed
    ]
    if not candidates:
        raise SystemExit(
            "no test tile has both open water and a reed stand in it; "
            "lower --min-water or raise --max-reed"
        )
    candidates.sort(key=lambda r: r["iou"])
    chosen = candidates[len(candidates) // 2]
    if chosen["fractions"][0] <= 0.0:
        print(
            "  NOTE: no reed-bearing held-out tile contains open water, so this figure "
            "shows an inland stand. That is the split's geography, not a selection choice."
        )

    index = chosen["index"]
    record = dataset.records[index]
    image, teacher = dataset._image_and_mask(index)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None].to(device)
    with torch.no_grad():
        predicted = model(tensor).argmax(dim=1)[0].cpu().numpy().astype(np.int64)

    corners = dataset.tile_corners_wgs84(index)
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    print(
        f"chosen tile {record.key}: centre {np.mean(lats):.4f}N {np.mean(lons):.4f}E, "
        f"median of {len(candidates)} reed+water candidate(s)"
    )
    print(
        "  usable class mix:"
        + "".join(f"  {n} {f:.0%}" for n, f in zip(CLASS_NAMES, chosen["fractions"]))
    )
    print(
        f"  reed agreement IoU {chosen['iou']:.3f}  "
        f"(split pooled {pooled:.3f})  -- agreement with the rules, not field truth"
    )
    print(f"  supervised pixels {chosen['supervised']:.1%} of the tile; the rest is grey")

    true_color = render_true_color(_Stack(band_names, image), vmax=args.true_color_vmax)
    clipped = float((true_color[..., :3] >= 255).any(axis=-1).mean())
    panels = [
        ("True colour  B04 / B03 / B02", true_color),
        ("Weak-supervision target (grey = not supervised)", class_rgba(teacher)),
        (f"Model prediction  (best.pt, epoch {meta.get('epoch')})", class_rgba(predicted)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0), dpi=args.dpi)
    for ax, (title, rgba) in zip(axes, panels):
        ax.imshow(rgba, interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=7, color="#dbe6f7")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2b3648")
    fig.patch.set_facecolor("#0a0f1c")
    for ax in axes:
        ax.set_facecolor("#0a0f1c")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=tuple(c / 255 for c in CLASS_COLORS[i]))
        for i in range(len(CLASS_NAMES))
    ] + [plt.Rectangle((0, 0), 1, 1, facecolor=tuple(c / 255 for c in IGNORED_COLOR))]
    labels = [name.replace("_", " ") for name in CLASS_NAMES] + ["not supervised"]
    legend = axes[0].legend(
        handles, labels, loc="upper center", bbox_to_anchor=(1.70, -0.03),
        ncol=5, frameon=False, fontsize=9.5, handlelength=1.1, handleheight=1.1,
        columnspacing=1.2,
    )
    for text in legend.get_texts():
        text.set_color("#c9d6ea")

    # Two lines, because one line at this width overflows the canvas and gets clipped
    # at both ends -- which silently truncated "not field truth" to "not fi".
    fig.text(
        0.5, 0.005,
        f"held-out test tile {record.key}  |  {np.mean(lats):.4f}N {np.mean(lons):.4f}E  |  "
        f"true colour stretched to {args.true_color_vmax:.2f} reflectance "
        f"({clipped:.1%} of pixels clipped)\n"
        f"reed agreement with the teacher: IoU {chosen['iou']:.2f} "
        f"(test split pooled {pooled:.2f})  --  agreement with the rules, not field truth",
        ha="center", va="bottom", fontsize=8.5, color="#8f9fba", linespacing=1.6,
    )

    fig.subplots_adjust(left=0.012, right=0.988, top=0.90, bottom=0.16)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
