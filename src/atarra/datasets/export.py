"""Annotation packs: export tiles for human labelling, then score against them.

Why this exists. The training labels come from a rule engine, so any metric computed
against them measures agreement with that engine rather than detection accuracy. The
only way to a defensible number is a set of pixels a human actually read -- and the
rule engine already knows which pixels those should be, because it recorded every one
it declined to be confident about.

A pack is that handover made concrete: georeferenced GeoTIFF chips that open in QGIS,
the rule engine's current guess so the annotator corrects rather than starts from
nothing, a review mask showing where the guess is untrustworthy, and a GeoJSON index
so the tiles are findable on a map.

Two disciplines the pack enforces, because both are easy to get wrong by accident:

**Reserved tiles are excluded from training.** A held-out set that the model trained
on is not held out. The pack records the tile keys it exported and ``atarra train
--exclude-pack`` drops them before splitting, so the annotated pixels are unseen by
construction rather than by good intentions.

**Scoring reads the annotator's file, never the rule engine's.** ``score_annotation_pack``
loads ``labels/<key>.tif`` (or ``.geojson``) and reports against that. If nothing has
been labelled yet it says so instead of silently falling back to the weak labels,
which would produce exactly the circular number this whole exercise exists to avoid.
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
from atarra.datasets.weak_labels import CLASS_NAMES, NUM_CLASSES, PHRAGMITES_CODE

log = get_logger("datasets.export")

PACK_VERSION = 1
WGS84 = "EPSG:4326"

#: The reflectance convention a pack's imagery is written in. Recorded in the pack so
#: the scorer can tell a float chip from a stored-integer one instead of guessing --
#: reading the wrong one scales every input without raising anything.
CHIP_UNITS = "float32 reflectance, 1.0 = 100% reflectance"
#: Reflectance is physically close to 1 at most. A value far above this is a stored
#: integer (uint16 at 1e-4 scale reaches 10000), not a bright surface.
MAX_PLAUSIBLE_REFLECTANCE = 2.0

#: Below this many reed pixels across a pack, the per-class IoU is not a measurement.
#: One run of this project reported "reed IoU 0.0" from 36 support pixels, which says
#: nothing about the model. Reed is ~1.8% of this imagery, so a pack selected without
#: regard for reed content can easily fall under this.
MIN_REED_PIXELS_FOR_IOU = 500
#: Value meaning "no human has labelled this pixel". Anything outside 0..3 is ignored
#: by the scorer, but 255 is the convention written into the pack README because a
#: label raster's natural fill value of 0 would silently mean "open water".
UNLABELLED = 255

#: QGIS reads these straight out of the GeoTIFF colour table, so the classes are
#: visually distinguishable without the annotator configuring anything.
CLASS_COLORS = {
    0: (58, 110, 165, 255),  # open water      - blue
    1: (196, 160, 90, 255),  # crops / soil    - tan
    2: (150, 130, 170, 255),  # mixed halophyte - muted violet
    3: (60, 150, 80, 255),  # phragmites      - green
}
REVIEW_COLORS = {0: (70, 70, 70, 255), 1: (230, 120, 40, 255)}


def _as_reflectance(stack: np.ndarray, *, source: str) -> np.ndarray:
    """Confirm a chip is in the pack's reflectance convention, or say what it is in.

    The failure this guards against is silent: dividing an already-decoded chip by the
    storage scale produces finite, well-shaped, plausible-looking inputs, and the model
    answers them with a broadly plausible segmentation that happens to be wrong. The
    score would read as a mediocre model rather than as a unit error.
    """
    if not stack.size:
        raise AtarraError(f"{source} has no pixels")
    peak = float(np.nanmax(stack))
    if peak > MAX_PLAUSIBLE_REFLECTANCE:
        raise AtarraError(
            f"{source} holds values up to {peak:.4g}, which is not reflectance (the "
            f"pack format is {CHIP_UNITS}). This looks like the stored-integer "
            "convention; re-export the pack rather than scoring it, because the model "
            "would receive inputs orders of magnitude outside its training range."
        )
    return stack


def _safe_name(key: str) -> str:
    """A tile key rendered as a flat filename.

    Store keys are ``date/composite/row_col`` shaped, so they contain separators. Used
    raw they would either nest directories or, worse, collapse two different tiles onto
    one filename and silently overwrite one of them.
    """
    return key.replace("/", "__")


def _write_raster(
    path: Path,
    array: np.ndarray,
    *,
    transform,
    crs,
    descriptions: Sequence[str] | None = None,
    colormap: dict | None = None,
    nodata=None,
) -> None:
    """Write a GeoTIFF that QGIS opens with the right georeferencing."""
    import rasterio

    data = np.asarray(array)
    if data.ndim == 2:
        data = data[None, :, :]
    count, height, width = data.shape

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": False,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data)
        for index, name in enumerate(descriptions or [], start=1):
            if index <= count:
                dataset.set_band_description(index, name)
        if colormap:
            for band in range(1, count + 1):
                dataset.write_colormap(band, colormap)


def _polygon_feature(
    corners: list[list[float]], properties: dict, *, ring: bool = True
) -> dict:
    """A GeoJSON Polygon from four corners.

    The ring is closed by repeating the first position. RFC 7946 requires it, and a
    ring that is merely implied works in some parsers and produces a degenerate
    feature in others -- not a failure worth risking in a file whose entire purpose is
    to be opened by a different program.
    """
    coordinates = [list(corner) for corner in corners]
    if ring:
        coordinates.append(list(corners[0]))
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
        "properties": properties,
    }


def export_annotation_pack(
    store: Path | str,
    out_dir: Path | str,
    *,
    limit: int = 20,
    min_score: float = 0.0,
    imagery_bands: Sequence[str] | None = None,
    overwrite: bool = False,
    strategy: str = "uncertainty",
    seed: int = 0,
) -> dict:
    """Write georeferenced chips, a GeoJSON index, and a labelling README.

    ``strategy`` chooses what "top-ranked" means -- see
    :meth:`TileStoreDataset.annotation_ranking`. ``min_score`` filters on that score,
    so its units depend on the strategy (a review fraction, or a reed pixel count).

    Returns the pack manifest.
    """
    store = Path(store)
    out_dir = Path(out_dir)

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise AtarraError(
            f"{out_dir} already exists and is not empty; pass overwrite=True to "
            "replace it, or choose another directory. An annotation pack is a record "
            "of someone's work — silently overwriting it would destroy labels."
        )

    dataset = TileStoreDataset(store, band_names=imagery_bands)
    ranking = dataset.annotation_ranking(strategy=strategy, seed=seed)
    if not ranking:
        raise AtarraError(
            f"no tile qualifies under strategy {strategy!r}. Under 'uncertainty' this "
            "means nothing was flagged for review (possible for uniformly clear "
            "imagery); under 'reed' it means no stored tile contains reed pixels."
        )

    selected = [(score, index) for score, index in ranking if score >= min_score][:limit]
    if not selected:
        raise AtarraError(
            f"no tile reaches min_score={min_score} under strategy {strategy!r}; "
            f"the highest score is {ranking[0][0]:.4f}"
        )

    imagery_bands = list(dataset.band_names)
    chips_dir = out_dir / "chips"
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[dict] = []
    features: list[dict] = []

    reed_total = 0
    for rank, (score, index) in enumerate(selected, start=1):
        record = dataset.records[index]
        transform = dataset.tile_transform(index)
        grid = dataset.grid()
        size = int(dataset.manifest["tile_size"])

        item = dataset[index]
        image = item["image"]  # (C, H, W) float32 reflectance
        usable = item["usable"]

        # Nodata is decided by `usable`, never by the training mask. An ambiguous pixel
        # is -1 in the training mask too, so masking on that would blank out precisely
        # the pixels the annotator is being asked to rule on.
        weak = np.where(usable, dataset.rule_labels_at(index), UNLABELLED).astype(np.uint8)

        review = dataset.review_at(index)
        review_raster = np.where(usable, np.where(review, 1, 0), UNLABELLED).astype(np.uint8)

        stem = _safe_name(record.key)
        chip_path = chips_dir / f"{stem}.tif"
        weak_path = chips_dir / f"{stem}_weak_labels.tif"
        review_path = chips_dir / f"{stem}_review.tif"

        _write_raster(
            chip_path,
            image,
            transform=transform,
            crs=grid.crs,
            descriptions=imagery_bands,
            nodata=None,
        )
        _write_raster(
            weak_path,
            weak,
            transform=transform,
            crs=grid.crs,
            descriptions=["rule_engine_label"],
            colormap=CLASS_COLORS,
            nodata=UNLABELLED,
        )
        _write_raster(
            review_path,
            review_raster,
            transform=transform,
            crs=grid.crs,
            descriptions=["needs_human_review"],
            colormap=REVIEW_COLORS,
            nodata=UNLABELLED,
        )

        corners = dataset.tile_corners_wgs84(index)
        reed_px = int((dataset.label_at(index) == PHRAGMITES_CODE).sum())
        reed_total += reed_px
        properties = {
            "key": record.key,
            "date": dataset._shards[record.shard]["date"],
            "rank": rank,
            "score": round(score, 5),
            "review_fraction": round(float(review.mean()), 5),
            "reed_px": reed_px,
            "image_chip": f"chips/{stem}.tif",
            "weak_labels": f"chips/{stem}_weak_labels.tif",
            "review_mask": f"chips/{stem}_review.tif",
            "label_output": f"labels/{stem}.tif",
            "filename_stem": stem,
        }
        features.append(_polygon_feature(corners, properties))
        tiles.append({**properties, "corners_wgs84": corners})

    pack = {
        "format_version": PACK_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "store": str(store),
        "area": dataset.manifest["area"],
        "gsd": dataset.manifest["gsd"],
        "tile_size": int(dataset.manifest["tile_size"]),
        "imagery_bands": imagery_bands,
        "imagery_units": CHIP_UNITS,
        "class_names": CLASS_NAMES,
        "phragmites_class_code": PHRAGMITES_CODE,
        "unlabelled_value": UNLABELLED,
        "strategy": strategy,
        "tiles": tiles,
        "reserved_keys": [tile["key"] for tile in tiles],
        "total_candidates_available": len(ranking),
        "reed_px_total": reed_total,
        "reed_px_sufficient": reed_total >= MIN_REED_PIXELS_FOR_IOU,
    }

    if reed_total < MIN_REED_PIXELS_FOR_IOU:
        log.warning(
            "this pack holds only %d reed pixel(s); the reed IoU measured against it "
            "will not be a meaningful number. Raise --limit, or use "
            "--strategy reed to select tiles that actually contain the class.",
            reed_total,
        )

    (out_dir / "pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "annotation.geojson").write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features}, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_pack_readme(pack), encoding="utf-8")

    log.info(
        "annotation pack written to %s: %d chip(s), selected by strategy %r",
        out_dir,
        len(tiles),
        strategy,
    )
    return pack


def _pack_readme(pack: dict) -> str:
    tile_count = len(pack["tiles"])
    columns = "\n".join(
        "| {code} | `{name}` |".format(code=code, name=name)
        for code, name in enumerate(pack["class_names"])
    )
    rows = "\n".join(
        "| {rank} | `{key}` | {date} | {review:.1%} |".format(
            rank=tile["rank"],
            key=tile["key"],
            date=tile["date"],
            review=tile["review_fraction"],
        )
        for tile in pack["tiles"]
    )
    return f"""# Annotation pack — {pack['area']}

{tile_count} tiles at {pack['gsd']:g} m, ordered by how much a human is needed.
Generated {pack['created']}.

## What each file is

| File | Contents |
| --- | --- |
| `chips/<key>.tif` | Imagery. Bands: {', '.join(pack['imagery_bands'])} (float32 reflectance) |

The imagery chip has **no nodata value set**: ground outside the satellite swath reads
as `0`, and a genuine reflectance of `0` looks identical. Use the review mask instead
to see which ground is real - its nodata ({pack['unlabelled_value']}) marks exactly the
areas the swath did not cover, and those areas must not be labelled.
| `chips/<key>_weak_labels.tif` | The rule engine's current guess. **Correct this, do not start from blank.** |
| `chips/<key>_review.tif` | `1` where the rule engine was not confident. Focus here. |
| `annotation.geojson` | One polygon per tile, with its lon/lat corners |
| `labels/` | Empty. Your answers go here. |

Both label rasters use a colour table, so QGIS displays them class-coloured with no
configuration.

## Class codes

| Code | Class |
| --- | --- |
{columns}

Every other value is treated as **unlabelled** ({pack['unlabelled_value']} by
convention). This matters: a raster's natural fill is `0`, which in this scheme means
`open_water`. If you leave areas unpainted, set them to {pack['unlabelled_value']} —
otherwise untouched ground is scored as water.

## Labelling in QGIS

Draw polygons, which is what QGIS is actually good at. Two supported outputs per tile,
either or both:

1. **Polygons** — create a polygon layer, add an integer field named `class_code`, and
   save it as `labels/<key>.geojson` with class values from the table above. The scorer
   rasterises it onto the chip's grid.
2. **A raster** — save as `labels/<key>.tif` on exactly the same grid as the chip, with
   class codes and {pack['unlabelled_value']} for anything untouched.

In both cases use the `label_output` value from `annotation.geojson` as the filename.
Tile keys contain `/` separators, so filenames substitute `__` for them; do not
reconstruct the name by hand, or your labels will land where the scorer will not find
them.

Restrict yourself to areas the review mask flags if time is short. Labelling all of a
tile is slower and most of it agrees with the rule engine anyway, which teaches you
nothing.

## Why these tiles are reserved

The tiles below are **excluded from training** once this pack exists:

```
atarra train <store> --exclude-pack <this directory>
```

That is the whole point of a held-out set. If you train on these tiles first, the
resulting accuracy measures memorisation, not detection.

## Scoring

```
atarra annotation score <this directory> --checkpoint <run>/best.pt
```

This reads only your `labels/` files. If none exist yet it refuses rather than falling
back to the rule engine's guesses, because a score computed against the training labels
is the circular number this pack exists to avoid.

## Tiles

| # | key | date | review |
| --- | --- | --- | --- |
{rows}
"""


def load_pack(pack_dir: Path | str) -> dict:
    """Read and version-check a pack manifest."""
    path = Path(pack_dir) / "pack.json"
    if not path.exists():
        raise AtarraError(f"{pack_dir} is not an annotation pack (no pack.json)")
    pack = json.loads(path.read_text(encoding="utf-8"))
    if pack.get("format_version") != PACK_VERSION:
        raise AtarraError(
            f"pack format {pack.get('format_version')} is not supported "
            f"(this build reads {PACK_VERSION})"
        )
    return pack


def reserved_keys(pack_dir: Path | str) -> set[str]:
    """Tile keys a pack has claimed for human labelling."""
    return set(load_pack(pack_dir).get("reserved_keys", []))


def _reproject_geometry(geometry: dict, transformer) -> dict:
    """Reproject a GeoJSON geometry's coordinates.

    GeoJSON is WGS84 by specification (RFC 7946), but the chip's transform is in the
    store's projected CRS. Rasterising lon/lat coordinates against a UTM transform does
    not fail -- it silently produces garbage, because the numbers are in a different
    space entirely. So the geometry is reprojected first.
    """

    def walk(node):
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (int, float)):
                x, y = transformer.transform(node[0], node[1])
                return [x, y]
            return [walk(item) for item in node]
        return node

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def load_annotations(
    pack_dir: Path | str, key: str, *, shape: tuple[int, int], transform, crs
) -> np.ndarray | None:
    """Load one tile's human labels, from a raster or from polygons.

    Returns ``None`` when nothing has been annotated for this tile yet, which the
    scorer treats as "not yet", never as "use the weak labels instead".
    """
    pack_dir = Path(pack_dir)

    stem = _safe_name(key)
    raster_path = pack_dir / "labels" / f"{stem}.tif"
    if raster_path.exists():
        import rasterio

        with rasterio.open(raster_path) as dataset:
            if (dataset.height, dataset.width) != shape:
                raise AtarraError(
                    f"{raster_path.name} is {dataset.height}x{dataset.width} but the chip "
                    f"is {shape[0]}x{shape[1]}; labels must be on the chip's grid"
                )
            return dataset.read(1)

    vector_path = pack_dir / "labels" / f"{stem}.geojson"
    if vector_path.exists():
        from pyproj import Transformer
        from rasterio.features import rasterize

        if crs is None:
            raise AtarraError(
                f"{vector_path.name}: polygon labels need the chip's CRS to reproject "
                "into; the chip appears to carry none"
            )
        to_chip = Transformer.from_crs(WGS84, crs, always_xy=True)

        document = json.loads(vector_path.read_text(encoding="utf-8"))
        features = document.get("features", [])
        shapes = []
        for feature in features:
            properties = feature.get("properties", {}) or {}
            code = properties.get("class_code")
            if code is None:
                name = properties.get("class_name")
                if name in CLASS_NAMES:
                    code = CLASS_NAMES.index(name)
            if code is None and properties.get("class") is not None:
                value = properties["class"]
                code = CLASS_NAMES.index(value) if value in CLASS_NAMES else int(value)
            if code is None:
                raise AtarraError(
                    f"{vector_path.name}: a feature has no class. Add an integer "
                    "`class_code` field (see the pack README for the codes)."
                )
            geometry = feature.get("geometry")
            if geometry:
                shapes.append((_reproject_geometry(geometry, to_chip), int(code)))

        if not shapes:
            return None

        return rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=UNLABELLED,
            dtype="uint8",
        )

    return None


def _check_independence(provenance: dict, pack_keys: Sequence[str]) -> dict:
    """Decide whether the annotated ground can be *shown* to be unseen.

    An annotation pack only measures detection if the model never saw its ground --
    during training, or while its checkpoint was being chosen. The two failures differ
    and are reported apart: training on it makes the score meaningless, whereas
    selecting the checkpoint on it makes the score optimistic while still describing
    the model's ability. Neither can be checked without the checkpoint's own record of
    which tiles went where, so an older checkpoint is reported as unverified rather
    than assumed innocent.
    """
    annotated = set(pack_keys)
    trained = provenance.get("train_keys")
    validated = provenance.get("val_keys")
    if not trained and not validated:
        return {
            "status": "unverified",
            "detail": (
                "the checkpoint records no split provenance, so nothing here shows the "
                "annotated ground was excluded from training or from checkpoint "
                "selection. It may have been; it cannot be demonstrated. Re-run "
                "`atarra train` to record it."
            ),
        }

    overlap = sorted(annotated & set(trained or ()))
    if overlap:
        return {
            "status": "trained_on",
            "detail": (
                f"{len(overlap)} annotated tile(s) are in the checkpoint's training "
                f"split, e.g. {overlap[0]}. These metrics measure recall of ground the "
                "model was fitted on, which is not detection accuracy."
            ),
        }

    selected_on = sorted(annotated & set(validated or ()))
    if selected_on:
        return {
            "status": "used_for_selection",
            "detail": (
                f"{len(selected_on)} annotated tile(s) are in the checkpoint's "
                f"validation split, e.g. {selected_on[0]}, and the checkpoint was "
                "chosen by its score there. The model did not train on them, but the "
                "score is optimistic rather than held out."
            ),
        }

    return {
        "status": "verified",
        "detail": (
            f"none of the {len(annotated)} annotated tile(s) appears in the "
            "checkpoint's training or validation splits."
        ),
        "checkpoint_train_tiles": len(trained or ()),
    }


def _assess(report: dict, predicted_classes: set[int], independence: dict) -> dict:
    """Decide whether a score is a validation at all, before anyone quotes it.

    A report can be arithmetically perfect and completely empty. Labelling a single
    class and scoring a model that predicts that class everywhere yields mIoU 1.0 and
    "targets met" -- from a model that has learned nothing. The same happens when a
    handful of reed pixels happens to land under a uniform prediction.

    So the numbers are only called a validation when the annotations actually cover
    more than one class, contain enough of the target class to measure, and are on
    ground the checkpoint cannot be shown to have seen. Otherwise the reason is stated
    instead of the pass.
    """
    support = {entry["class_name"]: int(entry["support_px"]) for entry in report["per_class"]}
    present = [name for name, pixels in support.items() if pixels > 0]
    absent = [name for name, pixels in support.items() if pixels == 0]
    reed_support = support.get("phragmites_australis", 0)

    blocking: list[str] = []
    if independence["status"] != "verified":
        blocking.append(independence["detail"])
    if len(present) < 2:
        blocking.append(
            f"the annotations cover only {len(present)} class(es) "
            f"({', '.join(present) or 'none'}), so a model that predicts that class "
            "everywhere scores a perfect IoU on every pixel"
        )
    if reed_support < MIN_REED_PIXELS_FOR_IOU:
        blocking.append(
            f"only {reed_support} reed pixel(s) are annotated; fewer than "
            f"{MIN_REED_PIXELS_FOR_IOU} cannot measure the reed IoU the proposal "
            "quotes"
        )
    if len(predicted_classes) < 2 and len(present) >= 2:
        blocking.append(
            f"the model predicted a single class across every annotated pixel "
            f"(class {sorted(predicted_classes)[0] if predicted_classes else '?'}), "
            "which is a degenerate prediction rather than a segmentation"
        )

    assessable = not blocking
    if assessable:
        verdict = (
            f"Measured against {len(present)} annotated class(es) with {reed_support} "
            "reed pixels. This is an independent check of detection accuracy."
        )
    else:
        reasons = "; ".join(blocking)
        verdict = (
            "NOT A VALIDATION of the model: "
            + reasons[0].upper()
            + reasons[1:]
            + ". Annotate tiles that cover water, cropland and reed -- enough of each "
            "that the per-class score is a measurement -- before quoting any of these "
            "numbers."
        )

    return {
        "assessable": assessable,
        "verdict": verdict,
        "blocking_reasons": blocking,
        "classes_present": present,
        "classes_absent": absent,
        "reed_support_px": reed_support,
        "predicted_classes": sorted(int(c) for c in predicted_classes),
        "min_reed_pixels": MIN_REED_PIXELS_FOR_IOU,
        "independence": independence,
    }


def score_annotation_pack(
    pack_dir: Path | str,
    checkpoint: Path | str,
    *,
    device: str | None = None,
    write: bool = True,
) -> dict:
    """Score a trained model against human labels in a pack.

    This is the only number in the project that can honestly be called detection
    accuracy, because it is the only one not measured against the training labels.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AtarraError("scoring needs torch, which is not installed") from exc

    import torch

    from atarra.models.segmentation import build_model
    from atarra.train.metrics import (
        ConfusionAccumulator,
        meets_targets,
        unassessable_targets,
    )
    from atarra.train.trainer import load_checkpoint, resolve_device

    pack_dir = Path(pack_dir)
    pack = load_pack(pack_dir)
    resolved = resolve_device(device)

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise AtarraError(f"checkpoint not found: {checkpoint}")

    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = metadata.get("in_channels")
    bands = list(pack["imagery_bands"])
    if expected is not None and int(expected) != len(bands):
        raise AtarraError(
            f"checkpoint expects {expected} input channels but the pack holds {len(bands)} "
            f"bands {bands}. Export the pack with the band set the model was trained on "
            "(`--bands`), or score with a matching checkpoint."
        )

    # Order matters as much as the count. Eight channels of the right shape in the wrong
    # order is not a shape error the framework can catch -- B08 where B04 belongs
    # produces a confident, entirely wrong answer -- so it is checked against the band
    # names the checkpoint was written with, and named as unverifiable when absent.
    trained_bands = metadata.get("band_names")
    if trained_bands and list(trained_bands) != bands:
        raise AtarraError(
            f"the checkpoint was trained on bands {list(trained_bands)} but this pack "
            f"holds {bands}. The channel count matches, so scoring would proceed and be "
            "wrong. Re-export the pack with the training band set."
        )
    if not trained_bands:
        log.warning(
            "this checkpoint predates recorded band names, so the chip's band order "
            "cannot be checked against the model's inputs; the channel count is the "
            "only agreement available"
        )

    independence = _check_independence(
        metadata.get("provenance") or {}, [tile["key"] for tile in pack["tiles"]]
    )
    if independence["status"] == "trained_on":
        raise AtarraError(
            f"refusing to score: {independence['detail']} Exclude the pack's ground from "
            "training (`atarra train --exclude-pack`) and score that run instead."
        )

    model = build_model(in_channels=len(bands), num_classes=NUM_CLASSES, variant="unet")
    load_checkpoint(checkpoint, model, device=resolved)

    accumulator = ConfusionAccumulator(num_classes=NUM_CLASSES)
    per_tile: list[dict] = []
    annotated = 0
    missing: list[str] = []
    predicted_classes: set[int] = set()

    missing_channels = list(bands)

    for tile in pack["tiles"]:
        key = tile["key"]
        chip_path = pack_dir / tile["image_chip"]
        if not chip_path.exists():
            raise AtarraError(f"chip missing from pack: {chip_path}")

        import rasterio

        with rasterio.open(chip_path) as source:
            descriptions = list(source.descriptions)
            stack = source.read().astype(np.float32)
            grid_transform = source.transform
            crs = source.crs
            shape = (source.height, source.width)

        if None in descriptions:
            raise AtarraError(
                f"{chip_path.name} has unnamed bands, so the pack's band order cannot be "
                "verified; re-export the pack"
            )
        order = [descriptions.index(name) for name in missing_channels]
        # The chips are written *as* reflectance -- see the `_write_raster` call in
        # `export_annotation_pack` -- and the model's normalisation buffers were fitted
        # to that convention. Decoding a second time divides an already-decoded 0.2 by
        # 10,000: no exception, no shape change, just inputs four orders of magnitude
        # below anything the model trained on, scoring like a much worse model.
        image = _as_reflectance(stack, source=chip_path.name)[order]

        labels = load_annotations(
            pack_dir, key, shape=shape, transform=grid_transform, crs=crs
        )
        if labels is None:
            missing.append(key)
            continue

        with torch.no_grad():
            tensor = torch.from_numpy(image[None, ...]).to(resolved)
            logits = model(tensor)
            prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)

        target = np.asarray(labels, dtype=np.int64)
        # Anything outside the class set is unlabelled and must not be scored.
        target = np.where((target >= 0) & (target < NUM_CLASSES), target, -1)

        labelled_px = int((target >= 0).sum())
        if labelled_px == 0:
            missing.append(key)
            continue

        accumulator.update(prediction, target, ignore_index=-1)
        predicted_classes.update(np.unique(prediction[target >= 0]).tolist())
        agreement = float((prediction[target >= 0] == target[target >= 0]).mean())
        per_tile.append(
            {
                "key": key,
                "labelled_px": labelled_px,
                "agreement": round(agreement, 4),
            }
        )
        annotated += 1

    if annotated == 0:
        raise AtarraError(
            f"nothing in {pack_dir / 'labels'} has been annotated yet, so there is no "
            "independent truth to score against. Label at least one tile (see the pack "
            "README; `labels/<key>.geojson` with a class_code field, or a raster on the "
            "chip's grid) and run this again. Refusing to fall back to the weak labels, "
            "because a score against them is exactly the circular number this pack "
            "exists to avoid."
        )

    report = accumulator.report()
    assessment = _assess(report, predicted_classes, independence)
    result = {
        "pack": str(pack_dir),
        "checkpoint": str(checkpoint),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(resolved),
        "bands": bands,
        "band_order_verified": bool(trained_bands),
        "tiles_reserved": len(pack["tiles"]),
        "tiles_annotated": annotated,
        "tiles_still_unlabelled": missing,
        "report": report,
        # A verdict, or an explicit statement that none was reached. Never a boolean
        # standing in for a comparison that did not happen.
        "targets": (
            meets_targets(report)
            if assessment["assessable"]
            else unassessable_targets(assessment["verdict"])
        ),
        "assessable": assessment["assessable"],
        "assessment": assessment,
        "per_tile": per_tile,
        "truth": (
            "Scored against human annotations in labels/, not against the rule engine. "
            "This is an independent measure of detection accuracy -- provided the "
            "annotations cover enough classes and the checkpoint can be shown never to "
            "have seen this ground; see `assessable` and "
            "`assessment.independence`."
        ),
    }

    if write:
        (pack_dir / "score.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )

    log.info(
        "scored against %d annotated tile(s): mIoU %.4f | phragmites IoU %s | F1 %s",
        annotated,
        report["mean_iou"],
        report["phragmites_iou"],
        report["phragmites_f1"],
    )
    if not assessment["assessable"]:
        log.warning("this score is NOT a validation: %s", assessment["verdict"])
    if missing:
        log.warning(
            "%d reserved tile(s) still have no labels: %s",
            len(missing),
            ", ".join(missing[:5]) + ("..." if len(missing) > 5 else ""),
        )
    return result
