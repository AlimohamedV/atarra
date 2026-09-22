"""Weak-supervision labelling.

Hand-digitising reed beds in QGIS for months is the single most likely way for a
project like this to run out of time, and it is not even the best use of a human's
attention. Instead, physical rules over spectral indices generate candidate labels
with a confidence for each pixel, and a human verifies only the uncertain ones.

**These labels are a bootstrap, not ground truth.** Everything here is tuned to be
*conservative*: a pixel is only labelled confidently when several independent
signals agree, and anything ambiguous is explicitly handed to the review queue
rather than guessed at. A confidently wrong label is far more damaging than an
abstained one, because the model learns it as fact.

What makes the rules work, physically:

  * Emergent reeds are the only class that is simultaneously *wet*, *tall* and
    *high in chlorophyll* for the whole season. Water fails the vegetation test,
    crops fail the wetness test, and bare soil fails both.
  * The red-edge index is what separates reed from cropland. Toward late summer
    NDVI saturates for both, but red-edge reflectance keeps responding to canopy
    structure and chlorophyll, so NDRE holds the contrast after NDVI has flattened.

External priors (ESA WorldCover, Dynamic World, JRC Global Surface Water) slot in
as additional agreement votes where they are available; see ``prior_agreement``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger

log = get_logger("datasets.weak_labels")

OPEN_WATER = 0
CROPS_SOIL = 1
MIXED_HALOPHYTES = 2
PHRAGMITES = 3

# The class order is fixed pipeline-wide: the labeller, the loss, and every metric
# index into it. This is the only definition -- metrics and the model import these
# rather than repeating the list, because three copies that agree today are three
# copies that can disagree tomorrow, and the failure mode is silently scoring one
# class with another's numbers.
CLASS_NAMES = ["open_water", "crops_soil", "mixed_halophytes", "phragmites_australis"]
NUM_CLASSES = len(CLASS_NAMES)
PHRAGMITES_CODE = PHRAGMITES

# Below this per-pixel confidence a label is unreliable and belongs in the human
# review queue. Chosen so that the review set stays a manageable fraction of the
# scene while still capturing the genuinely ambiguous margins.
REVIEW_THRESHOLD = 0.60


@dataclass
class WeakLabelConfig:
    """Thresholds for the rule engine. All are tunable against review data."""

    water_ndwi: float = 0.10
    vegetation_ndvi: float = 0.35
    # Red-edge contrast: dense intact canopy versus cropland.
    reed_ndre: float = 0.28
    reed_ndvi: float = 0.55
    # Reed stands stay moist even at peak summer; crops do not.
    reed_ndmi: float = 0.08
    # A reed bed is never deep open water.
    reed_max_ndwi: float = 0.25
    halophyte_ndvi: float = 0.20
    min_agreement: int = 2
    # Drop genuinely ambiguous pixels from the loss mask (standard confidence
    # filtering for weak supervision). Set False to train on every usable pixel.
    drop_ambiguous: bool = True
    # The ambiguity test is deliberately NOT the review test. REVIEW_THRESHOLD is
    # tuned so the human review queue stays a manageable size, and the per-class
    # scores have different ceilings by design -- the crop and halophyte rules
    # saturate low. Using it as a training filter drops the entire cropland class
    # at 0.597 against a 0.60 cut and every halophyte at 0.501, leaving a 2-class
    # problem wearing a 4-class label. Margin is what actually separates a coin
    # flip from a decision: a clear reed pixel scores 0.23 margin, the reed/crop
    # confusion 0.067, cropland 0.52. So margin and agreement do the filtering.
    train_margin: float = 0.08
    train_max_agreement: int = 2
    notes: list[str] = field(default_factory=list)


def _water_score(ndvi: np.ndarray, ndwi: np.ndarray, cfg: WeakLabelConfig) -> np.ndarray:
    """Confidence that a pixel is open water.

    High NDWI is the primary signal; low NDVI corroborates it, which matters over
    turbid delta water where NDWI alone weakens.
    """
    wet = np.clip((ndwi - cfg.water_ndwi) / 0.35, 0.0, 1.0)
    not_green = np.clip((cfg.vegetation_ndvi - ndvi) / 0.35, 0.0, 1.0)
    return np.clip(0.65 * wet + 0.35 * not_green, 0.0, 1.0)


def _phragmites_score(
    ndvi: np.ndarray, ndre: np.ndarray, ndmi: np.ndarray, ndwi: np.ndarray, cfg: WeakLabelConfig
) -> np.ndarray:
    """Confidence that a pixel is *Phragmites australis*.

    All four signals must lean the same way, so the *minimum* rather than the mean
    drives the score. A pixel that looks like a textbook reed by NDVI but not by
    NDRE is exactly the crop confusion this project has to avoid, and averaging
    would let the strong signal carry the weak one.
    """
    vigour = np.clip((ndvi - cfg.reed_ndvi) / 0.30, 0.0, 1.0)
    red_edge = np.clip((ndre - cfg.reed_ndre) / 0.25, 0.0, 1.0)
    moisture = np.clip((ndmi - cfg.reed_ndmi) / 0.25, 0.0, 1.0)
    not_deep_water = np.clip((cfg.reed_max_ndwi - ndwi) / 0.30, 0.0, 1.0)
    return np.minimum(np.minimum(vigour, red_edge), np.minimum(moisture, not_deep_water))


def _crops_score(
    ndvi: np.ndarray, ndre: np.ndarray, ndwi: np.ndarray, cfg: WeakLabelConfig
) -> np.ndarray:
    """Confidence that a pixel is cropland or bare soil.

    Vigorous vegetation that is *not* wet and does *not* show the red-edge
    structure of a dense natural stand. The subtraction here is what keeps
    cropland out of the reed class.
    """
    vigour = np.clip((ndvi - cfg.vegetation_ndvi) / 0.35, 0.0, 1.0)
    dry = np.clip((0.0 - ndwi) / 0.35, 0.0, 1.0)
    less_red_edge = np.clip((cfg.reed_ndre - ndre) / 0.25, 0.0, 1.0)
    return np.clip(0.45 * vigour + 0.25 * dry + 0.30 * less_red_edge, 0.0, 1.0)


def _halophyte_score(
    ndvi: np.ndarray, ndre: np.ndarray, ndwi: np.ndarray, cfg: WeakLabelConfig
) -> np.ndarray:
    """Confidence that a pixel is mixed aquatic halophyte.

    The residual class: something is growing, it is not plainly water, and it does
    not commit to either the crop or the reed signature. Deliberately given the
    lowest ceiling -- ambiguous pixels should stay ambiguous and reach review
    rather than being confidently mislabelled as a real class.
    """
    some_growth = np.clip((ndvi - cfg.halophyte_ndvi) / 0.25, 0.0, 1.0)
    not_full_reed = np.clip((cfg.reed_ndvi - ndvi) / 0.20, 0.0, 1.0) + np.clip(
        (cfg.reed_ndre - ndre) / 0.20, 0.0, 1.0
    )
    near_water = np.clip((ndwi + 0.20) / 0.40, 0.0, 1.0)
    return np.clip(0.55 * np.minimum(some_growth, np.clip(not_full_reed, 0.0, 1.0)) + 0.45 * near_water, 0.0, 0.7)


def weak_label(
    indices: dict[str, np.ndarray],
    *,
    config: WeakLabelConfig | None = None,
    valid: np.ndarray | None = None,
    prior_agreement: dict[int, np.ndarray] | None = None,
) -> dict:
    """Assign candidate class labels and confidences.

    Returns a dict with ``labels`` (int8), ``confidence`` (float32), ``review``
    (boolean mask of pixels needing human verification) and per-class scores.
    """
    cfg = config or WeakLabelConfig()
    required = {"ndvi", "ndwi", "ndre", "ndmi"}
    missing = required - set(indices)
    if missing:
        raise AtarraError(
            f"weak_label needs indices {sorted(required)}; missing {sorted(missing)}"
        )

    ndvi = np.asarray(indices["ndvi"], dtype=np.float32)
    ndwi = np.asarray(indices["ndwi"], dtype=np.float32)
    ndre = np.asarray(indices["ndre"], dtype=np.float32)
    ndmi = np.asarray(indices["ndmi"], dtype=np.float32)

    scores = {
        OPEN_WATER: _water_score(ndvi, ndwi, cfg),
        CROPS_SOIL: _crops_score(ndvi, ndre, ndwi, cfg),
        MIXED_HALOPHYTES: _halophyte_score(ndvi, ndre, ndwi, cfg),
        PHRAGMITES: _phragmites_score(ndvi, ndre, ndmi, ndwi, cfg),
    }

    if prior_agreement:
        # An external land-cover product voting for a class adds directly to that
        # class's confidence. Priors that disagree simply fail to add.
        for code, bonus in prior_agreement.items():
            if code in scores:
                scores[code] = np.clip(scores[code] + bonus, 0.0, 1.0)

    stack = np.stack([scores[c] for c in (OPEN_WATER, CROPS_SOIL, MIXED_HALOPHYTES, PHRAGMITES)])
    labels = np.argmax(stack, axis=0).astype(np.int8)
    confidence = np.max(stack, axis=0).astype(np.float32)

    # Agreement count: how many independent class scores are within a small margin
    # of the winner. A pixel where two classes tie is not confident however high
    # the winning score happens to be.
    sorted_scores = np.sort(stack, axis=0)
    margin = sorted_scores[-1] - sorted_scores[-2]
    agree = (stack >= (sorted_scores[-1][None, :] - 0.10)).sum(axis=0)

    confident = (confidence >= REVIEW_THRESHOLD) & (margin >= 0.08) & (agree <= 2)

    # Which pixels carry information about their own label, as opposed to being
    # coin flips between two classes. See the note on `train_margin`.
    ambiguous = (margin < cfg.train_margin) | (agree > cfg.train_max_agreement)

    # The mixed-halophyte class is the residual bucket: "something is growing here
    # and it is not plainly water, crop or reed". By construction we cannot be
    # confident about it from spectra alone, so it always goes to human review --
    # that class needs actual annotation.
    halophyte = labels == MIXED_HALOPHYTES

    finite = np.isfinite(stack).all(axis=0)
    if valid is not None:
        usable = valid & finite
    else:
        usable = finite

    # `review` is the annotation worklist: pixels that carry a label and that a human
    # should adjudicate. Restricted to usable pixels on purpose. Nodata is not a
    # question for a human, it is simply absent -- and counting it as review made the
    # review *density* of a tile measure how much of its bounding box falls outside the
    # rotated Sentinel-2 swath (70%+ is typical). Ranking by that selects the emptiest
    # tiles instead of the most uncertain ones, which is the opposite of the intent.
    review = (~confident | halophyte) & usable

    # `trainable` is the loss mask, and it is deliberately NOT the same mask.
    #
    # Halophyte pixels stay trainable even though the class is flagged for review.
    # This docstring used to read as though the class should simply be withheld,
    # but that does not survive contact with the loss: a class with zero examples
    # is not conservative, it is unlearnable. Its IoU would be undefined and the
    # 4-class mIoU would quietly be arithmetic over three classes. So the class is
    # trained on, its weakness is reported per class instead of hidden, and it is
    # the top annotation priority.
    #
    # Ambiguity is the genuinely different case: a pixel that is a coin flip
    # between reed and crop teaches the model to confuse exactly the pair this
    # project exists to separate. Those are dropped from the loss and become the
    # annotation queue instead.
    if cfg.drop_ambiguous:
        trainable = usable & ~ambiguous
    else:
        trainable = usable

    labels = np.where(usable, labels, -1).astype(np.int8)

    return {
        "labels": labels,
        "confidence": confidence,
        "review": review,
        "usable": usable,
        "trainable": trainable,
        "ambiguous": ambiguous,
        "scores": scores,
        "margin": margin,
    }


def label_statistics(result: dict, *, class_names: list[str] | None = None) -> dict:
    """Summarise a labelling pass, including how much needs human review."""
    labels = result["labels"]
    usable = result["usable"]
    total = int(usable.sum())
    if total == 0:
        return {"usable_px": 0, "review_px": 0, "review_fraction": 0.0, "classes": []}

    classes = []
    for code, name in enumerate(class_names or CLASS_NAMES):
        count = int(((labels == code) & usable).sum())
        classes.append(
            {
                "class_code": code,
                "class_name": name,
                "pixels": count,
                "fraction": round(count / total, 5),
            }
        )

    review_px = int(result["review"].sum())
    trainable_px = int(result["trainable"].sum())
    return {
        "usable_px": total,
        "trainable_px": trainable_px,
        "dropped_px": total - trainable_px,
        "dropped_fraction": round((total - trainable_px) / max(1, total), 5),
        "review_px": review_px,
        # Over usable pixels, not over the frame: the frame includes the swath's empty
        # corners, which would make this ratio a measure of tile geometry.
        "review_fraction": round(review_px / max(1, total), 5),
        "mean_confidence": round(float(result["confidence"][usable].mean()), 4),
        "classes": classes,
    }
