"""Spectral index computation.

Sentinel-2 L2A reflectance is bottom-of-atmosphere, so these are the standard
BOA-form indices. Every function here returns ``NaN`` where inputs are invalid
rather than propagating a value.

That NaN discipline matters more than it looks: the naive implementation
``(nir - red) / (nir + red)`` divides by zero wherever both bands are fill, and
numpy answers with ``inf`` or ``nan`` depending on the numerator. An ``inf``
survives a mean, a percentile, and a loss function, and only shows up much later
as a suspiciously good or ``nan`` metric. Guarding at the source is cheap.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from atarra.core.errors import ImageryError

# index name -> (positive band, negative band), expressed as
# (a - b) / (a + b).
INDEX_BANDS: dict[str, tuple[str, str]] = {
    "ndvi": ("B08", "B04"),  # NIR vs red       -- green vegetation vigour
    "ndwi": ("B03", "B08"),  # green vs NIR     -- open water
    "ndre": ("B08", "B05"),  # NIR vs red-edge1 -- dense canopy / reed discrimination
    "ndmi": ("B08", "B11"),  # NIR vs SWIR1     -- canopy moisture
}

_INDEX_DESCRIPTIONS = {
    "ndvi": "Normalised Difference Vegetation Index",
    "ndwi": "Normalised Difference Water Index",
    "ndre": "Red-Edge Normalised Difference Vegetation Index",
    "ndmi": "Normalised Difference Moisture Index",
}


def normalized_difference(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Compute ``(a - b) / (a + b)`` safely.

    Where the denominator is degenerate (both bands fill, or both exactly zero),
    the result is ``NaN`` -- an explicit "no data here" -- instead of ``inf`` or a
    fabricated zero.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ImageryError(f"index bands have mismatched shapes: {a.shape} vs {b.shape}")

    denominator = a + b
    result = np.full(a.shape, np.nan, dtype=np.float32)
    usable = np.isfinite(denominator) & (np.abs(denominator) > eps)
    np.divide(a - b, denominator, out=result, where=usable)
    return result


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NIR vs red. Dense reed canopy sits high; turbid water and soil sit low."""
    return normalized_difference(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Green vs NIR (McFeeters). Positive over open water."""
    return normalized_difference(green, nir)


def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """NIR vs red-edge 1.

    The reason this project is not an RGB project: red-edge reflectance keeps
    rising with chlorophyll well past the point where red saturates, so NDRE
    separates dense reed beds from vigorously growing crops long after NDVI has
    flattened out for both.
    """
    return normalized_difference(nir, red_edge)


def ndmi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """NIR vs SWIR1. Tracks canopy water content, which drives harvest scheduling."""
    return normalized_difference(nir, swir)


def compute_indices(
    stack,
    which: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute indices from a :class:`~atarra.preprocess.reader.BandStack`.

    Indices whose bands are absent from the stack are skipped rather than raising,
    so a 3-band RGB stack yields a useful (if smaller) result instead of an error.
    That is what lets the RGB baseline run through the same code path as the
    8-band model.
    """
    names = list(which) if which is not None else list(INDEX_BANDS)
    available = set(stack.band_names)

    results: dict[str, np.ndarray] = {}
    for name in names:
        key = name.lower()
        if key not in INDEX_BANDS:
            raise ImageryError(
                f"unknown index {name!r}; known: {sorted(INDEX_BANDS)}"
            )
        band_a, band_b = INDEX_BANDS[key]
        if band_a not in available or band_b not in available:
            continue
        results[key] = normalized_difference(stack.band(band_a), stack.band(band_b))
    return results


def describe(name: str) -> str:
    """Human-readable index name."""
    return _INDEX_DESCRIPTIONS.get(name.lower(), name.upper())
