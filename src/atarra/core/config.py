"""Typed loaders for the YAML configuration files.

Configuration is validated eagerly and with specific messages. A typo in a band
name would otherwise surface much later as a missing STAC asset or a tensor with
the wrong channel count -- both of which look like model bugs rather than config
bugs, and cost far more to debug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from atarra.core.errors import ConfigError
from atarra.core.grids import BBox
from atarra.core.settings import get_settings

VALID_RESAMPLE = {"bilinear", "nearest", "cubic", "average", "mode"}

# How the archive encodes reflectance. See configs/bands.yaml for the measurements
# that decided which one is the default.
VALID_REFLECTANCE_MODES = {"dn_scale", "offset_applied"}


@dataclass(frozen=True)
class BandSpec:
    """One spectral band: where to fetch it and how to place it on the grid."""

    name: str
    asset: str
    gsd: float
    resample: str


@dataclass(frozen=True)
class BandsConfig:
    """The full band contract for the pipeline."""

    bands: dict[str, BandSpec]
    mask: BandSpec
    bands_8: list[str]
    bands_10: list[str]
    bands_rgb: list[str]
    reflectance_mode: str
    reflectance_scale: float
    declared_offset: float
    max_negative_fraction: float
    nodata: int
    scl_invalid_classes: list[int]
    scl_water_classes: list[int]
    scl_vegetation_classes: list[int] = field(default_factory=list)

    def spec(self, name: str) -> BandSpec:
        try:
            return self.bands[name]
        except KeyError as exc:  # pragma: no cover - guarded at load time
            raise ConfigError(f"unknown band {name!r}; known: {sorted(self.bands)}") from exc

    def resolve(self, names: list[str]) -> list[BandSpec]:
        return [self.spec(n) for n in names]

    def asset_names(self, names: list[str]) -> list[str]:
        """STAC asset names for a band list, in order."""
        return [self.spec(n).asset for n in names]

    @property
    def reflectance_offset(self) -> float:
        """Offset to apply when converting stored integers to reflectance."""
        if self.reflectance_mode == "offset_applied":
            return self.declared_offset
        return 0.0

    def reflectance(self, storage_values):
        """Convert stored integers to reflectance under the validated convention."""
        return storage_values * self.reflectance_scale + self.reflectance_offset


def _require(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise ConfigError(f"{context}: missing required key {key!r}")
    return mapping[key]


def load_bands(path: Path | None = None) -> BandsConfig:
    """Load ``configs/bands.yaml``."""
    path = path or (get_settings().configs_dir / "bands.yaml")
    if not path.exists():
        raise ConfigError(f"band config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")

    bands: dict[str, BandSpec] = {}
    for name, spec in (_require(raw, "bands", str(path)) or {}).items():
        resample = str(spec.get("resample", "bilinear"))
        if resample not in VALID_RESAMPLE:
            raise ConfigError(
                f"{path}: band {name} has invalid resample {resample!r}; "
                f"expected one of {sorted(VALID_RESAMPLE)}"
            )
        bands[name] = BandSpec(
            name=name,
            asset=str(_require(spec, "asset", f"band {name}")),
            gsd=float(_require(spec, "gsd", f"band {name}")),
            resample=resample,
        )
    if not bands:
        raise ConfigError(f"{path}: no bands defined")

    mask_raw = _require(raw, "mask_band", str(path))
    mask = BandSpec(
        name=str(mask_raw.get("name", "SCL")),
        asset=str(_require(mask_raw, "asset", "mask_band")),
        gsd=float(mask_raw.get("gsd", 20)),
        resample=str(mask_raw.get("resample", "nearest")),
    )
    if mask.resample != "nearest":
        # Not merely a preference: SCL holds class codes, and interpolating them
        # invents codes that do not exist (e.g. a blend of "cloud" and "water").
        raise ConfigError(
            f"{path}: mask band must use nearest resampling, got {mask.resample!r}"
        )

    def _band_list(key: str) -> list[str]:
        values = list(raw.get(key) or [])
        unknown = [v for v in values if v not in bands]
        if unknown:
            raise ConfigError(f"{path}: {key} references unknown bands {unknown}")
        if not values:
            raise ConfigError(f"{path}: {key} is empty")
        return values

    reflectance_raw = raw.get("reflectance") or {}
    mode = str(reflectance_raw.get("mode", "dn_scale"))
    if mode not in VALID_REFLECTANCE_MODES:
        raise ConfigError(
            f"{path}: reflectance.mode must be one of "
            f"{sorted(VALID_REFLECTANCE_MODES)}, got {mode!r}"
        )

    return BandsConfig(
        bands=bands,
        mask=mask,
        bands_8=_band_list("bands_8"),
        bands_10=_band_list("bands_10"),
        bands_rgb=_band_list("bands_rgb"),
        reflectance_mode=mode,
        reflectance_scale=float(reflectance_raw.get("scale", 0.0001)),
        declared_offset=float(reflectance_raw.get("declared_offset", -0.1)),
        max_negative_fraction=float(reflectance_raw.get("max_negative_fraction", 0.02)),
        nodata=int(raw.get("nodata", 0)),
        scl_invalid_classes=[int(c) for c in (raw.get("scl_invalid_classes") or [])],
        scl_water_classes=[int(c) for c in (raw.get("scl_water_classes") or [])],
        scl_vegetation_classes=[int(c) for c in (raw.get("scl_vegetation_classes") or [])],
    )


@dataclass(frozen=True)
class StudyArea:
    """A monitored zone."""

    key: str
    name: str
    role: str
    bbox: BBox
    crs: str

    @property
    def is_wetland(self) -> bool:
        return self.role == "wetland"


def load_study_areas(path: Path | None = None) -> dict[str, StudyArea]:
    """Load ``configs/study_areas.yaml``."""
    path = path or (get_settings().configs_dir / "study_areas.yaml")
    if not path.exists():
        raise ConfigError(f"study area config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    areas_raw = _require(raw, "study_areas", str(path)) or {}
    if not areas_raw:
        raise ConfigError(f"{path}: no study areas defined")

    areas: dict[str, StudyArea] = {}
    for key, spec in areas_raw.items():
        bbox_values = _require(spec, "bbox", f"study area {key}")
        if len(bbox_values) != 4:
            raise ConfigError(
                f"{path}: study area {key} bbox must be [west, south, east, north], "
                f"got {bbox_values!r}"
            )
        try:
            bbox = BBox.from_sequence(tuple(bbox_values), crs="EPSG:4326")
        except ValueError as exc:
            raise ConfigError(f"{path}: study area {key}: {exc}") from exc

        areas[key] = StudyArea(
            key=key,
            name=str(spec.get("name", key)),
            role=str(spec.get("role", "wetland")),
            bbox=bbox,
            crs=str(spec.get("crs", "EPSG:32636")),
        )
    return areas


@lru_cache(maxsize=1)
def get_bands() -> BandsConfig:
    """Cached band configuration."""
    return load_bands()


@lru_cache(maxsize=1)
def get_study_areas() -> dict[str, StudyArea]:
    """Cached study-area configuration."""
    return load_study_areas()


def get_study_area(key: str) -> StudyArea:
    areas = get_study_areas()
    try:
        return areas[key]
    except KeyError as exc:
        raise ConfigError(
            f"unknown study area {key!r}; known: {sorted(areas)}"
        ) from exc
