"""Typed application settings.

All values come from the environment with an ``ATARRA_`` prefix (optionally via a
``.env`` file at the project root). Defaults are chosen so a fresh checkout works
against the credential-free STAC/AWS imagery path with no configuration at all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# settings.py -> core -> atarra -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATARRA_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    configs_dir: Path = Field(default=PROJECT_ROOT / "configs")

    # --- Imagery discovery ---------------------------------------------------
    # Earth Search (Element 84) mirrors ESA Sentinel-2 L2A as public COGs on S3.
    # Verified live: no credentials, byte-range reads, 2022-2026 coverage over
    # the Nile Delta. See README for why this is preferred over sentinelsat.
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    stac_collection: str = "sentinel-2-l2a"
    stac_timeout_s: float = 60.0

    # --- Quality gates -------------------------------------------------------
    max_cloud_cover: float = 10.0

    # --- Target raster grid --------------------------------------------------
    # Everything is resampled onto a single projected grid at this resolution.
    target_gsd: float = 10.0
    tile_size: int = 256

    # --- Storage guard rails -------------------------------------------------
    # Hard ceiling for the derived-product cache. The drive this was developed on
    # had under 3 GB free, so an unbounded cache is a real failure mode, not a
    # theoretical one.
    max_cache_gb: float = 2.0

    # --- Optional secondary imagery source (ESA CDSE OAuth) ------------------
    cdse_username: str | None = None
    cdse_password: str | None = None

    # --- Derived paths -------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def tiles_dir(self) -> Path:
        return self.data_dir / "tiles"

    @property
    def previews_dir(self) -> Path:
        return self.data_dir / "previews"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def spatial_db_path(self) -> Path:
        return self.data_dir / "atarra.gpkg"

    def ensure_dirs(self) -> None:
        """Create the data directories. Safe to call repeatedly."""
        for path in (
            self.data_dir,
            self.cache_dir,
            self.tiles_dir,
            self.previews_dir,
            self.checkpoints_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (cached)."""
    return Settings()
