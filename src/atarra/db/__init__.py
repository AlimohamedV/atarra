"""Spatial persistence with a PostGIS-shaped schema."""

from atarra.db.repository import (
    Detection,
    GeoPackageRepository,
    PostgisRepository,
    get_repository,
)

__all__ = [
    "Detection",
    "GeoPackageRepository",
    "PostgisRepository",
    "get_repository",
]
