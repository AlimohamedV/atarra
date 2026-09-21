"""Spatial persistence.

Two backends behind one interface. ``GeoPackageRepository`` is the working
default: a single file, no server, no extensions, readable by QGIS, GeoPandas and
GDAL. ``PostgisRepository`` implements the target schema in ``schema.sql`` for when
the dataset outgrows a file.

The interface exists so that swap is genuinely mechanical. Nothing above this
module knows which one is in use.

A note on area: every area is computed in a projected CRS, never in EPSG:4326.
An area in degrees is a number that looks reasonable and is meaningless, and it
would silently corrupt every hectare figure the decision engine reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

from atarra.core.config import StudyArea, get_study_areas
from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger
from atarra.core.settings import get_settings

log = get_logger("db")

# Class codes and names, matching the segmentation contract.
CLASS_NAMES = {0: "open_water", 1: "crops_soil", 2: "mixed_halophytes", 3: "phragmites_australis"}


@dataclass
class Detection:
    """One classified patch from the segmentation model."""

    area_key: str
    scene_date: Date
    class_code: int
    geometry: object  # shapely geometry in EPSG:4326
    confidence: float | None = None
    model_version: str | None = None
    area_m2: float | None = None
    composite_id: int | None = None

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_code, f"class_{self.class_code}")

    def as_geojson(self) -> dict:
        return {
            "type": "Feature",
            "geometry": mapping(self.geometry),
            "properties": {
                "area_key": self.area_key,
                "scene_date": self.scene_date.isoformat(),
                "class_code": self.class_code,
                "class_name": self.class_name,
                "confidence": self.confidence,
                "area_m2": self.area_m2,
                "area_ha": (self.area_m2 / 10000.0) if self.area_m2 else None,
                "model_version": self.model_version,
            },
        }


def geometry_area_m2(geometry, src_crs: str | CRS, area: StudyArea) -> float:
    """Area in square metres, computed in the study area's projected CRS.

    Using an equal-area-aware projected CRS rather than computing in degrees: the
    difference over a 30 km reed bed is not a rounding error.
    """
    transformer = Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(area.crs), always_xy=True)
    projected = shapely_transform(lambda x, y: transformer.transform(x, y), geometry)
    return float(projected.area)


class SpatialRepository(Protocol):
    """What every persistence backend must provide."""

    def ensure_schema(self) -> None: ...

    def upsert_study_areas(self, areas: Sequence[StudyArea]) -> int: ...

    def insert_detections(self, detections: Sequence[Detection]) -> int: ...

    def detections(self, area_key: str, *, on: Date | None = None) -> list[Detection]: ...

    def coverage_summary(self, area_key: str) -> list[dict]: ...


class GeoPackageRepository:
    """Store ATARRA entities in a GeoPackage (SQLite + PostGIS-style geometry types).

    Two layers: ``study_areas`` for polygons, and ``detections`` for model output.
    Alphanumeric attributes live in ordinary SQLite tables so the file stays
    readable by anything that speaks GeoPackage.
    """

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        settings.ensure_dirs()
        self.path = Path(path or settings.spatial_db_path)

    # --- schema --------------------------------------------------------------
    def ensure_schema(self) -> None:
        """Create the file and its tables if they do not exist.

        GDAL needs at least one feature to have a schema to work from, so we write
        an empty layer with the right columns and geometry type.
        """
        if self.path.exists():
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_layer(self.path, "study_areas", _empty_features("footprint", "Polygon"))
        _write_layer(self.path, "detections", _empty_features("geometry", "Polygon"))

        import sqlite3

        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS threat_scores (
                    detection_id INTEGER PRIMARY KEY,
                    score REAL NOT NULL,
                    coverage_component REAL NOT NULL,
                    width_component REAL NOT NULL,
                    priority_component REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS harvest_windows (
                    area_key TEXT NOT NULL,
                    season_year INTEGER NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    peak_date TEXT NOT NULL,
                    rationale TEXT,
                    PRIMARY KEY (area_key, season_year)
                )
                """
            )
            connection.commit()
        log.info("initialised spatial database at %s", self.path)

    # --- study areas ---------------------------------------------------------
    def upsert_study_areas(self, areas: Sequence[StudyArea] | None = None) -> int:
        self.ensure_schema()
        areas = list(areas or get_study_areas().values())
        rows = pd.DataFrame(
            [
                {
                    "key": area.key,
                    "name": area.name,
                    "role": area.role,
                    "crs": area.crs,
                    "footprint": shape(area.bbox.as_geojson()),
                }
                for area in areas
            ]
        )
        _write_layer(self.path, "study_areas", rows, mode="w")
        return len(rows)

    # --- detections ----------------------------------------------------------
    def insert_detections(self, detections: Sequence[Detection]) -> int:
        self.ensure_schema()
        detections = list(detections)
        if not detections:
            return 0

        areas = get_study_areas()
        rows = []
        for detection in detections:
            area = areas.get(detection.area_key)
            area_m2 = detection.area_m2
            if area_m2 is None and area is not None:
                area_m2 = geometry_area_m2(detection.geometry, "EPSG:4326", area)
            rows.append(
                {
                    "area_key": detection.area_key,
                    "scene_date": detection.scene_date.isoformat(),
                    "class_code": detection.class_code,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "area_m2": area_m2,
                    "model_version": detection.model_version,
                    "geometry": detection.geometry,
                }
            )

        frame = pd.DataFrame(rows)
        mode = "a" if self.path.exists() and _layer_has_features(self.path, "detections") else "w"
        _write_layer(self.path, "detections", frame, mode=mode)
        log.info("stored %d detection(s) in %s", len(rows), self.path.name)
        return len(rows)

    def detections(self, area_key: str, *, on: Date | None = None) -> list[Detection]:
        if not self.path.exists():
            return []
        query = f"SELECT * FROM detections WHERE area_key = '{area_key}'"
        if on is not None:
            query += f" AND scene_date = '{on.isoformat()}'"
        data = _read_query(self.path, query)
        results: list[Detection] = []
        for record in data:
            geometry = record.get("geometry")
            if geometry is None:
                continue
            results.append(
                Detection(
                    area_key=record["area_key"],
                    scene_date=Date.fromisoformat(str(record["scene_date"])),
                    class_code=int(record["class_code"]),
                    geometry=shape(geometry) if isinstance(geometry, dict) else geometry,
                    confidence=record.get("confidence"),
                    model_version=record.get("model_version"),
                    area_m2=record.get("area_m2"),
                )
            )
        return results

    def coverage_summary(self, area_key: str) -> list[dict]:
        """Infested area and patch count per date, for the trend view."""
        if not self.path.exists():
            return []
        query = f"""
            SELECT scene_date,
                   COUNT(*) AS patch_count,
                   SUM(area_m2) AS total_area_m2,
                   SUM(area_m2) / 10000.0 AS total_area_ha,
                   AVG(confidence) AS mean_confidence
            FROM detections
            WHERE area_key = '{area_key}' AND class_code = 3
            GROUP BY scene_date
            ORDER BY scene_date
        """
        return _read_query(self.path, query)

    def store_threat_scores(self, scores: Sequence[dict]) -> int:
        """Persist threat scores alongside the components that produced them."""
        self.ensure_schema()
        import sqlite3

        with sqlite3.connect(self.path) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO threat_scores
                    (detection_id, score, coverage_component, width_component, priority_component)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(s["detection_id"]),
                        float(s["score"]),
                        float(s["coverage_component"]),
                        float(s["width_component"]),
                        float(s["priority_component"]),
                    )
                    for s in scores
                ],
            )
            connection.commit()
        return len(scores)


class PostgisRepository:
    """Target backend: the schema in ``schema.sql``, served by PostgreSQL + PostGIS.

    Not the default, and deliberately so -- the development environment has no
    PostGIS, and this module does not pretend otherwise. It fails loudly with an
    actionable message rather than silently degrading.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise AtarraError(
                "PostGIS backend requires the `psycopg` driver, which is not "
                "installed. Install it with `pip install 'psycopg[binary]'` and "
                "provide a DSN, or use the default GeoPackage backend."
            ) from exc

    def ensure_schema(self) -> None:
        import psycopg

        schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
        with psycopg.connect(self.dsn) as connection:
            connection.execute(schema)
            connection.commit()
        log.info("applied PostGIS schema")

    def upsert_study_areas(self, areas: Sequence[StudyArea] | None = None) -> int:
        import json

        import psycopg

        areas = list(areas or get_study_areas().values())
        with psycopg.connect(self.dsn) as connection:
            for area in areas:
                connection.execute(
                    """
                    INSERT INTO study_areas (key, name, role, crs, footprint)
                    VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                    ON CONFLICT (key) DO UPDATE
                        SET name = EXCLUDED.name,
                            role = EXCLUDED.role,
                            crs = EXCLUDED.crs,
                            footprint = EXCLUDED.footprint
                    """,
                    (
                        area.key,
                        area.name,
                        area.role,
                        area.crs,
                        json.dumps(area.bbox.as_geojson()),
                    ),
                )
            connection.commit()
        return len(areas)

    def insert_detections(self, detections: Sequence[Detection]) -> int:
        import json

        import psycopg

        areas = get_study_areas()
        with psycopg.connect(self.dsn) as connection:
            for detection in detections:
                area = areas.get(detection.area_key)
                area_m2 = detection.area_m2
                if area_m2 is None and area is not None:
                    area_m2 = geometry_area_m2(detection.geometry, "EPSG:4326", area)
                connection.execute(
                    """
                    INSERT INTO detections
                        (area_key, scene_date, class_code, class_name, confidence,
                         geometry, area_m2, model_version)
                    VALUES (%s, %s, %s, %s, %s,
                            ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s, %s)
                    """,
                    (
                        detection.area_key,
                        detection.scene_date,
                        detection.class_code,
                        detection.class_name,
                        detection.confidence,
                        json.dumps(mapping(detection.geometry)),
                        area_m2,
                        detection.model_version,
                    ),
                )
            connection.commit()
        return len(detections)

    def detections(self, area_key: str, *, on: Date | None = None) -> list[Detection]:
        import json

        import psycopg

        query = (
            "SELECT area_key, scene_date, class_code, confidence, area_m2, model_version, "
            "ST_AsGeoJSON(geometry) AS geometry FROM detections WHERE area_key = %s"
        )
        params: list = [area_key]
        if on is not None:
            query += " AND scene_date = %s"
            params.append(on)

        with psycopg.connect(self.dsn) as connection:
            rows = connection.execute(query, params).fetchall()

        return [
            Detection(
                area_key=row[0],
                scene_date=row[1],
                class_code=int(row[2]),
                confidence=row[3],
                area_m2=row[4],
                model_version=row[5],
                geometry=shape(json.loads(row[6])),
            )
            for row in rows
        ]

    def coverage_summary(self, area_key: str) -> list[dict]:
        import psycopg

        with psycopg.connect(self.dsn) as connection:
            rows = connection.execute(
                """
                SELECT scene_date, count(*) AS patch_count,
                       sum(area_m2) AS total_area_m2,
                       sum(area_m2) / 10000.0 AS total_area_ha,
                       avg(confidence) AS mean_confidence
                FROM detections
                WHERE area_key = %s AND class_code = 3
                GROUP BY scene_date ORDER BY scene_date
                """,
                [area_key],
            ).fetchall()
        keys = ["scene_date", "patch_count", "total_area_m2", "total_area_ha", "mean_confidence"]
        return [dict(zip(keys, row)) for row in rows]


# --- GDAL helpers ------------------------------------------------------------
def _empty_features(geometry_column: str, geometry_type: str) -> pd.DataFrame:
    """An empty frame with the right dtypes, so GDAL can infer a layer schema."""
    frame = pd.DataFrame(
        {
            "key": pd.Series(dtype="object"),
            "name": pd.Series(dtype="object"),
            "role": pd.Series(dtype="object"),
            "crs": pd.Series(dtype="object"),
            "area_key": pd.Series(dtype="object"),
            "scene_date": pd.Series(dtype="object"),
            "class_code": pd.Series(dtype="int32"),
            "class_name": pd.Series(dtype="object"),
            "confidence": pd.Series(dtype="float64"),
            "area_m2": pd.Series(dtype="float64"),
            "model_version": pd.Series(dtype="object"),
            geometry_column: pd.Series(dtype="object"),
        }
    )
    return frame


def _write_layer(path: Path, layer: str, frame: pd.DataFrame, *, mode: str = "w") -> None:
    """Write or append a layer to a GeoPackage."""
    import pyogrio

    # Only keep the columns this layer actually uses; GDAL writes whatever it is
    # given, and a sparse layer with mostly-null columns is painful to query.
    if layer == "study_areas":
        frame = frame[["key", "name", "role", "crs", "footprint"]].copy()
        geometry_column = "footprint"
    else:
        frame = frame[
            [
                "area_key",
                "scene_date",
                "class_code",
                "class_name",
                "confidence",
                "area_m2",
                "model_version",
                "geometry",
            ]
        ].copy()
        geometry_column = "geometry"

    frame = frame.rename(columns={geometry_column: "geometry"})
    pyogrio.write_dataframe(
        frame,
        path,
        layer=layer,
        driver="GPKG",
        append=(mode == "a"),
        geometry_type="Polygon" if layer == "detections" else None,
    )


def _layer_has_features(path: Path, layer: str) -> bool:
    try:
        import pyogrio

        info = pyogrio.read_info(path, layer=layer)
        return int(info.get("features", 0)) > 0
    except Exception:  # pragma: no cover - layer may not exist yet
        return False


def _read_query(path: Path, sql: str) -> list[dict]:
    """Run SQL against the GeoPackage and return dicts (geometry as GeoJSON)."""
    import json
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description]
        results = []
        for row in cursor.fetchall():
            record = {name: row[name] for name in columns}
            blob = record.get("geom") or record.get("geometry")
            if isinstance(blob, bytes):
                try:
                    from osgeo import ogr  # type: ignore

                    geometry = ogr.CreateGeometryFromWkb(blob)
                    record["geometry"] = json.loads(geometry.ExportToJson())
                except ImportError:
                    # Without the osgeo bindings we cannot decode the blob; return
                    # attributes only rather than failing the whole query.
                    record["geometry"] = None
            results.append(record)
        return results


def get_repository(backend: str | None = None, **kwargs) -> SpatialRepository:
    """Construct a repository. Defaults to the GeoPackage backend."""
    key = (backend or "geopackage").strip().lower()
    if key in {"geopackage", "gpkg", "sqlite"}:
        return GeoPackageRepository(**kwargs)
    if key in {"postgis", "postgres", "postgresql"}:
        return PostgisRepository(**kwargs)
    raise AtarraError(f"unknown storage backend {backend!r}; expected 'geopackage' or 'postgis'")
