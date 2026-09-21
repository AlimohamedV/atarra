-- ATARRA spatial schema (PostgreSQL + PostGIS).
--
-- This is the target schema. The working implementation in
-- `repository.py` writes the same entities to a GeoPackage via GDAL, because the
-- development machine has no PostGIS and the data volumes involved (a few hundred
-- detections per AOI) do not justify standing one up. Both backends speak the same
-- repository interface, so migrating is a swap rather than a rewrite.
--
-- Design notes that matter:
--
--  * Geometries are stored in EPSG:4326 so the API can hand them straight to a web
--    map, but every geometric *computation* (area, length, intersection) casts to
--    a projected CRS. Computing area in degrees is a classic silent error: the
--    number looks plausible and is meaningless.
--  * `detections` keeps `scene_date` separate from `created_at`. One is when the
--    satellite looked; the other is when we processed it. Conflating them makes a
--    reprocessing run look like a change on the ground.
--  * Threat scores are stored per detection rather than recomputed on read, so a
--    score can be audited against the inputs that produced it.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS study_areas (
    key             TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('wetland', 'canal')),
    crs             TEXT NOT NULL,
    footprint       geometry(Polygon, 4326) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS study_areas_footprint_idx
    ON study_areas USING GIST (footprint);

-- ---------------------------------------------------------------------------
-- Imagery provenance
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scenes (
    id                  TEXT PRIMARY KEY,
    source              TEXT NOT NULL,
    platform            TEXT,
    grid_code           TEXT,
    acquired            TIMESTAMPTZ NOT NULL,
    cloud_cover         REAL,
    footprint           geometry(Polygon, 4326) NOT NULL,
    reflectance_mode    TEXT NOT NULL,
    reflectance_scale   REAL NOT NULL,
    reflectance_offset  REAL NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS scenes_acquired_idx ON scenes (acquired DESC);
CREATE INDEX IF NOT EXISTS scenes_footprint_idx ON scenes USING GIST (footprint);

-- A composite is one AOI + date window, expressed as the set of scenes that made it.
CREATE TABLE IF NOT EXISTS composites (
    id               BIGSERIAL PRIMARY KEY,
    area_key         TEXT NOT NULL REFERENCES study_areas (key) ON DELETE CASCADE,
    target_date      DATE NOT NULL,
    window_days      INTEGER NOT NULL,
    gsd              REAL NOT NULL,
    crs              TEXT NOT NULL,
    coverage         REAL NOT NULL,
    negative_refl_fraction REAL NOT NULL,
    scene_ids        TEXT[] NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (area_key, target_date, window_days, gsd)
);

-- ---------------------------------------------------------------------------
-- Model output
-- ---------------------------------------------------------------------------
-- Class codes match the segmentation contract in `atarra.models.segmentation`.
CREATE TABLE IF NOT EXISTS detections (
    id              BIGSERIAL PRIMARY KEY,
    composite_id    BIGINT REFERENCES composites (id) ON DELETE CASCADE,
    area_key        TEXT NOT NULL REFERENCES study_areas (key) ON DELETE CASCADE,
    scene_date      DATE NOT NULL,
    class_code      SMALLINT NOT NULL CHECK (class_code BETWEEN 0 AND 3),
    class_name      TEXT NOT NULL,
    confidence      REAL CHECK (confidence BETWEEN 0 AND 1),
    geometry        geometry(Polygon, 4326) NOT NULL,
    -- Area is denormalised in metres^2, computed in a projected CRS at write time.
    area_m2         DOUBLE PRECISION NOT NULL,
    perimeter_m     DOUBLE PRECISION,
    model_version   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS detections_geometry_idx ON detections USING GIST (geometry);
CREATE INDEX IF NOT EXISTS detections_area_date_idx ON detections (area_key, scene_date DESC);
CREATE INDEX IF NOT EXISTS detections_class_idx ON detections (class_code);

-- ---------------------------------------------------------------------------
-- Decision support
-- ---------------------------------------------------------------------------
-- Blockage threat per detection, 0-100, with its inputs preserved for audit.
CREATE TABLE IF NOT EXISTS threat_scores (
    detection_id         BIGINT PRIMARY KEY REFERENCES detections (id) ON DELETE CASCADE,
    score                REAL NOT NULL CHECK (score BETWEEN 0 AND 100),
    coverage_component   REAL NOT NULL,
    width_component      REAL NOT NULL,
    priority_component   REAL NOT NULL,
    computed_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Modelled seasonal biomass, used to place the harvest window.
CREATE TABLE IF NOT EXISTS biomass_observations (
    id           BIGSERIAL PRIMARY KEY,
    area_key     TEXT NOT NULL REFERENCES study_areas (key) ON DELETE CASCADE,
    observed_on  DATE NOT NULL,
    -- Index integral used as the biomass proxy. Not a mass measurement; see
    -- `atarra.forecast.growth` for the caveat.
    proxy_value  DOUBLE PRECISION NOT NULL,
    proxy_kind   TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (area_key, observed_on, proxy_kind)
);

-- Harvest windows and the dispatch payloads generated from them.
CREATE TABLE IF NOT EXISTS harvest_windows (
    id            BIGSERIAL PRIMARY KEY,
    area_key      TEXT NOT NULL REFERENCES study_areas (key) ON DELETE CASCADE,
    season_year   INTEGER NOT NULL,
    window_start  DATE NOT NULL,
    window_end    DATE NOT NULL,
    peak_date     DATE NOT NULL,
    rationale     TEXT,
    UNIQUE (area_key, season_year)
);

CREATE TABLE IF NOT EXISTS dispatches (
    id                BIGSERIAL PRIMARY KEY,
    harvest_window_id BIGINT REFERENCES harvest_windows (id) ON DELETE SET NULL,
    area_key          TEXT NOT NULL REFERENCES study_areas (key) ON DELETE CASCADE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    threat_score      REAL NOT NULL,
    estimated_tonnes  REAL,
    geometry          geometry(Polygon, 4326) NOT NULL,
    payload           JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS dispatches_geometry_idx ON dispatches USING GIST (geometry);

-- ---------------------------------------------------------------------------
-- Convenience view: infested area per AOI per date, in hectares.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_phragmites_coverage AS
SELECT
    d.area_key,
    d.scene_date,
    COUNT(*)                                        AS patch_count,
    SUM(d.area_m2)                                  AS total_area_m2,
    SUM(d.area_m2) / 10000.0                        AS total_area_ha,
    AVG(d.confidence)                               AS mean_confidence
FROM detections d
WHERE d.class_code = 3
GROUP BY d.area_key, d.scene_date
ORDER BY d.area_key, d.scene_date DESC;
