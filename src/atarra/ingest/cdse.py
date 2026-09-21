"""ESA Copernicus Data Space Ecosystem (CDSE) imagery source.

This is the second backend behind the same interface, giving the platform direct
ESA provenance rather than an AWS mirror of ESA data.

Honest status: the OAuth flow, the OData catalogue query, and the SAFE band-file
mapping are implemented against the documented CDSE API, but they could not be
executed end-to-end here because that requires a CDSE account, which this project
deliberately does not ship with. Treat it as the fallback path and keep the STAC
source as the default until you have run it once with your own credentials.

The important structural difference from STAC: CDSE hands over whole zipped SAFE
products. There is no byte-range access to a single band, so a scene from this
backend costs a full product download (~800 MB for L2A) before a single pixel can
be read, and its bands must first be unzipped out of the archive. That is why
:attr:`Scene.remote_readable` exists -- the reader checks it and explains the
situation instead of failing obscurely.
"""

from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

import requests

from atarra.core.config import BandsConfig, get_bands
from atarra.core.errors import ImageryError
from atarra.core.grids import BBox
from atarra.core.logging import get_logger
from atarra.core.settings import Settings, get_settings
from atarra.ingest.base import Scene

log = get_logger("ingest.cdse")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

# CDSE nests bands inside the SAFE archive by resolution folder.
_RESOLUTION_FOLDER = {10: "R10m", 20: "R20m", 60: "R60m"}


class CdseImagerySource:
    """Discover and fetch Sentinel-2 L2A products from ESA's CDSE."""

    name = "cdse"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        settings: Settings | None = None,
        bands: BandsConfig | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bands = bands or get_bands()
        self.username = username or self.settings.cdse_username
        self.password = password or self.settings.cdse_password
        self._token: str | None = None

    # --- authentication ------------------------------------------------------
    def token(self) -> str:
        """Obtain (and cache) an OAuth access token."""
        if self._token:
            return self._token
        if not (self.username and self.password):
            raise ImageryError(
                "CDSE credentials are not configured. Set ATARRA_CDSE_USERNAME and "
                "ATARRA_CDSE_PASSWORD (see .env.example), or use the default STAC "
                "source which needs no account."
            )
        try:
            response = requests.post(
                TOKEN_URL,
                data={
                    "client_id": "cdse-public",
                    "grant_type": "password",
                    "username": self.username,
                    "password": self.password,
                },
                timeout=60,
            )
            response.raise_for_status()
            self._token = response.json()["access_token"]
        except (requests.RequestException, KeyError) as exc:
            raise ImageryError(f"CDSE authentication failed: {exc}") from exc
        return self._token

    # --- search --------------------------------------------------------------
    def search(
        self,
        aoi: BBox,
        start: datetime | str,
        end: datetime | str,
        *,
        max_cloud_cover: float | None = None,
        limit: int = 100,
    ) -> list[Scene]:
        """Query the CDSE OData catalogue for Sentinel-2 L2A products."""
        cloud = self.settings.max_cloud_cover if max_cloud_cover is None else max_cloud_cover
        start_iso, end_iso = _isoformat(start), _isoformat(end)
        bbox = aoi.to_crs("EPSG:4326")

        polygon = (
            f"POLYGON(({bbox.west} {bbox.south},{bbox.east} {bbox.south},"
            f"{bbox.east} {bbox.north},{bbox.west} {bbox.north},{bbox.west} {bbox.south}))"
        )
        clauses = [
            "Collection/Name eq 'SENTINEL-2'",
            "contains(Name,'MSIL2A')",  # Level-2A only
            f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')",
            f"ContentDate/Start gt {start_iso}",
            f"ContentDate/Start lt {end_iso}",
        ]
        if cloud is not None:
            clauses.append(
                "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
                f"and att/OData.CSC.DoubleAttribute/Value lt {float(cloud):.2f})"
            )

        params = {
            "$filter": " and ".join(clauses),
            "$orderby": "ContentDate/Start asc",
            "$top": int(min(limit, 1000)),
        }

        log.info("CDSE OData search | %s -> %s | cloud<%s", start_iso, end_iso, cloud)
        try:
            response = requests.get(
                ODATA_URL, params=params, headers={"Authorization": f"Bearer {self.token()}"}, timeout=120
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise ImageryError(f"CDSE catalogue query failed: {exc}") from exc

        scenes = [s for s in (self._to_scene(p) for p in payload.get("value", [])) if s is not None]
        log.info("CDSE search returned %d products", len(scenes))
        return scenes

    def _to_scene(self, product: dict) -> Scene | None:
        product_id = product.get("Id")
        content_date = (product.get("ContentDate") or {}).get("Start")
        footprint = product.get("GeoFootprint") or {}

        if not product_id or not content_date:
            return None
        try:
            acquired = datetime.fromisoformat(content_date.replace("Z", "+00:00"))
        except ValueError:
            return None

        coordinates = (footprint.get("coordinates") or [[]])[0]
        if not coordinates:
            return None
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]

        cloud = None
        for attribute in product.get("Attributes") or []:
            if attribute.get("Name") == "cloudCover":
                cloud = attribute.get("Value")

        return Scene(
            id=str(product_id),
            acquired=acquired,
            platform=str(product.get("Name", ""))[:3] or "unknown",
            cloud_cover=cloud,
            bbox=BBox(west=min(xs), south=min(ys), east=max(xs), north=max(ys), crs="EPSG:4326"),
            # Asset hrefs are only known once the SAFE archive is on disk; see
            # resolve_assets(). Until then the product id is the handle.
            assets={},
            epsg=32636,
            grid_code=_mgrs_from_name(str(product.get("Name", ""))),
            scale=self.bands.reflectance_scale,
            offset=self.bands.declared_offset,
            remote_readable=False,
            source="cdse",
            extra={"name": product.get("Name")},
        )

    # --- product retrieval ---------------------------------------------------
    def download_product(self, scene: Scene, dest_dir: Path, *, chunk_bytes: int = 1 << 20) -> Path:
        """Stream a product archive to disk. Returns the zip path."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f"{scene.id}.zip"
        if target.exists():
            log.info("CDSE product already cached: %s", target.name)
            return target

        url = ZIPPER_URL.format(product_id=scene.id)
        log.info("downloading CDSE product %s (~800 MB)", scene.id)
        tmp = target.with_suffix(".zip.part")
        try:
            with requests.get(
                url, headers={"Authorization": f"Bearer {self.token()}"}, stream=True, timeout=600
            ) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        handle.write(chunk)
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            raise ImageryError(f"CDSE download failed for {scene.id}: {exc}") from exc
        tmp.replace(target)
        return target

    def resolve_assets(self, archive: Path, scene: Scene) -> Scene:
        """Map SAFE band filenames inside an archive to internal band names.

        Returns a new :class:`Scene` whose ``assets`` point at the contained
        ``.jp2`` files, so the rest of the pipeline can treat it like any other
        scene.
        """
        assets: dict[str, str] = {}
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            for band_name, spec in self.bands.bands.items():
                folder = _RESOLUTION_FOLDER.get(int(spec.gsd))
                if folder is None:
                    continue
                match = _find_band(names, f"{band_name}_", folder)
                if match:
                    assets[band_name] = f"{archive}!/{match}"

            mask_folder = _RESOLUTION_FOLDER.get(int(self.bands.mask.gsd))
            mask_match = _find_band(names, f"{self.bands.mask.name}_", mask_folder)
            if mask_match:
                assets[self.bands.mask.name] = f"{archive}!/{mask_match}"

        missing = [b for b in self.bands.bands_8 if b not in assets]
        if missing:
            raise ImageryError(
                f"archive {archive.name} is missing required bands {missing}; "
                "expected an MSIL2A SAFE product"
            )
        return Scene(
            id=scene.id,
            acquired=scene.acquired,
            platform=scene.platform,
            cloud_cover=scene.cloud_cover,
            bbox=scene.bbox,
            assets=assets,
            epsg=scene.epsg,
            grid_code=scene.grid_code,
            scale=scene.scale,
            offset=scene.offset,
            remote_readable=False,
            source="cdse",
            extra=scene.extra,
        )


# --- helpers -----------------------------------------------------------------
def _isoformat(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        return value.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return value.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _mgrs_from_name(product_name: str) -> str | None:
    for token in product_name.split("_"):
        if len(token) == 5 and token[:2].isdigit() and token[2:].isalpha():
            return token
    return None


def _find_band(names: list[str], prefix: str, folder: str | None) -> str | None:
    """Find ``..._B08_10m.jp2``-style member inside the SAFE tree."""
    for name in names:
        if not name.endswith(".jp2"):
            continue
        if folder and f"/{folder}/" not in name:
            continue
        filename = name.rsplit("/", 1)[-1]
        # e.g. T36RUV_20230819T085141_B08_10m.jp2 -> looks for "_B08_"
        if f"_{prefix}" in filename or filename.startswith(prefix):
            return name
    return None
