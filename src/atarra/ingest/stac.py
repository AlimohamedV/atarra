"""STAC imagery discovery (Earth Search / AWS public Sentinel-2 COGs).

Verified live against ``https://earth-search.aws.element84.com/v1``: the
``sentinel-2-l2a`` collection covers the Nile Delta continuously from 2022 through
2026, assets are Cloud-Optimized GeoTIFFs on a public S3 bucket, and band files
support HTTP byte-range reads. No account, no API key, no cost.

Why not ``sentinelsat``? It targets the legacy Copernicus Open Access Hub, needs
credentials, and only yields whole-product downloads. This project needs to read
a 512x512 window out of a 68 MB band file, which is exactly what a public COG
gives us and what a zipped SAFE product does not. ESA provenance remains available
through :mod:`atarra.ingest.cdse`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator

import pystac_client

from atarra.core.config import BandsConfig, get_bands
from atarra.core.errors import ImageryError
from atarra.core.grids import BBox
from atarra.core.logging import get_logger
from atarra.core.settings import Settings, get_settings
from atarra.ingest.base import Scene

log = get_logger("ingest.stac")

# Earth Search paginates; asking for more than this per page invites a 400.
MAX_PAGE_SIZE = 250


def _isoformat(value: datetime | str) -> str:
    """Render a datetime as the RFC 3339 UTC string STAC expects."""
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        # Naive inputs are assumed UTC; the alternative (local time) would make
        # searches silently timezone-dependent.
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return value.astimezone(tz=None).strftime("%Y-%m-%dT%H:%M:%SZ")


class StacImagerySource:
    """Discover Sentinel-2 L2A scenes through a STAC API."""

    name = "stac"

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        *,
        settings: Settings | None = None,
        bands: BandsConfig | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bands = bands or get_bands()
        self.url = url or self.settings.stac_url
        self.collection = collection or self.settings.stac_collection
        self._client: pystac_client.Client | None = None

    # --- connection ----------------------------------------------------------
    @property
    def client(self) -> pystac_client.Client:
        if self._client is None:
            try:
                self._client = pystac_client.Client.open(
                    self.url, timeout=self.settings.stac_timeout_s
                )
            except Exception as exc:  # network, DNS, malformed catalogue
                raise ImageryError(f"could not open STAC API at {self.url}: {exc}") from exc
        return self._client

    def collections(self) -> list[str]:
        """List collection ids, for diagnostics."""
        try:
            return [c.id for c in self.client.get_collections()]
        except Exception as exc:
            raise ImageryError(f"could not list collections at {self.url}: {exc}") from exc

    # --- search --------------------------------------------------------------
    def search(
        self,
        aoi: BBox,
        start: datetime | str,
        end: datetime | str,
        *,
        max_cloud_cover: float | None = None,
        limit: int = 100,
        collection: str | None = None,
    ) -> list[Scene]:
        """Return scenes intersecting ``aoi`` within ``[start, end]``."""
        cloud = self.settings.max_cloud_cover if max_cloud_cover is None else max_cloud_cover
        page_size = max(1, min(int(limit), MAX_PAGE_SIZE))
        interval = f"{_isoformat(start)}/{_isoformat(end)}"

        query: dict[str, Any] = {}
        if cloud is not None:
            # `lt` on eo:cloud_cover is the one filter every Sentinel-2 STAC
            # catalogue agrees on; CQL2 `filter` support is much patchier.
            query["eo:cloud_cover"] = {"lt": float(cloud)}

        log.info(
            "STAC search %s | %s | bbox=%s | cloud<%s | limit=%d",
            collection or self.collection,
            interval,
            [round(v, 3) for v in aoi.as_stac_query()],
            cloud,
            page_size,
        )

        try:
            results = self.client.search(
                collections=[collection or self.collection],
                bbox=aoi.as_stac_query(),
                datetime=interval,
                query=query or None,
                limit=page_size,
                max_items=page_size,
            )
            items = list(results.items())
        except Exception as exc:
            raise ImageryError(f"STAC search failed against {self.url}: {exc}") from exc

        scenes = [s for s in (self._to_scene(item) for item in items) if s is not None]
        log.info("STAC search returned %d items -> %d usable scenes", len(items), len(scenes))
        if len(items) >= page_size:
            # Surfacing this matters: a silently truncated result set looks like
            # "the archive has no more scenes" and quietly shortens a time series.
            log.warning(
                "STAC search hit the %d-item limit, so results are truncated. "
                "Raise `limit` or narrow the date range for complete coverage.",
                page_size,
            )
        return scenes

    def search_iter(
        self,
        aoi: BBox,
        start: datetime | str,
        end: datetime | str,
        *,
        max_cloud_cover: float | None = None,
    ) -> Iterator[Scene]:
        """Stream scenes without materialising the whole result set."""
        for scene in self.search(aoi, start, end, max_cloud_cover=max_cloud_cover, limit=MAX_PAGE_SIZE):
            yield scene

    # --- conversion ----------------------------------------------------------
    def _reflectance(self, item) -> tuple[float, float]:
        """Read the DN -> reflectance scale and offset from item metadata.

        Reading it from the item rather than trusting the config matters: the
        -1000 BOA offset introduced in processing baseline 04.00 is invisible in
        the pixel values, so a hardcoded conversion silently biases every index
        when the archive's convention changes. The config values are only a
        fallback for assets that omit ``raster:bands``.
        """
        for band_name in self.bands.bands_8:
            spec = self.bands.bands[band_name]
            asset = item.assets.get(spec.asset)
            if asset is None:
                continue
            meta = asset.extra_fields.get("raster:bands")
            if isinstance(meta, list) and meta and "scale" in meta[0]:
                return float(meta[0].get("scale", 1.0)), float(meta[0].get("offset", 0.0))

        # Fallback to the configured convention. Recorded on the Scene for
        # traceability; the reader resolves the convention itself (see
        # configs/bands.yaml) because this metadata is not trustworthy.
        return self.bands.reflectance_scale, self.bands.declared_offset

    def _to_scene(self, item) -> Scene | None:
        """Convert a STAC item, or ``None`` if it carries no usable bands."""
        assets: dict[str, str] = {}
        for band_name, spec in self.bands.bands.items():
            asset = item.assets.get(spec.asset)
            if asset is not None:
                assets[band_name] = asset.href

        mask_asset = item.assets.get(self.bands.mask.asset)
        if mask_asset is not None:
            assets[self.bands.mask.name] = mask_asset.href

        if not assets:
            log.debug("skipping item %s: no recognised band assets", item.id)
            return None

        acquired = item.datetime
        if acquired is None:
            log.debug("skipping item %s: no acquisition datetime", item.id)
            return None

        props = item.properties
        bounds = item.bbox or []
        if len(bounds) != 4:
            # Fall back to the item geometry's envelope if bbox is absent.
            try:
                bounds = list(item.geometry["coordinates"][0])
                xs = [c[0] for c in bounds]
                ys = [c[1] for c in bounds]
                bounds = [min(xs), min(ys), max(xs), max(ys)]
            except (KeyError, TypeError, ValueError):
                log.debug("skipping item %s: no usable footprint", item.id)
                return None

        scale, offset = self._reflectance(item)
        # The projection extension renamed proj:epsg to proj:code in v1.1.0; recent
        # items carry only the latter. Read either, and parse the "EPSG:xxxx" form.
        epsg = props.get("proj:epsg")
        if epsg is None:
            code = props.get("proj:code")
            if isinstance(code, str) and code.upper().startswith("EPSG:"):
                try:
                    epsg = int(code.split(":", 1)[1])
                except ValueError:
                    epsg = None

        return Scene(
            id=item.id,
            acquired=acquired,
            platform=str(props.get("platform") or props.get("constellation") or "unknown"),
            cloud_cover=props.get("eo:cloud_cover"),
            bbox=BBox.from_sequence(tuple(bounds), crs="EPSG:4326"),
            assets=assets,
            epsg=epsg,
            grid_code=props.get("grid:code") or props.get("s2:mgrs_tile"),
            scale=scale,
            offset=offset,
            remote_readable=True,
            source="stac",
            extra={
                "collection": item.collection_id,
                "processing_baseline": props.get("processing:version") or props.get("s2:processing_baseline"),
            },
        )
