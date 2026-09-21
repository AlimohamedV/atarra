"""The imagery-source contract shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Protocol, runtime_checkable

from atarra.core.errors import ImageryError
from atarra.core.grids import BBox


@dataclass(frozen=True)
class Scene:
    """One satellite acquisition, normalised across backends.

    ``assets`` is keyed by ATARRA's *internal* band names (``B02``, ``B08``, ...)
    rather than whatever the provider happens to call them, so downstream code
    never learns which archive the pixels came from. That is the whole point of
    this abstraction.
    """

    id: str
    acquired: datetime
    platform: str
    cloud_cover: float | None
    bbox: BBox
    assets: dict[str, str]
    epsg: int | None = None
    grid_code: str | None = None
    scale: float = 0.0001
    offset: float = -0.1
    # True when bands can be read with HTTP byte ranges (COG). False means the
    # backend can only hand over whole products, so a download is unavoidable.
    remote_readable: bool = True
    source: str = "stac"
    extra: dict = field(default_factory=dict)

    @property
    def date(self):
        return self.acquired.date()

    @property
    def month_key(self) -> str:
        return self.acquired.strftime("%Y-%m")

    def asset(self, band: str) -> str:
        """Return the href for an internal band name."""
        try:
            return self.assets[band]
        except KeyError as exc:
            raise ImageryError(
                f"scene {self.id} has no asset for band {band!r}; "
                f"available: {sorted(self.assets)}"
            ) from exc

    def has_bands(self, bands: Iterable[str]) -> bool:
        return all(b in self.assets for b in bands)

    def describe(self) -> str:
        cloud = "n/a" if self.cloud_cover is None else f"{self.cloud_cover:.2f}%"
        return (
            f"{self.id} | {self.acquired:%Y-%m-%d} | {self.platform} | "
            f"cloud {cloud} | tile {self.grid_code} | {len(self.assets)} bands"
        )


@runtime_checkable
class ImagerySource(Protocol):
    """What every imagery backend must provide."""

    name: str

    def search(
        self,
        aoi: BBox,
        start: datetime | str,
        end: datetime | str,
        *,
        max_cloud_cover: float | None = None,
        limit: int = 100,
    ) -> list[Scene]:
        """Return scenes intersecting ``aoi`` within ``[start, end]``."""
        ...


def select_best_per_period(
    scenes: Iterable[Scene],
    *,
    period: str = "month",
    require_bands: Iterable[str] | None = None,
) -> list[Scene]:
    """Pick the least-cloudy usable scene in each period.

    A time series wants even temporal spacing far more than it wants every
    acquisition: two scenes three days apart add almost no phenological
    information, while a missing month leaves a hole in the growth curve. So we
    collapse each period to its clearest scene.

    ``require_bands`` drops scenes missing any band we actually need, which keeps
    a scene that is clear but incomplete from being selected and then failing
    deep inside the reader.
    """
    if period not in {"month", "day"}:
        raise ImageryError(f"unsupported period {period!r}; expected 'month' or 'day'")

    required = list(require_bands or [])
    buckets: dict[str, Scene] = {}

    for scene in scenes:
        if required and not scene.has_bands(required):
            continue
        key = scene.month_key if period == "month" else scene.acquired.strftime("%Y-%m-%d")
        incumbent = buckets.get(key)
        if incumbent is None:
            buckets[key] = scene
            continue
        # None cloud cover sorts last: an unquantified scene must not displace a
        # measured clear one.
        new_cloud = scene.cloud_cover if scene.cloud_cover is not None else float("inf")
        old_cloud = incumbent.cloud_cover if incumbent.cloud_cover is not None else float("inf")
        if new_cloud < old_cloud:
            buckets[key] = scene

    return [buckets[k] for k in sorted(buckets)]
