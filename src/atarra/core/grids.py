"""Raster grid algebra.

Why this module exists
----------------------
Every scene, index layer, model tile, and prediction mask in ATARRA has to agree
on one thing: which pixel is which. Sentinel-2 hands us bands at two different
resolutions (10 m and 20 m) inside scenes whose native grids are only *incidentally*
similar. If tiles from two dates are even one pixel out of phase, a time series
acquires fake change and a segmentation mask lands offset from the imagery it
describes -- both failure modes that produce plausible-looking numbers while
being completely wrong.

So we pin an explicit target grid and derive everything from it.

Alignment note: Sentinel-2 L2A UTM pixel grids originate at absolute multiples of
the 10 m pixel size (a real tile sits at an origin of exactly 300000/3500040 with
10 m pixels, verified live). Snapping our AOI bounds outward to multiples of the
ground sample distance therefore lands on the same lattice the satellite uses,
which is what lets a windowed read hit pixel boundaries exactly. `tests/test_grids.py`
asserts this against a live scene rather than assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from affine import Affine
from pyproj import CRS, Transformer
from rasterio.windows import Window

WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True)
class BBox:
    """An axis-aligned bounding box in a known CRS."""

    west: float
    south: float
    east: float
    north: float
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        if self.west >= self.east or self.south >= self.north:
            raise ValueError(
                f"degenerate bbox: west={self.west} east={self.east} "
                f"south={self.south} north={self.north}"
            )

    @classmethod
    def from_sequence(cls, values: tuple[float, float, float, float], crs: str = "EPSG:4326") -> "BBox":
        """Build from a ``[west, south, east, north]`` sequence (e.g. YAML config)."""
        west, south, east, north = (float(v) for v in values)
        return cls(west=west, south=south, east=east, north=north, crs=crs)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)

    def to_crs(self, dst_crs: str | CRS) -> "BBox":
        """Reproject the box, taking the hull of the four reprojected corners.

        A bbox that is axis-aligned in UTM is not axis-aligned in lon/lat, so
        transforming only the lower-left and upper-right corners under-covers the
        footprint. Transforming all four and taking min/max is the conservative
        choice, which is what we want for a *search* box.
        """
        dst = CRS.from_user_input(dst_crs)
        src = CRS.from_user_input(self.crs)
        if dst == src:
            return self

        transformer = Transformer.from_crs(src, dst, always_xy=True)
        corners = [
            (self.west, self.south),
            (self.west, self.north),
            (self.east, self.south),
            (self.east, self.north),
        ]
        xs, ys = zip(*(transformer.transform(x, y) for x, y in corners))
        return BBox(west=min(xs), south=min(ys), east=max(xs), north=max(ys), crs=dst.to_string())

    def as_stac_query(self) -> list[float]:
        """Return ``[west, south, east, north]`` in WGS84, as STAC requires."""
        box = self.to_crs(WGS84)
        return [box.west, box.south, box.east, box.north]

    def as_geojson(self) -> dict:
        """Return an RFC 7946 Polygon geometry (WGS84 required by the spec)."""
        box = self.to_crs(WGS84)
        ring = [
            [box.west, box.south],
            [box.east, box.south],
            [box.east, box.north],
            [box.west, box.north],
            [box.west, box.south],
        ]
        return {"type": "Polygon", "coordinates": [ring]}


@dataclass(frozen=True)
class TileWindow:
    """One non-overlapping tile cut from a grid."""

    row: int
    col: int
    window: Window
    transform: Affine
    size: int

    @property
    def name(self) -> str:
        return f"r{self.row}_c{self.col}"


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid: CRS, affine transform, and pixel dimensions."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def resolution(self) -> tuple[float, float]:
        """Pixel size as ``(x, y)`` in CRS units."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> BBox:
        left = self.transform.c
        top = self.transform.f
        right = left + self.width * self.transform.a
        bottom = top + self.height * self.transform.e  # e is negative
        return BBox(west=left, south=bottom, east=right, north=top, crs=self.crs.to_string())

    def corners_wgs84(self) -> list[list[float]]:
        """The four grid corners as ``[lon, lat]``, in MapLibre image-source order.

        Corners rather than a bounding box: a rectangle that is axis-aligned in a
        UTM zone becomes a slightly rotated quadrilateral in lon/lat, and using a
        lon/lat bbox would shear the rendered overlay off the imagery it describes.
        MapLibre's ``image`` source accepts an arbitrary quad, so the distortion is
        avoidable at no cost.
        """
        box = self.bounds
        transformer = Transformer.from_crs(self.crs, WGS84, always_xy=True)
        ordered = [
            (box.west, box.north),  # top-left
            (box.east, box.north),  # top-right
            (box.east, box.south),  # bottom-right
            (box.west, box.south),  # bottom-left
        ]
        return [
            [round(x, 7), round(y, 7)]
            for x, y in (transformer.transform(x, y) for x, y in ordered)
        ]

    def pixel_center(self, col: float, row: float) -> tuple[float, float]:
        """Map a fractional pixel position to CRS coordinates."""
        x = self.transform.c + (col + 0.5) * self.transform.a
        y = self.transform.f + (row + 0.5) * self.transform.e
        return (x, y)

    def window_for(self, bbox: BBox) -> Window:
        """Return the pixel window covering ``bbox``, clipped to this grid."""
        box = bbox.to_crs(self.crs)
        inv = ~self.transform
        col0, row0 = inv @ (box.west, box.north)
        col1, row1 = inv @ (box.east, box.south)

        col_off = max(0, math.floor(col0))
        row_off = max(0, math.floor(row0))
        col_end = min(self.width, math.ceil(col1))
        row_end = min(self.height, math.ceil(row1))
        if col_end <= col_off or row_end <= row_off:
            raise ValueError(f"bbox {box} does not intersect grid bounds {self.bounds}")
        return Window(col_off, row_off, col_end - col_off, row_end - row_off)

    def tiles(self, size: int, *, full_only: bool = True) -> Iterator[TileWindow]:
        """Yield tiles row-major.

        ``full_only`` drops ragged edge tiles. Training wants square tiles of a
        single shape so batches stack; inference pads instead (see
        :mod:`atarra.preprocess.tiler`).
        """
        if size <= 0:
            raise ValueError("tile size must be positive")
        for row in range(0, self.height, size):
            for col in range(0, self.width, size):
                w = min(size, self.width - col)
                h = min(size, self.height - row)
                if full_only and (w != size or h != size):
                    continue
                window = Window(col, row, w, h)
                yield TileWindow(
                    row=row // size,
                    col=col // size,
                    window=window,
                    transform=self.transform @ Affine.translation(col, row),
                    size=size,
                )


def snap_bounds(box: BBox, gsd: float) -> BBox:
    """Expand ``box`` outward to land on multiples of ``gsd``.

    Expanding outward (floor the minima, ceil the maxima) guarantees the snapped
    grid fully contains the requested AOI, so no part of the area of interest is
    silently dropped at the edge.
    """
    if gsd <= 0:
        raise ValueError("gsd must be positive")
    return BBox(
        west=math.floor(box.west / gsd) * gsd,
        south=math.floor(box.south / gsd) * gsd,
        east=math.ceil(box.east / gsd) * gsd,
        north=math.ceil(box.north / gsd) * gsd,
        crs=box.crs,
    )


def grid_from_bbox(box: BBox, crs: str | CRS, gsd: float, *, snap: bool = True) -> Grid:
    """Build the canonical target grid for an AOI.

    Snapped by default so that two callers asking for the same AOI at the same
    resolution get bit-identical grids and can therefore share cached tiles.
    """
    dst = CRS.from_user_input(crs)
    projected = box.to_crs(dst)
    if snap:
        projected = snap_bounds(projected, gsd)

    width = int(round((projected.east - projected.west) / gsd))
    height = int(round((projected.north - projected.south) / gsd))
    if width <= 0 or height <= 0:
        raise ValueError(f"AOI {projected} is smaller than one {gsd} m pixel")

    transform = Affine(gsd, 0.0, projected.west, 0.0, -gsd, projected.north)
    return Grid(crs=dst, transform=transform, width=width, height=height)
