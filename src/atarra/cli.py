"""Command-line interface.

Uses argparse from the standard library so the CLI has no extra dependency. Every
subcommand is a thin wrapper over :mod:`atarra.pipeline`, which keeps behaviour
identical whether you call it from a shell, the API, or a notebook.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from atarra import __version__
from atarra.core.config import get_bands, get_study_areas
from atarra.core.errors import AtarraError
from atarra.core.logging import get_logger
from atarra.core.settings import get_settings
from atarra.pipeline import DEFAULT_PREVIEW_GSD, available_dates, get_cache, load_composite
from atarra.preprocess.indices import INDEX_BANDS
from atarra.viz.render import apply_colormap, legend, render_true_color, save_png

log = get_logger("cli")


def _parse_date(value: str) -> Date:
    try:
        return Date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from None


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("area", help="study area key (see `atarra areas`)")
    parser.add_argument("--date", type=_parse_date, default=None, help="YYYY-MM-DD of interest")
    parser.add_argument(
        "--window-days",
        type=int,
        default=3,
        help="days either side of --date used to fill tile gaps (default 3)",
    )
    parser.add_argument(
        "--gsd", type=float, default=DEFAULT_PREVIEW_GSD, help="target resolution in metres"
    )


# --- commands ----------------------------------------------------------------
def cmd_areas(args: argparse.Namespace) -> int:
    areas = get_study_areas()
    print(f"{len(areas)} study area(s):\n")
    for area in areas.values():
        box = area.bbox.as_stac_query()
        print(f"  {area.key}")
        print(f"      {area.name}")
        print(f"      role={area.role}  crs={area.crs}")
        print(f"      bbox (WGS84) = {[round(v, 3) for v in box]}")
    return 0


def cmd_scenes(args: argparse.Namespace) -> int:
    end = args.end or Date.today()
    start = args.start or (end - timedelta(days=30 * max(1, args.months)))
    items = available_dates(args.area, start, end, max_cloud_cover=args.max_cloud)

    print(f"{args.area}: {len(items)} date(s) with usable imagery ({start} -> {end})\n")
    for item in items:
        cloud = item["best_cloud_cover"]
        cloud_text = "  n/a" if cloud is None else f"{cloud:5.2f}%"
        print(
            f"  {item['date']}  scenes={item['scenes']:2d}  cloud={cloud_text}  "
            f"tiles={','.join(item['tiles'])}"
        )
    return 0


def cmd_indices(args: argparse.Namespace) -> int:
    target = args.date or Date(2023, 8, 26)
    composite = load_composite(
        args.area, target, window_days=args.window_days, gsd=args.gsd, max_size=args.max_size
    )

    print(f"{composite.area_name}")
    print(f"  date          {target}  (+/- {args.window_days} days)")
    print(f"  grid          {composite.grid.width} x {composite.grid.height} px @ "
          f"{composite.grid.resolution[0]:g} m, {composite.grid.crs}")
    print(f"  coverage      {composite.coverage * 100:.1f}%")
    print(f"  scenes        {len(composite.scene_ids)}")
    print(f"  reflectance   mode={composite.reflectance.get('mode', 'dn_scale')} "
          f"negative={composite.reflectance['negative_fraction'] * 100:.2f}%  "
          f"median={composite.reflectance['median']:+.4f}")

    print("\n  index    p10      median   p90      min      max")
    for name, values in sorted(composite.indices.items()):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            print(f"  {name:7s} (no valid pixels)")
            continue
        print(
            f"  {name:7s} {np.percentile(finite, 10):+.3f}  {np.median(finite):+.3f}   "
            f"{np.percentile(finite, 90):+.3f}   {finite.min():+.3f}  {finite.max():+.3f}"
        )

    for warning in composite.warnings:
        print(f"\n  WARNING: {warning}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    target = args.date or Date(2023, 8, 26)
    out_dir = Path(args.out) if args.out else (get_settings().previews_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.index == "truecolor":
        composite = load_composite(
            args.area, target, window_days=args.window_days, gsd=args.gsd,
            max_size=args.max_size, bands=["B02", "B03", "B04", "B08"], indices=["ndvi"],
        )
        rgba = render_true_color(composite.stack)
        stem = f"{args.area}_{target}_truecolor"
    else:
        key = args.index.lower()
        if key not in INDEX_BANDS:
            print(f"unknown index {args.index!r}; known: {sorted(INDEX_BANDS)} or 'truecolor'")
            return 2
        composite = load_composite(
            args.area, target, window_days=args.window_days, gsd=args.gsd,
            max_size=args.max_size, indices=[key],
        )
        rgba = apply_colormap(composite.index(key), key)
        stem = f"{args.area}_{target}_{key}"

    path = out_dir / f"{stem}.png"
    save_png(path, rgba)
    scale = legend(args.index.lower()) if args.index.lower() in INDEX_BANDS else None
    print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} kB)")
    print(f"  grid     {composite.grid.width} x {composite.grid.height} px")
    print(f"  coverage {composite.coverage * 100:.1f}%")
    print(f"  corners  {composite.grid.corners_wgs84()}")
    if scale:
        print(f"  scale    {scale['vmin']} .. {scale['vmax']}")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    cache = get_cache()
    if args.clear:
        before = cache.size_bytes()
        cache.clear()
        print(f"cleared {before / 1e6:.1f} MB from the composite cache")
    stats = cache.stats()
    print(f"cache root   {stats['root']}")
    print(f"entries      {stats['entries']}")
    print(f"size         {stats['megabytes']} MB of {stats['max_bytes'] / 1e6:.0f} MB ceiling")
    if stats["usage_fraction"] is not None:
        print(f"utilisation  {stats['usage_fraction'] * 100:.1f}%")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    bands = get_bands()
    settings = get_settings()
    print("ATARRA configuration report")
    print(f"  version            {__version__}")
    print(f"  imagery source     {settings.stac_url}")
    print(f"  collection         {settings.stac_collection}")
    print(f"  max cloud cover    {settings.max_cloud_cover}%")
    print(f"  data dir           {settings.data_dir}")
    print(f"  cache ceiling      {settings.max_cache_gb} GB")
    print(f"  tile size          {settings.tile_size} px @ {settings.target_gsd} m")
    print(f"  reflectance mode   {bands.reflectance_mode} "
          f"(scale {bands.reflectance_scale}, offset {bands.reflectance_offset})")
    print(f"  8-band input       {', '.join(bands.bands_8)}")
    print(f"  RGB baseline       {', '.join(bands.bands_rgb)}")
    print(f"  indices            {', '.join(sorted(INDEX_BANDS))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atarra",
        description="ATARRA - aquatic invasive weed tracking from satellite telemetry",
    )
    parser.add_argument("--version", action="version", version=f"atarra {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("areas", help="list configured study areas")
    p.set_defaults(func=cmd_areas)

    p = sub.add_parser("scenes", help="show which dates have usable imagery")
    p.add_argument("area")
    p.add_argument("--months", type=int, default=3, help="how far back to look (default 3)")
    p.add_argument("--start", type=_parse_date, default=None)
    p.add_argument("--end", type=_parse_date, default=None)
    p.add_argument("--max-cloud", type=float, default=None)
    p.set_defaults(func=cmd_scenes)

    p = sub.add_parser("indices", help="compute and report spectral index statistics")
    _add_common(p)
    p.add_argument("--max-size", type=int, default=1024)
    p.set_defaults(func=cmd_indices)

    p = sub.add_parser("preview", help="render an index composite to a PNG")
    _add_common(p)
    p.add_argument("--index", default="ndvi", help=f"{', '.join(sorted(INDEX_BANDS))} or truecolor")
    p.add_argument("--max-size", type=int, default=1024)
    p.add_argument("--out", default=None, help="output directory")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("cache", help="report or clear the composite cache")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("report", help="print the effective configuration")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AtarraError as exc:
        # Expected, actionable failures: report cleanly rather than as a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
