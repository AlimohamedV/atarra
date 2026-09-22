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
from atarra.datasets.weak_labels import CLASS_NAMES
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


def _evenly_spaced(items: list, count: int) -> list:
    """Pick ``count`` entries spread across ``items``.

    Spread rather than "the N most recent" on purpose: a model trained only on
    late-summer imagery has seen the single season in which reed is easiest to
    separate from cropland, and would be weakest exactly where it will be deployed.
    """
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def cmd_dataset_build(args: argparse.Namespace) -> int:
    """Fetch imagery, tile it, and write a store that training can reuse."""
    from atarra.datasets.store import build_store

    end = args.end or Date.today()
    start = args.start or (end - timedelta(days=30 * max(1, args.months)))
    found = available_dates(args.area, start, end, max_cloud_cover=args.max_cloud)
    if not found:
        print(f"error: no usable imagery for {args.area} between {start} and {end}", file=sys.stderr)
        return 1

    available = [Date.fromisoformat(item["date"]) for item in found]
    selected = _evenly_spaced(available, args.dates)

    print(f"{args.area}: {len(available)} date(s) available, building {len(selected)} shard(s)")
    print(f"  range   {start} -> {end}")
    print(f"  gsd     {args.gsd:g} m   tile {args.tile_size}px   stride {args.stride or args.tile_size}")
    print(f"  out     {args.out}")
    print(f"  dates   {', '.join(d.isoformat() for d in selected)}")
    print("\nThis performs real ranged reads of the archive and takes minutes.\n")

    manifest = build_store(
        Path(args.out),
        args.area,
        selected,
        gsd=args.gsd,
        tile_size=args.tile_size,
        stride=args.stride,
        max_size=args.max_size,
        window_days=args.window_days,
        drop_empty=args.drop_empty,
        max_cloud_cover=args.max_cloud,
        first=args.rebuild,
    )

    totals = manifest["totals"]
    print(f"\nwrote {totals['tiles']} tiles across {totals['shards']} shard(s) to {args.out}")
    print(f"  trainable pixels  {totals['trainable_px']:,}")
    print("  class balance (pixels):")
    for code, name in enumerate(CLASS_NAMES):
        count = totals["class_counts"][code]
        share = count / max(1, totals["trainable_px"]) * 100
        print(f"    {name:<24} {count:>12,}  {share:5.1f}%")
        if count == 0:
            print(f"      WARNING: no {name} pixels; that class cannot be learned")
    for entry in manifest["skipped"]:
        print(f"  skipped {entry['date']}: {entry['reason']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a segmentation model from a tile store."""
    from atarra.train.run import format_report, train_from_store

    bands = get_bands()
    selected = {
        "8": bands.bands_8,
        "10": bands.bands_10,
        "rgb": bands.bands_rgb,
    }[args.bands]

    metrics = train_from_store(
        args.store,
        bands=selected,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        out_dir=args.out,
        run_name=args.name,
        seed=args.seed,
        num_workers=args.workers,
        device=args.device,
        patience=args.patience,
        max_tiles=args.max_tiles,
        exclude_pack=args.exclude_pack,
    )
    print("\n" + format_report(metrics))
    return 0


def cmd_annotation_export(args: argparse.Namespace) -> int:
    """Export the highest-need tiles as georeferenced chips for labelling."""
    from atarra.datasets.export import export_annotation_pack

    pack = export_annotation_pack(
        args.store,
        args.out,
        limit=args.limit,
        min_score=args.min_score,
        overwrite=args.overwrite,
        strategy=args.strategy,
        seed=args.seed,
    )

    print(f"annotation pack written to {args.out}")
    print(f"  area           {pack['area']}")
    print(f"  tiles reserved {len(pack['tiles'])}  (of {pack['total_candidates_available']} "
          f"candidates under strategy {pack['strategy']!r})")
    print(f"  gsd            {pack['gsd']:g} m")
    print(f"  bands          {', '.join(pack['imagery_bands'])}")
    print(f"  reed pixels    {pack['reed_px_total']:,} across the pack")
    if not pack["reed_px_sufficient"]:
        print("                 WARNING: too few to measure a reed IoU -- the per-class")
        print("                 score would not mean anything. Raise --limit, or select")
        print("                 with --strategy reed.")
    print()
    print("  tiles, best first under the chosen strategy:")
    for tile in pack["tiles"][:5]:
        print(f"    {tile['rank']:>2}. {tile['key']:<16} {tile['date']}  "
              f"score {tile['score']:>7.4f}  review {tile['review_fraction'] * 100:5.1f}%  "
              f"reed {tile['reed_px']:>5} px")
    if len(pack["tiles"]) > 5:
        print(f"    ... {len(pack['tiles']) - 5} more, see annotation.geojson")

    print()
    print("next steps")
    print("  1. open the chips in QGIS and draw polygons; save to labels/<key>.geojson")
    print("     with an integer class_code field (codes are in the pack README)")
    print(f"  2. retrain WITHOUT these tiles:  atarra train {args.store} --exclude-pack {args.out}")
    print(f"  3. score against your labels:    atarra annotation score {args.out} "
          "--checkpoint <run>/best.pt")
    return 0


def cmd_annotation_score(args: argparse.Namespace) -> int:
    """Score a trained model against the human labels in a pack."""
    from atarra.datasets.export import score_annotation_pack

    result = score_annotation_pack(
        args.pack, args.checkpoint, device=args.device, write=not args.no_write
    )
    report = result["report"]

    print(f"scored {result['tiles_annotated']} annotated tile(s) of "
          f"{result['tiles_reserved']} reserved")
    print(f"  checkpoint     {result['checkpoint']}")
    print(f"  bands          {result['bands']}")
    print()
    print("independent report (against human labels, not the rule engine)")
    print(f"  mean IoU       {report['mean_iou']}")
    print(f"  reed IoU       {report['phragmites_iou']}")
    print(f"  reed F1        {report['phragmites_f1']}")
    print(f"  pixel acc.     {report['pixel_accuracy']}")
    print()
    print("per class (IoU / F1 / support px)")
    for entry in report["per_class"]:
        print(f"  {entry['class_name']:<24} {entry['iou']} / {entry['f1']} / "
              f"{entry['support_px']}")

    if result["per_tile"]:
        print()
        print("per tile")
        for entry in result["per_tile"]:
            print(f"  {entry['key']:<18} agreement {entry['agreement']:.3f}  "
                  f"labelled {entry['labelled_px']:,} px")

    assessment = result["assessment"]
    print()
    if result["assessable"]:
        targets = result["targets"]
        print(f"proposal targets (reed IoU >= {targets['target_iou']}, "
              f"F1 >= {targets['target_f1']}): "
              f"{'met' if targets['both_met'] else 'not met'}")
        print()
        print(assessment["verdict"])
    else:
        # Deliberately does not print a target verdict. A single-class annotation can
        # produce a perfect IoU from a model that predicts one class everywhere, and
        # reporting that as "met" is the most misleading thing this tool could do.
        print("targets        NOT ASSESSED -- this score is not a validation")
        print()
        print(f"  classes present  {', '.join(assessment['classes_present']) or 'none'}")
        print(f"  reed annotated   {assessment['reed_support_px']:,} px "
              f"(needs {assessment['min_reed_pixels']:,})")
        print(f"  model predicted  {assessment['predicted_classes']}")
        print()
        print(assessment["verdict"])

    if result["tiles_still_unlabelled"]:
        print()
        print(f"  {len(result['tiles_still_unlabelled'])} reserved tile(s) still have no "
              "labels; the score above covers the rest")
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

    p = sub.add_parser("dataset", help="build a reusable tile store for training")
    dataset_sub = p.add_subparsers(dest="dataset_command", required=True)
    q = dataset_sub.add_parser("build", help="fetch imagery, tile it, and write a store")
    q.add_argument("area")
    q.add_argument("--out", default="data/dataset", help="store directory")
    q.add_argument("--dates", type=int, default=12, help="how many shards to build (default 12)")
    q.add_argument("--months", type=int, default=24, help="how far back to search (default 24)")
    q.add_argument("--start", type=_parse_date, default=None)
    q.add_argument("--end", type=_parse_date, default=None)
    q.add_argument("--max-cloud", type=float, default=None)
    q.add_argument(
        "--gsd",
        type=float,
        default=20.0,
        help="resolution in metres; 20 keeps ~12 dates inside Colab's free RAM budget",
    )
    q.add_argument("--tile-size", type=int, default=256)
    q.add_argument(
        "--stride",
        type=int,
        default=None,
        help="defaults to --tile-size; a smaller value overlaps tiles for more samples",
    )
    q.add_argument("--max-size", type=int, default=8192, help="largest AOI dimension in px")
    q.add_argument("--window-days", type=int, default=3)
    q.add_argument("--drop-empty", action="store_true", help="skip tiles without reed")
    q.add_argument("--rebuild", action="store_true", help="replace an existing store")
    q.set_defaults(func=cmd_dataset_build)

    p = sub.add_parser("annotation", help="export tiles for labelling, and score against them")
    annotation_sub = p.add_subparsers(dest="annotation_command", required=True)
    q = annotation_sub.add_parser("export", help="write georeferenced chips + a GeoJSON index")
    q.add_argument("store", help="store directory built by `atarra dataset build`")
    q.add_argument("--out", default="data/annotation", help="pack directory")
    q.add_argument("--limit", type=int, default=20, help="how many tiles to reserve (default 20)")
    q.add_argument(
        "--strategy",
        choices=["uncertainty", "reed", "random"],
        default="uncertainty",
        help="uncertainty = hardest ground (default), reed = tiles richest in the "
        "target class, random = a representative sample",
    )
    q.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="only export tiles at or above this score; units follow --strategy "
        "(a review fraction, or a reed pixel count)",
    )
    q.add_argument("--seed", type=int, default=0, help="seed for --strategy random")
    q.add_argument("--overwrite", action="store_true", help="replace an existing pack")
    q.set_defaults(func=cmd_annotation_export)

    q = annotation_sub.add_parser("score", help="score a checkpoint against human labels")
    q.add_argument("pack", help="pack directory containing labels/")
    q.add_argument("--checkpoint", required=True, help="path to best.pt from a training run")
    q.add_argument("--device", default=None)
    q.add_argument("--no-write", action="store_true", help="do not write score.json")
    q.set_defaults(func=cmd_annotation_score)

    p = sub.add_parser("train", help="train a segmentation model from a tile store")
    p.add_argument("store", help="store directory built by `atarra dataset build`")
    p.add_argument(
        "--exclude-pack",
        default=None,
        help="annotation pack whose reserved tiles must not be trained on",
    )
    p.add_argument(
        "--bands",
        choices=["8", "10", "rgb"],
        default="8",
        help="8 = multispectral (default), 10 = all, rgb = the 3-band control arm",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="data/checkpoints", help="run output directory")
    p.add_argument("--name", default=None, help="run name (default: derived from area and bands)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu, cuda, or unset for automatic")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-tiles", type=int, default=None, help="truncate for a smoke run")
    p.set_defaults(func=cmd_train)

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
