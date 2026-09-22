"""Weak-supervision labelling, tile loading, and the on-disk tile store."""

from atarra.datasets.weak_labels import (
    CLASS_NAMES,
    NUM_CLASSES,
    PHRAGMITES_CODE,
    WeakLabelConfig,
    label_statistics,
    weak_label,
)
from atarra.datasets.tile_dataset import (
    CompositeTileDataset,
    augment_tile,
    class_weights_from_counts,
    geometric_split,
)
from atarra.datasets.store import (
    TileStoreDataset,
    build_store,
    decode_reflectance,
    encode_reflectance,
    load_manifest,
)
from atarra.datasets.export import (
    export_annotation_pack,
    load_annotations,
    load_pack,
    reserved_keys,
    score_annotation_pack,
)

__all__ = [
    "CLASS_NAMES",
    "NUM_CLASSES",
    "PHRAGMITES_CODE",
    "CompositeTileDataset",
    "TileStoreDataset",
    "WeakLabelConfig",
    "augment_tile",
    "build_store",
    "class_weights_from_counts",
    "decode_reflectance",
    "encode_reflectance",
    "export_annotation_pack",
    "geometric_split",
    "label_statistics",
    "load_annotations",
    "load_manifest",
    "load_pack",
    "reserved_keys",
    "score_annotation_pack",
    "weak_label",
]
