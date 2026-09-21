"""ATARRA — automated tracking and remediation planning for aquatic invasive weeds.

Layered architecture, each layer independently testable:

    ingest/      satellite imagery discovery (STAC primary, CDSE adapter)
    preprocess/  windowed raster reads, reflectance conversion, spectral indices, tiling
    datasets/    weak-supervision labelling and tile loading
    models/      multispectral semantic segmentation
    train/       training loops and evaluation metrics
    forecast/    patch growth modelling and spread projection
    decision/    threat scoring and harvest-window optimisation
    db/          PostGIS-shaped spatial persistence
    api/         FastAPI service
    viz/         server-side rendering of index composites and masks
"""

__version__ = "0.1.0"
