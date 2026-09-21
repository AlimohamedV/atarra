# ATARRA

**Automated Tracking, Analysis, and Remediation Recommendation Architecture for aquatic invasive weeds using multispectral satellite telemetry and deep learning.**

A GeoAI platform that finds, tracks, and prioritises *Phragmites australis* (common reed) infestations across Egypt's Nile Delta canals and coastal lagoons using free multispectral satellite imagery — so maintenance shifts from reactive dredging to early intervention.

---

## Status

| Layer | State |
|---|---|
| `core/` — settings, grid algebra, capped cache | complete |
| `ingest/` — STAC source (live-verified), CDSE adapter | complete |
| `preprocess/` — windowed reads, indices, tiling | complete |
| `viz/` — server-side rendering | complete |
| `api/` — FastAPI service | complete |
| `db/` — PostGIS-shaped persistence | complete |
| `datasets/` — weak-supervision labelling | scaffolded |
| `models/`, `train/` — segmentation | scaffolded |
| `forecast/`, `decision/` — growth + threat scoring | scaffolded |
| `web/` — Next.js + MapLibre dashboard | complete |

"Scaffolded" means the interfaces, contracts, and algorithms are real and unit-tested, but the models are untrained — that requires the labelled dataset and a training run, which is the next milestone.

---

## Why the imaging choices are what they are

These are measured facts from this environment, not assumptions. Each one changed a design decision.

**Sentinel-2 L2A is reachable with no account and no cost.**
The Element 84 Earth Search STAC API (`https://earth-search.aws.element84.com/v1`) serves the `sentinel-2-l2a` collection over the Nile Delta continuously from **2022 through 2026**. A single August 2023 search over Lake Burullus returned **29 scenes under 10 % cloud**.

**Bands are read with HTTP byte ranges, not downloaded.**
Band files are Cloud-Optimized GeoTIFFs on `sentinel-cogs` (public S3). A full 10 m band is ~68 MB, but the files advertise `Accept-Ranges: bytes` and serve `206 Partial Content`. A 512×512 window is therefore ~0.5 MB of transfer. This is why the pipeline never needs to store imagery: the development machine had **under 3 GB free**.

**The archive's reflectance metadata contradicts its own pixels, and the pixels win.**
Earth Search's collection definition declares `scale: 0.0001, offset: 0` (so `reflectance = DN / 10000`), but individual items declare `offset: -0.1` (so `(DN - 1000) / 10000`). Both cannot be right. Applying the item-level offset to a Burullus tile produces **74 % negative reflectance** — physically impossible — whereas `DN / 10000` yields a clean 0–0.33 range and agrees with deep clear ocean, where NIR reflectance is known to be ≈ 0 (measured `+0.032` against `−0.068`). The pipeline therefore uses `DN / 10000` and deliberately **does not trust the item metadata**. Because that means a wrong guess would silently bias every index, each composite is re-validated at runtime and flagged if more than 2 % of valid pixels convert to negative — which is exactly what a real change in the archive's convention would look like.

**Bands arrive on two different pixel grids.**
B02/B03/B04/B08 are 10 m; red-edge and SWIR bands are 20 m. Everything is resampled onto one canonical grid before stacking, with **nearest-neighbour for the SCL mask** — interpolating class codes invents classes that do not exist.

**One date does not cover a study area.**
Lake Burullus spans four MGRS tiles on different relative orbits (36RTV covers 51 %, 36RUV 34 %, 36STA 14 %, 36SUA 23 %), so a single-date mosaic legitimately has holes — a first attempt reached only 57 % coverage. Compositing over a ±3 day window raises that to **100 %**, with clearer scenes painted first and an early exit once coverage is complete.

**`sentinelsat` is deliberately not used.** It targets the retired Open Access Hub, requires credentials, and only offers whole-product downloads. ESA provenance remains available via the CDSE adapter.

---

## Architecture

```
ingest/      scene discovery        stac.py (default)  |  cdse.py (ESA OAuth)
preprocess/  reader.py  -> windowed COG reads -> reflectance
             indices.py -> NDVI / NDWI / NDRE / NDMI
             tiler.py   -> non-overlapping model tiles
datasets/    weak-supervision labels + tile loading
models/      multispectral segmentation (U-Net / SegFormer)
train/       training loops, mIoU / F1 / pixel accuracy
forecast/    patch growth curves, expansion vectors
decision/    blockage threat score, harvest-window optimisation
db/          PostGIS-shaped spatial persistence
api/         FastAPI service
viz/         server-side PNG rendering of index composites
web/         Next.js + MapLibre GL dashboard
```

### The grid is the contract

Every layer agrees on one raster grid, described by `core/grids.py`. Tiles from two dates that are one pixel out of phase would fabricate change in a time series and offset every segmentation mask. AOI bounds are snapped **outward** to multiples of the ground sample distance, which lands on the same lattice Sentinel-2 itself uses (a real tile sits at an origin of exactly 300000/3500040 with 10 m pixels). `tests/test_core.py` pins the snapping algebra, and the live suite checks it against a real scene rather than trusting it.

---

## Quickstart

Requires **Python 3.11** (not 3.12+ — the geospatial wheel set is most reliable here) and Node 18+.

```bash
# 1. Environment. --system-site-packages reuses an existing CUDA PyTorch install
#    instead of re-downloading ~2.5 GB.
py -3.11 -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install wheel
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e . --no-deps --no-build-isolation

# 2. API
.venv/Scripts/python -m uvicorn atarra.api.main:app --reload --port 8000

# 3. Dashboard
cd web && npm install && npm run dev
```

Then open <http://localhost:3000>.

### Command line

```bash
atarra areas                          # list configured study zones
atarra scenes burullus --months 3     # search the archive and show scene availability
atarra indices burullus --date 2023-08-19   # compute + report index statistics
atarra preview burullus --date 2023-08-19 --index ndvi   # write a PNG
atarra cache                          # report cache size against its ceiling
```

### Tests

```bash
.venv/Scripts/python -m pytest              # offline: fast, deterministic
.venv/Scripts/python -m pytest -m network   # hits the live archive
```

Network tests are marked and deselected by default so the suite never fails merely because a satellite catalogue is unreachable.

---

## Design rules that are load-bearing

These are the decisions that quietly determine whether reported metrics hold up.

1. **Split train/validation/test by geography, never randomly.** Adjacent satellite tiles are heavily autocorrelated; a random split puts near-duplicates of test pixels in training and inflates mIoU into fiction.
2. **A 3-band RGB baseline is trained on identical splits and seed.** This is how the spec's "multispectral gain" claim gets substantiated rather than asserted.
3. **SCL masking happens before index computation.** Cloud, shadow, cirrus and snow otherwise enter the phenology features as spurious signal.
4. **`numpy < 2` is pinned.** The installed CUDA PyTorch build is compiled against numpy 1.26.4; upgrading shadows it and breaks `import torch`.
5. **The disk cache has a hard byte ceiling** with LRU eviction, reporting its size on every write.
6. **Biomass is a proxy, not a measurement.** Canopy biomass cannot be observed directly from orbit; the harvest-window logic integrates NDVI/NDRE over the season as a documented proxy. Stated plainly rather than presented as ground truth.

---

## Known limitations

- **No field validation yet.** Labels are bootstrapped by weak supervision and need a verification pass before reported accuracy means anything.
- **Biomass is modelled, not measured** (see rule 6).
- **The CDSE adapter is unexercised end-to-end** — it needs a real account and downloads ~800 MB per product. It is the fallback; STAC is the default.
- **The repo currently lives inside a OneDrive folder**, which syncs `.venv/` and `data/`. OneDrive can lock files mid-write and add sync latency; moving the checkout outside OneDrive (or pausing sync for `data/`) is recommended.
