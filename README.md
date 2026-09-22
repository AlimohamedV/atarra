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
| `datasets/` — weak-supervision labelling, tile store, geographic splits, annotation queue | complete |
| `models/`, `train/` — segmentation, training loop, metrics, provenance | complete, models untrained |
| `forecast/`, `decision/` — growth + threat scoring | scaffolded |
| `web/` — Next.js + MapLibre dashboard | complete |

"Complete" means implemented and exercised end to end; "scaffolded" means the interfaces, contracts, and algorithms are real and unit-tested but nothing reachable calls them yet.

**No model has been trained.** `data/checkpoints/` is empty, which is why none of the proposal's accuracy targets (§6) has a number behind it yet. The dataset and training path now exist to produce one — see *Training* below.

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

# training path
atarra dataset build burullus --gsd 20 --stride 128 --dates 12 --out data/dataset
atarra train data/dataset --bands 8 --epochs 40
```

### Training (on Google Colab)

Training needs a GPU and several gigabytes of imagery, so it belongs on Colab rather than here: this machine has a **4.29 GB** RTX 2050 and had under 3 GB of free disk, while a full Burullus 8-band composite at 10 m is 644 MB *per date*. Colab's runtime is Python **3.12** with numpy **2.0.2** and PyTorch **2.11**; both packaging constraints that would have blocked it (`requires-python < 3.12`, `numpy < 2`) are now scoped to where they actually apply.

`notebooks/atarra_colab_train.ipynb` drives it in **stages**, and the split is deliberate — free-tier sessions disconnect when idle, so a fetch combined with a training run means one disconnect discards both.

| Stage | What runs | How often |
|---|---|---|
| Build | `atarra dataset build burullus --gsd 20 --stride 128 --dates 12 --out <drive>/store_20m` | once, 30–60 min |
| Reserve | `atarra annotation export <store> --out <drive>/annotation_pack --strategy reed` | once, **before any training** |
| Pilot | `atarra train <store> --epochs 4 --exclude-pack <pack>` | first, to check the split |
| Train | `atarra train <store> --bands 8 --epochs 40 --exclude-pack <pack>` | repeatedly, minutes |
| Score | `atarra annotation score <pack> --checkpoint <run>/best.pt` | once labelled |

Reserving before training is the load-bearing order: a test set the model has already seen is not a test set, and `--exclude-pack` removes the pack's **ground** — the same reed bed on every date it was imaged, plus its neighbours within half a tile — rather than only the tile keys it names.

The store build itself is **not** resumable: an interrupted build restarts, and an incomplete store (manifest present, shards missing) is detected and discarded rather than built on. Neither is a training run — there is no checkpoint-resume path — which is exactly why the store exists as a separate stage. Run directories are timestamped and `atarra train` refuses to write into one that already holds a run, because two 40-epoch runs whose `metrics.json` overwrote each other are indistinguishable afterwards.

The build writes **tile shards**, not imagery: reflectance as uint16 at the archive's own 1e-4 scale, which round-trips exactly at a quarter of float32's size. (float16 would be smaller still, but carries ~3 significant digits at magnitude 1.0 — the same order as the red-edge differences that separate reed from cropland.) Training memory-maps the shards, so a store larger than RAM still works. The 8-band model and the 3-band RGB control arm are cut from the **same** store, so the multispectral-gain comparison costs no second download.

Two parameters matter more than they look:

- **`--gsd 20`** keeps twelve dates to 1.9 GB of composites. At 10 m it is 7.6 GB, which does not fit a free-tier session — `CompositeTileDataset` holds every composite in memory.
- **`--stride 128`** overlaps tiles 2× for roughly 4× the training samples at no extra download cost, because imagery is fetched once and cut afterwards. Overlapping tiles are separate samples for the loss, which is the point — but they must never straddle the train/test fence. The split is therefore built from tile *footprints*, not from a `(row // tile_size, col // tile_size)` block index, and tiles that cross a boundary are dropped rather than assigned. See rule 1.

Store builds are refused outright rather than warned about when the grid was coarsened to fit `--max-size`, or when a second shard disagrees with the first about the grid. Either would produce a manifest that misdescribes its own shards.

#### The labels measure agreement, not truth

Training labels come from the weak-supervision rule engine in `datasets/weak_labels.py`, so a metric computed against them scores **agreement with that rule engine** — if the rules are wrong about a pixel, a model that reproduces them faithfully is still scored correct. Every `metrics.json` carries that caveat as a field, so the number cannot be quoted without it.

Two related decisions were bugs first, and are worth stating:

- The rule engine records every pixel it declined to be confident about. That mask is persisted alongside the tiles and *is* the annotation worklist: `TileStoreDataset.annotation_tiles()` ranks tiles by how much a human is needed and returns lon/lat corners for each, so the hardest cases open directly in QGIS. A handful of annotated tiles, held out, is what makes a defensible mIoU possible.
- The loss mask is deliberately **not** the review mask. `REVIEW_THRESHOLD` is tuned for queue size, and the per-class scores saturate at different ceilings by design — the crop rule tops out at 0.597 against a 0.60 cut. Reusing it as a training filter deleted the **entire cropland class** by 0.003 of confidence, silently turning a 4-class problem into a 2-class one. Ambiguity is decided on **margin** instead (0.067 for a reed/crop coin flip, versus 0.23–0.52 for real decisions), and `test_every_class_survives_confidence_filtering` exists to keep it that way.

#### Building a test set a panel will accept

The rule engine already knows which pixels it cannot call, so annotating its uncertainty is cheaper than annotating at random:

```bash
atarra annotation export data/dataset --out data/annotation --limit 20 --strategy reed
atarra train data/dataset --exclude-pack data/annotation     # the reserved tiles stay unseen
atarra annotation score data/annotation --checkpoint data/checkpoints/<run>/best.pt
```

The export writes three georeferenced GeoTIFFs per tile — the imagery, the rule engine's current guess (so you **correct** it rather than start from blank), and a review mask showing where that guess is untrustworthy — plus `annotation.geojson` with one polygon per tile carrying its lon/lat corners, and a generated README.

`--strategy` decides what "top-ranked" means, and the choice matters:

| Strategy | Picks | Use it for |
|---|---|---|
| `uncertainty` (default) | ground where the rules are least confident | improving the *training* labels |
| `reed` | tiles richest in the target class | a test set that can actually measure reed IoU |
| `random` | a seeded sample | a representative, non-adversarial test set |

The default is a **hard-case** selection, so a test set built from it measures the model on the hardest pixels in the area. That is defensible, but it is not a representative sample, and it is why `reed` exists: reed is ~1.8 % of this imagery, and one arbitrary selection produced "reed IoU 0.0" from **36 support pixels**, which measures nothing.

Labels can be polygons (`labels/<key>.geojson` with an integer `class_code` field — draw them in QGIS, which is what it is good at) or a raster on the chip's grid. Polygon geometry is reprojected from WGS84 into the chip's CRS before rasterising; GeoJSON is WGS84 by specification, and rasterising lon/lat against a UTM transform does not fail, it silently produces garbage.

##### The scorer refuses to flatter you

A metric can be arithmetically perfect and completely empty. Measured during development: a model trained for 2 epochs on 7 tiles reported **reed IoU 1.0, "proposal targets met"** — because the annotation covered a single class, and a model predicting that class everywhere scores 1.0 on every pixel of it.

So `score.json` carries an `assessable` flag with the blocking reason, and the CLI will not print a target verdict when the score is not a validation:

```
targets        NOT ASSESSED -- this score is not a validation
  classes present  phragmites_australis
  reed annotated   32,768 px (needs 500)
  model predicted  [3]
NOT A VALIDATION of the model: the annotations cover only 1 class(es)...
```

Three ways a report is rejected as a validation: fewer than two classes annotated, fewer than 500 reed pixels annotated, or the model predicting a single class across every annotated pixel. Scorer and scores both stop at "not assessable" rather than "passed".

### Tests

```bash
.venv/Scripts/python -m pytest              # offline: fast, deterministic
.venv/Scripts/python -m pytest -m network   # hits the live archive
```

Network tests are marked and deselected by default so the suite never fails merely because a satellite catalogue is unreachable.

`tests/test_integrity.py` is the regression suite for *whether a reported score means anything*: splits sharing ground, a report computed from discarded weights, chips read at the wrong scale, normalisation drawn from held-out imagery, augmented validation inputs, a holdout covering one date, an unearned target verdict, an unseeded initialisation, dropped accumulation gradients, and a dataset that cannot cross into a worker process. Several of its tests carry a negative control — the superseded behaviour reproduced and asserted to *fail* the same check — because a guard that cannot fail tests nothing.

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request.

| Job | Runner | Trigger | Covers |
|---|---|---|---|
| `test` | ubuntu-latest | every push | offline suite; **0 skips tolerated** |
| `web` | ubuntu-latest | every push | `tsc --noEmit` + production build |
| `live` | ubuntu-latest | manual dispatch | live archive suite, ~2 min |
| `gpu` | self-hosted | manual dispatch + `run_gpu` | GPU tests on real CUDA hardware |

Network tests deliberately stay off the push path: they make real ranged reads against the public archive, so an upstream catalogue outage would fail a commit that is perfectly correct. Dispatch the workflow from the Actions tab when you want them.

CI installs **CPU-only** PyTorch. Without torch every model test would hit `pytest.importorskip` and *skip*, so the job would report success having validated none of the machine learning. A dedicated step fails the run if anything skips, so that false-green cannot pass unnoticed.

#### Enabling the GPU job

Hosted runners have no CUDA device, so the GPU-marked tests would otherwise only ever run on one developer's machine. Give them a home by registering a self-hosted runner that carries the `gpu` label:

```bash
./config.sh --url https://github.com/<owner>/<repo> --token <token> --labels gpu
```

Then dispatch the workflow with `run_gpu` ticked. If no runner carries that label the job **queues and waits** rather than failing — cancel it and check the label. The job installs the CUDA PyTorch wheel (~2.5 GB) once; a self-hosted runner keeps its pip cache, so later runs reinstall from cache instead of re-downloading.

Before running anything the job asserts that `torch.cuda.is_available()` is true. Because every GPU test skips itself when no device is present, a mislabelled CPU runner would otherwise pass while testing nothing.

---

## Design rules that are load-bearing

These are the decisions that quietly determine whether reported metrics hold up.

1. **Split train/validation/test by geography, never randomly — and verify it.** Adjacent satellite tiles are heavily autocorrelated, and with `--stride 128` they literally overlap: a split that assigns *tiles* still leaves shared *pixels* on both sides. So partitions are defined on the ground, tiles crossing a boundary are dropped, every run calls `assert_disjoint` on the result, and the gap actually applied is recorded in `metrics.json` (it can be narrower than requested — disjointness is the invariant, the gap is a knob). Block-index splits look spatial and are not: `TestGeographicSplit::test_the_check_has_teeth` reproduces that scheme and asserts it *fails* the same check. The date a tile came from never moves ground between splits, so reserving one date's tile reserves that position on all of them.
2. **A 3-band RGB baseline is trained on identical splits and seed, and normalised from its own training split.** This is how the spec's "multispectral gain" claim gets substantiated rather than asserted — and the arms have to differ in the band set and in nothing else. `build_model(variant="rgb")` used to override the caller's statistics with hard-coded constants, which quietly made every reported gain part band set and part preprocessing; the constants are now only a fallback for callers that supply nothing.
3. **The reported number describes the checkpoint that was kept.** `train()` restores `best.pt` before its final validation and before the test pass. Without that, the run selects an epoch on validation, writes it to disk, and then reports whichever epoch happened to run last — two different models in the same directory. `metrics.training.evaluated_epoch` records which one was evaluated.
4. **Nothing outside the training split touches a training decision.** Loss weights (rule 6), normalisation statistics, and augmentation all come from or apply to training tiles only. `band_statistics` streams raw usable pixels from a supplied index set rather than sampling the store, and validation/test views are built with `augment=False` — jitter on the images a metric is computed from moves the metric. Only the training view augments, and its transform advances every epoch: a fixed "random" flip repeated forty times is not augmentation.
5. **Reflectance has one convention, and the reader asserts it.** Exported chips are float32 reflectance and the scorer consumes them as-is. Applying the storage decode a second time would divide an already-decoded `0.2` by 10,000 — no exception, no shape change, just inputs four orders of magnitude outside the training range. `_as_reflectance` rejects anything that looks like stored integers and names the convention instead.
6. **Initialisation is seeded before construction, not after.** `set_seed` runs before `build_model`, so two runs of one configuration are the same run; loader shuffling uses an explicit generator for the same reason, since the global RNG's state depends on how much randomness model construction happened to consume.
7. **A batch with no supervised pixel is skipped, and an epoch with no such batch fails.** Cross-entropy over an empty label set is NaN, and one NaN batch would write NaN into every weight while the loss curve kept printing. Non-finite losses abort the run. The final partial gradient-accumulation group is flushed with the correct scaling rather than discarded at the epoch boundary.
8. **Checkpoints carry provenance, not just weights.** Ordered band names (a channel *count* cannot tell B08 from B04 — the shape matches and every number is plausible) and the tile keys of every split. That is what lets `annotation score` check that annotated ground was never trained on, and report independence as **unverified** rather than assumed when the record is absent.
9. **No target verdict is published for an unearned comparison.** `metrics.json` from a training run writes `targets.assessable: false` with the reason, because those metrics score the model against the rule engine that produced its labels. `score.json` does the same whenever its assessment is rejected. `both_met` is `null`, not `false`: the comparison was not made, and a boolean either way reads as an answer.
10. **SCL masking happens before index computation.** Cloud, shadow, cirrus and snow otherwise enter the phenology features as spurious signal.
11. **`numpy < 2` is pinned locally, and only locally.** The development machine's CUDA PyTorch build is compiled against numpy 1.26.4, so `requirements.txt` caps it. That is a property of one machine, not of ATARRA: `pyproject.toml` declares no upper bound and `requirements-colab.txt` leaves numpy alone, because Colab ships 2.0.2 alongside a numpy-2-native torch and forcing the downgrade there would churn its preinstalled pandas/scipy for no reason.
12. **The disk cache has a hard byte ceiling** with LRU eviction, reporting its size on every write.
13. **Loss weights are computed from the training split alone.** Reed is ~1.8 % of pixels on real Burullus imagery, so unweighted cross-entropy is minimised by predicting "not reed" everywhere. The inverse-frequency correction is necessary — but computing it over the whole store imports the validation and test class balance into a training-time decision.
14. **Biomass is a proxy, not a measurement.** Canopy biomass cannot be observed directly from orbit; the harvest-window logic integrates NDVI/NDRE over the season as a documented proxy. Stated plainly rather than presented as ground truth.
15. **Pixel tallies state their unit.** Class counts and `trainable_px` are summed per tile, so overlapping tiles appear more than once; `usable_px` and `review_px` are counted once per date on the common grid. Mixing the two bases is what once let a review *fraction* exceed 1, and a filtered view now reports `null` for the store-level figures rather than a nearby number.

---

## Known limitations

- **No field validation yet.** Labels are bootstrapped by weak supervision and need a verification pass before reported accuracy means anything.
- **The Colab notebook has never run on Colab.** Every cell has been executed locally against a synthetic store with the Drive paths redirected, which checks the logic and the printed claims but not the runtime, the Drive mount, or the download.
- **No model has been trained to convergence, and no real-data training run has been scored.** The offline suite trains for one or two epochs on a 48 px synthetic store to prove the plumbing; `data/checkpoints/` is still empty. Nothing here is evidence about accuracy.
- **The GPU CI job is unexercised.** The workflow for it exists, but no self-hosted runner has ever picked it up.
- **`--strategy reed` biases the test set on purpose** toward the class the proposal quotes a number for. That makes the reed IoU measurable; it also means the set is not a representative sample of the area, and the two claims are not the same claim.
- **Biomass is modelled, not measured** (see rule 14).
- **The CDSE adapter is unexercised end-to-end** — it needs a real account and downloads ~800 MB per product. It is the fallback; STAC is the default.
- **The repo currently lives inside a OneDrive folder**, which syncs `.venv/` and `data/`. OneDrive can lock files mid-write and add sync latency; moving the checkout outside OneDrive (or pausing sync for `data/`) is recommended.
