# Project Memory

# CLAUDE.md — Project Instructions for Claude Code

## Project
Seagrass Health Predictive Pipeline — Q1 journal submission.
3-class classification (Healthy / Intermediate / Not Healthy) of seagrass
health in Florida Bay using SFWMD DBHYDRO water quality data + Sentinel-2
spectral indices (NDAVI, WAVI, NDVI, SSII).

## Architecture
8-stage modular pipeline: ingest → spectral index → spatial join →
feature engineering → preprocessing → models (RF + XGBoost) → SHAP → comparison.
All parameters live in config/config.yaml — no hardcoded values in source.

Stage modules live in src/ and are named `<name>_<NN>.py` (e.g. `ingest_01.py`,
`spectral_index_02.py`), not `NN_name.py` — Python module names can't start
with a digit. run_pipeline.py imports each stage module by this convention as
it's built; unbuilt stages still log an explicit placeholder message.

Standalone data-acquisition scripts (not pipeline stages — they populate
data/raw/ before Stage 1/2 can run) live in scripts/:
- `scripts/download_dbhydro_wq.py` — SFWMD DBHYDRO water-quality export
- `scripts/download_sentinel2.py` — Sentinel-2 L2A imagery via Copernicus
  Data Space Ecosystem
Both require real internet access and must be run in a normal terminal or
local Jupyter kernel — see "Environment notes" below.

## Code standards
- NumPy-style docstrings on every function (parameters, returns, raises, notes, references)
- Module-level header docstrings with purpose, inputs, outputs, references, author placeholder
- Inline comments explain *why* (scientific rationale), not *what*
- Literature citations where methodological choices are made
- All logging via `from src.utils.logging_config import get_logger`
- All config via `from src.utils.config_loader import load_config`
- Validation functions in src/utils/validation.py

## Key commands
- Run tests: `pytest tests/ -v`
- Run pipeline: `python run_pipeline.py --config config/config.yaml`
- Acquire WQ data: `python scripts/download_dbhydro_wq.py` (run outside the
  agent sandbox — see "Environment notes")
- Acquire Sentinel-2 imagery: `python scripts/download_sentinel2.py` (same)

## Data
- WQ source: SFWMD DBHYDRO, real export in hand at
  `data/raw/wq_florida_bay_dbhydro.csv` (Monroe + Miami-Dade counties,
  2016-01-01 to 2026-01-01). DBHYDRO's export is LONG format (one row per
  station x date x parameter, matrix=SA for saltwater/estuarine) — Stage 1
  pivots it to wide format. 22 stations survive the Florida Bay bounding-box
  filter: 19 `FLAB*` (in-bay) + 3 `C111*` (canal inflow, kept intentionally).
  11 usable parameters: chlorophyll-a, TPO4, TOTN, NOx (nitrate+nitrite),
  NH4 (ammonia), DO, salinity, turbidity, temp, pH, Secchi depth. TSS and
  Color exist in DBHYDRO generally but are not measured at these specific
  stations. See config/config.yaml's `ingestion` section for the full
  column-rename map and known data-quality caveats (longitude sign
  convention, depth outliers).
- Raster: Sentinel-2 L2A via Copernicus Data Space Ecosystem (free account,
  no CCM add-on needed). 120 scenes acquired (2016-2025, one per ~30-day
  window, cloud cover <20%), bands B02/B04/B08 (10m) in
  `data/raw/sentinel2/<scene_id>/`. Landsat backfill for pre-2016 imagery
  was considered and dropped, since the WQ export in hand only goes back to
  2016 anyway.
- **Target label (health_status) is NOT yet acquired.** DBHYDRO has no
  seagrass condition data. Source: SEACAR Program 4049 (FWC-FWRI,
  https://data.florida-seacar.org/programs/details/4049, self-serve
  download "SAV - 4049.zip", contact brad.furman@myfwc.com) — Braun-Blanquet
  cover-abundance score + shoot density, not a pre-made 3-class label.
  Proposed (unvalidated) threshold scheme: BB 0-1 = Not Healthy, 2-3 =
  Intermediate, 4-5 = Healthy — needs checking against the real score
  distribution once downloaded.
- Study area: Florida Bay (~24.85–25.25°N, ~80.25–81.20°W)
- Under consideration: adding SSI-I/SSI-II indices (Liang et al. 2023,
  Optics Express, DOI 10.1364/OE.498901) alongside NDAVI/WAVI/NDVI, verified
  against the paper's rendered equations:
  `SSI-I = Rrs(NIR) - [Rrs(red) + (Rrs(SWIR1)-Rrs(red))*(NIR-red)/(SWIR1-red)]`
  (continuum-removal/baseline-height index, same family as the Floating
  Algae Index) and `SSI-II = (Rrs(green)-Rrs(red)) / Rrs(green)`. Landsat
  band roles map to Sentinel-2 as: green->B3, red->B4, NIR->B8A (20m, NOT
  the 10m B8 already used for NDAVI/WAVI/NDVI — B8A is the correct
  narrow-NIR analog to Landsat's NIR band), SWIR1->B11 (20m). SSI-II can
  stay 10m-native (B3+B4 only); SSI-I is inherently 20m-bound. Needs 3 more
  Sentinel-2 bands not yet downloaded (B03, B8A, B11), and the paper's
  Landsat-calibrated thresholds (-0.0034/0.1627 for TM/ETM+, -0.0021/0.2071
  for OLI/OLI-2) need recalibration against SEACAR ground truth, not reused
  as-is — though if these are only used as continuous ML features (not a
  rule-based presence/absence classifier), the threshold question may not
  block using them at all.

- HAB (harmful algal bloom) data: FWC/NOAA HABSOS export in hand at
  `data/raw/habsos_20240430.csv` (all Gulf states, 211K rows -- filtered to
  Florida Bay bbox + 2016-2026 study period by `src/hab_matching.py`, 589
  raw observations, all *Karenia brevis*). HABSOS has no fixed station
  network (252 unique lat/lon pairs); each observation is assigned to its
  nearest DBHYDRO station (mean 8.0 km, max 23.1 km) so it can reuse the
  same per-station 5-day resampling grid as the main WQ pipeline. Cell
  count is heavily zero-inflated (median 0, spikes to 103,000 cells/L) --
  resampled via nearest-observation carry-forward, NOT linear
  interpolation, to avoid fabricating a smooth ramp into/out of a bloom;
  salinity/temperature (partial coverage, also reported by HABSOS) ARE
  linearly interpolated. 3 stations (FLAB48, C111JB, FLAB08) have zero HAB
  data within reach. Outputs: `data/interim/hab_florida_bay.parquet` and
  `data/interim/hab_resampled_5day.parquet`.

## Environment notes
- **The coding-agent sandbox (Bash/PowerShell tools) has no outbound network
  access.** Confirmed via DNS resolution failure against external hosts.
  Any script hitting a real API (DBHYDRO, Copernicus) must be run by the user
  in a normal terminal or a local-kernel Jupyter notebook, not by the agent.
- Bare `python`/`python3` on this machine resolve to broken Windows Store
  stub aliases. Use the real interpreter explicitly:
  `C:\Users\ianra\miniconda3\python.exe` (conda `base` env — already has
  pandas, numpy, pyyaml, requests, lxml, pytest, pyarrow, rasterio,
  matplotlib installed as of this pipeline's development).
- Stage 2 output rasters are stored as compressed int16 (scaled by
  `spectral_index.output_scale_factor`, DEFLATE + predictor=2), not float32
  — an uncompressed float32 run filled a 465 GB disk. Keep this in mind if
  adding new raster-writing code.

## Dependencies
See requirements.txt (pip) or environment.yml (conda).
Python 3.10+.