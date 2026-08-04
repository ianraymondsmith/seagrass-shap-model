# Seagrass Health Predictive Pipeline
### Machine Learning Classification of Seagrass Health Status in Florida Bay Using Water Quality Parameters and Sentinel-2 Spectral Indices

---

## Overview

This repository contains a fully reproducible, modular Python pipeline for classifying seagrass health status (Healthy / Intermediate / Not Healthy) across Florida Bay monitoring stations. The pipeline integrates in-situ water quality (WQ) measurements from SFWMD DBHYDRO and supporting federal monitoring programs with Sentinel-2 multispectral imagery to derive spectral vegetation indices optimized for submerged aquatic vegetation (SAV). Two ensemble classifiers — Random Forest and XGBoost — are trained, tuned, and compared using SHAP (SHapley Additive exPlanations) to identify the dominant physicochemical and spectral drivers of seagrass condition.

This pipeline is designed to meet Q1 journal reproducibility standards (e.g., *Remote Sensing of Environment*, *Estuarine, Coastal and Shelf Science*, *Science of the Total Environment*). Every processing decision is documented with scientific rationale and literature citations.

---

## Study Area

**Florida Bay, Upper Florida Keys** — a shallow, oligotrophic estuary within Everglades National Park bounded approximately by:

| Boundary | Coordinate |
|----------|------------|
| West     | -81.20° W  |
| East     | -80.25° W  |
| South    | 24.85° N   |
| North    | 25.25° N   |

Florida Bay supports extensive *Thalassia testudinum*–dominated seagrass meadows that have experienced episodic die-off events (1987–1991; 2015 bleaching) linked to elevated turbidity, hypersalinity, and nutrient enrichment (Fourqurean & Robblee, 1999; Hall et al., 1999). Long-term monitoring through SEACAR and SFWMD DBHYDRO provides the WQ time series used here.

---

## Data Sources

| Dataset | Agency | Parameters | Access |
|---------|--------|-----------|--------|
| SFWMD DBHYDRO | South Florida Water Management District | Chl-a, TPO4, TOTN, DO, salinity, turbidity, temp, pH, Secchi depth | https://www.sfwmd.gov/science-data/dbhydro |
| NPS Florida Bay Monitoring | National Park Service | Salinity, hydrology, seagrass co-location | https://www.nps.gov/ever/learn/nature/floridabay.htm |
| FCE-LTER / EDI | Florida Coastal Everglades LTER | Nutrient dynamics, long-term WQ | https://edirepository.org |
| SEACAR DDI | Florida Fish and Wildlife Conservation Commission | Seagrass health labels, co-located WQ | https://SEACAR.net |
| Sentinel-2 MSI (Level-2A) | ESA Copernicus | Multispectral reflectance (10 m) | https://scihub.copernicus.eu |

**Sentinel-2 bands used:**

| Band | Name | Wavelength (nm) | Resolution (m) |
|------|------|-----------------|----------------|
| B2   | Blue | 490             | 10             |
| B4   | Red  | 665             | 10             |
| B8   | NIR  | 842             | 10             |

---

## Spectral Indices

Three indices are computed (Stage 2); NDAVI/WAVI are the primary indices for SAV:

| Index | Formula | Reference |
|-------|---------|-----------|
| NDAVI | (NIR − Blue) / (NIR + Blue) | Villa et al. (2013) |
| WAVI  | (1 + L)(NIR − Blue) / (NIR + Blue + L), L = 0.5 | Villa et al. (2014) |
| NDVI  | (NIR − Red) / (NIR + Red) | Rouse et al. (1974) |

NDAVI and WAVI are preferred over NDVI for submerged vegetation because they substitute the blue band for red, reducing sensitivity to terrestrial vegetation signal and improving discrimination under optically shallow water conditions (Villa et al., 2013; Roelfsema et al., 2014).

---

## Pipeline Architecture

```
seagrass_pipeline/
├── README.md
├── requirements.txt
├── environment.yml
├── config/
│   └── config.yaml                    # All paths, parameters, hyperparameters
├── data/
│   ├── raw/                           # Read-only: WQ CSVs, GeoTIFFs
│   ├── interim/                       # Intermediate files (spectral index rasters, joined GDFs)
│   └── processed/                     # Final feature table (features.parquet)
├── src/
│   ├── __init__.py
│   ├── 01_ingest.py                   # Load & validate WQ CSV + Sentinel-2 raster
│   ├── 02_spectral_index.py           # Compute NDAVI, WAVI, NDVI
│   ├── 03_spatial_join.py             # Join WQ points to spectral index values
│   ├── 04_feature_engineering.py      # Build unified feature dataframe
│   ├── 05_preprocessing.py            # Scaling, encoding, train/test splits
│   ├── 06_models.py                   # RF + XGBoost training with GridSearchCV
│   ├── 07_shap_analysis.py            # SHAP global + per-class + interaction plots
│   ├── 08_comparison.py               # Side-by-side metrics, McNemar's test
│   └── utils/
│       ├── logging_config.py          # Timestamped, reproducible logging
│       ├── validation.py              # Data integrity checks
│       └── plotting.py                # Publication-quality figure helpers
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_results_review.ipynb
├── tests/
│   └── test_*.py
├── outputs/
│   ├── figures/                       # 300 dpi publication figures
│   ├── models/                        # Pickled trained models
│   └── reports/                       # Metric tables, SHAP summaries
└── run_pipeline.py                    # Master end-to-end execution script
```

### Stage Summary

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `01_ingest.py` | Load WQ CSV, Sentinel-2 raster; validate CRS, columns, date alignment |
| 2 | `02_spectral_index.py` | Compute NDAVI, WAVI, NDVI; output masked GeoTIFFs |
| 3 | `03_spatial_join.py` | Extract/buffer/zonal-stat spectral values at WQ sample locations |
| 4 | `04_feature_engineering.py` | Engineer features (N:P ratio, turbidity×depth, seasonal); encode target |
| 5 | `05_preprocessing.py` | Impute, scale, stratified split, handle class imbalance |
| 6 | `06_models.py` | Train RF + XGBoost with 5-fold CV, GridSearchCV, report full metrics |
| 7 | `07_shap_analysis.py` | Global, per-class, and interaction SHAP analysis for both models |
| 8 | `08_comparison.py` | Side-by-side metrics, McNemar's test, Spearman importance correlation |

---

## Installation

### Option A: pip

```bash
git clone https://github.com/<your-org>/seagrass_pipeline.git
cd seagrass_pipeline
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: conda (recommended for geospatial dependencies)

```bash
conda env create -f environment.yml
conda activate seagrass_pipeline
```

**Python version:** 3.10+

---

## Usage

### End-to-end run

```bash
python run_pipeline.py --config config/config.yaml
```

### Individual stages

```bash
python src/01_ingest.py
python src/02_spectral_index.py
# ... etc.
```

### Reproduce figures only

```bash
python src/08_comparison.py --figures-only
```

---

## Configuration

All paths and parameters are controlled via `config/config.yaml`. No hardcoded values exist in source code. Key sections:

```yaml
data:
  wq_csv: data/raw/wq_florida_bay.csv
  raster: data/raw/sentinel2_florida_bay.tif

spatial_join:
  strategy: buffered_mean    # point_extraction | buffered_mean | zonal_stats
  buffer_radius_m: 20        # ~2× Sentinel-2 pixel size (10 m)

model:
  random_seed: 42
  test_size: 0.30
  cv_folds: 5
```

---

## Reproducibility

- All random seeds set in `config.yaml` and logged at runtime
- Package versions pinned in `requirements.txt` and `environment.yml`
- All pipeline runs produce a timestamped log in `outputs/reports/`
- Intermediate outputs cached in `data/interim/` to allow stage-level re-runs

---

## Expected Outputs

| Output | Location | Description |
|--------|----------|-------------|
| `ndavi.tif`, `wavi.tif`, `ndvi.tif` | `data/interim/` | Spectral index rasters |
| `features.parquet` | `data/processed/` | Final feature table |
| `rf_model.pkl`, `xgb_model.pkl` | `outputs/models/` | Trained model objects |
| `metrics_comparison.csv` | `outputs/reports/` | Accuracy, F1, kappa, AUC |
| `metrics_comparison.tex` | `outputs/reports/` | LaTeX-ready metric table |
| `shap_summary_rf.png`, etc. | `outputs/figures/` | 300 dpi SHAP plots |
| `confusion_matrices.png` | `outputs/figures/` | Normalized confusion matrices |
| `pipeline_run_<timestamp>.log` | `outputs/reports/` | Full run log |

---

## Citation

If you use this pipeline, please cite:

> [Author(s)]. (Year). Seagrass Health Predictive Pipeline: Machine Learning Classification of Seagrass Condition in Florida Bay Using Water Quality and Sentinel-2 Spectral Indices. *[Journal]*. DOI: [pending]

---

## References

Fourqurean, J.W., & Robblee, M.B. (1999). Florida Bay: A history of disturbance and recovery. *Estuaries*, 22(2B), 345–357. https://doi.org/10.2307/1353203

Hall, M.O., Durako, M.J., Fourqurean, J.W., & Zieman, J.C. (1999). Decadal changes in seagrass distribution and abundance in Florida Bay. *Estuaries*, 22(2B), 445–459.

Huete, A.R. (1988). A soil-adjusted vegetation index (SAVI). *Remote Sensing of Environment*, 25(3), 295–309. https://doi.org/10.1016/0034-4257(88)90106-X

Roelfsema, C., Kovacs, E., Saunders, M.I., Phinn, S., Lyons, M., & Maxwell, P. (2014). Challenges of remote sensing for quantifying changes in large complex seagrass environments. *Estuarine, Coastal and Shelf Science*, 133, 161–171. https://doi.org/10.1016/j.ecss.2013.08.026

Rouse, J.W., Haas, R.H., Schell, J.A., & Deering, D.W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Proceedings of the 3rd Earth Resources Technology Satellite Symposium*, 1, 309–317.

Villa, P., Bresciani, M., Braga, F., & Bolpagni, R. (2013). A rule-based approach for mapping macrophyte communities using multi-temporal aquatic vegetation indices. *Remote Sensing of Environment*, 133, 148–160. https://doi.org/10.1016/j.rse.2013.02.007

Villa, P., Bresciani, M., Bolpagni, R., Pinardi, M., & Giardino, C. (2014). A rule-based approach for mapping macrophyte communities using multi-temporal aquatic vegetation indices. *Remote Sensing*, 6(8), 7181–7202.

---

## License

[MIT / CC BY 4.0 — to be confirmed by PI]

---

*Last updated: 2026 | Contact: [PI email placeholder]*
