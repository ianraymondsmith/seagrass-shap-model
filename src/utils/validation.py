"""
validation.py
=============
Data integrity checks for the Seagrass Health Predictive Pipeline.

PURPOSE:
    Provides reusable validation functions called by ingestion (Stage 1),
    spatial join (Stage 3), and feature engineering (Stage 4) modules.
    Failing fast with informative errors is preferable to propagating
    silent data quality issues into model training.

    Validation is a critical reproducibility safeguard: it documents the
    expected data contract between field/satellite data providers and the
    pipeline, making the methodology auditable by journal reviewers.

VALIDATION CATEGORIES:
    1. Column completeness — required columns present and correctly typed
    2. Coordinate validity — latitude/longitude within study area bounds
    3. CRS consistency — WQ points and raster share compatible reference systems
    4. Date alignment — WQ samples temporally matched to raster acquisition
    5. Value ranges — physical plausibility of WQ parameters
    6. Class distribution — health label set matches expected categories
    7. Feature matrix — no leakage of target into features

REFERENCES:
    Wickham, H. (2014). Tidy data. Journal of Statistical Software, 59(10), 1–23.
    https://doi.org/10.18637/jss.v059.i10

    Wilson, G. et al. (2017). Good enough practices in scientific computing.
    PLOS Computational Biology, 13(6), e1005510.
    https://doi.org/10.1371/journal.pcbi.1005510

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Physical plausibility bounds for Florida Bay WQ parameters.
# Sources:
#   - Temperature: SFWMD DBHYDRO historical range; Boyer et al. (1997)
#   - Salinity: Florida Bay hypersalinity events can reach 70 ppt
#     (Fourqurean & Robblee, 1999)
#   - Turbidity: oligotrophic baseline <5 NTU; storm events >200 NTU
#   - Chl-a: background <2 µg/L; bloom events up to ~50 µg/L (Boyer et al., 1997)
#   - DO: hypoxic below 2 mg/L; supersaturation up to ~15 mg/L
#   - pH: marine range 7.5–8.5; carbonate system in Florida Bay
#   - TPO4: oligotrophic baseline <0.01 mg/L; enriched up to ~0.5 mg/L
#   - TOTN: typical range 0.1–3.0 mg/L
#   - Secchi depth: 0.1 m (very turbid) to ~10 m (clear)
# ---------------------------------------------------------------------------
WQ_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temp_C":        (10.0,  40.0),
    "salinity_ppt":  (0.0,   75.0),
    "turbidity_NTU": (0.0,  500.0),
    "chla_ug_L":     (0.0,  100.0),
    "do_mg_L":       (0.0,   20.0),
    "ph":            (6.5,    9.5),
    "tpo4_mg_L":     (0.0,    2.0),
    "totn_mg_L":     (0.0,   10.0),
    "secchi_m":      (0.0,   20.0),
    # Bioavailable N fractions (DBHYDRO parameters NITRATE+NITRITE-N,
    # AMMONIA-N). Typical oligotrophic Florida Bay range <0.05 mg/L;
    # bound set loosely to admit episodic canal-discharge enrichment
    # near the C-111 stations without treating it as an error.
    "nox_mg_L":      (0.0,    5.0),
    "nh4_mg_L":      (0.0,    5.0),
}

# Maximum plausible sample depth (metres) for Florida Bay grab samples.
# Florida Bay is a shallow lagoon (typically <2 m, rarely >4 m in channels);
# a reported depth beyond this is almost certainly a unit/entry error
# (e.g. feet recorded as metres) rather than a real measurement.
MAX_PLAUSIBLE_DEPTH_M = 5.0

# Expected Florida Bay bounding box in WGS84 decimal degrees.
# Adjust if study area extends beyond this region.
FLORIDA_BAY_BOUNDS = {
    "lon_min": -81.20,
    "lon_max": -80.25,
    "lat_min":  24.85,
    "lat_max":  25.25,
}


# ---------------------------------------------------------------------------
# 1. Column validation
# ---------------------------------------------------------------------------

def validate_wq_columns(
    df: pd.DataFrame,
    required_columns: List[str],
) -> None:
    """
    Raise ``ValueError`` if required water quality columns are absent.

    Checks column presence after the rename map in config has been applied
    (Stage 1, 01_ingest.py).  Type enforcement is lenient — numeric columns
    are coerced in ingestion; a non-numeric value triggers a separate warning
    rather than a hard failure here.

    Parameters
    ----------
    df : pd.DataFrame
        Water quality DataFrame (post-rename).
    required_columns : list of str
        Column names that must be present.  Sourced from
        ``config.ingestion.required_wq_columns``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If one or more required columns are missing.

    Examples
    --------
    >>> validate_wq_columns(df, config["ingestion"]["required_wq_columns"])
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"WQ DataFrame is missing required columns: {missing}.\n"
            f"Present columns: {list(df.columns)}.\n"
            "Check the column_rename_map in config/config.yaml."
        )
    logger.info("Column validation PASSED — all %d required columns present",
                len(required_columns))


# ---------------------------------------------------------------------------
# 2. Coordinate validation
# ---------------------------------------------------------------------------

def validate_coordinates(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    bounds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Check and report coordinate validity for WQ sample points.

    Identifies rows with NaN coordinates, coordinates outside Florida Bay
    bounds, and coordinates at exactly (0, 0) — a common sentinel value for
    missing GPS data.  Out-of-bounds rows are logged and dropped from the
    returned DataFrame rather than raising an exception, because occasional
    bad-fix GPS records are common in field monitoring datasets.

    Parameters
    ----------
    df : pd.DataFrame
        WQ DataFrame containing latitude and longitude columns.
    lat_col : str, optional
        Name of the latitude column.  Default: ``"latitude"``.
    lon_col : str, optional
        Name of the longitude column.  Default: ``"longitude"``.
    bounds : dict, optional
        Bounding box dict with keys ``lon_min``, ``lon_max``, ``lat_min``,
        ``lat_max``.  Defaults to ``FLORIDA_BAY_BOUNDS``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with invalid-coordinate rows removed.

    Raises
    ------
    ValueError
        If the coordinate columns are not present in ``df``.

    Notes
    -----
    Dropped rows are logged at WARNING level.  The count of dropped rows
    must be reported in the Methods section as part of the data quality
    description (e.g., "N = X records were excluded due to invalid GPS
    coordinates").

    Examples
    --------
    >>> df_clean = validate_coordinates(df)
    """
    if bounds is None:
        bounds = FLORIDA_BAY_BOUNDS

    for col in [lat_col, lon_col]:
        if col not in df.columns:
            raise ValueError(f"Coordinate column '{col}' not found in DataFrame.")

    n_initial = len(df)

    # Null coordinates
    null_mask = df[lat_col].isna() | df[lon_col].isna()
    if null_mask.any():
        logger.warning(
            "Dropping %d rows with null coordinates", null_mask.sum()
        )
        df = df[~null_mask].copy()

    # Zero-zero sentinel check (common missing-GPS artefact)
    zero_mask = (df[lat_col] == 0.0) & (df[lon_col] == 0.0)
    if zero_mask.any():
        logger.warning(
            "Dropping %d rows with (0.0, 0.0) coordinates — likely missing GPS",
            zero_mask.sum(),
        )
        df = df[~zero_mask].copy()

    # Bounding box check
    out_of_bounds = (
        (df[lon_col] < bounds["lon_min"])
        | (df[lon_col] > bounds["lon_max"])
        | (df[lat_col] < bounds["lat_min"])
        | (df[lat_col] > bounds["lat_max"])
    )
    if out_of_bounds.any():
        logger.warning(
            "Dropping %d rows with coordinates outside Florida Bay bounding box "
            "(lon: [%.2f, %.2f], lat: [%.2f, %.2f])",
            out_of_bounds.sum(),
            bounds["lon_min"], bounds["lon_max"],
            bounds["lat_min"], bounds["lat_max"],
        )
        df = df[~out_of_bounds].copy()

    n_dropped = n_initial - len(df)
    if n_dropped == 0:
        logger.info("Coordinate validation PASSED — all %d records within bounds",
                    n_initial)
    else:
        logger.warning(
            "Coordinate validation: %d of %d records retained after dropping "
            "%d invalid-coordinate rows",
            len(df), n_initial, n_dropped,
        )

    return df


# ---------------------------------------------------------------------------
# 3. CRS consistency
# ---------------------------------------------------------------------------

def validate_crs_match(
    wq_crs: str,
    raster_crs: str,
) -> None:
    """
    Warn if WQ point CRS and raster CRS differ without reprojection.

    The pipeline reprojects WQ points to the raster CRS before spatial
    operations (Stage 3), so a mismatch is not fatal.  This function logs
    the detected CRS values so the decision is traceable.

    Parameters
    ----------
    wq_crs : str
        CRS string or EPSG code of the WQ GeoDataFrame
        (e.g., ``"EPSG:4326"``).
    raster_crs : str
        CRS string or EPSG code read from the raster file.

    Returns
    -------
    None

    Notes
    -----
    Stage 3 (03_spatial_join.py) reprojects to the projected CRS specified
    in ``config.spatial_join.projected_crs`` before buffering.  Both inputs
    are ultimately reprojected, so a mismatch here is expected and handled.
    """
    logger.info("WQ CRS:     %s", wq_crs)
    logger.info("Raster CRS: %s", raster_crs)
    if wq_crs != raster_crs:
        logger.warning(
            "CRS mismatch (WQ: %s vs Raster: %s). "
            "Stage 3 will reproject to projected_crs before spatial join.",
            wq_crs, raster_crs,
        )
    else:
        logger.info("CRS validation PASSED — WQ and raster share CRS: %s", wq_crs)


# ---------------------------------------------------------------------------
# 4. Date alignment
# ---------------------------------------------------------------------------

def validate_date_alignment(
    wq_dates: pd.Series,
    raster_acquisition_date: pd.Timestamp,
    max_offset_days: int = 30,
) -> pd.Series:
    """
    Flag WQ records whose sample date exceeds the temporal matching window.

    The temporal match window (default ±30 days) reflects the assumption
    that seagrass physiological condition integrates environmental stress
    over weeks to months rather than days (Fourqurean & Robblee, 1999).
    Records outside the window are not dropped — the analyst should inspect
    them and decide — but they are flagged in a returned boolean Series.

    Parameters
    ----------
    wq_dates : pd.Series
        Parsed date column from the WQ DataFrame (dtype: datetime64).
    raster_acquisition_date : pd.Timestamp
        Acquisition date of the Sentinel-2 image.
    max_offset_days : int, optional
        Maximum allowable offset in days.  Default: ``30``.
        Sourced from ``config.ingestion.max_date_offset_days``.

    Returns
    -------
    pd.Series
        Boolean Series (same index as ``wq_dates``) where ``True`` means
        the record is *outside* the temporal window.

    Notes
    -----
    Report the fraction of records outside the window in the Methods section.
    If >20% of records fall outside the window, consider tightening the
    Sentinel-2 image selection criteria.

    References
    ----------
    Fourqurean, J.W., & Robblee, M.B. (1999). Florida Bay: A history of
    disturbance and recovery. Estuaries, 22(2B), 345–357.
    https://doi.org/10.2307/1353203

    Examples
    --------
    >>> outside_window = validate_date_alignment(df["date"], raster_date)
    >>> df_matched = df[~outside_window]
    """
    offset_days = (wq_dates - raster_acquisition_date).abs().dt.days
    outside_window = offset_days > max_offset_days

    n_outside = outside_window.sum()
    n_total = len(wq_dates)
    pct_outside = 100 * n_outside / n_total if n_total > 0 else 0

    if n_outside == 0:
        logger.info(
            "Date alignment PASSED — all %d records within ±%d days of "
            "raster acquisition date (%s)",
            n_total, max_offset_days, raster_acquisition_date.date(),
        )
    else:
        logger.warning(
            "Date alignment: %d / %d records (%.1f%%) fall outside the "
            "±%d-day window of raster acquisition date (%s). "
            "Inspect before proceeding.",
            n_outside, n_total, pct_outside, max_offset_days,
            raster_acquisition_date.date(),
        )

    return outside_window


# ---------------------------------------------------------------------------
# 5. WQ value range checks
# ---------------------------------------------------------------------------

def validate_wq_ranges(
    df: pd.DataFrame,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Detect out-of-range values in WQ columns based on physical plausibility.

    Values outside the plausibility bounds are NOT dropped — they may be
    valid extreme events (e.g., hypersalinity during the 2015 drought).
    Instead, the fraction of out-of-range values is reported so the analyst
    can make an informed decision about trimming or Winsorisation.

    Parameters
    ----------
    df : pd.DataFrame
        WQ DataFrame with renamed columns.
    bounds : dict, optional
        Mapping ``{column_name: (min_val, max_val)}``.
        Defaults to ``WQ_BOUNDS`` (physical plausibility for Florida Bay).

    Returns
    -------
    pd.DataFrame
        Input DataFrame unchanged (side-effect: logs warnings).

    Notes
    -----
    The full list of out-of-range values and their row indices is logged at
    DEBUG level for post-hoc audit.

    References
    ----------
    Boyer, J.N., Fourqurean, J.W., & Jones, R.D. (1997). Spatial characterization
    of water quality in Florida Bay and Whitewater Bay by multivariate analyses:
    zones of similar influence. Estuaries, 20(4), 743–758.

    Examples
    --------
    >>> df = validate_wq_ranges(df)
    """
    if bounds is None:
        bounds = WQ_BOUNDS

    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue  # Optional columns may not be present

        col_data = df[col].dropna()
        out_low = (col_data < lo).sum()
        out_high = (col_data > hi).sum()
        n_valid = len(col_data)

        if out_low > 0 or out_high > 0:
            logger.warning(
                "Range check | %-20s: %d below min (%.1f), %d above max (%.1f) "
                "out of %d non-null values",
                col, out_low, lo, out_high, hi, n_valid,
            )
        else:
            logger.debug("Range check | %-20s PASSED (%d values)", col, n_valid)

    return df


# ---------------------------------------------------------------------------
# 6. Health label validation
# ---------------------------------------------------------------------------

def validate_health_labels(
    series: pd.Series,
    expected_labels: Optional[List[str]] = None,
) -> None:
    """
    Verify that the health status column contains only expected class labels.

    Parameters
    ----------
    series : pd.Series
        The health_status column from the WQ DataFrame.
    expected_labels : list of str, optional
        Valid label strings.  Default: ``["Healthy", "Intermediate", "Not Healthy"]``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If unexpected labels are present.

    Notes
    -----
    The class distribution (n per class, % per class) is logged at INFO
    level and should be reported in the paper's Data section.

    Examples
    --------
    >>> validate_health_labels(df["health_status"])
    """
    if expected_labels is None:
        expected_labels = ["Healthy", "Intermediate", "Not Healthy"]

    unique_found = set(series.dropna().unique())
    unexpected = unique_found - set(expected_labels)

    if unexpected:
        raise ValueError(
            f"Unexpected health labels found: {unexpected}. "
            f"Expected: {expected_labels}. "
            "Update config.feature_engineering.target_encoding."
        )

    # Log class distribution — required for Methods section
    dist = series.value_counts()
    logger.info("Health label validation PASSED. Class distribution:")
    for label in expected_labels:
        n = dist.get(label, 0)
        pct = 100 * n / len(series) if len(series) > 0 else 0
        logger.info("  %-15s  n = %4d  (%.1f%%)", label, n, pct)


# ---------------------------------------------------------------------------
# 7. Feature matrix integrity
# ---------------------------------------------------------------------------

def validate_no_target_leakage(
    feature_df: pd.DataFrame,
    target_col: str = "health_status",
) -> None:
    """
    Check that the target variable is not present in the feature matrix.

    Data leakage — where the target label is directly or indirectly included
    as a predictor — is a common modelling error that produces artificially
    inflated accuracy and fails peer review scrutiny.  This check raises a
    hard error if the target column is found in the feature DataFrame.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature matrix (X) before splitting.
    target_col : str, optional
        Target column name.  Default: ``"health_status"``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``target_col`` is present in ``feature_df``.

    Examples
    --------
    >>> validate_no_target_leakage(X, target_col="health_label")
    """
    if target_col in feature_df.columns:
        raise ValueError(
            f"Target column '{target_col}' is present in the feature matrix. "
            "Remove it before passing to the model. "
            "This is a data leakage error."
        )
    logger.info("Target leakage check PASSED — '%s' not in feature matrix", target_col)


def validate_no_constant_features(
    feature_df: pd.DataFrame,
) -> List[str]:
    """
    Identify and warn about constant (zero-variance) features.

    Constant features provide no discriminative information and can cause
    numerical issues in some classifiers.  They are not dropped here — the
    analyst should verify whether a constant column indicates a data error
    — but all are logged.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature matrix.

    Returns
    -------
    list of str
        Names of columns with zero variance.

    Examples
    --------
    >>> constant_cols = validate_no_constant_features(X)
    """
    constant_cols = [
        col for col in feature_df.select_dtypes(include=[np.number]).columns
        if feature_df[col].nunique(dropna=False) <= 1
    ]
    if constant_cols:
        logger.warning(
            "Constant (zero-variance) features detected — these will be "
            "uninformative to the model: %s",
            constant_cols,
        )
    else:
        logger.info(
            "Constant-feature check PASSED — no zero-variance columns found "
            "in %d-feature matrix",
            feature_df.shape[1],
        )
    return constant_cols


def validate_missing_fractions(
    df: pd.DataFrame,
    max_missing_fraction: float = 0.40,
) -> List[str]:
    """
    Identify columns that exceed the allowable missing-data threshold.

    Columns above ``max_missing_fraction`` are flagged for removal during
    preprocessing (Stage 5).  The decision threshold (default 40%) is a
    commonly used heuristic in environmental ML studies; below this level
    median imputation is considered acceptable (Sterne et al., 2009).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (WQ or feature matrix).
    max_missing_fraction : float, optional
        Maximum fraction of NaN values allowed per column (0.0–1.0).
        Default: ``0.40``.

    Returns
    -------
    list of str
        Names of columns exceeding the threshold.

    References
    ----------
    Sterne, J.A.C. et al. (2009). Multiple imputation for missing data in
    epidemiological and clinical research: potential and pitfalls. BMJ, 338.
    https://doi.org/10.1136/bmj.b2393

    Examples
    --------
    >>> cols_to_drop = validate_missing_fractions(df, max_missing_fraction=0.4)
    """
    missing_frac = df.isnull().mean()
    high_missing = missing_frac[missing_frac > max_missing_fraction]

    logger.info("Missing value audit (threshold = %.0f%%):", max_missing_fraction * 100)
    for col, frac in missing_frac.items():
        marker = " *** EXCEEDS THRESHOLD" if frac > max_missing_fraction else ""
        logger.debug("  %-30s %.1f%% missing%s", col, frac * 100, marker)

    if not high_missing.empty:
        logger.warning(
            "%d columns exceed the %.0f%% missing-data threshold and will be "
            "dropped in preprocessing: %s",
            len(high_missing), max_missing_fraction * 100,
            list(high_missing.index),
        )

    return list(high_missing.index)
