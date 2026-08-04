"""
ingest_01.py
============
Stage 1 of the Seagrass Health Predictive Pipeline: load and validate the
SFWMD DBHYDRO water-quality export.

PURPOSE:
    DBHYDRO Insights exports are LONG format -- one row per
    station x collectDate x parameter measurement -- rather than one row
    per station-date with a column per parameter. This module filters the
    raw export down to the relevant water body and sample type, corrects
    a longitude sign convention, pivots to wide format, renames columns
    to the pipeline's internal schema, and validates the result.

INPUTS:
    config["paths"]["wq_csv"]        -- raw DBHYDRO CSV (long format)
    config["ingestion"]              -- filter/rename parameters (see config.yaml)

OUTPUTS:
    config["paths"]["wq_clean"]      -- cleaned, wide-format WQ table (parquet)

NOTES:
    The target label (health_status) is NOT present in DBHYDRO data and is
    therefore not validated or required here. It is merged in from a
    separate labelled source (e.g. SEACAR) at Stage 4 and validated there.

REFERENCES:
    Wilson, G. et al. (2017). Good enough practices in scientific computing.
    PLOS Computational Biology, 13(6), e1005510.

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.logging_config import get_logger
from src.utils.validation import (
    MAX_PLAUSIBLE_DEPTH_M,
    validate_coordinates,
    validate_missing_fractions,
    validate_wq_columns,
    validate_wq_ranges,
)

logger = get_logger(__name__)


def load_raw_wq(path: str) -> pd.DataFrame:
    """
    Load the raw DBHYDRO CSV export, skipping its leading comment block.

    Parameters
    ----------
    path : str
        Path to the raw DBHYDRO CSV (long format).

    Returns
    -------
    pd.DataFrame
        Raw long-format WQ data, one row per station x date x parameter.
    """
    df = pd.read_csv(path, comment="#", low_memory=False)
    logger.info("Loaded raw DBHYDRO export: %d rows, %d columns", len(df), len(df.columns))
    return df


def filter_matrix_and_sample_type(
    df: pd.DataFrame,
    matrix_filter: str,
    sample_type_filter: str,
) -> pd.DataFrame:
    """
    Keep only the water-body matrix and sample type relevant to this study.

    DBHYDRO's `matrix` field distinguishes water body types under one
    export (e.g. `SA` = saltwater/estuarine ambient vs `SW` = freshwater
    inland canal, `GW` = groundwater); `sampleType` distinguishes real
    samples (`SAMP`) from QC replicates/duplicates (`RS`, `FD`) and blanks
    (`EB`, `FB`, `FCEB`). Both must be filtered before pivoting, otherwise
    QC records would be double-counted as independent observations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format WQ data.
    matrix_filter : str
        Value of `matrix` to keep (e.g. ``"SA"``).
    sample_type_filter : str
        Value of `sampleType` to keep (e.g. ``"SAMP"``).

    Returns
    -------
    pd.DataFrame
        Filtered subset.
    """
    n_before = len(df)
    df = df[(df["matrix"] == matrix_filter) & (df["sampleType"] == sample_type_filter)].copy()
    logger.info(
        "Matrix/sampleType filter (matrix=%s, sampleType=%s): %d / %d rows retained",
        matrix_filter, sample_type_filter, len(df), n_before,
    )
    return df


def fix_longitude_sign(df: pd.DataFrame, longitude_stored_as_positive: bool) -> pd.DataFrame:
    """
    Correct DBHYDRO's unsigned longitude convention to signed decimal degrees.

    DBHYDRO stores longitude as an unsigned magnitude (implicitly
    "degrees West") rather than a signed WGS84 value -- e.g. ``80.52``
    instead of ``-80.52``. Left uncorrected, every Florida Bay bounding-box
    check silently excludes all rows, since the raw values never fall
    within a negative-longitude range.

    Parameters
    ----------
    df : pd.DataFrame
        WQ data with a `longitude` column.
    longitude_stored_as_positive : bool
        If True, apply ``longitude = -abs(longitude)``.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with `longitude` corrected.
    """
    df = df.copy()
    if longitude_stored_as_positive:
        df["longitude"] = -df["longitude"].abs()
        logger.info("Applied longitude sign correction (stored as unsigned West magnitude)")
    return df


def flag_implausible_depths(df: pd.DataFrame, max_depth_m: float = MAX_PLAUSIBLE_DEPTH_M) -> pd.DataFrame:
    """
    Log (without dropping) rows whose sample depth exceeds a plausible bound.

    Florida Bay is a shallow lagoon (typically <2 m); a reported depth
    beyond `max_depth_m` is almost certainly a unit or entry error (e.g.
    feet mislabeled as metres) rather than a real measurement. Following
    the rest of this pipeline's validation philosophy, suspect rows are
    logged for analyst review rather than silently dropped or corrected.

    Parameters
    ----------
    df : pd.DataFrame
        WQ data with a `depth` column (metres).
    max_depth_m : float, optional
        Plausibility threshold. Default: ``MAX_PLAUSIBLE_DEPTH_M``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame, unchanged.
    """
    suspect = df["depth"] > max_depth_m
    if suspect.any():
        logger.warning(
            "%d rows have depth > %.1f m (implausible for Florida Bay) -- "
            "stations/dates: %s",
            suspect.sum(), max_depth_m,
            df.loc[suspect, ["station", "collectDate", "depth"]].to_dict("records"),
        )
    return df


def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape long-format (station x date x parameter) rows into one row per
    station-date sample event, with one column per parameter.

    Duplicate (station, date, parameter) combinations -- e.g. from repeat
    depths within the same site visit -- are averaged rather than raising
    an error, since they represent the same sampling event rather than
    independent observations.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered, sign-corrected long-format WQ data.

    Returns
    -------
    pd.DataFrame
        Wide-format table: one row per (station, date), columns = raw
        DBHYDRO `parameter` names plus `station`, `date`, `latitude`,
        `longitude`.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["collectDate"]).dt.date

    group_cols = ["station", "date"]
    wide = df.pivot_table(index=group_cols, columns="parameter", values="value", aggfunc="mean")
    wide = wide.reset_index()

    latlon = df.groupby(group_cols)[["latitude", "longitude"]].mean().reset_index()
    wide = wide.merge(latlon, on=group_cols, how="left")
    wide = wide.rename(columns={"station": "station_id"})

    logger.info(
        "Pivoted to wide format: %d sample events (station x date), %d parameter columns",
        len(wide), wide.shape[1] - 4,
    )
    return wide


def run(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute Stage 1: load, filter, pivot, rename, and validate WQ data.

    Parameters
    ----------
    config : dict
        Parsed configuration (see ``config/config.yaml``).

    Returns
    -------
    pd.DataFrame
        Cleaned, wide-format WQ table, also written to
        ``config["paths"]["wq_clean"]``.
    """
    ing = config["ingestion"]

    df = load_raw_wq(config["paths"]["wq_csv"])
    df = filter_matrix_and_sample_type(df, ing["matrix_filter"], ing["sample_type_filter"])
    df = fix_longitude_sign(df, ing["longitude_stored_as_positive"])
    df = flag_implausible_depths(df, ing["max_plausible_depth_m"])

    wide = pivot_to_wide(df)
    wide = wide.rename(columns=ing["column_rename_map"])

    wide = validate_coordinates(wide)
    validate_wq_columns(wide, ing["required_wq_columns"])
    wide = validate_wq_ranges(wide)
    validate_missing_fractions(wide)

    out_path = Path(config["paths"]["wq_clean"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_path, index=False)
    logger.info("Stage 1 complete -- saved %d rows to %s", len(wide), out_path)

    return wide
