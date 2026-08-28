"""
resample_5day.py
=================
Optional derived-data step: resample Stage 3's WQ+spectral joined table
onto a regular grid at Sentinel-2's nominal revisit cadence.

PURPOSE:
    This is NOT one of the 8 numbered pipeline stages and is never run by
    run_pipeline.py -- it produces a separate, clearly-labelled analytical
    view of the Stage 3 output for time-series work that needs evenly
    spaced samples (e.g. seasonal decomposition, smooth trend
    visualisation), without touching the original irregular-interval data.

    IMPORTANT CAVEAT: this step does not create new measurements. WQ
    stations are sampled roughly monthly, so most points on a 5-day grid
    fall between real observations and are linearly interpolated across
    gaps far larger than the grid spacing itself. Every output row carries
    `is_observed` (True only if it coincides with a real sample date) and
    `days_to_nearest_obs` (distance to the nearest real measurement) so
    downstream use can distinguish real from interpolated values rather
    than treating the resampled series as equivalent-quality data.

INPUTS:
    config["paths"]["joined_gdf"]        -- Stage 3 output (untouched)
    config["temporal_resampling"]        -- resample_freq_days,
                                             interpolation_method, max_gap_days

OUTPUTS:
    config["paths"]["resampled_5day"]    -- new parquet file; Stage 3's
        output is never modified or overwritten.

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order columns for readability: identifiers/WQ first, resampling-quality
    flags next (right after latitude/longitude), scene/spectral metadata
    and index values last.

    Parameters
    ----------
    df : pd.DataFrame
        Resampled table with the default groupby/concat column order.

    Returns
    -------
    pd.DataFrame
        Same data, columns reordered. Any column not named below (e.g. a
        WQ parameter) keeps its relative position in the leading group.
    """
    quality_cols = ["is_observed", "days_to_nearest_obs", "cloud_cover"]
    scene_spectral_cols = [
        "matched_scene_id", "matched_scene_datetime", "days_offset",
        "ndavi", "ndavi_n_pixels", "wavi", "wavi_n_pixels", "ndvi", "ndvi_n_pixels",
    ]
    trailing = [c for c in quality_cols + scene_spectral_cols if c in df.columns]
    leading = [c for c in df.columns if c not in trailing]

    # latitude/longitude anchor where the quality columns get inserted;
    # everything else in `leading` keeps its original relative order.
    insert_at = leading.index("longitude") + 1 if "longitude" in leading else len(leading)
    final_order = (
        leading[:insert_at]
        + [c for c in quality_cols if c in df.columns]
        + leading[insert_at:]
        + [c for c in scene_spectral_cols if c in df.columns]
    )
    return df[final_order]


def resample_station(
    station_df: pd.DataFrame,
    freq_days: int,
    interpolation_method: str,
    max_gap_days: int,
    interpolate_cols: list,
    nearest_attribution_cols: list,
) -> pd.DataFrame:
    """
    Resample one station's irregular time series onto a fixed-interval grid.

    Parameters
    ----------
    station_df : pd.DataFrame
        Rows for a single station, indexed by `date` (datetime64, unique,
        sorted).
    freq_days : int
        Grid spacing in days (config: temporal_resampling.resample_freq_days).
    interpolation_method : str
        Passed to ``pd.DataFrame.interpolate`` (e.g. ``"time"``).
    max_gap_days : int
        Grid points farther than this from the nearest real observation on
        either side are dropped rather than kept as a low-confidence
        interpolation.
    interpolate_cols : list of str
        Continuous physical quantities (WQ parameters, spectral indices)
        that are linearly interpolated between real observations.
    nearest_attribution_cols : list of str
        Quality/provenance columns (`cloud_cover`, `days_offset`, the
        `*_n_pixels` counts) that describe WHICH real scene/sample was
        used rather than a smoothly varying quantity -- averaging two
        different scenes' cloud cover via interpolation would be
        meaningless, so each grid point instead takes these values
        directly from whichever real observation is nearest in time.
        Remaining non-numeric columns (station_id, etc.) are treated as
        station-constant and forward/back-filled.

    Returns
    -------
    pd.DataFrame
        One row per grid point actually kept, with `is_observed` and
        `days_to_nearest_obs` added.
    """
    real_dates = station_df.index
    grid = pd.date_range(real_dates.min(), real_dates.max(), freq=f"{freq_days}D")
    union_index = real_dates.union(grid)

    combined = station_df.reindex(union_index)
    combined[interpolate_cols] = combined[interpolate_cols].interpolate(
        method=interpolation_method, limit_area="inside"
    )

    # Station-constant metadata (station_id, coordinates, etc.) -- fill
    # rather than interpolate or nearest-attribute.
    constant_cols = [
        c for c in station_df.columns
        if c not in interpolate_cols and c not in nearest_attribution_cols
    ]
    combined[constant_cols] = combined[constant_cols].ffill().bfill()

    nearest_real_date = [real_dates[np.abs((real_dates - d).to_numpy()).argmin()] for d in grid]
    grid_only = combined.loc[grid].copy()
    for col in nearest_attribution_cols:
        grid_only[col] = [station_df.at[d, col] for d in nearest_real_date]

    grid_only["is_observed"] = grid.isin(real_dates)
    grid_only["days_to_nearest_obs"] = [
        int(np.abs((real_dates - d).to_numpy()).min() / np.timedelta64(1, "D"))
        for d in grid
    ]

    return grid_only[grid_only["days_to_nearest_obs"] <= max_gap_days]


def run(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the 5-day-resampled derived table from Stage 3's output.

    Parameters
    ----------
    config : dict
        Parsed configuration (see ``config/config.yaml``).

    Returns
    -------
    pd.DataFrame
        Resampled table (geometry dropped -- grid points don't have their
        own real coordinates the way observed rows do), also written to
        ``config["paths"]["resampled_5day"]``. The original
        ``config["paths"]["joined_gdf"]`` file is never modified.
    """
    tr_config = config["temporal_resampling"]
    freq_days = tr_config["resample_freq_days"]
    interpolation_method = tr_config["interpolation_method"]
    max_gap_days = tr_config["max_gap_days"]

    gdf = gpd.read_file(config["paths"]["joined_gdf"])
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["date"] = pd.to_datetime(df["date"])

    # Quality/provenance columns describe WHICH real scene/sample produced
    # a row (which satellite pass, how cloudy it was, how many pixels went
    # into a buffer average) rather than a smoothly varying quantity --
    # interpolating "20% cloud" and "5% cloud" into "12.5%" for a synthetic
    # date would be meaningless, since no such scene exists. These get
    # attributed from the nearest real observation instead (see
    # resample_station's nearest_attribution_cols).
    nearest_attribution_cols = [
        c for c in df.columns if c == "cloud_cover" or c == "days_offset" or c.endswith("_n_pixels")
    ]
    interpolate_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in nearest_attribution_cols
    ]

    logger.info(
        "Resampling %d stations' worth of records onto a %d-day grid (method=%s)",
        df["station_id"].nunique(), freq_days, interpolation_method,
    )

    resampled = []
    for station_id, group in df.groupby("station_id"):
        station_df = group.set_index("date").sort_index()
        station_df = station_df[~station_df.index.duplicated(keep="first")]
        resampled.append(
            resample_station(
                station_df, freq_days, interpolation_method, max_gap_days,
                interpolate_cols, nearest_attribution_cols,
            )
        )

    result = pd.concat(resampled).reset_index(names="date")
    result = reorder_columns(result)

    n_observed = result["is_observed"].sum()
    logger.info(
        "Resampled table: %d rows total, %d (%.1f%%) coincide with a real observation, "
        "%d (%.1f%%) are interpolated",
        len(result), n_observed, 100 * n_observed / len(result),
        len(result) - n_observed, 100 * (1 - n_observed / len(result)),
    )

    out_path = Path(config["paths"]["resampled_5day"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    logger.info("Saved resampled table -> %s (original joined_gdf untouched)", out_path)

    return result
