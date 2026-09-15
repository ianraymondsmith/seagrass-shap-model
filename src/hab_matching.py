"""
hab_matching.py
================
Match FWC/NOAA HABSOS harmful algal bloom observations to this project's
Florida Bay bounding box, study period, DBHYDRO station network, and
5-day resampling grid.

PURPOSE:
    HABSOS (Harmful Algal BloomS Observing System) has no fixed station
    network like DBHYDRO or SEACAR -- 252 unique lat/lon pairs across 589
    Florida-Bay-bbox observations, each essentially a one-off sampling
    point named by free-text location description, not a station ID. To
    resample it onto the same per-station 5-day grid as everything else,
    each observation is assigned to its nearest DBHYDRO station (same
    nearest-neighbour approach already used and validated in the SEACAR
    WQ proxy check).

    Cell count is heavily zero-inflated with rare extreme spikes (in this
    export: median 0, max 103,000 cells/L) -- linearly interpolating
    between a real zero and a real bloom spike would fabricate a smooth
    ramp into a bloom that never happened. So cell count (and the
    categorical/QA columns describing it) are carried forward from the
    nearest real observation via `resample_5day.resample_station`'s
    nearest-attribution mechanism, not interpolated. Salinity and water
    temperature -- the two smoothly-varying ancillary fields HABSOS also
    reports -- ARE linearly interpolated, same as the main WQ pipeline.

INPUTS:
    data/raw/habsos_20240430.csv     -- raw HABSOS export (all Gulf states)
    data/interim/wq_florida_bay_clean.parquet -- DBHYDRO station coordinates

OUTPUTS:
    data/interim/hab_florida_bay.parquet         -- filtered + station-
        assigned raw observations (one row per real HABSOS sample)
    data/interim/hab_resampled_5day.parquet      -- resampled onto the
        same 5-day grid as the main WQ+spectral pipeline

REFERENCES:
    HABSOS data dictionary: data/raw/HAB 0120767.8.8/.../
    HABSOS_Data_Dictionary.xlsx (FWC-FWRI / NOAA NCCOS)

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger
from src.utils.validation import FLORIDA_BAY_BOUNDS
from src.resample_5day import resample_station

logger = get_logger(__name__)

# Matches the study period used throughout this pipeline (DBHYDRO export,
# Sentinel-2 acquisition) -- not itself a HABSOS-specific constant.
DATE_MIN = "2016-01-01"
DATE_MAX = "2026-01-01"

RAW_HABSOS_PATH = "data/raw/habsos_20240430.csv"


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) from one point to an array of points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_and_filter_hab(
    raw_path: str = RAW_HABSOS_PATH,
    bounds: Dict[str, float] = FLORIDA_BAY_BOUNDS,
    date_min: str = DATE_MIN,
    date_max: str = DATE_MAX,
) -> pd.DataFrame:
    """
    Load the raw HABSOS export and filter to the Florida Bay bbox + study period.

    Parameters
    ----------
    raw_path : str
        Path to the raw HABSOS CSV.
    bounds : dict
        Bounding box with lat_min/lat_max/lon_min/lon_max keys.
        Default: ``FLORIDA_BAY_BOUNDS``.
    date_min, date_max : str
        Study period bounds (inclusive), matching the rest of this pipeline.

    Returns
    -------
    pd.DataFrame
        Filtered HABSOS rows, `SAMPLE_DATE` parsed to datetime.
    """
    df = pd.read_csv(raw_path, low_memory=False)
    df["SAMPLE_DATE"] = pd.to_datetime(df["SAMPLE_DATE"], format="%Y/%m/%d", errors="coerce")

    in_bbox = df[
        df["LATITUDE"].between(bounds["lat_min"], bounds["lat_max"])
        & df["LONGITUDE"].between(bounds["lon_min"], bounds["lon_max"])
        & df["SAMPLE_DATE"].between(date_min, date_max)
    ].copy()

    logger.info(
        "HABSOS: %d / %d rows in Florida Bay bbox + study period (%s to %s)",
        len(in_bbox), len(df), date_min, date_max,
    )
    return in_bbox


def assign_nearest_station(hab_df: pd.DataFrame, wq_stations: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each HABSOS observation to its nearest DBHYDRO station.

    HABSOS has no fixed station network of its own (252 unique lat/lon
    pairs across 589 bbox observations) -- this borrows the same
    nearest-neighbour approach already validated for SEACAR WQ proxying,
    so downstream resampling can reuse the per-station 5-day grid logic.

    Parameters
    ----------
    hab_df : pd.DataFrame
        Filtered HABSOS rows with LATITUDE/LONGITUDE columns.
    wq_stations : pd.DataFrame
        Must have station_id, latitude, longitude columns (one row per
        DBHYDRO station).

    Returns
    -------
    pd.DataFrame
        `hab_df` with `station_id` and `dist_to_station_km` columns added.
    """
    stations = wq_stations.drop_duplicates("station_id")[["station_id", "latitude", "longitude"]].reset_index(drop=True)

    nearest_station, nearest_dist = [], []
    for _, row in hab_df.iterrows():
        dists = haversine_km(row["LATITUDE"], row["LONGITUDE"], stations["latitude"].values, stations["longitude"].values)
        idx = int(np.argmin(dists))
        nearest_station.append(stations.iloc[idx]["station_id"])
        nearest_dist.append(dists[idx])

    hab_df = hab_df.copy()
    hab_df["station_id"] = nearest_station
    hab_df["dist_to_station_km"] = nearest_dist

    logger.info(
        "Nearest-station assignment: mean distance %.2f km (median %.2f km, max %.2f km)",
        np.mean(nearest_dist), np.median(nearest_dist), np.max(nearest_dist),
    )
    return hab_df


def build_clean_table(hab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename/select HABSOS columns to this pipeline's naming convention and
    aggregate to one row per station-date.

    Multiple HABSOS observations can be assigned to the same station on
    the same date (several nearby one-off sampling points all snapping to
    one DBHYDRO station); cell count is aggregated by MAX (not mean) so a
    real bloom reading is never diluted by co-occurring zero readings from
    other nearby points, while salinity/temperature use mean.

    Parameters
    ----------
    hab_df : pd.DataFrame
        Output of `assign_nearest_station`.

    Returns
    -------
    pd.DataFrame
        One row per (station_id, date), columns: station_id, date,
        cellcount, category, salinity_ppt, temp_C, n_observations,
        dist_to_station_km (mean, if multiple source points collapsed).
    """
    df = hab_df.copy()
    df["date"] = df["SAMPLE_DATE"].dt.date

    # CATEGORY is ordinal-ish text ("not observed" < "very low" < ... <
    # "high"); keep whichever row has the max CELLCOUNT as the
    # representative category for that station-date.
    idx_max_cellcount = df.groupby(["station_id", "date"])["CELLCOUNT"].idxmax()
    category_by_group = df.loc[idx_max_cellcount].set_index(["station_id", "date"])["CATEGORY"]

    agg = df.groupby(["station_id", "date"]).agg(
        cellcount=("CELLCOUNT", "max"),
        salinity_ppt=("SALINITY", "mean"),
        temp_C=("WATER_TEMP", "mean"),
        dist_to_station_km=("dist_to_station_km", "mean"),
        n_observations=("CELLCOUNT", "size"),
    )
    agg["category"] = category_by_group
    agg = agg.reset_index()
    agg["date"] = pd.to_datetime(agg["date"])

    return agg


def run(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Execute the full HAB matching + resampling pipeline.

    Parameters
    ----------
    config : dict
        Parsed configuration (see ``config/config.yaml``); reuses
        `temporal_resampling.resample_freq_days`/`interpolation_method`
        from the main pipeline, but applies its own
        `temporal_resampling.hab_max_gap_days` (tighter than the shared
        `max_gap_days`) since bloom dynamics shift faster than the
        weekly-to-monthly timescale that justifies 30 days for WQ.

    Returns
    -------
    dict of str -> pd.DataFrame
        ``{"raw": <station-date table>, "resampled": <5-day grid table>}``,
        also written to `data/interim/hab_florida_bay.parquet` and
        `data/interim/hab_resampled_5day.parquet`.
    """
    wq = pd.read_parquet(config["paths"]["wq_clean"])

    hab = load_and_filter_hab()
    hab = assign_nearest_station(hab, wq)
    clean = build_clean_table(hab)

    raw_out = Path("data/interim/hab_florida_bay.parquet")
    clean.to_parquet(raw_out, index=False)
    logger.info("Saved HAB station-date table -> %s (%d rows)", raw_out, len(clean))

    tr_config = config["temporal_resampling"]
    interpolate_cols = ["salinity_ppt", "temp_C"]
    nearest_attribution_cols = ["cellcount", "category", "n_observations", "dist_to_station_km"]

    # HAB-specific override, tighter than the shared 30-day WQ/spectral
    # standard -- see temporal_resampling.hab_max_gap_days in config.yaml
    # for the rationale (K. brevis blooms shift over days-to-weeks, not
    # the weekly-to-monthly timescale that justifies 30 days elsewhere).
    hab_max_gap_days = tr_config["hab_max_gap_days"]

    resampled = []
    for station_id, group in clean.groupby("station_id"):
        station_df = group.set_index("date").sort_index()
        station_df = station_df[~station_df.index.duplicated(keep="first")]
        resampled.append(
            resample_station(
                station_df, tr_config["resample_freq_days"], tr_config["interpolation_method"],
                hab_max_gap_days, interpolate_cols, nearest_attribution_cols,
            )
        )
    resampled_df = pd.concat(resampled).reset_index(names="date")

    resampled_out = Path("data/interim/hab_resampled_5day.parquet")
    resampled_df.to_parquet(resampled_out, index=False)
    logger.info(
        "Saved HAB resampled table -> %s (%d rows, %d stations)",
        resampled_out, len(resampled_df), clean["station_id"].nunique(),
    )

    return {"raw": clean, "resampled": resampled_df}
