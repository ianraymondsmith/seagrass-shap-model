"""
spatial_join_03.py
===================
Stage 3 of the Seagrass Health Predictive Pipeline: match each water-quality
sample to its nearest-in-time Sentinel-2 scene and extract buffered-mean
spectral index values at the station's location.

PURPOSE:
    Stage 1 produces one row per WQ sample event (station x date). Stage 2
    produces one set of NDAVI/WAVI/NDVI rasters per Sentinel-2 scene
    (roughly one per ~30-day window). This stage bridges them: for each WQ
    row, find the scene whose acquisition date is closest in time (within
    the configured tolerance), then extract the mean index value within a
    circular buffer around the station's coordinates -- the buffer
    accounts for GPS uncertainty and sub-pixel variability rather than
    trusting a single pixel's value (Roelfsema et al., 2014).

INPUTS:
    config["paths"]["wq_clean"]                -- Stage 1 output
    config["paths"]["spectral_indices_manifest"] -- Stage 2 output
    config["spatial_join"]                      -- buffer_radius_m, projected_crs
    config["ingestion"]["max_date_offset_days"]  -- temporal matching window
    config["ingestion"]["wq_crs"]               -- WQ point CRS (WGS84)
    config["spectral_index"]                    -- compute list, nodata_value,
                                                    output_scale_factor

OUTPUTS:
    config["paths"]["joined_gdf"] -- GeoPackage with all WQ columns plus,
        per index: the buffered-mean value, the pixel count averaged, the
        matched scene ID/date, and the day offset between sample and scene.

REFERENCES:
    Roelfsema, C. et al. (2014). Challenges of remote sensing for
    quantifying changes in large complex seagrass environments.
    Estuarine, Coastal and Shelf Science, 133, 161-171.

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from pyproj import Transformer
from shapely.geometry import Point

from src.utils.logging_config import get_logger
from src.utils.validation import validate_crs_match

logger = get_logger(__name__)


def load_scene_manifest(path: str) -> pd.DataFrame:
    """
    Load the Stage 2 spectral-indices manifest.

    Parameters
    ----------
    path : str
        Path to ``spectral_indices_manifest.csv``.

    Returns
    -------
    pd.DataFrame
        Manifest with `datetime` parsed to a timezone-naive Timestamp
        (matching the WQ data's naive dates).
    """
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df


def match_nearest_scene(
    wq_date: Any, manifest: pd.DataFrame, max_offset_days: int
) -> Optional[pd.Series]:
    """
    Find the manifest scene whose datetime is closest to a WQ sample date.

    Parameters
    ----------
    wq_date : date-like
        The WQ sample's collection date.
    manifest : pd.DataFrame
        Scene manifest (see `load_scene_manifest`).
    max_offset_days : int
        Maximum allowable |offset| in days; beyond this, no match is
        returned (config: ingestion.max_date_offset_days).

    Returns
    -------
    pd.Series or None
        The matched manifest row, or None if no scene falls within the
        tolerance window.
    """
    offsets = (manifest["datetime"] - pd.Timestamp(wq_date)).abs()
    within = offsets <= pd.Timedelta(days=max_offset_days)
    if not within.any():
        return None
    return manifest.loc[offsets[within].idxmin()]


def buffer_mean_from_dataset(
    src: rasterio.io.DatasetReader,
    transformer: Transformer,
    lon: float,
    lat: float,
    buffer_radius_m: float,
    nodata_value: float,
    scale_factor: float,
) -> Tuple[Optional[float], int]:
    """
    Extract the mean index value within a circular buffer around a point.

    Reads only the small pixel window covering the buffer (not the full
    raster), reprojects the query point into the raster's CRS, and averages
    valid (non-nodata) pixels whose center falls within `buffer_radius_m`
    of the point -- a true circular buffer, not just the bounding window.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Already-open raster (one index, one scene).
    transformer : pyproj.Transformer
        Transformer from the WQ point CRS to `src.crs`, reused across calls
        against the same raster for efficiency.
    lon, lat : float
        Station coordinates in the WQ point CRS (WGS84 decimal degrees).
    buffer_radius_m : float
        Buffer radius in metres (config: spatial_join.buffer_radius_m).
    nodata_value : float
        Sentinel value for masked/invalid pixels (already in scaled int16
        units, matching the raster's stored nodata).
    scale_factor : float
        Divides the raw stored (scaled int16) mean back to true index
        units (config: spectral_index.output_scale_factor).

    Returns
    -------
    mean_value : float or None
        Mean index value in true (unscaled) units, or None if the buffer
        contains no valid pixels (e.g. entirely cloud-masked, or the
        station falls outside the raster's extent).
    n_pixels : int
        Number of valid pixels averaged.
    """
    x, y = transformer.transform(lon, lat)

    window = rasterio.windows.from_bounds(
        x - buffer_radius_m, y - buffer_radius_m,
        x + buffer_radius_m, y + buffer_radius_m,
        transform=src.transform,
    ).round_offsets().round_lengths()
    raster_window = rasterio.windows.Window(0, 0, src.width, src.height)

    # A station can fall entirely outside this scene's tile -- Florida Bay
    # spans multiple Sentinel-2 UTM tiles, and Stage 2 picked one
    # lowest-cloud scene per time window without guaranteeing it covers
    # every station. rasterio raises rather than returning an empty
    # window in this case, so treat that as "no data here" rather than a
    # fatal error.
    try:
        window = window.intersection(raster_window)
    except rasterio.errors.WindowError:
        return None, 0

    if window.width <= 0 or window.height <= 0:
        return None, 0

    arr = src.read(1, window=window)
    win_transform = src.window_transform(window)

    rows, cols = np.indices(arr.shape)
    xs, ys = rasterio.transform.xy(win_transform, rows.ravel(), cols.ravel())
    dist = np.sqrt((np.array(xs) - x) ** 2 + (np.array(ys) - y) ** 2).reshape(arr.shape)

    valid = (arr != nodata_value) & (dist <= buffer_radius_m)
    if not valid.any():
        return None, 0

    return float(arr[valid].mean()) / scale_factor, int(valid.sum())


def run(config: Dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Execute Stage 3: temporal matching + buffered spatial extraction.

    Parameters
    ----------
    config : dict
        Parsed configuration (see ``config/config.yaml``).

    Returns
    -------
    geopandas.GeoDataFrame
        WQ records with matched-scene metadata and buffered-mean spectral
        index columns, also written to ``config["paths"]["joined_gdf"]``.
    """
    paths = config["paths"]
    sj_config = config["spatial_join"]
    ing_config = config["ingestion"]
    sp_config = config["spectral_index"]

    index_names = sp_config["compute"]
    buffer_radius_m = sj_config["buffer_radius_m"]
    wq_crs = ing_config["wq_crs"]
    max_offset_days = ing_config["max_date_offset_days"]
    nodata_value = sp_config["nodata_value"]
    scale_factor = sp_config["output_scale_factor"]

    wq = pd.read_parquet(paths["wq_clean"]).reset_index(drop=True)
    wq["date"] = pd.to_datetime(wq["date"])
    manifest = load_scene_manifest(paths["spectral_indices_manifest"])
    manifest_indexed = manifest.set_index("scene_id")

    logger.info("Matching %d WQ records to nearest scene within +/-%d days", len(wq), max_offset_days)

    matches = [match_nearest_scene(d, manifest, max_offset_days) for d in wq["date"]]
    wq["matched_scene_id"] = [s["scene_id"] if s is not None else None for s in matches]
    wq["matched_scene_datetime"] = [s["datetime"] if s is not None else pd.NaT for s in matches]
    wq["days_offset"] = (wq["matched_scene_datetime"] - wq["date"]).abs().dt.days

    n_unmatched = wq["matched_scene_id"].isna().sum()
    if n_unmatched:
        logger.warning(
            "%d / %d WQ records have no Sentinel-2 scene within +/-%d days -- "
            "spectral columns will be NaN for these",
            n_unmatched, len(wq), max_offset_days,
        )

    for name in index_names:
        wq[name] = np.nan
        wq[f"{name}_n_pixels"] = 0

    crs_checked = False
    n_extracted = 0
    n_empty_buffer = 0

    for scene_id, group in wq[wq["matched_scene_id"].notna()].groupby("matched_scene_id"):
        scene_row = manifest_indexed.loc[scene_id]
        for name in index_names:
            raster_path = scene_row[f"{name}_path"]
            with rasterio.open(raster_path) as src:
                if not crs_checked:
                    validate_crs_match(str(sj_config["projected_crs"]), str(src.crs))
                    crs_checked = True
                transformer = Transformer.from_crs(wq_crs, src.crs, always_xy=True)
                for idx in group.index:
                    mean_val, n_px = buffer_mean_from_dataset(
                        src, transformer,
                        wq.at[idx, "longitude"], wq.at[idx, "latitude"],
                        buffer_radius_m, nodata_value, scale_factor,
                    )
                    wq.at[idx, name] = mean_val
                    wq.at[idx, f"{name}_n_pixels"] = n_px
                    n_extracted += 1
                    if mean_val is None:
                        n_empty_buffer += 1

    if n_empty_buffer:
        logger.warning(
            "%d / %d station-scene-index extractions had zero valid pixels "
            "in the buffer (cloud-masked or outside raster extent)",
            n_empty_buffer, n_extracted,
        )

    geometry = [Point(lon, lat) for lon, lat in zip(wq["longitude"], wq["latitude"])]
    gdf = gpd.GeoDataFrame(wq, geometry=geometry, crs=wq_crs)

    out_path = Path(paths["joined_gdf"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")

    n_matched_rows = len(wq) - n_unmatched
    logger.info(
        "Stage 3 complete -- %d/%d records matched to a scene, saved -> %s",
        n_matched_rows, len(wq), out_path,
    )
    return gdf
