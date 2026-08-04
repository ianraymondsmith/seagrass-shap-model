"""
spectral_index_02.py
=====================
Stage 2 of the Seagrass Health Predictive Pipeline: compute NDAVI, WAVI,
and NDVI spectral indices from downloaded Sentinel-2 L2A band imagery.

PURPOSE:
    Reads the Blue (B02), Red (B04), and NIR (B08) 10 m bands for each
    scene listed in the Stage-0 acquisition manifest
    (scripts/download_sentinel2.py), converts them to surface reflectance,
    masks physically-invalid pixels, and computes three vegetation
    indices per scene. NDAVI and WAVI substitute the blue band for red
    relative to standard NDVI, which reduces sensitivity to the
    overlying water column and improves discrimination of submerged
    aquatic vegetation in optically shallow water (Villa et al., 2013;
    Roelfsema et al., 2014) -- this is why all three are computed rather
    than NDVI alone.

INPUTS:
    config["paths"]["sentinel2_manifest"]  -- scene_id/datetime/cloud_cover
    config["paths"]["sentinel2_dir"]       -- <scene_id>/{B02,B04,B08}.jp2
    config["spectral_index"]               -- wavi_L, nodata_value,
                                               reflectance scale/valid range

OUTPUTS:
    config["paths"]["spectral_indices_dir"]/<scene_id>/{ndavi,wavi,ndvi}.tif
    config["paths"]["spectral_indices_manifest"] -- scene_id, datetime,
        cloud_cover, and the three output raster paths, for Stage 3's
        temporal matching.

FORMULAS:
    NDAVI = (NIR - Blue) / (NIR + Blue)                  (Villa et al., 2013)
    WAVI  = (1 + L)(NIR - Blue) / (NIR + Blue + L)        (Villa et al., 2014)
    NDVI  = (NIR - Red) / (NIR + Red)                     (Rouse et al., 1974)

REFERENCES:
    Rouse, J.W. et al. (1974). Monitoring vegetation systems in the Great
    Plains with ERTS. NASA SP-351, 309-317.

    Villa, P. et al. (2013). A rule-based approach for mapping macrophyte
    communities using multi-temporal aquatic vegetation indices. Remote
    Sensing of Environment, 133, 148-160.

    Villa, P. et al. (2014). A rule-based approach for mapping macrophyte
    communities using multi-temporal aquatic vegetation indices. Remote
    Sensing, 6(8), 7181-7202.

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import rasterio

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BAND_FILENAMES = {"blue": "B02.jp2", "red": "B04.jp2", "nir": "B08.jp2"}


def load_band_as_reflectance(
    path: Path,
    scale_factor: float,
    valid_min: int,
    valid_max: int,
    nodata_value: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a single Sentinel-2 band and convert it to masked surface reflectance.

    Parameters
    ----------
    path : Path
        Path to the band's `.jp2` file.
    scale_factor : float
        Multiplier converting the raw integer DN to reflectance (0-1).
    valid_min, valid_max : int
        Raw DN range considered physically plausible; pixels outside this
        range (e.g. cloud/glint saturation) are masked rather than
        silently included, since they would distort the index formulas.
    nodata_value : float
        Sentinel value written for masked pixels.

    Returns
    -------
    np.ndarray (float32)
        Reflectance array, same shape as the source band.
    dict
        Rasterio profile from the source band, used to georeference the
        output index rasters identically.
    """
    with rasterio.open(path) as src:
        raw = src.read(1)
        profile = src.profile

    valid_mask = (raw >= valid_min) & (raw <= valid_max)
    reflectance = np.where(valid_mask, raw.astype(np.float32) * scale_factor, nodata_value)
    return reflectance, profile


def compute_indices(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    wavi_l: float,
    nodata_value: float,
) -> Dict[str, np.ndarray]:
    """
    Compute NDAVI, WAVI, and NDVI from co-registered reflectance arrays.

    A pixel is masked in the output if it was masked (nodata) in any input
    band, or if the corresponding index denominator is zero (which would
    otherwise produce a spurious +/-inf rather than a genuine index value).

    Parameters
    ----------
    blue, red, nir : np.ndarray (float32)
        Reflectance arrays from `load_band_as_reflectance`, same shape.
    wavi_l : float
        WAVI soil-adjustment parameter (config: spectral_index.wavi_L).
    nodata_value : float
        Sentinel value for masked pixels, matching the input arrays'.

    Returns
    -------
    dict of str -> np.ndarray
        Keys "ndavi", "wavi", "ndvi".
    """
    valid = (blue != nodata_value) & (red != nodata_value) & (nir != nodata_value)

    with np.errstate(divide="ignore", invalid="ignore"):
        nir_blue_sum = nir + blue
        ndavi = np.where(
            valid & (nir_blue_sum != 0), (nir - blue) / nir_blue_sum, nodata_value
        )

        wavi_denom = nir + blue + wavi_l
        wavi = np.where(
            valid & (wavi_denom != 0),
            (1 + wavi_l) * (nir - blue) / wavi_denom,
            nodata_value,
        )

        nir_red_sum = nir + red
        ndvi = np.where(
            valid & (nir_red_sum != 0), (nir - red) / nir_red_sum, nodata_value
        )

    return {"ndavi": ndavi, "wavi": wavi, "ndvi": ndvi}


def process_scene(
    scene_id: str,
    scenes_dir: Path,
    output_dir: Path,
    sp_config: Dict[str, Any],
) -> Dict[str, str]:
    """
    Compute and write NDAVI/WAVI/NDVI rasters for one Sentinel-2 scene.

    Parameters
    ----------
    scene_id : str
        Scene identifier, matching a subfolder of `scenes_dir`.
    scenes_dir : Path
        Root directory containing per-scene band subfolders.
    output_dir : Path
        Root directory to write per-scene index-raster subfolders into.
    sp_config : dict
        ``config["spectral_index"]`` section.

    Returns
    -------
    dict of str -> str
        Output raster paths keyed by index name ("ndavi", "wavi", "ndvi").

    Raises
    ------
    FileNotFoundError
        If any of the three required band files is missing.
    ValueError
        If the three bands do not share the same array shape (they should,
        since all three were requested at 10 m resolution).
    """
    scene_dir = scenes_dir / scene_id
    band_paths = {b: scene_dir / fname for b, fname in BAND_FILENAMES.items()}
    missing = [str(p) for p in band_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing band file(s) for scene {scene_id}: {missing}")

    bands = {}
    profile = None
    for name, path in band_paths.items():
        bands[name], profile = load_band_as_reflectance(
            path,
            sp_config["reflectance_scale_factor"],
            sp_config["reflectance_valid_min"],
            sp_config["reflectance_valid_max"],
            sp_config["nodata_value"],
        )

    shapes = {name: arr.shape for name, arr in bands.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Band shape mismatch for scene {scene_id}: {shapes}")

    indices = compute_indices(
        bands["blue"], bands["red"], bands["nir"],
        sp_config["wavi_L"], sp_config["nodata_value"],
    )

    nodata_value = sp_config["nodata_value"]
    scale_factor = sp_config["output_scale_factor"]

    out_profile = profile.copy()
    out_profile.update(
        dtype="int16",
        count=1,
        nodata=nodata_value,
        driver="GTiff",
        compress=sp_config["output_compression"],
        predictor=2,  # horizontal differencing -- standard GDAL practice that
                      # improves DEFLATE ratios on integer rasters, lossless
        tiled=True,
    )

    scene_out_dir = output_dir / scene_id
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {}
    for index_name in sp_config["compute"]:
        arr = indices[index_name]
        # Scale to int16 for storage (~4-8x smaller than float32 -- an
        # uncompressed float32 run of all 120 scenes filled a 465 GB disk).
        # Masked pixels are re-applied as the integer nodata sentinel
        # *after* scaling, since scaling the float nodata value itself
        # would produce a nonsense integer rather than the sentinel.
        scaled = np.where(
            arr == nodata_value, nodata_value, np.round(arr * scale_factor)
        ).astype(np.int16)

        out_path = scene_out_dir / f"{index_name}.tif"
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(scaled, 1)
        out_paths[index_name] = str(out_path)

    return out_paths


def run(config: Dict[str, Any]) -> list:
    """
    Execute Stage 2: compute spectral indices for every scene in the
    Sentinel-2 acquisition manifest.

    Parameters
    ----------
    config : dict
        Parsed configuration (see ``config/config.yaml``).

    Returns
    -------
    list of dict
        One row per successfully processed scene (also written to
        ``config["paths"]["spectral_indices_manifest"]``).
    """
    sp_config = config["spectral_index"]
    paths = config["paths"]

    scenes_dir = Path(paths["sentinel2_dir"])
    output_dir = Path(paths["spectral_indices_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(paths["sentinel2_manifest"]).open(newline="") as f:
        scenes = list(csv.DictReader(f))

    logger.info("Computing spectral indices for %d scenes", len(scenes))

    results = []
    for i, scene in enumerate(scenes, start=1):
        scene_id = scene["id"]
        try:
            out_paths = process_scene(scene_id, scenes_dir, output_dir, sp_config)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("[%d/%d] %s -- skipping (%s)", i, len(scenes), scene_id, e)
            continue

        logger.info("[%d/%d] %s (%s) -- done", i, len(scenes), scene_id, scene["datetime"])
        results.append(
            {
                "scene_id": scene_id,
                "datetime": scene["datetime"],
                "cloud_cover": scene["cloud_cover"],
                **{f"{name}_path": path for name, path in out_paths.items()},
            }
        )

    manifest_path = Path(paths["spectral_indices_manifest"])
    fieldnames = ["scene_id", "datetime", "cloud_cover"] + [
        f"{name}_path" for name in sp_config["compute"]
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(
        "Stage 2 complete -- %d/%d scenes processed, manifest -> %s",
        len(results), len(scenes), manifest_path,
    )
    return results
