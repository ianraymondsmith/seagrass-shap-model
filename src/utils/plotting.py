"""
plotting.py
===========
Publication-quality figure helpers for the Seagrass Health Predictive Pipeline.

PURPOSE:
    Provides a consistent visual style and reusable figure components used
    across Stages 6–8 (model evaluation, SHAP, comparison).  All functions
    save figures at 300 dpi in the format specified by
    ``config.figures.format`` (default: PNG).

    Consistent, high-quality figures are a submission prerequisite for Q1
    journals such as Remote Sensing of Environment (300 dpi minimum;
    guidelines: https://www.elsevier.com/authors/policies-and-guidelines/
    artwork-and-media-instructions).

STYLE PHILOSOPHY:
    - Colour-blind-safe palettes (Wong, 2011)
    - Minimal chart junk — no gratuitous gridlines, no 3-D effects
    - All text at ≥11 pt for legibility in print
    - Health class colours consistent across all figures:
        Healthy       → Green  (#2CA02C)
        Intermediate  → Orange (#FF7F0E)
        Not Healthy   → Red    (#D62728)

REFERENCES:
    Wong, B. (2011). Color blindness. Nature Methods, 8(6), 441.
    https://doi.org/10.1038/nmeth.1618

    Tufte, E.R. (2001). The Visual Display of Quantitative Information
    (2nd ed.). Graphics Press.

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Colour palette for health classes — consistent across all pipeline figures.
# Palette is perceptually ordered (green → orange → red) and passes
# standard colour-blind simulations (deuteranopia, protanopia).
# Reference: Wong (2011), Nature Methods.
# ---------------------------------------------------------------------------
CLASS_COLOURS: Dict[str, str] = {
    "Healthy":      "#2CA02C",   # Green
    "Intermediate": "#FF7F0E",   # Orange
    "Not Healthy":  "#D62728",   # Red
}

# Integer-encoded labels → colours (for confusion matrix axes)
CLASS_COLOURS_INT: Dict[int, str] = {2: "#2CA02C", 1: "#FF7F0E", 0: "#D62728"}
CLASS_LABELS: List[str] = ["Not Healthy", "Intermediate", "Healthy"]  # ascending ordinal


def set_publication_style(
    font_size_axis: int = 11,
    font_size_title: int = 12,
    font_family: str = "sans-serif",
) -> None:
    """
    Apply a publication-ready Matplotlib style globally.

    Sets rcParams for font size, line weights, and background.  Call once
    at the start of any module that produces figures; the style persists
    for the Python session.

    Parameters
    ----------
    font_size_axis : int, optional
        Axis tick and label font size (pt).  Default: ``11``.
    font_size_title : int, optional
        Title font size (pt).  Default: ``12``.
    font_family : str, optional
        Font family string.  Default: ``"sans-serif"``.

    Returns
    -------
    None

    Examples
    --------
    >>> set_publication_style()
    """
    plt.rcParams.update({
        "font.family":         font_family,
        "font.size":           font_size_axis,
        "axes.titlesize":      font_size_title,
        "axes.labelsize":      font_size_axis,
        "xtick.labelsize":     font_size_axis - 1,
        "ytick.labelsize":     font_size_axis - 1,
        "legend.fontsize":     font_size_axis - 1,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.grid":           False,
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "savefig.bbox":        "tight",
        "savefig.dpi":         300,
        "lines.linewidth":     1.5,
        "patch.edgecolor":     "none",
    })
    logger.debug("Publication style applied (font_size_axis=%d)", font_size_axis)


def save_figure(
    fig: plt.Figure,
    filename: str,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    fmt: str = "png",
) -> Path:
    """
    Save a Matplotlib figure to disk at publication resolution.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Output filename without extension (e.g., ``"confusion_matrix_rf"``).
    figures_dir : str, optional
        Directory path.  Created if absent.  Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution in dots per inch.  Default: ``300``.
    fmt : str, optional
        File format: ``"png"``, ``"pdf"``, or ``"svg"``.  Default: ``"png"``.

    Returns
    -------
    pathlib.Path
        Full path to the saved figure.

    Examples
    --------
    >>> path = save_figure(fig, "shap_summary_rf")
    """
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename}.{fmt}"
    fig.savefig(out_path, dpi=dpi, format=fmt, bbox_inches="tight")
    logger.info("Figure saved: %s", out_path)
    return out_path


# Shared between plot_environmental_time_series and
# plot_environmental_scatter so the two companion figures stay visually
# consistent. Each panel gets its own pair of genuinely different (not
# shades of one hue) colours, and no colour repeats anywhere across all
# 12 panels x 2 groups = 24 total colour slots.
_ENV_PANELS = [
    ("chla_ug_L",     "Chlorophyll-a (µg/L)",       "#1f77b4", "#ff7f0e"),
    ("salinity_ppt",  "Salinity (ppt)",              "#d62728", "#17becf"),
    ("temp_C",        "Water Temperature (°C)",      "#e377c2", "#bcbd22"),
    ("do_mg_L",       "Dissolved Oxygen (mg/L)",     "#9467bd", "#8c564b"),
    ("ph",            "pH",                          "#2ca02c", "#ff1493"),
    ("turbidity_NTU", "Turbidity (NTU)",             "#000000", "#daa520"),
    ("secchi_m",      "Secchi Depth (m)",            "#008080", "#dc143c"),
    ("tpo4_mg_L",     "Total Phosphorus (mg/L)",     "#000080", "#32cd32"),
    ("totn_mg_L",     "Total Nitrogen (mg/L)",       "#800000", "#40e0d0"),
    ("nox_mg_L",      "Nitrate+Nitrite-N (mg/L)",    "#4b0082", "#ff7f50"),
    ("nh4_mg_L",      "Ammonia-N (mg/L)",            "#006400", "#da70d6"),
    ("ndavi",         "NDAVI",                       "#6a5acd", "#ff8c00"),
]


def _prep_station_group_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add `station_group` (FLAB vs C-111) and month-bucket columns."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["station_group"] = np.where(
        data["station_id"].str.startswith("FLAB"), "Open Bay (FLAB)", "Canal Inflow (C111)"
    )
    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp()
    return data


def plot_environmental_time_series(
    df: pd.DataFrame,
    figures_dir: str = "outputs/figures",
    dpi: int = 120,
    fmt: str = "png",
) -> plt.Figure:
    """
    Multi-panel time series of key water-quality and spectral variables.

    Produces a 6x2 grid, one panel per variable, each showing two monthly-
    mean lines: open-bay (FLAB*) stations vs. C-111 canal-inflow stations,
    plotted separately rather than blended into one bay-wide average.
    FLAB and C-111 are known to be hydrologically distinct (open estuarine
    water vs. freshwater canal discharge with systematically different
    nutrient chemistry -- see config.yaml's ingestion notes); averaging
    them together would obscure that real difference rather than reveal it.
    Each panel has its own pair of genuinely distinct colours for FLAB vs
    C-111 (not shades of one hue), with no colour reused across any of
    the 12 panels, and a legend on each panel.

    Uses boxed axes, gridlines, and wider panel spacing rather than this
    module's default minimal-chart-junk style (see `set_publication_style`)
    -- deliberately overridden locally for this figure only, modelled on
    the multi-panel environmental time series figures common in harmful-
    algal-bloom and water-quality driver studies.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 3 output (or equivalent): one row per station-date, with
        `station_id`, `date`, WQ parameter columns, and `ndavi`.
    figures_dir : str, optional
        Output directory. Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution. Default: ``300``.
    fmt : str, optional
        File format. Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_environmental_time_series(joined_df)
    """
    data = _prep_station_group_data(df)

    fig, axes = plt.subplots(6, 2, figsize=(13, 19), sharex=True)
    axes = axes.flatten()

    for ax, (col, ylabel, flab_colour, c111_colour) in zip(axes, _ENV_PANELS):
        if col not in data.columns:
            ax.set_visible(False)
            continue
        group_colours = {"Open Bay (FLAB)": flab_colour, "Canal Inflow (C111)": c111_colour}
        c111_end_date = None
        for group, colour in group_colours.items():
            series = (
                data[data["station_group"] == group]
                .groupby("month")[col].mean().dropna()
            )
            ax.plot(
                series.index, series.values,
                color=colour, label=group,
                linewidth=2.5, solid_capstyle="round",
            )
            if group == "Canal Inflow (C111)" and not series.empty:
                c111_end_date = series.index.max()
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

        # C-111 monitoring genuinely ended (confirmed against the raw
        # DBHYDRO export, not a pipeline artifact) -- mark it explicitly,
        # in one uniform colour/style across every panel, so the line
        # stopping doesn't read as a data-processing error. Labelled (not
        # annotated with on-plot text) so it shows up in each legend.
        if c111_end_date is not None:
            ax.axvline(
                c111_end_date, color="red", linestyle="--", linewidth=1.5,
                alpha=0.8, label="C-111 monitoring ended",
            )

        # Boxed axes + gridlines, overriding this module's default
        # minimal-chart-junk style for this figure specifically.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.grid(True, alpha=0.4, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    for ax in axes[-2:]:
        ax.set_xlabel("Date", fontweight="bold")

    fig.suptitle("Water Quality and Spectral Index Time Series — Florida Bay", y=0.995, fontweight="bold")
    fig.subplots_adjust(top=0.97, bottom=0.03, left=0.06, right=0.98, hspace=0.45, wspace=0.3)

    save_figure(fig, "environmental_time_series", figures_dir, dpi, fmt)
    return fig


def plot_environmental_scatter(
    df: pd.DataFrame,
    figures_dir: str = "outputs/figures",
    dpi: int = 120,
    fmt: str = "png",
) -> plt.Figure:
    """
    Multi-panel scatter of raw (unaggregated) water-quality and spectral
    samples -- companion to `plot_environmental_time_series`.

    Same 6x2 layout, panel set, and colour scheme as the time-series
    figure, but plots every individual sample as a point rather than a
    monthly mean line. Monthly aggregation hides real sample-to-sample
    variability and outliers; this figure shows the actual data density
    and spread instead. Modelled on the raw-sample scatter plots (coloured
    by station group) used alongside smoothed time series in harmful-
    algal-bloom driver studies.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 3 output (or equivalent): one row per station-date, with
        `station_id`, `date`, WQ parameter columns, and `ndavi`.
    figures_dir : str, optional
        Output directory. Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution. Default: ``300``.
    fmt : str, optional
        File format. Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_environmental_scatter(joined_df)
    """
    data = _prep_station_group_data(df)

    fig, axes = plt.subplots(6, 2, figsize=(13, 19), sharex=True)
    axes = axes.flatten()

    for ax, (col, ylabel, flab_colour, c111_colour) in zip(axes, _ENV_PANELS):
        if col not in data.columns:
            ax.set_visible(False)
            continue
        group_colours = {"Open Bay (FLAB)": flab_colour, "Canal Inflow (C111)": c111_colour}
        c111_end_date = None
        for group, colour in group_colours.items():
            sub = data[(data["station_group"] == group) & data[col].notna()]
            ax.scatter(
                sub["date"], sub[col],
                color=colour, label=group,
                s=18, alpha=0.6, edgecolors="none",
            )
            if group == "Canal Inflow (C111)" and not sub.empty:
                c111_end_date = sub["date"].max()
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

        if c111_end_date is not None:
            ax.axvline(
                c111_end_date, color="red", linestyle="--", linewidth=1.5,
                alpha=0.8, label="C-111 monitoring ended",
            )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.grid(True, alpha=0.4, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    for ax in axes[-2:]:
        ax.set_xlabel("Date", fontweight="bold")

    fig.suptitle("Water Quality and Spectral Index Raw Samples — Florida Bay", y=0.995, fontweight="bold")
    fig.subplots_adjust(top=0.97, bottom=0.03, left=0.06, right=0.98, hspace=0.45, wspace=0.3)

    save_figure(fig, "environmental_scatter", figures_dir, dpi, fmt)
    return fig


def _simple_basemap_figure(
    bounds: Dict[str, float], figsize: tuple = (11, 11)
):
    """
    Shared cartographic scaffold for this module's "simple" reference maps:
    a Cartopy PlateCarree axes zoomed to `bounds` (+15% pad), flat land/water
    fill, land border -- styled after NOAA NCEI's own HABSOS accession maps
    (data/raw/HAB .../about/0120767_map.jpg). Used by both
    `plot_ground_truth_map` (basemap=False) and `plot_dbhydro_station_map`
    so the two stay visually identical.

    Returns
    -------
    (fig, ax, proj, main_extent)
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    # Zoomed to the data's own bounding box (+15% pad) so points stay
    # readable -- the wider South Florida shape for context is instead
    # shown via the locator inset (`_add_locator_inset`).
    lon_pad = (bounds["lon_max"] - bounds["lon_min"]) * 0.15
    lat_pad = (bounds["lat_max"] - bounds["lat_min"]) * 0.15
    main_extent = [
        bounds["lon_min"] - lon_pad, bounds["lon_max"] + lon_pad,
        bounds["lat_min"] - lat_pad, bounds["lat_max"] + lat_pad,
    ]
    ax.set_extent(main_extent, crs=proj)
    # Explicit axes background as the ocean colour (confirmed by isolated
    # testing that land+ocean features interact oddly via z-order when both
    # are added together) -- land polygon drawn on top, at zorder=2 (not 1)
    # since GeoAxes' background patch also defaults to zorder=1, and a tie
    # leaves the land feature hidden behind the background on some render
    # paths. Border drawn as the land patch's own edgecolor, not a separate
    # coastline feature: that line-only geometry gets filled with a default
    # colour unless told facecolor="none", and even fixed it read as visual
    # noise here.
    ax.set_facecolor("#cfe8f3")
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m"),
        facecolor="#e8e4d8", edgecolor="black", linewidth=0.7, zorder=2,
    )
    return fig, ax, proj, main_extent


def _add_locator_inset(fig: plt.Figure, ax: plt.Axes, proj, main_extent: list) -> plt.Axes:
    """
    Small South-Florida-peninsula locator map tucked into the main map's
    own bottom-left (open-water) corner, with a red rectangle marking where
    the main map's extent sits within the wider region -- same two-tier
    layout as the reference NOAA HABSOS map.
    """
    import cartopy.feature as cfeature

    main_pos = ax.get_position()
    inset_ax = fig.add_axes(
        (main_pos.x0 + 0.006, main_pos.y0 + 0.006, 0.18, 0.18), projection=proj
    )
    inset_ax.set_extent([-82.9, -79.9, 24.4, 27.0], crs=proj)
    inset_ax.set_facecolor("#cfe8f3")
    inset_ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m"),
        facecolor="#e8e4d8", edgecolor="black", linewidth=0.5, zorder=2,
    )
    inset_ax.add_patch(
        plt.Rectangle(
            (main_extent[0], main_extent[2]),
            main_extent[1] - main_extent[0], main_extent[3] - main_extent[2],
            transform=proj, edgecolor="red", facecolor="none", linewidth=1.3, zorder=5,
        )
    )
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)
    return inset_ax


def plot_ground_truth_map(
    seacar_df: pd.DataFrame,
    scene_dir: Optional[str] = None,
    basemap: bool = True,
    bounds: Optional[Dict[str, float]] = None,
    pad_km: float = 2.0,
    figures_dir: str = "outputs/figures",
    dpi: int = 150,
    fmt: str = "png",
) -> plt.Figure:
    """
    Map of SEACAR ground-truth sampling locations, colour-coded by health
    classification (same palette as every other figure in this module --
    see `CLASS_COLOURS`).

    Two styles, both showing the same points/classification:

    - ``basemap=True`` (default): plotted over a real Sentinel-2 colour-
      infrared composite (R=NIR/B08, G=Red/B04, B=Blue/B02) cropped to the
      study bounding box, in projected metres (UTM 17N). Colour-infrared
      (rather than true colour, which would need the B03 green band this
      pipeline hasn't downloaded) renders vegetation/mangrove in red tones
      and open water in dark blue/black -- visually striking and gives
      real geographic texture, at the cost of busier surroundings around
      the points themselves.
    - ``basemap=False``: basic reference cartography via Cartopy -- flat
      land/water fill, coastline, lat/lon gridlines -- styled after NOAA
      NCEI's own HABSOS accession maps rather than either the busy
      satellite composite or a bare axes-only scatter. Requires Cartopy's
      Natural Earth shapefiles (downloaded once, then cached locally); no
      `scene_dir` needed since no Sentinel-2 imagery is loaded.

    Parameters
    ----------
    seacar_df : pd.DataFrame
        Must have `Latitude`, `Longitude`, `Braun-Blanquet Score` columns
        (WGS84 decimal degrees) -- e.g. `04_seacar_spectral_validation.csv`.
    scene_dir : str, optional
        Path to a scene folder containing B02.jp2/B04.jp2/B08.jp2 (see
        `data/raw/sentinel2/<scene_id>/`). Required when ``basemap=True``;
        pick a low-cloud-cover scene from `scene_manifest.csv`.
    basemap : bool, optional
        See above. Default: ``True``.
    bounds : dict, optional
        lat_min/lat_max/lon_min/lon_max. Default: `FLORIDA_BAY_BOUNDS`.
    pad_km : float, optional
        Padding around the bounding box (basemap version only) so edge
        points aren't flush against the image border. Default: ``2.0``.
    figures_dir : str, optional
        Output directory. Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution. Default: ``150`` (a basemap image doesn't
        benefit from 300 dpi print resolution the way a line chart does).
    fmt : str, optional
        File format. Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> df = pd.read_csv("outputs/consolidated_export/04_seacar_spectral_validation.csv")
    >>> fig = plot_ground_truth_map(df, "data/raw/sentinel2/S2A_..._T17RNH_...")
    >>> fig_simple = plot_ground_truth_map(df, basemap=False)
    """
    from src.utils.validation import FLORIDA_BAY_BOUNDS

    if bounds is None:
        bounds = FLORIDA_BAY_BOUNDS

    def classify(score: float) -> str:
        if score <= 1.0:
            return "Not Healthy"
        elif score <= 3.0:
            return "Intermediate"
        return "Healthy"

    df = seacar_df.dropna(subset=["Latitude", "Longitude", "Braun-Blanquet Score"]).copy()
    df["health_class"] = df["Braun-Blanquet Score"].apply(classify)

    if not basemap:
        fig, ax, proj, main_extent = _simple_basemap_figure(bounds)

        for cls in CLASS_LABELS:
            sub = df[df["health_class"] == cls]
            ax.scatter(
                sub["Longitude"], sub["Latitude"], c=CLASS_COLOURS[cls], s=12, alpha=0.8,
                label=f"{cls} (n={len(sub)})", edgecolors="white", linewidth=0.3,
                transform=proj, zorder=5,
            )

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = gl.right_labels = False
        ax.set_title("SEACAR Ground-Truth Sampling Locations — Florida Bay", pad=10)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)
        _add_locator_inset(fig, ax, proj, main_extent)

        save_figure(fig, "ground_truth_map_simple", figures_dir, dpi, fmt)
        return fig

    if scene_dir is None:
        raise ValueError("scene_dir is required when basemap=True")

    import rasterio
    import rasterio.windows
    from pyproj import Transformer

    with rasterio.open(f"{scene_dir}/B08.jp2") as src:
        crs = src.crs
        transform_to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x_min, y_min = transform_to_utm.transform(bounds["lon_min"], bounds["lat_min"])
        x_max, y_max = transform_to_utm.transform(bounds["lon_max"], bounds["lat_max"])
        pad_m = pad_km * 1000
        window = rasterio.windows.from_bounds(
            x_min - pad_m, y_min - pad_m, x_max + pad_m, y_max + pad_m, transform=src.transform
        ).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        win_transform = src.window_transform(window)
        win_bounds = rasterio.windows.bounds(window, src.transform)

    def load_band_stretched(band_file: str) -> np.ndarray:
        with rasterio.open(f"{scene_dir}/{band_file}") as src:
            arr = src.read(1, window=window).astype(np.float32)
        valid = arr[arr > 0]
        lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
        return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)

    rgb = np.dstack([load_band_stretched("B08.jp2"), load_band_stretched("B04.jp2"), load_band_stretched("B02.jp2")])

    x, y = transform_to_utm.transform(df["Longitude"].values, df["Latitude"].values)
    df["_x"], df["_y"] = x, y

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(rgb, extent=(win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3]), origin="upper")

    for cls in CLASS_LABELS:
        sub = df[df["health_class"] == cls]
        ax.scatter(
            sub["_x"], sub["_y"], c=CLASS_COLOURS[cls], s=14, alpha=0.75,
            label=f"{cls} (n={len(sub)})", edgecolors="white", linewidth=0.3,
        )

    ax.set_xlim(win_bounds[0], win_bounds[2])
    ax.set_ylim(win_bounds[1], win_bounds[3])
    ax.set_xlabel("Easting (m, UTM Zone 17N)")
    ax.set_ylabel("Northing (m, UTM Zone 17N)")
    ax.set_title("SEACAR Ground-Truth Sampling Locations — Florida Bay", pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    save_figure(fig, "ground_truth_map", figures_dir, dpi, fmt)
    return fig


# Distinct from CLASS_COLOURS (health classification) since this is a
# different categorical variable -- station network membership -- and the
# two maps may sit side by side in the same notebook.
_STATION_GROUP_COLOURS = {
    "Open Bay (FLAB)": "#2166AC",
    "Canal Inflow (C111)": "#762A83",
}


def plot_dbhydro_station_map(
    wq_df: pd.DataFrame,
    bounds: Optional[Dict[str, float]] = None,
    label_stations: bool = True,
    figures_dir: str = "outputs/figures",
    dpi: int = 150,
    fmt: str = "png",
) -> plt.Figure:
    """
    Map of DBHYDRO water-quality monitoring station locations, colour-coded
    by network membership (open-bay FLAB stations vs. C-111 canal-inflow
    stations).

    Same "simple" reference-map style as `plot_ground_truth_map`
    (basemap=False): flat land/water cartography, gridlines, and a South
    Florida locator inset, so the two figures read as a matched pair.

    Parameters
    ----------
    wq_df : pd.DataFrame
        Must have `Station`, `Latitude`, `Longitude` columns (WGS84 decimal
        degrees) -- e.g. `01_wq_florida_bay_clean.csv`. One row per station
        is plotted (duplicates from repeated station-date rows are dropped).
    bounds : dict, optional
        lat_min/lat_max/lon_min/lon_max. Default: `FLORIDA_BAY_BOUNDS`.
    label_stations : bool, optional
        Annotate each point with its station ID. With only 22 stations
        (unlike the thousands of SEACAR points) labels stay readable and
        add real value. Default: ``True``.
    figures_dir : str, optional
        Output directory. Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution. Default: ``150``.
    fmt : str, optional
        File format. Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> df = pd.read_csv("outputs/consolidated_export/01_wq_florida_bay_clean.csv")
    >>> fig = plot_dbhydro_station_map(df)
    """
    from src.utils.validation import FLORIDA_BAY_BOUNDS

    if bounds is None:
        bounds = FLORIDA_BAY_BOUNDS

    stations = wq_df.dropna(subset=["Latitude", "Longitude"]).drop_duplicates("Station").copy()
    stations["station_group"] = np.where(
        stations["Station"].str.startswith("FLAB"), "Open Bay (FLAB)", "Canal Inflow (C111)"
    )

    fig, ax, proj, main_extent = _simple_basemap_figure(bounds)

    for group, colour in _STATION_GROUP_COLOURS.items():
        sub = stations[stations["station_group"] == group]
        ax.scatter(
            sub["Longitude"], sub["Latitude"], c=colour, s=45, alpha=0.9,
            label=f"{group} (n={len(sub)})", edgecolors="white", linewidth=0.6,
            transform=proj, zorder=5,
        )

    if label_stations:
        # Default offset overlaps in the tight FLAB04/05/06/08/10 + C111JB
        # cluster (all within ~0.15 deg of each other, upper right of the
        # bay) -- nudged individually so those labels stay legible.
        label_offsets = {
            "FLAB08": (4, 10),
            "C111JB": (-8, 14),
            "FLAB10": (8, -14),
            "FLAB06": (4, -12),
            "FLAB05": (4, -10),
            "FLAB04": (6, 2),
        }
        for _, row in stations.iterrows():
            ax.annotate(
                row["Station"], (row["Longitude"], row["Latitude"]),
                xytext=label_offsets.get(row["Station"], (4, 3)),
                textcoords="offset points", fontsize=7, color="#333333",
                transform=proj, zorder=6,
            )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    ax.set_title("DBHYDRO Water Quality Monitoring Stations — Florida Bay", pad=10)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    _add_locator_inset(fig, ax, proj, main_extent)

    save_figure(fig, "dbhydro_station_map", figures_dir, dpi, fmt)
    return fig


def plot_sentinel2_footprint_map(
    scene_dir: str,
    bounds: Optional[Dict[str, float]] = None,
    figures_dir: str = "outputs/figures",
    dpi: int = 150,
    fmt: str = "png",
) -> plt.Figure:
    """
    Map of the acquired Sentinel-2 tile footprint against the Florida Bay
    study area, for context on spectral data coverage.

    Every one of the 120 acquired scenes (`data/raw/sentinel2/
    scene_manifest.csv`) is the same MGRS tile (T17RNH) at different dates,
    so there is exactly one footprint to show, not 120. Same "simple"
    reference-map style as `plot_ground_truth_map`/`plot_dbhydro_station_map`
    (flat land/water cartography, gridlines, South Florida locator inset).

    Parameters
    ----------
    scene_dir : str
        Path to any one acquired scene folder containing `B08.jp2` (its
        footprint is identical for every scene of the same tile) -- e.g.
        `data/raw/sentinel2/<scene_id>/`.
    bounds : dict, optional
        Study-area lat_min/lat_max/lon_min/lon_max, drawn as a dashed
        reference outline. Default: `FLORIDA_BAY_BOUNDS`.
    figures_dir : str, optional
        Output directory. Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution. Default: ``150``.
    fmt : str, optional
        File format. Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_sentinel2_footprint_map(
    ...     "data/raw/sentinel2/S2A_..._T17RNH_..."
    ... )
    """
    import re

    import rasterio
    from pyproj import Transformer

    from src.utils.validation import FLORIDA_BAY_BOUNDS

    if bounds is None:
        bounds = FLORIDA_BAY_BOUNDS

    tile_match = re.search(r"_(T\d\d[A-Z]{3})_", Path(scene_dir).name)
    tile_id = tile_match.group(1) if tile_match else "Sentinel-2 tile"

    with rasterio.open(f"{scene_dir}/B08.jp2") as src:
        transform_to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        tile_lon_min, tile_lat_min = transform_to_wgs84.transform(src.bounds.left, src.bounds.bottom)
        tile_lon_max, tile_lat_max = transform_to_wgs84.transform(src.bounds.right, src.bounds.top)

    # Zoom extent must cover the tile AND the study area -- the tile is far
    # larger than Florida Bay, but a corner of the bay's own bounding box
    # (west of ~81.0 deg W, e.g. FLAB25) actually falls just outside it, so
    # neither box alone is guaranteed to contain the other.
    combined_bounds = {
        "lon_min": min(bounds["lon_min"], tile_lon_min),
        "lon_max": max(bounds["lon_max"], tile_lon_max),
        "lat_min": min(bounds["lat_min"], tile_lat_min),
        "lat_max": max(bounds["lat_max"], tile_lat_max),
    }

    fig, ax, proj, main_extent = _simple_basemap_figure(combined_bounds)

    ax.add_patch(
        plt.Rectangle(
            (tile_lon_min, tile_lat_min), tile_lon_max - tile_lon_min, tile_lat_max - tile_lat_min,
            transform=proj, facecolor="#2166AC", alpha=0.15, edgecolor="#2166AC", linewidth=1.5,
            label=f"Sentinel-2 tile {tile_id} footprint", zorder=4,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (bounds["lon_min"], bounds["lat_min"]),
            bounds["lon_max"] - bounds["lon_min"], bounds["lat_max"] - bounds["lat_min"],
            transform=proj, facecolor="none", edgecolor="#D62728", linewidth=1.5, linestyle="--",
            label="Study area (Florida Bay)", zorder=5,
        )
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    ax.set_title("Sentinel-2 Tile Coverage — Florida Bay", pad=10)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    _add_locator_inset(fig, ax, proj, main_extent)

    save_figure(fig, "sentinel2_footprint_map", figures_dir, dpi, fmt)
    return fig


def plot_habsos_overview_map(
    raw_habsos_path: str = "data/raw/habsos_20240430.csv",
    bounds: Optional[Dict[str, float]] = None,
    figures_dir: str = "outputs/figures",
    dpi: int = 150,
    fmt: str = "png",
) -> plt.Figure:
    """
    Gulf-of-Mexico-wide map of every raw HABSOS observation location, with
    this project's Florida Bay study area marked -- built in the same
    cartographic style as `plot_ground_truth_map`/`plot_dbhydro_station_map`
    /`plot_sentinel2_footprint_map`, rather than annotating NOAA NCEI's own
    accession map image (`data/raw/HAB .../0120767_map.jpg`) directly.

    Plots the full, unfiltered accession (211K rows, all Gulf states) so the
    coastal density pattern matches that reference map -- `hab_matching.py`'s
    589-row Florida-Bay-bbox-filtered subset is what actually feeds the
    pipeline (see `plot_dbhydro_station_map` for where those land relative
    to the DBHYDRO network) and would be a nearly invisible speck at this
    scale.

    Parameters
    ----------
    raw_habsos_path : str, optional
        Path to the raw HABSOS export CSV.
    bounds : dict, optional
        Study-area lat_min/lat_max/lon_min/lon_max, drawn as a dashed
        reference outline. Default: `FLORIDA_BAY_BOUNDS`.
    figures_dir : str, optional
        Output directory. Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution. Default: ``150``.
    fmt : str, optional
        File format. Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_habsos_overview_map()
    """
    import cartopy.feature as cfeature

    from src.utils.validation import FLORIDA_BAY_BOUNDS

    if bounds is None:
        bounds = FLORIDA_BAY_BOUNDS

    habsos = pd.read_csv(raw_habsos_path, usecols=["LATITUDE", "LONGITUDE"], low_memory=False).dropna()

    data_bounds = {
        "lon_min": habsos["LONGITUDE"].min(), "lon_max": habsos["LONGITUDE"].max(),
        "lat_min": habsos["LATITUDE"].min(), "lat_max": habsos["LATITUDE"].max(),
    }
    fig, ax, proj, main_extent = _simple_basemap_figure(data_bounds, figsize=(11, 9))

    ax.scatter(
        habsos["LONGITUDE"], habsos["LATITUDE"], c="#D62728", s=3, alpha=0.35,
        edgecolors="none", label=f"HABSOS observation (n={len(habsos):,})",
        transform=proj, zorder=4,
    )
    ax.add_patch(
        plt.Rectangle(
            (bounds["lon_min"], bounds["lat_min"]),
            bounds["lon_max"] - bounds["lon_min"], bounds["lat_max"] - bounds["lat_min"],
            transform=proj, facecolor="none", edgecolor="black", linewidth=1.8, linestyle="--",
            label="Study area (Florida Bay)", zorder=5,
        )
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    ax.set_title("HABSOS Harmful Algal Bloom Observations — Gulf of Mexico", pad=10)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, markerscale=3)

    # Zoomed-in detail inset on the study area itself -- at Gulf-wide scale
    # its observations are a barely-visible speck -- tucked over open Gulf
    # water (clear of both coastline and every dot cluster) rather than the
    # bottom-left corner used elsewhere, which here is Mexican coastline.
    inset_lon_pad = (bounds["lon_max"] - bounds["lon_min"]) * 0.6
    inset_lat_pad = (bounds["lat_max"] - bounds["lat_min"]) * 0.6
    inset_extent = [
        bounds["lon_min"] - inset_lon_pad, bounds["lon_max"] + inset_lon_pad,
        bounds["lat_min"] - inset_lat_pad, bounds["lat_max"] + inset_lat_pad,
    ]
    # Anchored on-screen (not by its own data extent) to sit over deep,
    # observation-free open Gulf water south of the Louisiana/Mississippi
    # bloom band and west of Florida's -- (-92 deg W, 26 deg N).
    main_pos = ax.get_position()
    inset_size = 0.19
    target_lon, target_lat = -92.0, 26.0
    frac_x = (target_lon - main_extent[0]) / (main_extent[1] - main_extent[0])
    frac_y = (target_lat - main_extent[2]) / (main_extent[3] - main_extent[2])
    inset_ax = fig.add_axes(
        (
            main_pos.x0 + frac_x * main_pos.width - inset_size / 2,
            main_pos.y0 + frac_y * main_pos.height - inset_size / 2,
            inset_size, inset_size,
        ),
        projection=proj,
    )
    inset_ax.set_extent(inset_extent, crs=proj)
    inset_ax.set_facecolor("#cfe8f3")
    inset_ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m"),
        facecolor="#e8e4d8", edgecolor="black", linewidth=0.7, zorder=2,
    )
    inset_ax.scatter(
        habsos["LONGITUDE"], habsos["LATITUDE"], c="#D62728", s=10, alpha=0.6,
        edgecolors="none", transform=proj, zorder=4,
    )
    inset_ax.add_patch(
        plt.Rectangle(
            (bounds["lon_min"], bounds["lat_min"]),
            bounds["lon_max"] - bounds["lon_min"], bounds["lat_max"] - bounds["lat_min"],
            transform=proj, facecolor="none", edgecolor="black", linewidth=1.5, linestyle="--", zorder=5,
        )
    )
    inset_ax.set_title("Florida Bay detail", fontsize=9, pad=4)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    save_figure(fig, "habsos_overview_map", figures_dir, dpi, fmt)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    class_labels: Optional[List[str]] = None,
    normalise: bool = True,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    fmt: str = "png",
) -> plt.Figure:
    """
    Plot a single normalised or raw confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix from ``sklearn.metrics.confusion_matrix``.
        Expected order: ascending ordinal (Not Healthy → Intermediate → Healthy).
    model_name : str
        Short model identifier shown in title (e.g., ``"Random Forest"``).
    class_labels : list of str, optional
        Tick labels for axes.  Default: ``CLASS_LABELS``.
    normalise : bool, optional
        If ``True``, normalise rows to show recall per class.
        Default: ``True``.
    figures_dir : str, optional
        Output directory.  Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution.  Default: ``300``.
    fmt : str, optional
        File format.  Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Row-normalised confusion matrices (normalise=True) are preferred for
    imbalanced class problems because they show per-class recall regardless
    of class frequency (Stehman, 1997).

    References
    ----------
    Stehman, S.V. (1997). Selecting and interpreting measures of thematic
    classification accuracy. Remote Sensing of Environment, 62(1), 77–89.
    https://doi.org/10.1016/S0034-4257(97)00083-7

    Examples
    --------
    >>> fig = plot_confusion_matrix(cm, "Random Forest")
    """
    if class_labels is None:
        class_labels = CLASS_LABELS

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        value_fmt = ".2f"
        cbar_label = "Row-normalised recall"
    else:
        cm_plot = cm
        value_fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=value_fmt,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label", labelpad=8)
    ax.set_title(f"Confusion Matrix — {model_name}", pad=10)

    safe_name = model_name.lower().replace(" ", "_")
    save_figure(fig, f"confusion_matrix_{safe_name}", figures_dir, dpi, fmt)
    return fig


def plot_class_distribution(
    series: pd.Series,
    title: str = "Health Class Distribution",
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    fmt: str = "png",
) -> plt.Figure:
    """
    Bar chart of health class counts with percentage annotations.

    Parameters
    ----------
    series : pd.Series
        Health status column (string labels).
    title : str, optional
        Figure title.
    figures_dir : str, optional
        Output directory.
    dpi : int, optional
        Output resolution.  Default: ``300``.
    fmt : str, optional
        File format.  Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_class_distribution(df["health_status"])
    """
    counts = series.value_counts().reindex(
        ["Healthy", "Intermediate", "Not Healthy"]
    ).fillna(0)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[CLASS_COLOURS.get(c, "#888888") for c in counts.index],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, n in zip(bars, counts.values):
        pct = 100 * n / total if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            f"{int(n)}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Number of samples")
    ax.set_title(title, pad=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    save_figure(fig, "class_distribution", figures_dir, dpi, fmt)
    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    fmt: str = "png",
) -> plt.Figure:
    """
    Grouped bar chart comparing RF vs XGBoost performance metrics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics as rows and model names as columns.
        Index should be metric names (e.g., ``["Accuracy", "Macro-F1",
        "Cohen's Kappa", "ROC-AUC"]``).
    figures_dir : str, optional
        Output directory.  Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution.  Default: ``300``.
    fmt : str, optional
        File format.  Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_metrics_comparison(metrics_df)
    """
    n_metrics = len(metrics_df)
    x = np.arange(n_metrics)
    width = 0.35
    model_colours = {"Random Forest": "#1F77B4", "XGBoost": "#FF7F0E"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, col in enumerate(metrics_df.columns):
        offset = (i - (len(metrics_df.columns) - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            metrics_df[col].values,
            width,
            label=col,
            color=model_colours.get(col, f"C{i}"),
            edgecolor="white",
        )
        for rect in rects:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index, rotation=20, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison — RF vs XGBoost", pad=10)
    ax.legend(frameon=False)

    save_figure(fig, "metrics_comparison", figures_dir, dpi, fmt)
    return fig


def plot_feature_importance_comparison(
    rf_importances: pd.Series,
    xgb_importances: pd.Series,
    n_top: int = 15,
    figures_dir: str = "outputs/figures",
    dpi: int = 300,
    fmt: str = "png",
) -> plt.Figure:
    """
    Side-by-side horizontal bar charts of SHAP feature importances.

    Plots the top ``n_top`` features by mean |SHAP| for RF and XGBoost
    side by side.  Features are sorted by RF importance order so agreement
    between models is visually apparent.

    Parameters
    ----------
    rf_importances : pd.Series
        Mean absolute SHAP values for RF, indexed by feature name.
    xgb_importances : pd.Series
        Mean absolute SHAP values for XGBoost, indexed by feature name.
    n_top : int, optional
        Number of top features to display.  Default: ``15``.
    figures_dir : str, optional
        Output directory.  Default: ``"outputs/figures"``.
    dpi : int, optional
        Output resolution.  Default: ``300``.
    fmt : str, optional
        File format.  Default: ``"png"``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_feature_importance_comparison(rf_shap, xgb_shap)
    """
    # Align on RF ordering
    top_features = rf_importances.nlargest(n_top).index.tolist()
    rf_vals = rf_importances.reindex(top_features).fillna(0)
    xgb_vals = xgb_importances.reindex(top_features).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    y = np.arange(len(top_features))

    for ax, vals, name, colour in zip(
        axes,
        [rf_vals, xgb_vals],
        ["Random Forest", "XGBoost"],
        ["#1F77B4", "#FF7F0E"],
    ):
        ax.barh(y, vals.values[::-1], color=colour, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(name, pad=8)

    axes[0].invert_xaxis()
    axes[0].yaxis.set_label_position("right")
    fig.suptitle(
        f"Top {n_top} Features by Mean |SHAP| — RF vs XGBoost",
        y=1.01, fontsize=12,
    )
    plt.tight_layout()

    save_figure(fig, "shap_importance_comparison", figures_dir, dpi, fmt)
    return fig
