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
