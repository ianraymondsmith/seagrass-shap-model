"""
config_loader.py
================
YAML configuration loader for the Seagrass Health Predictive Pipeline.

PURPOSE:
    Provides a single ``load_config`` function that parses config/config.yaml
    and returns a validated Python dict.  Every pipeline module imports this
    function so that no file paths, parameters, or hyperparameters are
    hardcoded in source code.

    Config-driven design is a Q1 reproducibility requirement: a single
    config commit unambiguously captures all methodological choices for
    a given model run.

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_REQUIRED_TOP_LEVEL_KEYS = [
    "paths", "ingestion", "spectral_index", "spatial_join",
    "feature_engineering", "preprocessing", "models", "shap", "figures",
]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Parse and validate the pipeline YAML configuration file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML config file relative to the project root.
        Default: ``"config/config.yaml"``.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at ``config_path``.
    KeyError
        If any required top-level section is missing from the config.

    Examples
    --------
    >>> from src.utils.config_loader import load_config
    >>> config = load_config()
    >>> raster_path = config["paths"]["raster"]
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}\n"
            "Run from the project root directory or pass the correct path."
        )

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in config]
    if missing:
        raise KeyError(
            f"config.yaml is missing required sections: {missing}. "
            "Restore the missing sections from the template."
        )

    logger.info("Configuration loaded from: %s", path.resolve())
    return config
