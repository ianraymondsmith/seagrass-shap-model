"""
logging_config.py
=================
Configures reproducible, timestamped logging for the Seagrass Health
Predictive Pipeline.

PURPOSE:
    Provides a single ``get_logger`` entry point used by every pipeline
    module.  Each pipeline run writes to both the console (INFO level) and
    a timestamped file in ``outputs/reports/``.  The file log captures
    DEBUG-level detail so that post-hoc review of a run is possible without
    re-executing the pipeline.

    Logging all processing decisions (CRS detected, missing-value counts,
    split sizes, hyperparameters selected) is a reproducibility requirement
    for Q1 journal submission: reviewers and co-authors must be able to
    reconstruct the exact conditions of any reported result.

OUTPUTS:
    ``outputs/reports/pipeline_run_YYYYMMDD_HHMMSS.log``  (one per run)

USAGE:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Stage 1 — ingestion complete")

REFERENCES:
    Wilson, G. et al. (2017). Good enough practices in scientific computing.
    PLOS Computational Biology, 13(6), e1005510.
    https://doi.org/10.1371/journal.pcbi.1005510

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


# Module-level registry so repeated calls to get_logger() with the same
# run_id return the same file handler rather than creating duplicate log files.
_run_timestamp: Optional[str] = None
_file_handler: Optional[logging.FileHandler] = None


def get_logger(
    name: str,
    log_dir: str = "outputs/reports",
    level: str = "INFO",
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Return a configured logger for a pipeline module.

    Creates a logger that emits to both the console and (optionally) a
    shared timestamped log file.  All calls within a single Python session
    share the same log file, identified by a module-level timestamp set on
    the first call.

    Parameters
    ----------
    name : str
        Logger name — pass ``__name__`` from the calling module so that log
        records include the originating module path.
    log_dir : str, optional
        Directory where the log file is written.  Created if it does not
        exist.  Default: ``"outputs/reports"``.
    level : str, optional
        Minimum severity level for console output.
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
        File output is always ``DEBUG`` for full traceability.
        Default: ``"INFO"``.
    log_to_file : bool, optional
        If ``True``, attach a file handler in addition to the console
        handler.  Set ``False`` during unit tests to suppress disk I/O.
        Default: ``True``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    The timestamped filename format is
    ``pipeline_run_YYYYMMDD_HHMMSS.log``.  Using a fixed timestamp per
    session (rather than per call) ensures all modules write to the same
    file, producing a coherent run record.

    Examples
    --------
    >>> from src.utils.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Ingestion complete — %d records loaded", n_records)
    """
    global _run_timestamp, _file_handler

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called more than once
    # for the same module name within the session.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    # --- Formatter ----------------------------------------------------------
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler ----------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # --- File handler -------------------------------------------------------
    if log_to_file:
        # First call in session: create the shared timestamp and file handler.
        if _run_timestamp is None:
            _run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"pipeline_run_{_run_timestamp}.log"

            _file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            _file_handler.setLevel(logging.DEBUG)
            _file_handler.setFormatter(fmt)

        logger.addHandler(_file_handler)

    return logger


def log_config(logger: logging.Logger, config: dict) -> None:
    """
    Write the active configuration to the log at DEBUG level.

    Called once at pipeline start so that the run log contains a complete
    record of all parameters used.  This satisfies the reproducibility
    standard that every result must be traceable to a specific configuration
    state.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance (from ``get_logger``).
    config : dict
        Parsed configuration dictionary (from ``utils.config_loader``).

    Returns
    -------
    None

    Examples
    --------
    >>> log_config(logger, config)
    # Writes YAML-like representation of config to log at DEBUG level
    """
    import yaml  # lazy import — only needed at log time

    logger.debug("=" * 70)
    logger.debug("ACTIVE CONFIGURATION")
    logger.debug("=" * 70)
    for line in yaml.dump(config, default_flow_style=False).splitlines():
        logger.debug("  %s", line)
    logger.debug("=" * 70)


def log_environment(logger: logging.Logger) -> None:
    """
    Log Python version and key package versions for reproducibility.

    Captures the runtime environment so that readers of the log can verify
    that the pinned versions in ``requirements.txt`` were actually installed.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    None
    """
    import sys

    packages = [
        "numpy", "pandas", "rasterio", "geopandas",
        "sklearn", "xgboost", "shap", "rasterstats",
    ]

    logger.info("Python version: %s", sys.version.split()[0])
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            logger.info("  %-20s %s", pkg, version)
        except ImportError:
            logger.warning("  %-20s NOT INSTALLED", pkg)
