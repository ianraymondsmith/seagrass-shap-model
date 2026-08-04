"""
run_pipeline.py
===============
Master execution script for the Seagrass Health Predictive Pipeline.

PURPOSE:
    Runs all 8 pipeline stages in sequence, reading all parameters from
    config/config.yaml.  Each stage is called as a module-level function;
    intermediate outputs are written to disk so individual stages can be
    re-run without repeating upstream computation.

USAGE:
    # Full pipeline
    python run_pipeline.py

    # Custom config
    python run_pipeline.py --config config/config.yaml

    # Start from a specific stage (skips earlier stages if outputs exist)
    python run_pipeline.py --start-stage 5

    # Dry run (validate config and data only; no model training)
    python run_pipeline.py --dry-run

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logging_config import get_logger, log_config, log_environment

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seagrass Health Predictive Pipeline — end-to-end runner"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=range(1, 9),
        metavar="N",
        help="Start from stage N (1–8); earlier stages must have outputs on disk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data, then exit without training models",
    )
    return parser.parse_args()


def run_pipeline(config_path: str, start_stage: int = 1, dry_run: bool = False) -> None:
    """
    Execute the full pipeline from ``start_stage`` to stage 8.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    start_stage : int, optional
        First stage to execute (1–8).  Default: ``1``.
    dry_run : bool, optional
        If ``True``, run only validation (Stage 1 ingestion checks) and exit.
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    config = load_config(config_path)
    log_config(logger, config)
    log_environment(logger)

    logger.info("=" * 70)
    logger.info("SEAGRASS HEALTH PREDICTIVE PIPELINE")
    logger.info("Starting from stage %d | dry_run=%s", start_stage, dry_run)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1 — Ingestion
    # ------------------------------------------------------------------
    if start_stage <= 1:
        logger.info("--- Stage 1: Data Ingestion ---")
        from src import ingest_01 as ingest
        ingest.run(config)

    if dry_run:
        logger.info("Dry run complete. Exiting before modelling stages.")
        return

    # ------------------------------------------------------------------
    # Stage 2 — Spectral Index Computation
    # ------------------------------------------------------------------
    if start_stage <= 2:
        logger.info("--- Stage 2: Spectral Index Computation ---")
        from src import spectral_index_02

        spectral_index_02.run(config)

    # ------------------------------------------------------------------
    # Stage 3 — Spatial Join
    # ------------------------------------------------------------------
    if start_stage <= 3:
        logger.info("--- Stage 3: Spatial Join ---")
        from src import spatial_join_03

        spatial_join_03.run(config)

    # ------------------------------------------------------------------
    # Stages 4–8 — placeholders populated as modules are built
    # ------------------------------------------------------------------
    stage_labels = {
        4: "Feature Engineering",
        5: "Preprocessing",
        6: "Model Training",
        7: "SHAP Analysis",
        8: "Model Comparison",
    }
    for stage, label in stage_labels.items():
        if start_stage <= stage:
            logger.info("--- Stage %d: %s ---", stage, label)
            logger.info("Stage %d placeholder — module not yet built", stage)

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        config_path=args.config,
        start_stage=args.start_stage,
        dry_run=args.dry_run,
    )
