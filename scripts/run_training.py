#!/usr/bin/env python3
"""CLI runner for dataset preparation, model training, evaluation and ONNX export.

Reference:
- STAGE_3.md, Section 4.5
- 06_DATA_AND_MLOPS.md, Section 7
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.core.config import TrainingConfig
from src.ml.dataset import grouped_split_dataset, load_dataset_from_manifest
from src.ml.trainer import PieceClassifierTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run piece classifier training and export pipeline.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset/dataset_manifest.json",
        help="Path to AutoLabeller dataset manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18"],
        help="Backbone architecture",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    manifest_p = Path(args.manifest)
    if not manifest_p.is_file():
        logger.error("Manifest not found at %s. Generate samples via AutoLabeller first.", manifest_p)
        return 1

    logger.info("Loading samples from %s ...", manifest_p)
    samples = load_dataset_from_manifest(manifest_p)
    logger.info("Loaded %d total samples.", len(samples))

    train_s, val_s, test_s = grouped_split_dataset(samples, val_ratio=0.15, test_ratio=0.15)
    logger.info(
        "Grouped Split: Train=%d, Val=%d, Test=%d (unseen sets).",
        len(train_s),
        len(val_s),
        len(test_s),
    )

    cfg = TrainingConfig(
        dataset_dir=str(manifest_p.parent),
        output_dir=args.output_dir,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    trainer = PieceClassifierTrainer(config=cfg, output_dir=args.output_dir)
    trainer.log_hyperparameters()

    logger.info("Training pipeline initialized. Ready for execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

