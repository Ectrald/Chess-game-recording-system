#!/usr/bin/env python3
"""CLI utility to train or fine-tune YOLOv8 board detector.

Reference:
- STAGE_1.md, Section 4.1
- 04_VISION_PIPELINE.md, Section 4
- 06_DATA_AND_MLOPS.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_yolo_board(
    data_yaml: str,
    epochs: int = 30,
    imgsz: int = 640,
    batch_size: int = 16,
    base_model: str = "yolov8n.pt",
    output_weights: str = "data/weights/yolov8n_board.pt",
) -> bool:
    """Trains or fine-tunes YOLOv8 model for chessboard bounding box detection.

    Args:
        data_yaml: Path to dataset YAML configuration (Roboflow / YOLO format).
        epochs: Number of training epochs.
        imgsz: Image resolution for training (e.g. 640).
        batch_size: Mini-batch size.
        base_model: Pretrained YOLO backbone (default: 'yolov8n.pt').
        output_weights: Target destination path for trained weights.

    Returns:
        True if training succeeded and weights were copied, False otherwise.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed. Install via: pip install ultralytics")
        return False

    yaml_path = Path(data_yaml)
    if not yaml_path.is_file():
        logger.error("Dataset config not found at %s", yaml_path)
        return False

    logger.info("Initializing base YOLO model from: %s", base_model)
    model = YOLO(base_model)

    logger.info("Starting training on dataset %s for %d epochs ...", yaml_path, epochs)
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            save=True,
            verbose=True,
        )
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.is_file():
            dest = Path(output_weights)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_pt, dest)
            logger.info("Training complete! Best weights saved to: %s", dest)
            return True
        else:
            logger.warning("best.pt was not found in %s", results.save_dir)
            return False
    except Exception as exc:
        logger.error("YOLO training failed: %s", exc, exc_info=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8 chessboard bounding box detector.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml dataset definition file (YOLOv8 format)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/weights/yolov8n_board.pt",
        help="Output weights path (default: data/weights/yolov8n_board.pt)",
    )
    args = parser.parse_args()

    success = train_yolo_board(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        output_weights=args.output,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
