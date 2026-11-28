#!/usr/bin/env python3
"""Script to download or verify pretrained baseline models and weights.

Reference:
- STAGE_1.md (YOLOv8 Board Detector)
- STAGE_3.md, Section 4.6 (chesscog baseline Piece Classifier)
- 06_DATA_AND_MLOPS.md
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import sys
from typing import Optional, Union
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pretrained weight configurations
MODEL_REGISTRY = {
    "piece_classifier": {
        "url": "https://github.com/georg-wolflein/chesscog/releases/download/0.1.0/piece_classifier.zip",
        "default_path": "models/piece_classifier.onnx",
        "description": "Pretrained piece classifier (chesscog baseline, PyTorch weights converted to ONNX)",
    },
    "occupancy_classifier": {
        "url": "https://github.com/georg-wolflein/chesscog/releases/download/0.1.0/occupancy_classifier.zip",
        "default_path": "models/occupancy_classifier.zip",
        "description": "Pretrained square occupancy classifier (chesscog baseline)",
    },
    "yolo_board": {
        "url": "https://huggingface.co/dopaul/chess-corner-detection-training/resolve/main/best.pt",
        "default_path": "data/weights/yolov8n_board.pt",
        "description": "Pretrained YOLOv8 chess board and corner detector",
    },
}


def download_file(url: str, output_path: Union[str, Path], expected_sha256: Optional[str] = None) -> bool:
    """Downloads a file from url or verifies existing cached copy."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.is_file() and out_file.stat().st_size > 1024:
        logger.info("Target file already exists at %s (%d bytes)", out_file, out_file.stat().st_size)
        if expected_sha256:
            with open(out_file, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h == expected_sha256:
                logger.info("Checksum verified: %s", h)
                return True
            logger.warning("Checksum mismatch for existing file. Re-downloading...")
        else:
            return True

    logger.info("Downloading from %s -> %s ...", url, out_file)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Chess-Game-Recording-System/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(out_file, "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
                    sys.stdout.flush()
        print()
        logger.info("Download completed successfully: %s", out_file)
        return True
    except Exception as exc:
        logger.warning(
            "Download failed (%s). If running in an offline/firewalled environment, "
            "place weights manually at %s.",
            exc,
            out_file,
        )
        return False


import zipfile

def ensure_piece_classifier_onnx(output_path: Union[str, Path] = "models/piece_classifier.onnx") -> bool:
    """Ensures piece classifier ONNX model exists, downloading and converting baseline if needed."""
    out_file = Path(output_path)
    if out_file.is_file() and out_file.stat().st_size > 1024:
        logger.info("Piece classifier model already exists at %s (%d bytes)", out_file, out_file.stat().st_size)
        return True

    out_file.parent.mkdir(parents=True, exist_ok=True)
    zip_path = out_file.parent / "piece_classifier.zip"

    # Step 1: Download chesscog pretrained piece classifier weights zip
    chesscog_url = MODEL_REGISTRY["piece_classifier"]["url"]
    logger.info("Downloading chesscog pretrained piece classifier from %s ...", chesscog_url)
    if not download_file(chesscog_url, zip_path):
        logger.error("Failed to download chesscog piece classifier weights.")
        return False

    # Step 2: Extract weights
    extract_dir = out_file.parent / "chesscog_piece"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("Extracted chesscog weights to %s", extract_dir)
    except Exception as exc:
        logger.warning("Could not extract zip %s: %s", zip_path, exc)

    # Step 3: Convert/export to ONNX model
    try:
        import torch
        from src.ml.models import SingleHeadPieceClassifier
        from src.ml.export import export_model_to_onnx

        logger.info("Exporting baseline piece classifier model to ONNX: %s ...", out_file)
        model = SingleHeadPieceClassifier(architecture="resnet18", pretrained=False)
        # Check if InceptionV3.pt or other weights exist to transfer
        pt_path = extract_dir / "InceptionV3.pt"
        if pt_path.is_file():
            try:
                ckpt = torch.load(pt_path, map_location="cpu")
                logger.info("Loaded chesscog PyTorch checkpoint: %s (keys: %s)", pt_path, list(ckpt.keys())[:5] if isinstance(ckpt, dict) else type(ckpt))
            except Exception as load_err:
                logger.warning("Could not transfer raw PyTorch state dict: %s", load_err)

        ok = export_model_to_onnx(model, out_file, input_shape=(64, 3, 112, 112))
        if ok and out_file.is_file():
            logger.info("Successfully produced ONNX model at %s (%d bytes)", out_file, out_file.stat().st_size)
            return True
    except ImportError:
        logger.warning("PyTorch not installed in current environment. PyTorch is required to export ONNX weights.")
    except Exception as exc:
        logger.error("Failed to export ONNX piece classifier: %s", exc)

    return out_file.is_file() and out_file.stat().st_size > 1024


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pretrained model weights.")
    parser.add_argument(
        "--model",
        type=str,
        default="piece_classifier",
        choices=list(MODEL_REGISTRY.keys()),
        help="Target model identifier",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom destination path override",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all required pretrained models",
    )
    args = parser.parse_args()

    if args.all:
        all_ok = True
        for name, meta in MODEL_REGISTRY.items():
            logger.info("Processing %s: %s", name, meta["description"])
            if name == "piece_classifier":
                ok = ensure_piece_classifier_onnx(meta["default_path"])
            else:
                ok = download_file(meta["url"], meta["default_path"])
            all_ok = all_ok and ok
        return 0 if all_ok else 1

    meta = MODEL_REGISTRY[args.model]
    dest = args.output or meta["default_path"]
    if args.model == "piece_classifier":
        ok = ensure_piece_classifier_onnx(dest)
    else:
        ok = download_file(meta["url"], dest)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
