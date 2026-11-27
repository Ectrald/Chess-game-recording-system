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
    "chessq_vit_int8": {
        "url": "https://huggingface.co/joelseytre/chessq-lite/resolve/main/cq-vit-s-644-v6-seed1-onnx-int8-r1.onnx",
        "default_path": "models/chessq_vit_int8.onnx",
        "description": "Holistic ViT-S14 INT8 whole-board FEN & viewing-direction ONNX classifier (ChessQ-Lite)",
    },
    "chessq_vit_config": {
        "url": "https://huggingface.co/joelseytre/chessq-lite/resolve/main/cq-vit-s-644-v6-seed1-onnx-int8-r1.json",
        "default_path": "models/chessq_vit_int8.json",
        "description": "ChessQ-Lite ONNX preprocessing & output schema metadata",
    },
    "dinov2_small_int8": {
        "url": "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model_int8.onnx",
        "default_path": "models/dinov2_small_int8.onnx",
        "description": "DINOv2-Small INT8 patch embedding model for illumination-invariant change detection",
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
    piece_zip_path = out_file.parent / "piece_classifier.zip"
    occ_zip_path = out_file.parent / "occupancy_classifier.zip"

    # Step 1: Download chesscog pretrained weights zips
    chesscog_url = MODEL_REGISTRY["piece_classifier"]["url"]
    occ_url = MODEL_REGISTRY["occupancy_classifier"]["url"]
    logger.info("Downloading chesscog pretrained piece classifier from %s ...", chesscog_url)
    if not download_file(chesscog_url, piece_zip_path):
        logger.error("Failed to download chesscog piece classifier weights.")
        return False
    logger.info("Downloading chesscog pretrained occupancy classifier from %s ...", occ_url)
    if not download_file(occ_url, occ_zip_path):
        logger.error("Failed to download chesscog occupancy classifier weights.")
        return False

    # Step 2: Extract weights
    extract_piece_dir = out_file.parent / "chesscog_piece"
    extract_occ_dir = out_file.parent / "chesscog_occupancy"
    try:
        with zipfile.ZipFile(piece_zip_path, "r") as zf:
            zf.extractall(extract_piece_dir)
        with zipfile.ZipFile(occ_zip_path, "r") as zf:
            zf.extractall(extract_occ_dir)
        logger.info("Extracted chesscog weights to %s and %s", extract_piece_dir, extract_occ_dir)
    except Exception as exc:
        logger.warning("Could not extract chesscog zips: %s", exc)

    # Step 3: Load pickled chesscog PyTorch models and export combined 13-class ONNX
    try:
        import types
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from src.ml.export import export_model_to_onnx

        occ_pt = extract_occ_dir / "ResNet.pt"
        pc_pt = extract_piece_dir / "InceptionV3.pt"
        if not (occ_pt.is_file() and pc_pt.is_file()):
            logger.error("Expected chesscog checkpoints missing: %s, %s", occ_pt, pc_pt)
            return False

        for mod_name in [
            "chesscog",
            "chesscog.occupancy_classifier",
            "chesscog.occupancy_classifier.models",
            "chesscog.piece_classifier",
            "chesscog.piece_classifier.models",
        ]:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)

        class _StubModule(nn.Module):
            pass

        sys.modules["chesscog.occupancy_classifier.models"].ResNet = _StubModule
        sys.modules["chesscog.piece_classifier.models"].InceptionV3 = _StubModule

        occ_ckpt = torch.load(occ_pt, map_location="cpu", weights_only=False)
        pc_ckpt = torch.load(pc_pt, map_location="cpu", weights_only=False)
        occ_net = occ_ckpt.model.eval()
        pc_net = pc_ckpt.model.eval()
        if hasattr(pc_net, "aux_logits"):
            pc_net.aux_logits = False

        class _ChesscogCombined(nn.Module):
            def __init__(self, occ_m: nn.Module, pc_m: nn.Module) -> None:
                super().__init__()
                self.occ_model = occ_m
                self.pc_model = pc_m
                self.register_buffer(
                    "perm_idx",
                    torch.tensor([9, 8, 6, 11, 10, 7, 3, 2, 0, 5, 4, 1], dtype=torch.long),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x_occ = F.interpolate(x, size=(100, 100), mode="bilinear", align_corners=False)
                p_occ = F.softmax(self.occ_model(x_occ), dim=-1)
                x_pc = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
                p_pc = F.softmax(self.pc_model(x_pc), dim=-1).index_select(1, self.perm_idx)
                return torch.cat([p_occ[:, 0:1], p_occ[:, 1:2] * p_pc], dim=-1)

        combined = _ChesscogCombined(occ_net, pc_net).eval()
        # Also export standalone ResNet occupancy_classifier ONNX (expects tight 100x100 crops)
        occ_onnx_out = out_file.parent / "occupancy_classifier.onnx"
        export_model_to_onnx(occ_net, occ_onnx_out, input_shape=(64, 3, 100, 100))
        logger.info("Exporting combined chesscog 13-class classifier to ONNX: %s ...", out_file)
        ok = export_model_to_onnx(combined, out_file, input_shape=(64, 3, 112, 112))
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
