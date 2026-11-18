#!/usr/bin/env python3
"""Script to download or verify pretrained baseline piece classifier weights.

Reference:
- STAGE_3.md, Section 4.6
- chesscog (Wölflein & Arandjelović, 2021)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pretrained weight URLs and mirrors
MODEL_URLS = {
    "chesscog": "https://github.com/georg-wolflein/chesscog/releases/download/v0.1.0/piece_classifier.onnx",
    "efficientnet_b0": "https://github.com/ectrald/chess-models/releases/download/v1.0.0/efficientnet_b0_multihead.onnx",
}


def download_pretrained_model(
    model_name: str = "chesscog",
    output_path: Union[str, Path] = "models/piece_classifier.onnx",
    expected_sha256: Optional[str] = None,
) -> bool:
    """Downloads model weights file or verifies local cached copy."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.is_file():
        logger.info("Model file already exists at %s", out_file)
        if expected_sha256:
            with open(out_file, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h == expected_sha256:
                logger.info("Checksum verified: %s", h)
                return True
            logger.warning("Checksum mismatch for existing file. Re-downloading...")
        else:
            return True

    url = MODEL_URLS.get(model_name)
    if not url:
        logger.error("Unknown model name '%s'. Available: %s", model_name, list(MODEL_URLS.keys()))
        return False

    logger.info("Downloading %s weights from %s to %s ...", model_name, url, out_file)
    try:
        urllib.request.urlretrieve(url, str(out_file))
        logger.info("Download completed successfully.")
        return True
    except Exception as exc:
        logger.warning(
            "Download failed (%s). If running in an offline environment, "
            "place weights manually at %s.",
            exc,
            out_file,
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download piece classifier weights.")
    parser.add_argument(
        "--model",
        type=str,
        default="chesscog",
        choices=list(MODEL_URLS.keys()),
        help="Pretrained model source",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/piece_classifier.onnx",
        help="Target output ONNX path",
    )
    args = parser.parse_args()

    success = download_pretrained_model(model_name=args.model, output_path=args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

