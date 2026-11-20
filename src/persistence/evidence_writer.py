"""Evidence artifact writer for board observations and video frames.

Handles structured archival of visual evidence:
- Full normalized board crops in moment of move commitment
- Selective retention policy: compact JPEG for clear moves, PNG + meta for uncertain/occluded moves
- Metadata JSON sidecar files for reproducibility and auditing
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
import numpy as np

from src.core.datatypes import BoardObservation, MoveEvent

logger = logging.getLogger(__name__)


class EvidenceWriter:
    """Manages writing and indexing visual evidence artifacts for committed moves."""

    def __init__(
        self,
        evidence_dir: Path | str,
        policy: str = "selective",
        format: str = "jpg",
    ) -> None:
        """Initializes EvidenceWriter.

        Args:
            evidence_dir: Directory where evidence frames and metadata are saved.
            policy: Retention policy: 'all', 'selective', or 'none'.
            format: Image format: 'jpg' or 'png'.
        """
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy.lower()
        self.format = format.lower().replace(".", "")

    def save_evidence(
        self,
        event: MoveEvent,
        observation: BoardObservation,
        board_image: Optional[np.ndarray] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Saves visual evidence and metadata sidecar for a committed move event.

        Args:
            event: Confirmed MoveEvent.
            observation: Settled BoardObservation supporting the move.
            board_image: Optional normalized BGR board image [H, W, 3].
            extra_metadata: Optional dictionary with additional debug signals.

        Returns:
            Relative path to the primary saved evidence file, or None if skipped.
        """
        if self.policy == "none":
            return None

        base_stem = f"ply_{event.ply:04d}_{event.move_uci}_frame_{event.evidence_frame_id:06d}"
        img_rel_path: Optional[str] = None

        # 1. Save board image artifact if provided
        if board_image is not None and isinstance(board_image, np.ndarray) and board_image.size > 0:
            ext = self.format
            # Selective policy: use PNG for low-confidence or high-motion moves, JPG otherwise
            if self.policy == "selective" and (event.confidence < 0.25 or observation.motion_level > 0.05):
                ext = "png"

            img_filename = f"{base_stem}.{ext}"
            img_path = self.evidence_dir / img_filename

            if ext == "jpg" or ext == "jpeg":
                cv2.imwrite(str(img_path), board_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                cv2.imwrite(str(img_path), board_image)

            img_rel_path = str(img_path.relative_to(self.evidence_dir.parent))

        # 2. Save metadata sidecar JSON
        meta_filename = f"{base_stem}_meta.json"
        meta_path = self.evidence_dir / meta_filename

        metadata: Dict[str, Any] = {
            "event_id": event.event_id,
            "ply": event.ply,
            "move_uci": event.move_uci,
            "move_san": event.move_san,
            "confidence": event.confidence,
            "timestamp": event.timestamp,
            "evidence_frame_id": event.evidence_frame_id,
            "geometry_confidence": observation.geometry_confidence,
            "motion_detected": observation.motion_detected,
            "motion_level": observation.motion_level,
            "hand_present": observation.hand_present,
            "evidence_image": img_rel_path,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return img_rel_path or str(meta_path.relative_to(self.evidence_dir.parent))

