"""Differential motion detector in normalized chessboard space.

Implements rectified frame differencing, luminance normalization, morphological
clustering, and per-square motion analysis according to:
- STAGE_2.md, Section 4.1
- Explanatory Note, Section 2.4.2
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from src.core.config import TemporalConfig
from src.core.datatypes import BoardOrientation, CellROI, CHESS_SQUARES, MotionMetrics
from src.vision.roi_extractor import CellRoiExtractor


class MotionDetector:
    """Detects physical motion on the chessboard via rectified frame differencing."""

    def __init__(
        self,
        config: Optional[TemporalConfig] = None,
        pixel_diff_threshold: int = 25,
        motion_threshold: float = 0.04,
        ema_alpha: float = 0.35,
        blur_ksize: int = 5,
        board_size: int = 640,
    ) -> None:
        self.pixel_diff_threshold = pixel_diff_threshold
        self.motion_threshold = motion_threshold
        self.ema_alpha = ema_alpha
        self.blur_ksize = blur_ksize
        self.board_size = board_size

        if config is not None:
            self.motion_threshold = config.motion_diff_threshold
            self.pixel_diff_threshold = config.pixel_diff_threshold
            self.ema_alpha = config.motion_ema_alpha

        self.extractor = CellRoiExtractor(board_size=self.board_size)
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_mean_lum: float = 0.0
        self.smoothed_motion_score: float = 0.0
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def reset(self) -> None:
        """Clears previous frame history and resets temporal filter state."""
        self.prev_gray = None
        self.prev_mean_lum = 0.0
        self.smoothed_motion_score = 0.0

    def process_board_image(
        self,
        board_image: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> MotionMetrics:
        """Analyzes motion between consecutive normalized board frames.
        
        Args:
            board_image: Normalized board image array (BGR or grayscale).
            frame_id: Sequential video frame index.
            timestamp: Video timestamp in seconds.
            orientation: Current board orientation (WHITE_BOTTOM or BLACK_BOTTOM).
            
        Returns:
            MotionMetrics with global and per-square motion intensities.
        """
        if board_image.ndim == 3 and board_image.shape[2] == 3:
            gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        elif board_image.ndim == 3 and board_image.shape[2] == 4:
            gray = cv2.cvtColor(board_image, cv2.COLOR_BGRA2GRAY)
        elif board_image.ndim == 3 and board_image.shape[2] == 1:
            gray = board_image[:, :, 0].copy()
        else:
            gray = board_image.copy()
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Gaussian blur to filter sensor noise
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 1.0)
        curr_mean = float(np.mean(blurred))

        # First frame or shape change: initialize history
        if self.prev_gray is None or self.prev_gray.shape != blurred.shape:
            self.prev_gray = blurred
            self.prev_mean_lum = curr_mean
            self.smoothed_motion_score = 0.0
            zero_scores = {sq: 0.0 for sq in CHESS_SQUARES}
            empty_mask = np.zeros(gray.shape, dtype=np.uint8)
            return MotionMetrics(
                timestamp=timestamp,
                frame_id=frame_id,
                global_motion_score=0.0,
                square_motion_scores=zero_scores,
                is_motion_detected=False,
                motion_mask=empty_mask,
            )

        # Global luminance normalization (compensates for camera auto-exposure changes)
        if curr_mean > 1e-3 and self.prev_mean_lum > 1e-3:
            scale = self.prev_mean_lum / curr_mean
            # Limit scale to reasonable exposure range [0.7, 1.4]
            scale = max(0.7, min(1.4, scale))
            curr_norm = np.clip(blurred.astype(np.float32) * scale, 0.0, 255.0).astype(np.uint8)
        else:
            curr_norm = blurred

        # Absolute frame difference
        diff = cv2.absdiff(curr_norm, self.prev_gray)

        # Binary thresholding
        _, binary = cv2.threshold(diff, self.pixel_diff_threshold, 255, cv2.THRESH_BINARY)

        # Morphological closing to join sparse clusters
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel)

        # Calculate raw global motion ratio
        total_pixels = float(binary.size)
        active_pixels = float(np.count_nonzero(binary))
        raw_motion_score = active_pixels / total_pixels if total_pixels > 0 else 0.0

        # Temporal smoothing via Exponential Moving Average (EMA)
        self.smoothed_motion_score = (
            self.ema_alpha * raw_motion_score
            + (1.0 - self.ema_alpha) * self.smoothed_motion_score
        )

        # Ensure ROI grid matches incoming warped board image dimensions
        img_board_dim = int(min(gray.shape[0], gray.shape[1]))
        if img_board_dim > 0 and img_board_dim != self.extractor.board_size:
            self.extractor = CellRoiExtractor(board_size=img_board_dim)

        # Calculate per-square motion scores across all 64 squares
        square_scores: Dict[str, float] = {}
        for sq in CHESS_SQUARES:
            x1, y1, x2, y2 = self.extractor.get_square_coordinates(sq, orientation)
            x1_c = max(0, min(binary.shape[1], x1))
            y1_c = max(0, min(binary.shape[0], y1))
            x2_c = max(0, min(binary.shape[1], x2))
            y2_c = max(0, min(binary.shape[0], y2))

            cell_mask = binary[y1_c:y2_c, x1_c:x2_c]
            if cell_mask.size > 0:
                sq_active = float(np.count_nonzero(cell_mask))
                sq_score = sq_active / float(cell_mask.size)
            else:
                sq_score = 0.0
            square_scores[sq] = sq_score

        # Update reference frame
        self.prev_gray = blurred
        self.prev_mean_lum = curr_mean

        is_moving = self.smoothed_motion_score >= self.motion_threshold

        return MotionMetrics(
            timestamp=timestamp,
            frame_id=frame_id,
            global_motion_score=float(self.smoothed_motion_score),
            square_motion_scores=square_scores,
            is_motion_detected=is_moving,
            motion_mask=binary,
        )
