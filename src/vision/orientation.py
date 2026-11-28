"""Three-factor spatial orientation resolution for normalized chessboards.

Implements orientation cascade (corner square color, mass distribution, user config)
according to:
- STAGE_1.md, Section 4.4
- Explanatory Note, Section 2.4.1
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from src.core.config import OrientationConfig
from src.core.datatypes import BoardOrientation


class OrientationResolver:
    """Resolves spatial orientation of a normalized board image (640x640) relative to player camps."""

    def __init__(
        self,
        config: Optional[OrientationConfig] = None,
        color_diff_threshold: float = 15.0,
        mass_diff_threshold: float = 10.0,
    ) -> None:
        self.color_check_enabled = True
        self.mass_distribution_enabled = True
        self.allow_sideways = True
        self.fallback = BoardOrientation.UNKNOWN

        if config is not None:
            self.color_check_enabled = config.color_check_enabled
            self.mass_distribution_enabled = config.mass_distribution_enabled
            self.allow_sideways = getattr(config, "allow_sideways", True)
            try:
                self.fallback = BoardOrientation(config.fallback_orientation)
            except ValueError:
                self.fallback = BoardOrientation.UNKNOWN

        self.color_diff_threshold = color_diff_threshold
        self.mass_diff_threshold = mass_diff_threshold

    def check_corner_color_validity(self, board_image: np.ndarray) -> Tuple[bool, float, float]:
        """Verifies FIDE rule: bottom-right square (h1 or a8) MUST be a light square.
        
        Rejects 90° and 270° board rotations where bottom-right becomes a dark square.
        
        Returns:
            Tuple of (is_valid, br_intensity, bl_intensity).
        """
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY) if board_image.ndim == 3 else board_image
        h, w = gray.shape[:2]
        cell_w = w / 8.0
        cell_h = h / 8.0

        # Sample corners of squares outside the center circular piece area (rank 1 row: [7*cell_h, 8*cell_h])
        # Bottom-Right square: file 7 (x: 7*cell_w..8*cell_w), rank 0 (y: 7*cell_h..8*cell_h)
        br_x1 = int(round(7 * cell_w + 0.10 * cell_w))
        br_x2 = int(round(7 * cell_w + 0.25 * cell_w))
        br_y1 = int(round(7 * cell_h + 0.10 * cell_h))
        br_y2 = int(round(7 * cell_h + 0.25 * cell_h))

        # Bottom-Left square: file 0 (x: 0..cell_w), rank 0 (y: 7*cell_h..8*cell_h)
        bl_x1 = int(round(0.10 * cell_w))
        bl_x2 = int(round(0.25 * cell_w))
        bl_y1 = int(round(7 * cell_h + 0.10 * cell_h))
        bl_y2 = int(round(7 * cell_h + 0.25 * cell_h))

        br_patch = gray[br_y1:br_y2, br_x1:br_x2]
        bl_patch = gray[bl_y1:bl_y2, bl_x1:bl_x2]

        br_mean = float(np.mean(br_patch)) if br_patch.size > 0 else 0.0
        bl_mean = float(np.mean(bl_patch)) if bl_patch.size > 0 else 0.0

        # If manual outward corners landed on a dark outer bezel, fall back to central 4x4 checkerboard parity
        if min(br_mean, bl_mean) < 40.0:
            even_vals = []
            odd_vals = []
            for r in range(2, 6):
                cy1 = int(round((r + 0.30) * cell_h))
                cy2 = int(round((r + 0.70) * cell_h))
                for c in range(2, 6):
                    cx1 = int(round((c + 0.30) * cell_w))
                    cx2 = int(round((c + 0.70) * cell_w))
                    patch = gray[cy1:cy2, cx1:cx2]
                    if patch.size > 0:
                        val = float(np.mean(patch))
                        if (r + c) % 2 == 0:
                            even_vals.append(val)
                        else:
                            odd_vals.append(val)
            if even_vals and odd_vals:
                even_med = float(np.median(even_vals))
                odd_med = float(np.median(odd_vals))
                if max(even_med, odd_med) > max(br_mean, bl_mean):
                    br_mean, bl_mean = even_med, odd_med

        # Bottom-right must be significantly lighter than bottom-left
        is_valid = (br_mean - bl_mean) >= self.color_diff_threshold
        return is_valid, br_mean, bl_mean

    def check_mass_distribution(self, board_image: np.ndarray) -> Tuple[BoardOrientation, float, float]:
        """Analyzes luminance and piece mass between top ranks (rows 7-8) and bottom ranks (rows 1-2).
        
        On initial board:
        - White camp (ranks 1-2) has light pieces (higher luminance).
        - Black camp (ranks 7-8) has dark pieces (lower luminance).
        
        Returns:
            Tuple of (detected_orientation, top_mean, bot_mean).
        """
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY) if board_image.ndim == 3 else board_image
        h, _ = gray.shape[:2]
        rank_2h = int(round(h * 2.0 / 8.0))

        # Top 2 ranks (ranks 7-8 from White perspective, or 1-2 from Black perspective)
        top_ranks = gray[0:rank_2h, :]
        # Bottom 2 ranks (ranks 1-2 from White perspective, or 7-8 from Black perspective)
        bot_ranks = gray[h - rank_2h : h, :]

        top_mean = float(np.mean(top_ranks))
        bot_mean = float(np.mean(bot_ranks))

        diff = bot_mean - top_mean
        if abs(diff) < self.mass_diff_threshold:
            return BoardOrientation.UNKNOWN, top_mean, bot_mean

        if diff > 0:
            return BoardOrientation.WHITE_BOTTOM, top_mean, bot_mean
        else:
            return BoardOrientation.BLACK_BOTTOM, top_mean, bot_mean

    def check_sideways_mass_distribution(self, board_image: np.ndarray) -> Tuple[BoardOrientation, float, float]:
        """Analyzes piece luminance between left columns (0..2) and right columns (6..8) for sideways boards.

        Returns:
            Tuple of (detected_orientation, left_mean, right_mean).
        """
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY) if board_image.ndim == 3 else board_image
        _, w = gray.shape[:2]
        col_2w = int(round(w * 2.0 / 8.0))

        left_cols = gray[:, 0:col_2w]
        right_cols = gray[:, w - col_2w : w]

        left_mean = float(np.mean(left_cols))
        right_mean = float(np.mean(right_cols))

        diff = right_mean - left_mean
        if abs(diff) < self.mass_diff_threshold:
            return BoardOrientation.UNKNOWN, left_mean, right_mean

        if diff > 0:
            return BoardOrientation.WHITE_RIGHT, left_mean, right_mean
        else:
            return BoardOrientation.WHITE_LEFT, left_mean, right_mean

    def resolve(
        self,
        board_image: np.ndarray,
        user_preference: Optional[BoardOrientation] = None,
    ) -> BoardOrientation:
        """Resolves board orientation using the 3-factor cascade (0°, 90°, 180°, 270°).

        1. If user_preference is provided and valid, returns it immediately.
        2. Corner square color test: distinguishes standard (0°/180°) vs sideways (90°/270°).
        3. Mass distribution test: distinguishes WHITE_BOTTOM vs BLACK_BOTTOM (or WHITE_RIGHT vs WHITE_LEFT).
        4. Fallback to configured default or UNKNOWN.
        """
        # Factor 1: Explicit user preference
        if user_preference is not None and user_preference != BoardOrientation.UNKNOWN:
            return user_preference

        is_sideways = False
        # Factor 2: Corner square color validity
        if self.color_check_enabled:
            is_valid, br_mean, bl_mean = self.check_corner_color_validity(board_image)
            if not is_valid:
                # Bottom-left light and bottom-right dark -> 90° or 270° sideways board
                if self.allow_sideways and (bl_mean - br_mean) >= self.color_diff_threshold:
                    is_sideways = True
                else:
                    return BoardOrientation.UNKNOWN

        # Factor 3: Mass distribution analysis
        if self.mass_distribution_enabled:
            if is_sideways:
                detected, _, _ = self.check_sideways_mass_distribution(board_image)
            else:
                detected, _, _ = self.check_mass_distribution(board_image)
            if detected != BoardOrientation.UNKNOWN:
                return detected

        # Factor 4: Fallback
        return self.fallback

