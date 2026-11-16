"""64-square Cell ROI extraction and parallax height compensation.

Implements coordinate mapping and adaptive vertical crop expansion according to:
- STAGE_1.md, Section 4.5
- Explanatory Note, Section 2.4.1
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.core.config import VisionConfig
from src.core.datatypes import BoardOrientation, CellROI, CHESS_SQUARES
from src.vision.homography import GeometryTransformer


class CellRoiExtractor:
    """Extracts 64 square ROIs with parallax compensation from normalized 640x640 board image."""

    def __init__(
        self,
        board_size: int = 640,
        parallax_alpha: float = 1.25,
        output_cell_size: Tuple[int, int] = (112, 112),
        config: Optional[VisionConfig] = None,
    ) -> None:
        self.board_size = board_size
        self.parallax_alpha = parallax_alpha
        self.output_cell_size = output_cell_size

        if config is not None:
            self.board_size = config.target_board_size
            self.parallax_alpha = config.parallax_height_alpha

    def get_square_coordinates(
        self,
        square: str,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
    ) -> Tuple[int, int, int, int]:
        """Calculates normalized board square bounding box (xmin, ymin, xmax, ymax) in 640x640 space."""
        if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
            raise ValueError(f"Invalid algebraic square coordinate: '{square}'")

        file_idx = ord(square[0]) - ord("a")  # 0..7
        rank_idx = int(square[1])              # 1..8
        cell_size = self.board_size / 8.0

        if orientation == BoardOrientation.BLACK_BOTTOM:
            # Black camp at bottom (ranks 7-8 at bottom, ranks 1-2 at top)
            # File h at left (x=0), File a at right (x=board_size)
            x_min = int(round((7 - file_idx) * cell_size))
            x_max = int(round((7 - file_idx + 1) * cell_size))
            y_min = int(round((rank_idx - 1) * cell_size))
            y_max = int(round(rank_idx * cell_size))
        else:
            # White camp at bottom (ranks 1-2 at bottom, ranks 7-8 at top)
            # File a at left (x=0), File h at right (x=board_size)
            x_min = int(round(file_idx * cell_size))
            x_max = int(round((file_idx + 1) * cell_size))
            y_min = int(round((8 - rank_idx) * cell_size))
            y_max = int(round((8 - rank_idx + 1) * cell_size))

        return x_min, y_min, x_max, y_max

    def get_extended_coordinates(
        self,
        square: str,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
    ) -> Tuple[int, int, int, int]:
        """Calculates parallax-expanded bounding box (xmin, ymin_ext, xmax, ymax).
        
        Expands top edge upwards by delta_h to encompass heads of tall pieces,
        while anchoring bottom edge strictly to physical square base.
        """
        x_min, y_min, x_max, y_max = self.get_square_coordinates(square, orientation)
        cell_size = self.board_size / 8.0
        rank = int(square[1])

        # Farther ranks from camera experience greater perspective overlap
        if orientation == BoardOrientation.BLACK_BOTTOM:
            # For Black at bottom, rank 1 is far from camera, rank 8 is close
            distance_factor = 0.5 + 0.5 * ((8.0 - rank) / 7.0)
        else:
            # For White at bottom, rank 8 is far from camera, rank 1 is close
            distance_factor = 0.5 + 0.5 * ((rank - 1.0) / 7.0)

        delta_h = cell_size * (self.parallax_alpha - 1.0) * distance_factor
        y_min_ext = int(round(y_min - delta_h))

        return x_min, y_min_ext, x_max, y_max


    def build_cell_roi(
        self,
        square: str,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
        inverse_homography: Optional[np.ndarray] = None,
    ) -> CellROI:
        """Constructs a CellROI object with both board space and source frame space coordinates."""
        index = CHESS_SQUARES.index(square)
        bbox = self.get_square_coordinates(square, orientation)
        ext_bbox = self.get_extended_coordinates(square, orientation)

        # Center in board space
        cx_board = (bbox[0] + bbox[2]) / 2.0
        cy_board = (bbox[1] + bbox[3]) / 2.0

        # Center back-projected to original frame space
        center_image = (cx_board, cy_board)
        if inverse_homography is not None:
            proj = GeometryTransformer.project_board_to_image(
                np.array([[cx_board, cy_board]], dtype=np.float32),
                inverse_homography,
            )
            center_image = (float(proj[0, 0]), float(proj[0, 1]))

        return CellROI(
            square=square,
            index=index,
            board_bbox=bbox,
            extended_bbox=ext_bbox,
            center_image=center_image,
            center_board=(cx_board, cy_board),
        )

    def extract_cell_patch(
        self,
        board_image: np.ndarray,
        cell_roi: CellROI,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Crops cell image patch according to extended_bbox and resizes to target output_size."""
        target_size = output_size or self.output_cell_size
        x1, y1, x2, y2 = cell_roi.extended_bbox

        h_img, w_img = board_image.shape[:2]
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - h_img)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w_img)

        x1_c = max(0, min(w_img, x1))
        y1_c = max(0, min(h_img, y1))
        x2_c = max(0, min(w_img, x2))
        y2_c = max(0, min(h_img, y2))

        crop = board_image[y1_c:y2_c, x1_c:x2_c]
        if crop.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=board_image.dtype)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REFLECT_101,
            )


        interpolation = cv2.INTER_AREA if (crop.shape[1] > target_size[0]) else cv2.INTER_LINEAR
        return cv2.resize(crop, target_size, interpolation=interpolation)

    def extract_all_rois(
        self,
        board_image: np.ndarray,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
        inverse_homography: Optional[np.ndarray] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Tuple[CellROI, np.ndarray]]:
        """Extracts all 64 squares in canonical order a1..h8."""
        results: Dict[str, Tuple[CellROI, np.ndarray]] = {}
        for sq in CHESS_SQUARES:
            roi = self.build_cell_roi(sq, orientation=orientation, inverse_homography=inverse_homography)
            patch = self.extract_cell_patch(board_image, roi, output_size=output_size)
            results[sq] = (roi, patch)
        return results
