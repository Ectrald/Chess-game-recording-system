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
        elif orientation == BoardOrientation.WHITE_RIGHT:
            # Sideways 90° CCW: White at right (col 7=rank 1, col 0=rank 8), h at top (row 0), a at bottom (row 7)
            col_idx = 8 - rank_idx
            row_idx = 7 - file_idx
            x_min = int(round(col_idx * cell_size))
            x_max = int(round((col_idx + 1) * cell_size))
            y_min = int(round(row_idx * cell_size))
            y_max = int(round((row_idx + 1) * cell_size))
        elif orientation == BoardOrientation.WHITE_LEFT:
            # Sideways 90° CW: White at left (col 0=rank 1, col 7=rank 8), a at top (row 0), h at bottom (row 7)
            col_idx = rank_idx - 1
            row_idx = file_idx
            x_min = int(round(col_idx * cell_size))
            x_max = int(round((col_idx + 1) * cell_size))
            y_min = int(round(row_idx * cell_size))
            y_max = int(round((row_idx + 1) * cell_size))
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

        # Farther ranks/files from camera experience greater perspective overlap
        file_idx = ord(square[0]) - ord("a")
        if orientation == BoardOrientation.BLACK_BOTTOM:
            # For Black at bottom, rank 1 is far from camera, rank 8 is close
            distance_factor = 0.5 + 0.5 * ((8.0 - rank) / 7.0)
        elif orientation == BoardOrientation.WHITE_RIGHT:
            # For White at right, file h is at top (far from camera), file a is close
            distance_factor = 0.5 + 0.5 * (file_idx / 7.0)
        elif orientation == BoardOrientation.WHITE_LEFT:
            # For White at left, file a is at top (far from camera), file h is close
            distance_factor = 0.5 + 0.5 * ((7.0 - file_idx) / 7.0)
        else:
            # For White at bottom, rank 8 is far from camera, rank 1 is close
            distance_factor = 0.5 + 0.5 * ((rank - 1.0) / 7.0)

        delta_h = cell_size * (self.parallax_alpha - 1.0) * distance_factor
        y_min_ext = int(round(y_min - delta_h))

        return x_min, y_min_ext, x_max, y_max

    def get_base_core_bbox(
        self,
        square: str,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
    ) -> Tuple[int, int, int, int]:
        """Returns the base-focused contact core region within a square to minimize neighbor overhang."""
        x_min, y_min, x_max, y_max = self.get_square_coordinates(square, orientation)
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        return (
            int(round(x_min + 0.16 * w)),
            int(round(y_min + 0.28 * h)),
            int(round(x_max - 0.16 * w)),
            int(round(y_max - 0.12 * h)),
        )

    def get_clean_base_mask(
        self,
        square: str,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Returns binary uint8 elliptical mask over the unoccluded central base zone."""
        h_img, w_img = image_shape if image_shape is not None else (self.board_size, self.board_size)
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        x_min, y_min, x_max, y_max = self.get_square_coordinates(square, orientation)
        w = max(1.0, float(x_max - x_min))
        h = max(1.0, float(y_max - y_min))
        cx = int(round(x_min + 0.50 * w))
        cy = int(round(y_min + 0.56 * h))
        rx = max(2, int(round(0.26 * w)))
        ry = max(2, int(round(0.24 * h)))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        return mask


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
        """Crops cell image patch according to extended_bbox and resizes to target output_size.

        If board_image is padded (width/height > self.board_size), shifts coordinates by the padding
        offset so edge squares preserve real source-frame context instead of border reflection.
        """
        target_size = output_size or self.output_cell_size
        x1, y1, x2, y2 = cell_roi.extended_bbox

        h_img, w_img = board_image.shape[:2]
        pad_x = max(0, (w_img - self.board_size) // 2)
        pad_y = max(0, (h_img - self.board_size) // 2)
        x1 += pad_x
        x2 += pad_x
        y1 += pad_y
        y2 += pad_y

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
                cv2.BORDER_REPLICATE,
            )


        interpolation = (
            cv2.INTER_AREA
            if (crop.shape[1] > target_size[0] or crop.shape[0] > target_size[1])
            else cv2.INTER_LINEAR
        )
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

    def extract_occupancy_crops(
        self,
        board_image: np.ndarray,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
        output_size: Tuple[int, int] = (100, 100),
    ) -> Dict[str, np.ndarray]:
        """Extracts unexpanded 1:1 square crops matching chesscog occupancy_classifier (ResNet 100x100)."""
        h_img, w_img = board_image.shape[:2]
        crops: Dict[str, np.ndarray] = {}
        for sq in CHESS_SQUARES:
            x1, y1, x2, y2 = self.get_square_coordinates(sq, orientation)
            cx1 = max(0, min(w_img, x1))
            cy1 = max(0, min(h_img, y1))
            cx2 = max(0, min(w_img, x2))
            cy2 = max(0, min(h_img, y2))
            patch = board_image[cy1:cy2, cx1:cx2]
            if patch.size == 0:
                crops[sq] = np.zeros((output_size[1], output_size[0], 3), dtype=board_image.dtype)
            else:
                interp = cv2.INTER_AREA if (patch.shape[1] > output_size[0]) else cv2.INTER_LINEAR
                crops[sq] = cv2.resize(patch, output_size, interpolation=interp)
        return crops
