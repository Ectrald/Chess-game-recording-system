"""Unit tests for chessboard cell ROI extraction and parallax compensation in src/vision/roi_extractor.py."""

import numpy as np
import pytest

from src.core.datatypes import BoardOrientation, CHESS_SQUARES
from src.vision.homography import GeometryTransformer
from src.vision.roi_extractor import CellRoiExtractor


class TestCellRoiCoordinates:
    def test_all_64_squares_canonical(self):
        extractor = CellRoiExtractor(board_size=640)
        rois = extractor.extract_all_rois(np.zeros((640, 640, 3), dtype=np.uint8))
        assert len(rois) == 64
        assert list(rois.keys()) == CHESS_SQUARES

    def test_white_bottom_square_positions(self):
        extractor = CellRoiExtractor(board_size=640)
        # a1: bottom-left square
        a1_box = extractor.get_square_coordinates("a1", BoardOrientation.WHITE_BOTTOM)
        assert a1_box == (0, 560, 80, 640)

        # h1: bottom-right square
        h1_box = extractor.get_square_coordinates("h1", BoardOrientation.WHITE_BOTTOM)
        assert h1_box == (560, 560, 640, 640)

        # a8: top-left square
        a8_box = extractor.get_square_coordinates("a8", BoardOrientation.WHITE_BOTTOM)
        assert a8_box == (0, 0, 80, 80)

        # h8: top-right square
        h8_box = extractor.get_square_coordinates("h8", BoardOrientation.WHITE_BOTTOM)
        assert h8_box == (560, 0, 640, 80)

    def test_black_bottom_square_positions(self):
        extractor = CellRoiExtractor(board_size=640)
        # a1: top-right square from Black perspective
        a1_box = extractor.get_square_coordinates("a1", BoardOrientation.BLACK_BOTTOM)
        assert a1_box == (560, 0, 640, 80)

        # h8: bottom-left square from Black perspective
        h8_box = extractor.get_square_coordinates("h8", BoardOrientation.BLACK_BOTTOM)
        assert h8_box == (0, 560, 80, 640)

    def test_parallax_height_expansion(self):
        extractor = CellRoiExtractor(board_size=640, parallax_alpha=1.25)
        # Rank 1: close to camera
        r1_base = extractor.get_square_coordinates("e1", BoardOrientation.WHITE_BOTTOM)
        r1_ext = extractor.get_extended_coordinates("e1", BoardOrientation.WHITE_BOTTOM)
        assert r1_ext[3] == r1_base[3]  # Bottom baseline strictly anchored
        assert r1_ext[1] < r1_base[1]   # Top edge expanded upwards

        # Rank 8: far from camera
        r8_base = extractor.get_square_coordinates("e8", BoardOrientation.WHITE_BOTTOM)
        r8_ext = extractor.get_extended_coordinates("e8", BoardOrientation.WHITE_BOTTOM)
        assert r8_ext[3] == r8_base[3]  # Bottom baseline strictly anchored
        # Height expansion for rank 8 should be larger or equal to rank 1
        delta_h1 = r1_base[1] - r1_ext[1]
        delta_h8 = r8_base[1] - r8_ext[1]
        assert delta_h8 >= delta_h1

    def test_patch_extraction_shape(self):
        extractor = CellRoiExtractor(board_size=640, output_cell_size=(112, 112))
        dummy_board = np.ones((640, 640, 3), dtype=np.uint8) * 150
        roi = extractor.build_cell_roi("e4", BoardOrientation.WHITE_BOTTOM)
        patch = extractor.extract_cell_patch(dummy_board, roi)
        assert patch.shape == (112, 112, 3)

    def test_back_projection_with_homography(self, sample_corners):
        _, H_inv = GeometryTransformer.compute_homography(sample_corners, target_size=640)
        extractor = CellRoiExtractor(board_size=640)
        roi = extractor.build_cell_roi("e4", BoardOrientation.WHITE_BOTTOM, inverse_homography=H_inv)
        # Center image coordinate should fall within original frame boundaries
        cx, cy = roi.center_image
        assert 0.0 <= cx <= 1920.0
        assert 0.0 <= cy <= 1080.0

    def test_base_core_and_padded_edge_extraction(self):
        extractor = CellRoiExtractor(board_size=640, output_cell_size=(112, 112))
        # Base core stays strictly inside board_bbox
        bx1, by1, bx2, by2 = extractor.get_base_core_bbox("a1", BoardOrientation.WHITE_RIGHT)
        sq_x1, sq_y1, sq_x2, sq_y2 = extractor.get_square_coordinates("a1", BoardOrientation.WHITE_RIGHT)
        assert sq_x1 < bx1 < bx2 < sq_x2
        assert sq_y1 < by1 < by2 < sq_y2

        # Padded board warp (800x800, pad=80) supplies real top context for edge square h1 instead of reflection
        padded = np.zeros((800, 800, 3), dtype=np.uint8)
        # Put bright marker in top padding strip above h1 (x in [80+560..80+640], y in [40..80])
        padded[40:80, 640:720] = (10, 200, 10)
        roi_h1 = extractor.build_cell_roi("h1", BoardOrientation.WHITE_RIGHT)
        assert roi_h1.extended_bbox[1] < 0
        patch = extractor.extract_cell_patch(padded, roi_h1)
        assert patch.shape == (112, 112, 3)
        assert patch[:25, :, 1].mean() > 100.0

