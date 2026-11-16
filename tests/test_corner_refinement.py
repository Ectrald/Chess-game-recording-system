"""Unit tests for chessboard corner refinement in src/vision/corner_refinement.py."""

import math
import numpy as np
import pytest

from src.vision.corner_refinement import (
    CornerRefiner,
    detect_corners_hough_lines,
    order_corners_canonical,
    refine_subpixel_corners,
)
from src.vision.synthetic_board import SyntheticBoardGenerator


class TestCanonicalCornerOrdering:
    def test_canonical_ordering_clockwise(self):
        # TL, TR, BR, BL in arbitrary order
        raw_corners = [
            (268.0, 950.0),   # BL
            (1344.0, 162.0),  # TR
            (576.0, 162.0),   # TL
            (1651.0, 950.0),  # BR
        ]
        ordered = order_corners_canonical(raw_corners)
        assert ordered[0] == (576.0, 162.0)   # TL
        assert ordered[1] == (1344.0, 162.0)  # TR
        assert ordered[2] == (1651.0, 950.0)  # BR
        assert ordered[3] == (268.0, 950.0)   # BL

    def test_invalid_corner_count(self):
        with pytest.raises(ValueError):
            order_corners_canonical([(0.0, 0.0), (10.0, 10.0)])


class TestSubpixelRefinement:
    def test_refine_subpixel_corners(self):
        gen = SyntheticBoardGenerator()
        frame = gen.render_frame(0)
        gt_corners = gen.corners

        # Perturb ground truth slightly by 1-2 pixels
        perturbed = [(c[0] + 1.2, c[1] - 0.8) for c in gt_corners]
        refined = refine_subpixel_corners(frame.image, perturbed, win_size=11)

        assert len(refined) == 4
        # Verify corners are valid numbers
        for pt in refined:
            assert not math.isnan(pt[0])
            assert not math.isnan(pt[1])

        # Verify distance to ground truth is small
        diffs = [math.hypot(r[0] - g[0], r[1] - g[1]) for r, g in zip(refined, gt_corners)]
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / 4.0)
        assert rmse < 3.0


class TestCornerDetectorAndRefiner:
    def test_detect_hough_lines_on_synthetic_board(self):
        gen = SyntheticBoardGenerator()
        frame = gen.render_frame(0)
        gt = gen.corners

        corners = detect_corners_hough_lines(frame.image)
        assert corners is not None
        assert len(corners) == 4

        diffs = [math.hypot(c[0] - g[0], c[1] - g[1]) for c, g in zip(corners, gt)]
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / 4.0)
        # DoD requirement: RMSE <= 2.0 px
        assert rmse <= 2.0

    def test_corner_refiner_end_to_end(self):
        refiner = CornerRefiner()
        gen = SyntheticBoardGenerator()
        frame = gen.render_frame(0)
        gt = gen.corners

        corners = refiner.find_board_corners(frame.image)
        assert corners is not None
        assert len(corners) == 4

        diffs = [math.hypot(c[0] - g[0], c[1] - g[1]) for c, g in zip(corners, gt)]
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / 4.0)
        assert rmse <= 2.0

    def test_corner_refiner_with_roi(self):
        refiner = CornerRefiner()
        gen = SyntheticBoardGenerator()
        frame = gen.render_frame(0)
        gt = gen.corners

        # Tight ROI around the board
        roi_bbox = (200, 100, 1700, 1000)
        corners = refiner.find_board_corners(frame.image, roi_bbox=roi_bbox)
        assert corners is not None
        assert len(corners) == 4

        diffs = [math.hypot(c[0] - g[0], c[1] - g[1]) for c, g in zip(corners, gt)]
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / 4.0)
        assert rmse <= 2.0

    def test_blank_image_returns_none(self):
        refiner = CornerRefiner()
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        corners = refiner.find_board_corners(blank)
        assert corners is None

