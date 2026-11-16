"""Unit tests for chessboard orientation resolution in src/vision/orientation.py."""

import chess
import cv2
import numpy as np
import pytest

from src.core.config import OrientationConfig
from src.core.datatypes import BoardOrientation
from src.vision.orientation import OrientationResolver
from src.vision.synthetic_board import SyntheticBoardGenerator


@pytest.fixture
def initial_flat_board() -> np.ndarray:
    """Returns normalized flat 640x640 board in initial chess layout."""
    gen = SyntheticBoardGenerator(board_size=640)
    return gen.render_board_flat(chess.Board())


class TestOrientationResolver:
    def test_white_bottom_on_standard_layout(self, initial_flat_board):
        resolver = OrientationResolver()
        orientation = resolver.resolve(initial_flat_board)
        assert orientation == BoardOrientation.WHITE_BOTTOM

    def test_black_bottom_on_180_rotation(self, initial_flat_board):
        rot180 = cv2.rotate(initial_flat_board, cv2.ROTATE_180)
        resolver = OrientationResolver()
        orientation = resolver.resolve(rot180)
        assert orientation == BoardOrientation.BLACK_BOTTOM

    def test_reject_90_rotation(self, initial_flat_board):
        rot90 = cv2.rotate(initial_flat_board, cv2.ROTATE_90_CLOCKWISE)
        resolver = OrientationResolver()
        orientation = resolver.resolve(rot90)
        assert orientation == BoardOrientation.UNKNOWN

    def test_reject_270_rotation(self, initial_flat_board):
        rot270 = cv2.rotate(initial_flat_board, cv2.ROTATE_90_COUNTERCLOCKWISE)
        resolver = OrientationResolver()
        orientation = resolver.resolve(rot270)
        assert orientation == BoardOrientation.UNKNOWN

    def test_user_preference_override(self, initial_flat_board):
        # Even if board is 180 rotated, user override takes precedence
        rot180 = cv2.rotate(initial_flat_board, cv2.ROTATE_180)
        resolver = OrientationResolver()
        forced = resolver.resolve(rot180, user_preference=BoardOrientation.WHITE_BOTTOM)
        assert forced == BoardOrientation.WHITE_BOTTOM

    def test_corner_color_validity_metric(self, initial_flat_board):
        resolver = OrientationResolver()
        is_valid, br, bl = resolver.check_corner_color_validity(initial_flat_board)
        assert is_valid is True
        assert br > bl + 15.0  # Light h1 square is significantly brighter than dark a1

    def test_fallback_when_inconclusive(self):
        # Uniform gray image has zero mass difference and equal corner intensities
        uniform = np.full((640, 640, 3), 128, dtype=np.uint8)
        cfg = OrientationConfig(fallback_orientation="white_bottom")
        resolver = OrientationResolver(config=cfg)
        # With color check disabled on uniform image:
        resolver.color_check_enabled = False
        res = resolver.resolve(uniform)
        assert res == BoardOrientation.WHITE_BOTTOM

