"""Unit tests for differential motion detector in src/vision/motion_detector.py."""

import numpy as np
import pytest

from src.core.config import TemporalConfig
from src.core.datatypes import BoardOrientation, CHESS_SQUARES
from src.vision.homography import GeometryTransformer
from src.vision.motion_detector import MotionDetector
from src.vision.synthetic_board import SyntheticBoardGenerator


class TestMotionDetector:
    def test_first_frame_has_zero_motion(self):
        detector = MotionDetector()
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        metrics = detector.process_board_image(img, frame_id=0, timestamp=0.0)

        assert metrics.global_motion_score == 0.0
        assert metrics.is_motion_detected is False
        assert len(metrics.square_motion_scores) == 64
        assert all(s == 0.0 for s in metrics.square_motion_scores.values())

    def test_static_consecutive_frames(self):
        detector = MotionDetector(motion_threshold=0.03)
        gen = SyntheticBoardGenerator(board_size=640)
        f0 = gen.render_frame(0)
        H, _ = GeometryTransformer.compute_homography(gen.corners, 640)
        b0 = GeometryTransformer.warp_board(f0.image, H, 640)

        # Process frame 0
        detector.process_board_image(b0, frame_id=0, timestamp=0.0)
        # Process frame 1 (identical)
        metrics1 = detector.process_board_image(b0, frame_id=1, timestamp=0.033)

        assert metrics1.global_motion_score < 0.005
        assert metrics1.is_motion_detected is False

    def test_dynamic_motion_detection(self):
        detector = MotionDetector(motion_threshold=0.03, ema_alpha=0.5)
        gen = SyntheticBoardGenerator(board_size=640, frames_per_move=60)
        gen.set_game_from_moves(["e2e4"])

        H, _ = GeometryTransformer.compute_homography(gen.corners, 640)

        # Baseline static frame
        f_static = gen.render_frame(0)
        b_static = GeometryTransformer.warp_board(f_static.image, H, 640)
        detector.process_board_image(b_static, frame_id=0, timestamp=0.0)

        # Moving frame during half-move 1 transition (frame 70)
        f_moving = gen.render_frame(70)
        b_moving = GeometryTransformer.warp_board(f_moving.image, H, 640)
        metrics = detector.process_board_image(b_moving, frame_id=70, timestamp=2.33)

        assert metrics.global_motion_score > 0.03
        assert metrics.is_motion_detected is True

        # Check square-level motion: e2 and e4 must have significant motion
        e2_motion = metrics.square_motion_scores["e2"]
        e4_motion = metrics.square_motion_scores["e4"]
        a1_motion = metrics.square_motion_scores["a1"]
        h8_motion = metrics.square_motion_scores["h8"]

        assert e2_motion > 0.3
        assert e4_motion > 0.3
        assert a1_motion < 0.05
        assert h8_motion < 0.05

    def test_exposure_change_compensation(self):
        detector = MotionDetector(motion_threshold=0.04)
        base = np.full((640, 640, 3), 120, dtype=np.uint8)
        # Put some static texture
        base[200:400, 200:400] = 160

        detector.process_board_image(base, frame_id=0, timestamp=0.0)

        # Second frame: 15% brightness increase across the whole frame
        brighter = np.clip(base.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        metrics = detector.process_board_image(brighter, frame_id=1, timestamp=0.033)

        # With luminance compensation, global motion should remain below threshold
        assert metrics.global_motion_score < 0.04
        assert metrics.is_motion_detected is False

    def test_reset_detector(self):
        detector = MotionDetector()
        img = np.ones((640, 640, 3), dtype=np.uint8) * 100
        detector.process_board_image(img, frame_id=0, timestamp=0.0)
        assert detector.prev_gray is not None

        detector.reset()
        assert detector.prev_gray is None
        assert detector.smoothed_motion_score == 0.0
