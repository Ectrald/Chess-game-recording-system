"""Unit tests for chessboard localizer strategies in src/vision/board_localizer.py."""

import math
import cv2
import numpy as np
import pytest

from src.core.config import VisionConfig
from src.core.datatypes import Frame
from src.vision.board_localizer import (
    ArucoLocalizer,
    CoarseToFineLocalizer,
    ManualCornerLocalizer,
    PureHoughLocalizer,
    create_board_localizer,
)
from src.vision.synthetic_board import SyntheticBoardGenerator


class TestPureHoughLocalizer:
    def test_locate_synthetic_frame(self):
        gen = SyntheticBoardGenerator()
        frame = gen.render_frame(0)
        localizer = PureHoughLocalizer()

        geom = localizer.locate(frame)
        assert geom is not None
        assert geom.is_valid is True
        assert geom.detection_method == "pure_hough"
        assert geom.homography_matrix is not None
        assert geom.inverse_homography is not None
        assert len(geom.corners) == 4

        # Compare with ground truth
        gt = gen.corners
        diffs = [math.hypot(c[0] - g[0], c[1] - g[1]) for c, g in zip(geom.corners, gt)]
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / 4.0)
        assert rmse <= 2.0

    def test_locate_blank_frame_returns_none(self):
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = Frame(frame_id=0, timestamp=0.0, image=blank)
        localizer = PureHoughLocalizer()
        geom = localizer.locate(frame)
        assert geom is None


class TestCoarseToFineLocalizer:
    def test_locate_synthetic_frame(self):
        gen = SyntheticBoardGenerator()
        frame = gen.render_frame(0)
        localizer = CoarseToFineLocalizer()

        geom = localizer.locate(frame)
        assert geom is not None
        assert geom.is_valid is True
        assert geom.detection_method == "yolo_hough"

        gt = gen.corners
        diffs = [math.hypot(c[0] - g[0], c[1] - g[1]) for c, g in zip(geom.corners, gt)]
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / 4.0)
        assert rmse <= 2.0


class TestManualCornerLocalizer:
    def test_manual_localization(self, sample_corners, sample_frame):
        localizer = ManualCornerLocalizer(sample_corners)
        geom = localizer.locate(sample_frame)
        assert geom is not None
        assert geom.is_valid is True
        assert geom.detection_method == "manual"

    def test_update_corners(self, sample_frame):
        initial = [(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)]
        localizer = ManualCornerLocalizer(initial, subpixel_refinement=False)
        geom = localizer.locate(sample_frame)
        assert geom is not None
        assert geom.corners[0] == (100.0, 100.0)

        updated = [(200.0, 200.0), (600.0, 200.0), (600.0, 600.0), (200.0, 600.0)]
        localizer.update_corners(updated)
        geom2 = localizer.locate(sample_frame)
        assert geom2 is not None
        assert geom2.corners[0] == (200.0, 200.0)


class TestArucoLocalizer:
    def test_aruco_localization_with_markers(self):
        # Create an image canvas with 4 ArUco markers at known positions
        canvas = np.full((1000, 1000, 3), 240, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        marker_size = 80
        # Draw 4 markers: ID 0 at (200, 200), ID 1 at (800, 200), ID 2 at (800, 800), ID 3 at (200, 800)
        positions = {
            0: (200, 200),
            1: (800, 200),
            2: (800, 800),
            3: (200, 800),
        }
        for m_id, (cx, cy) in positions.items():
            marker_img = cv2.aruco.generateImageMarker(dictionary, m_id, marker_size)
            x1, y1 = cx - marker_size // 2, cy - marker_size // 2
            canvas[y1 : y1 + marker_size, x1 : x1 + marker_size] = cv2.cvtColor(
                marker_img, cv2.COLOR_GRAY2BGR
            )

        frame = Frame(frame_id=1, timestamp=0.033, image=canvas)
        localizer = ArucoLocalizer(marker_ids=(0, 1, 2, 3))
        geom = localizer.locate(frame)

        assert geom is not None
        assert geom.is_valid is True
        assert geom.detection_method == "aruco"
        # Check marker centers match within 1 pixel
        assert math.hypot(geom.corners[0][0] - 200.0, geom.corners[0][1] - 200.0) < 1.0
        assert math.hypot(geom.corners[1][0] - 800.0, geom.corners[1][1] - 200.0) < 1.0
        assert math.hypot(geom.corners[2][0] - 800.0, geom.corners[2][1] - 800.0) < 1.0
        assert math.hypot(geom.corners[3][0] - 200.0, geom.corners[3][1] - 800.0) < 1.0

    def test_aruco_missing_markers_returns_none(self):
        canvas = np.full((1000, 1000, 3), 240, dtype=np.uint8)
        frame = Frame(frame_id=1, timestamp=0.033, image=canvas)
        localizer = ArucoLocalizer()
        geom = localizer.locate(frame)
        assert geom is None


class TestFactory:
    def test_create_localizer_modes(self):
        cfg = VisionConfig()
        loc = create_board_localizer(cfg)
        assert isinstance(loc, CoarseToFineLocalizer)

        cfg.localizer.mode = "pure_hough"
        loc_hough = create_board_localizer(cfg)
        assert isinstance(loc_hough, PureHoughLocalizer)

        cfg.localizer.mode = "manual"
        loc_manual = create_board_localizer(cfg)
        assert isinstance(loc_manual, ManualCornerLocalizer)

        cfg.localizer.mode = "aruco"
        loc_aruco = create_board_localizer(cfg)
        assert isinstance(loc_aruco, ArucoLocalizer)


class TestGroundTruthBenchmarkFrames:
    def test_benchmark_standard_frame_rmse(self, repo_root):
        import json
        gt_path = repo_root / "data" / "ground_truth" / "board_corners_gt.json"
        frame_path = repo_root / "data" / "test_frames" / "standard_view_01.png"
        if not (gt_path.exists() and frame_path.exists()):
            pytest.skip("Benchmark frames not found")

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        img = cv2.imread(str(frame_path))
        frame = Frame(frame_id=1, timestamp=0.0, image=img)
        localizer = PureHoughLocalizer()
        geom = localizer.locate(frame)

        assert geom is not None
        gt_corners = gt_data["standard_view_01.png"]["corners"]
        diffs = [math.hypot(c[0] - g[0], c[1] - g[1]) for c, g in zip(geom.corners, gt_corners)]
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / 4.0)
        assert rmse <= 2.0

    def test_benchmark_aruco_frame_rmse(self, repo_root):
        import json
        gt_path = repo_root / "data" / "ground_truth" / "board_corners_gt.json"
        frame_path = repo_root / "data" / "test_frames" / "aruco_calibration_04.png"
        if not (gt_path.exists() and frame_path.exists()):
            pytest.skip("Benchmark frames not found")

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        img = cv2.imread(str(frame_path))
        frame = Frame(frame_id=4, timestamp=0.0, image=img)
        localizer = ArucoLocalizer()
        geom = localizer.locate(frame)

        assert geom is not None
        gt_corners = gt_data["aruco_calibration_04.png"]["corners"]
        diffs = [math.hypot(c[0] - g[0], c[1] - g[1]) for c, g in zip(geom.corners, gt_corners)]
        rmse = math.sqrt(sum(d ** 2 for d in diffs) / 4.0)
        assert rmse <= 2.0


class TestBoardGeometryTracker:
    def test_tracker_ignores_single_outlier_and_updates_on_sustained_shift(self):
        from src.vision.board_localizer import BoardGeometryTracker
        from src.vision.homography import GeometryTransformer

        base_corners = [(400.0, 200.0), (1000.0, 200.0), (1100.0, 800.0), (300.0, 800.0)]
        shifted_corners = [(415.0, 210.0), (1015.0, 210.0), (1115.0, 810.0), (315.0, 810.0)]
        outlier_corners = [(800.0, 400.0), (1400.0, 400.0), (1450.0, 950.0), (750.0, 950.0)]

        geom = GeometryTransformer.build_board_geometry(
            base_corners, target_size=640, image_shape=(1080, 1920), min_area_ratio=0.05
        )
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = Frame(frame_id=1, timestamp=0.1, image=dummy)

        seq = [outlier_corners, shifted_corners, shifted_corners, shifted_corners]

        class _StubRefiner:
            def find_board_corners(self, image, roi_bbox=None):
                return seq.pop(0) if seq else shifted_corners

        tracker = BoardGeometryTracker(refiner=_StubRefiner(), window_size=5, min_samples=3)
        # 1) Outlier (>95px) must be ignored
        g1, st1 = tracker.check_drift(frame, geom)
        assert st1 == "unchanged" and g1 is None
        # 2) First two shifted frames (< min_samples) wait for median stability
        g2, st2 = tracker.check_drift(frame, geom)
        assert st2 == "unchanged"
        g3, st3 = tracker.check_drift(frame, geom)
        assert st3 == "unchanged"
        # 3) 3rd valid sample crosses min_samples and updates cleanly
        g4, st4 = tracker.check_drift(frame, geom)
        assert st4 == "updated" and g4 is not None
        assert math.hypot(g4.corners[0][0] - 415.0, g4.corners[0][1] - 210.0) < 1.0

    def test_create_board_localizer_accepts_localizer_config(self):
        from src.core.config import VisionConfig, LocalizerConfig
        from src.vision.board_localizer import create_board_localizer, CoarseToFineLocalizer

        vis_cfg = VisionConfig(
            localizer=LocalizerConfig(mode="coarse_to_fine", yolo_model_path="custom/yolo.pt", yolo_confidence=0.65)
        )
        loc1 = create_board_localizer(vis_cfg)
        loc2 = create_board_localizer(vis_cfg.localizer)
        assert isinstance(loc1, CoarseToFineLocalizer)
        assert isinstance(loc2, CoarseToFineLocalizer)
        assert str(loc1.yolo_model_path) == "custom/yolo.pt"
        assert str(loc2.yolo_model_path) == "custom/yolo.pt"
        assert loc2.yolo_confidence == 0.65

