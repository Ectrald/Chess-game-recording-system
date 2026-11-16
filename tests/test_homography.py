"""Unit tests for geometry transformation and homography in src/vision/homography.py."""

import numpy as np
import pytest

from src.core.datatypes import BoardOrientation
from src.core.exceptions import BoardGeometryError
from src.vision.homography import GeometryTransformer


class TestHomographyComputation:
    def test_compute_homography_and_inverse(self, sample_corners):
        H, H_inv = GeometryTransformer.compute_homography(sample_corners, target_size=640)
        assert H.shape == (3, 3)
        assert H_inv.shape == (3, 3)

        # Check H * H_inv is close to identity matrix
        identity = np.eye(3, dtype=np.float64)
        prod = np.dot(H, H_inv)
        normalized_prod = prod / prod[2, 2]
        np.testing.assert_allclose(normalized_prod, identity, atol=1e-5)

    def test_roundtrip_point_projection(self, sample_corners):
        H, H_inv = GeometryTransformer.compute_homography(sample_corners, target_size=640)

        # Test points inside the board
        test_pts = np.array([
            [480.0, 200.0],    # TL
            [1440.0, 200.0],   # TR
            [1600.0, 920.0],   # BR
            [320.0, 920.0],    # BL
            [960.0, 560.0],    # center
        ], dtype=np.float32)

        board_pts = GeometryTransformer.project_image_to_board(test_pts, H)
        assert board_pts.shape == (5, 2)
        # Canonical corners must match target_size
        np.testing.assert_allclose(board_pts[0], [0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(board_pts[1], [640.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(board_pts[2], [640.0, 640.0], atol=1e-3)
        np.testing.assert_allclose(board_pts[3], [0.0, 640.0], atol=1e-3)

        # Reverse projection back to image space
        recovered_pts = GeometryTransformer.project_board_to_image(board_pts, H_inv)
        np.testing.assert_allclose(recovered_pts, test_pts, atol=1e-4)

    def test_invalid_corner_count_raises(self):
        with pytest.raises(BoardGeometryError):
            GeometryTransformer.compute_homography([(0.0, 0.0), (10.0, 10.0)])


class TestWarpBoard:
    def test_warp_board_shape(self, sample_corners):
        H, _ = GeometryTransformer.compute_homography(sample_corners, target_size=640)
        dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Put some pixel values
        dummy_img[500:600, 800:1000] = 255

        warped = GeometryTransformer.warp_board(dummy_img, H, target_size=640)
        assert warped.shape == (640, 640, 3)
        assert warped.dtype == np.uint8


class TestQuadrilateralValidation:
    def test_valid_convex_quadrilateral(self, sample_corners):
        is_valid, reason, metrics = GeometryTransformer.validate_quadrilateral(
            sample_corners,
            image_shape=(1080, 1920),
            min_area_ratio=0.15,
        )
        assert is_valid is True
        assert metrics["is_convex"] is True
        assert metrics["polygon_area"] > 500000.0
        assert metrics["condition_number"] < 1e5

    def test_reject_self_intersecting_butterfly(self):
        butterfly = [
            (100.0, 100.0),
            (500.0, 500.0),
            (500.0, 100.0),
            (100.0, 500.0),
        ]
        is_valid, reason, metrics = GeometryTransformer.validate_quadrilateral(butterfly)
        assert is_valid is False
        assert "non-convex or self-intersecting" in reason

    def test_reject_concave_dart(self):
        # Concave polygon
        dart = [
            (100.0, 100.0),
            (500.0, 100.0),
            (250.0, 200.0),  # indented vertex
            (100.0, 500.0),
        ]
        is_valid, reason, metrics = GeometryTransformer.validate_quadrilateral(dart)
        assert is_valid is False
        assert metrics["is_convex"] is False

    def test_reject_collinear_degenerate(self):
        collinear = [
            (100.0, 100.0),
            (200.0, 100.0),
            (300.0, 100.0),
            (400.0, 100.0),
        ]
        is_valid, reason, metrics = GeometryTransformer.validate_quadrilateral(collinear)
        assert is_valid is False

    def test_reject_area_ratio_below_threshold(self, sample_corners):
        # With huge frame size, sample_corners area will be < 50%
        is_valid, reason, metrics = GeometryTransformer.validate_quadrilateral(
            sample_corners,
            image_shape=(10800, 19200),
            min_area_ratio=0.50,
        )
        assert is_valid is False
        assert "below minimum required" in reason

    def test_reprojection_error(self, sample_corners):
        H, _ = GeometryTransformer.compute_homography(sample_corners, target_size=640)
        src = np.array(sample_corners, dtype=np.float32)
        dst = np.array([[0, 0], [640, 0], [640, 640], [0, 640]], dtype=np.float32)

        rmse = GeometryTransformer.compute_reprojection_error(src, dst, H)
        assert rmse < 1e-4

    def test_build_board_geometry(self, sample_corners):
        geom = GeometryTransformer.build_board_geometry(
            sample_corners,
            target_size=640,
            image_shape=(1080, 1920),
            min_area_ratio=0.15,
            orientation=BoardOrientation.WHITE_BOTTOM,
            detection_method="pure_hough",
        )
        assert geom.is_valid is True
        assert geom.homography_matrix is not None
        assert geom.inverse_homography is not None
        assert geom.orientation == BoardOrientation.WHITE_BOTTOM
        assert geom.detection_method == "pure_hough"

