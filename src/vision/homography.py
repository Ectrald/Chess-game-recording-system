"""Perspective homography computation, projection and quadrilateral validation.

Implements mathematical transformations and geometric validation according to:
- STAGE_1.md, Section 4.3
- Explanatory Note, Section 2.4.1
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.core.datatypes import BoardGeometry, BoardOrientation, Point2D, Quadrilateral
from src.core.exceptions import BoardGeometryError


class GeometryTransformer:
    """Handles perspective projection, homography computation, and geometry validation."""

    def __init__(self, target_size: int = 640) -> None:
        self.target_size = target_size

    @staticmethod
    def get_canonical_board_corners(target_size: int = 640) -> np.ndarray:
        """Returns canonical square board coordinates [(0,0), (S,0), (S,S), (0,S)] in float32."""
        s = float(target_size)
        return np.array(
            [[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]],
            dtype=np.float32,
        )

    @classmethod
    def compute_homography(
        cls,
        corners: Sequence[Tuple[float, float]],
        target_size: int = 640,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes direct H (image -> board) and inverse H_inv (board -> image).
        
        Args:
            corners: 4 corners [TL, TR, BR, BL] in image coordinates.
            target_size: Target square board resolution (default 640).
            
        Returns:
            Tuple of (H, H_inv) as float64 numpy arrays of shape (3, 3).
        """
        if len(corners) != 4:
            raise BoardGeometryError(f"Expected exactly 4 corners, got {len(corners)}")

        src_pts = np.array(corners, dtype=np.float32)
        dst_pts = cls.get_canonical_board_corners(target_size)

        # cv2.getPerspectiveTransform calculates matrix mapping src -> dst
        h_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts).astype(np.float64)

        try:
            h_inv = np.linalg.inv(h_matrix)
        except np.linalg.LinAlgError as err:
            raise BoardGeometryError(f"Singular homography matrix cannot be inverted: {err}") from err

        return h_matrix, h_inv

    @classmethod
    def compute_condition_number(
        cls,
        src_corners: Sequence[Tuple[float, float]],
        target_size: int = 640,
    ) -> float:
        """Calculates normalized condition number kappa(H_norm) = sigma_max / sigma_min via Hartley normalization.
        
        Properly invariant to image resolution and coordinate translation.
        """
        try:
            pts = np.asarray(src_corners, dtype=np.float64)
            c = np.mean(pts, axis=0)
            d = np.mean(np.sqrt(np.sum((pts - c) ** 2, axis=1)))
            if d < 1e-6:
                return float("inf")
            scale_src = np.sqrt(2.0) / d
            t_src = np.array(
                [
                    [scale_src, 0.0, -scale_src * c[0]],
                    [0.0, scale_src, -scale_src * c[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            dst = cls.get_canonical_board_corners(target_size).astype(np.float64)
            c_dst = np.mean(dst, axis=0)
            d_dst = np.mean(np.sqrt(np.sum((dst - c_dst) ** 2, axis=1)))
            scale_dst = np.sqrt(2.0) / d_dst
            t_dst = np.array(
                [
                    [scale_dst, 0.0, -scale_dst * c_dst[0]],
                    [0.0, scale_dst, -scale_dst * c_dst[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            h_raw = cv2.getPerspectiveTransform(pts.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
            h_norm = t_dst @ h_raw @ np.linalg.inv(t_src)
            if abs(h_norm[2, 2]) > 1e-12:
                h_norm /= h_norm[2, 2]

            _, s, _ = np.linalg.svd(h_norm)
            if s[-1] < 1e-12:
                return float("inf")
            return float(s[0] / s[-1])
        except Exception:
            return float("inf")


    @classmethod
    def validate_quadrilateral(
        cls,
        corners: Sequence[Tuple[float, float]],
        image_shape: Optional[Tuple[int, int]] = None,
        min_area_ratio: float = 0.15,
        max_condition_number: float = 1e5,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Performs rigorous mathematical validation on board quadrilateral.
        
        Checks:
        1. Exactly 4 points.
        2. Strict convexity and non-self-intersection (consecutive edge cross-products).
        3. Minimum polygon area threshold (Shoelace formula).
        4. Condition number of projective homography matrix kappa(H) < 10^5.
        
        Returns:
            Tuple of (is_valid, failure_reason, metrics_dict).
        """
        metrics: Dict[str, Any] = {
            "is_convex": False,
            "polygon_area": 0.0,
            "area_ratio": 0.0,
            "condition_number": float("inf"),
            "cross_products": [],
        }

        if len(corners) != 4:
            return False, f"Expected 4 corners, got {len(corners)}", metrics

        pts = list(corners)

        # 1. Check strict convexity and no self-intersection
        cross_products = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            p2 = pts[(i + 2) % 4]
            dx1 = p1[0] - p0[0]
            dy1 = p1[1] - p0[1]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            cross = dx1 * dy2 - dy1 * dx2
            cross_products.append(cross)

        metrics["cross_products"] = cross_products
        all_pos = all(cp > 1e-5 for cp in cross_products)
        all_neg = all(cp < -1e-5 for cp in cross_products)

        if not (all_pos or all_neg):
            return (
                False,
                f"Quadrilateral is non-convex or self-intersecting (cross_products={cross_products})",
                metrics,
            )
        metrics["is_convex"] = True

        # 2. Polygon area via Shoelace formula
        area = 0.5 * abs(
            sum(
                pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1]
                for i in range(4)
            )
        )
        metrics["polygon_area"] = float(area)

        if area <= 1.0:
            return False, f"Degenerate polygon area: {area} px^2", metrics

        if image_shape is not None:
            img_h, img_w = image_shape[:2]
            total_img_area = float(img_h * img_w)
            area_ratio = area / max(total_img_area, 1.0)
            metrics["area_ratio"] = float(area_ratio)

            if area_ratio < min_area_ratio:
                return (
                    False,
                    f"Board area ratio {area_ratio:.3f} is below minimum required {min_area_ratio}",
                    metrics,
                )

        # 3. Homography conditioning
        try:
            cond = cls.compute_condition_number(corners, target_size=640)
            metrics["condition_number"] = cond
            if cond > max_condition_number or math.isnan(cond):
                return (
                    False,
                    f"Ill-conditioned homography matrix (condition number = {cond:.2e} > {max_condition_number:.2e})",
                    metrics,
                )
        except Exception as err:
            return False, f"Homography computation failed: {err}", metrics


        return True, "Valid quadrilateral geometry", metrics

    @staticmethod
    def warp_board(
        image: np.ndarray,
        homography_matrix: np.ndarray,
        target_size: int = 640,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Warps image using homography matrix to normalized square board of shape (target_size, target_size, C)."""
        return cv2.warpPerspective(
            image,
            homography_matrix,
            (target_size, target_size),
            flags=interpolation,
        )

    @staticmethod
    def project_image_to_board(points: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
        """Projects array of 2D points (N, 2) from source frame space to board space."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, homography_matrix.astype(np.float64))
        return transformed.reshape(-1, 2)

    @staticmethod
    def project_board_to_image(points: np.ndarray, inverse_homography: np.ndarray) -> np.ndarray:
        """Projects array of 2D points (N, 2) from normalized board space back to source frame space."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, inverse_homography.astype(np.float64))
        return transformed.reshape(-1, 2)

    @classmethod
    def compute_reprojection_error(
        cls,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        homography_matrix: np.ndarray,
    ) -> float:
        """Computes RMSE reprojection error between ground truth dst_points and H * src_points."""
        projected = cls.project_image_to_board(src_points, homography_matrix)
        dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
        diffs = projected - dst
        mse = np.mean(np.sum(diffs ** 2, axis=1))
        return float(np.sqrt(mse))

    @classmethod
    def build_board_geometry(
        cls,
        corners: Sequence[Tuple[float, float]],
        target_size: int = 640,
        image_shape: Optional[Tuple[int, int]] = None,
        min_area_ratio: float = 0.15,
        orientation: BoardOrientation = BoardOrientation.UNKNOWN,
        detection_method: str = "unknown",
        reprojection_error_px: float = 0.0,
    ) -> BoardGeometry:
        """Constructs and validates a full BoardGeometry instance."""
        is_valid, reason, metrics = cls.validate_quadrilateral(
            corners=corners,
            image_shape=image_shape,
            min_area_ratio=min_area_ratio,
        )

        h_matrix, h_inv = None, None
        if is_valid:
            try:
                h_mat, h_i = cls.compute_homography(corners, target_size=target_size)
                h_matrix = h_mat.tolist()
                h_inv = h_i.tolist()
            except Exception:
                is_valid = False

        corner_list = [(float(c[0]), float(c[1])) for c in corners]
        return BoardGeometry(
            corners=corner_list,
            corners_image=corner_list,
            homography_matrix=h_matrix,
            inverse_homography=h_inv,
            target_size=target_size,
            orientation=orientation,
            detection_method=detection_method,  # type: ignore[arg-type]
            reprojection_error_px=reprojection_error_px,
            polygon_area_px2=metrics.get("polygon_area", 1.0),
            confidence=1.0 if is_valid else 0.0,
            geometry_quality=1.0 if is_valid else 0.0,
            is_valid=is_valid,
        )
