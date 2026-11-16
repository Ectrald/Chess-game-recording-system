"""Subpixel corner refinement and grid detection for chessboards.

Implements Canny edge detection, probabilistic Hough transform, line clustering,
boundary line intersection, RANSAC grid modeling and cv2.cornerSubPix according to:
- STAGE_1.md, Section 4.2
- Explanatory Note, Section 2.4.1
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.vision.homography import GeometryTransformer


def order_corners_canonical(corners: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Orders 4 quadrilateral corners canonically: [TL, TR, BR, BL].
    
    1. Sorts points by y coordinate into 2 top corners and 2 bottom corners.
    2. Sorts top corners by x coordinate: smaller x is TL, larger x is TR.
    3. Sorts bottom corners by x coordinate: larger x is BR, smaller x is BL.
    """
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    pts = sorted(corners, key=lambda p: p[1])
    top_two = sorted(pts[:2], key=lambda p: p[0])
    bot_two = sorted(pts[2:], key=lambda p: p[0])

    tl = top_two[0]
    tr = top_two[1]
    br = bot_two[1]  # larger x
    bl = bot_two[0]  # smaller x

    return [
        (float(tl[0]), float(tl[1])),
        (float(tr[0]), float(tr[1])),
        (float(br[0]), float(br[1])),
        (float(bl[0]), float(bl[1])),
    ]


def refine_subpixel_corners(
    image: np.ndarray,
    corners: Sequence[Tuple[float, float]],
    win_size: int = 5,
    max_shift: float = 12.0,
) -> List[Tuple[float, float]]:

    """Refines corner coordinates to subpixel accuracy using cv2.cornerSubPix.
    
    Args:
        image: Source image (BGR or grayscale ndarray).
        corners: 4 initial corner estimates.
        win_size: Half of the side length of the search window (win_size x win_size).
        max_shift: Maximum allowable deviation in pixels to reject false runaway convergence.
        
    Returns:
        List of 4 refined (x, y) tuples.
    """
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    pts_in = np.array(corners, dtype=np.float32)

    try:
        refined = cv2.cornerSubPix(
            gray,
            pts_in.copy(),
            (win_size, win_size),
            (-1, -1),
            criteria,
        )
    except Exception:
        return [(float(c[0]), float(c[1])) for c in corners]

    result: List[Tuple[float, float]] = []
    for init_pt, ref_pt in zip(pts_in, refined):
        shift = math.hypot(ref_pt[0] - init_pt[0], ref_pt[1] - init_pt[1])
        if shift <= max_shift and not (math.isnan(ref_pt[0]) or math.isnan(ref_pt[1])):
            result.append((float(ref_pt[0]), float(ref_pt[1])))
        else:
            result.append((float(init_pt[0]), float(init_pt[1])))

    return result


def fit_line_cluster(line_pts: np.ndarray) -> Tuple[float, float, float]:
    """Fits 2D line in standard form ax + by + c = 0 to a set of 2D points."""
    vx, vy, x0, y0 = cv2.fitLine(line_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    a = float(vy)
    b = float(-vx)
    c = float(vx * y0 - vy * x0)
    return a, b, c


def intersect_lines_2d(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
    """Calculates intersection point of two 2D lines in standard form ax + by + c = 0."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return float(x), float(y)


def detect_corners_hough_lines(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_thresh: int = 50,
    min_line_len: int = 50,
    max_line_gap: int = 20,
    cluster_tol: float = 6.0,
) -> Optional[List[Tuple[float, float]]]:
    """Detects 4 outer chessboard corners by clustering Hough line segments.
    
    1. Computes Canny edges on Gaussian-blurred grayscale image.
    2. Detects line segments with cv2.HoughLinesP.
    3. Separates lines into horizontal-like and vertical-like bundles.
    4. Clusters the outermost lines on each of the 4 borders and fits best-fit lines.
    5. Computes pairwise intersections of adjacent boundary lines.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h_img, w_img = gray.shape[:2]
    mid_x = w_img / 2.0
    mid_y = h_img / 2.0

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    if lines is None or len(lines) < 8:
        return None

    h_lines: List[Tuple[float, Tuple[float, float, float, float]]] = []
    v_lines: List[Tuple[float, Tuple[float, float, float, float]]] = []

    for l in lines:
        x1, y1, x2, y2 = [float(v) for v in l.flatten()]
        dx = x2 - x1
        dy = y2 - y1
        ang = np.degrees(np.arctan2(dy, dx)) % 180.0
        if ang > 90.0:
            ang -= 180.0

        if abs(ang) < 32.0 and abs(dx) > 1e-3:
            # y-intercept at mid_x
            y_mid = y1 + (mid_x - x1) * (dy / dx)
            h_lines.append((y_mid, (x1, y1, x2, y2)))
        elif abs(ang) > 38.0 and abs(dy) > 1e-3:
            # x-intercept at mid_y
            x_mid = x1 + (mid_y - y1) * (dx / dy)
            v_lines.append((x_mid, (x1, y1, x2, y2)))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h_lines.sort(key=lambda it: it[0])
    v_lines.sort(key=lambda it: it[0])

    # Cluster boundary lines
    top_items = [it for it in h_lines if abs(it[0] - h_lines[0][0]) <= cluster_tol]
    bot_items = [it for it in h_lines if abs(it[0] - h_lines[-1][0]) <= cluster_tol]
    left_items = [it for it in v_lines if abs(it[0] - v_lines[0][0]) <= cluster_tol]
    right_items = [it for it in v_lines if abs(it[0] - v_lines[-1][0]) <= cluster_tol]

    def extract_pts(items: List[Tuple[float, Tuple[float, float, float, float]]]) -> np.ndarray:
        pts = []
        for it in items:
            pts.append([it[1][0], it[1][1]])
            pts.append([it[1][2], it[1][3]])
        return np.array(pts, dtype=np.float32)

    l_top = fit_line_cluster(extract_pts(top_items))
    l_bot = fit_line_cluster(extract_pts(bot_items))
    l_left = fit_line_cluster(extract_pts(left_items))
    l_right = fit_line_cluster(extract_pts(right_items))

    tl = intersect_lines_2d(l_top, l_left)
    tr = intersect_lines_2d(l_top, l_right)
    br = intersect_lines_2d(l_bot, l_right)
    bl = intersect_lines_2d(l_bot, l_left)

    if tl is None or tr is None or br is None or bl is None:
        return None

    corners = [tl, tr, br, bl]
    # Check bounds
    for x, y in corners:
        if x < -w_img * 0.2 or x > w_img * 1.2 or y < -h_img * 0.2 or y > h_img * 1.2:
            return None

    is_valid, _, _ = GeometryTransformer.validate_quadrilateral(corners)
    if not is_valid:
        return None

    return order_corners_canonical(corners)


def detect_corners_contour_hull(
    image: np.ndarray,
    min_area_ratio: float = 0.10,
) -> Optional[List[Tuple[float, float]]]:
    """Fallback detector using largest convex quadrilateral contour approximation."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h_img, w_img = gray.shape[:2]
    total_area = float(h_img * w_img)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        if cv2.contourArea(cnt) < min_area_ratio * total_area:
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps_factor in [0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
            if len(approx) == 4:
                pts = [tuple(p) for p in approx.reshape(-1, 2).astype(float)]
                is_valid, _, _ = GeometryTransformer.validate_quadrilateral(pts)
                if is_valid:
                    return order_corners_canonical(pts)

    return None


class CornerRefiner:
    """End-to-end chessboard corner detector and subpixel refiner."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_thresh: int = 50,
        min_line_len: int = 50,
        max_line_gap: int = 20,
        subpixel_win_size: int = 5,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_thresh = hough_thresh
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.subpixel_win_size = subpixel_win_size

    def find_board_corners(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        initial_corners: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Optional[List[Tuple[float, float]]]:
        """Detects 4 subpixel board corners in canonical order [TL, TR, BR, BL].
        
        Args:
            image: Full frame (BGR numpy array).
            roi_bbox: Optional bounding box (xmin, ymin, xmax, ymax) to constrain detection.
            initial_corners: Optional coarse corner estimates (e.g. from previous frame or manual).
            
        Returns:
            List of 4 refined subpixel (x, y) coordinates or None if detection failed.
        """
        if initial_corners is not None and len(initial_corners) == 4:
            ordered = order_corners_canonical(initial_corners)
            return refine_subpixel_corners(image, ordered, win_size=self.subpixel_win_size)

        target_img = image
        offset_x, offset_y = 0.0, 0.0

        if roi_bbox is not None:
            xmin, ymin, xmax, ymax = roi_bbox
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(image.shape[1], xmax)
            ymax = min(image.shape[0], ymax)
            if xmax - xmin > 50 and ymax - ymin > 50:
                target_img = image[ymin:ymax, xmin:xmax]
                offset_x, offset_y = float(xmin), float(ymin)

        # 1. Try Hough line clustering
        corners = detect_corners_hough_lines(
            target_img,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            hough_thresh=self.hough_thresh,
            min_line_len=self.min_line_len,
            max_line_gap=self.max_line_gap,
        )

        # 2. Fall back to contour convex hull if lines failed
        if corners is None:
            corners = detect_corners_contour_hull(target_img)

        if corners is None:
            return None

        # Offset back to full image coordinate system
        shifted_corners = [(c[0] + offset_x, c[1] + offset_y) for c in corners]
        ordered = order_corners_canonical(shifted_corners)

        # Subpixel refinement on full image
        return refine_subpixel_corners(image, ordered, win_size=self.subpixel_win_size)
