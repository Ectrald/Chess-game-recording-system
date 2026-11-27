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


def _select_grid_boundary_candidates(
    sorted_lines: List[Tuple[float, float, float, Tuple[float, float, float, float]]],
    dim: int,
    cluster_tol: float,
    top_k: int = 5,
    angle_tol: float = 6.0,
) -> List[
    Tuple[
        float,
        List[Tuple[float, float, float, Tuple[float, float, float, float]]],
        List[Tuple[float, float, float, Tuple[float, float, float, float]]],
    ]
]:
    """Selects candidate outer checkerboard boundary cluster pairs using 1D projective cross-ratio fitting.

    Clusters segments by both spatial intercept and line orientation so crossing artifacts never corrupt
    line fitting, then fits the 1D homography t(k) = k / (k + r * (1 - k)) weighted by segment support.
    """
    # Each entry: [mean_pos, mean_ang, items]
    raw_clusters: List[List[object]] = []
    for item in sorted_lines:
        pos, ang, _, _ = item
        placed = False
        for cl in raw_clusters:
            if abs(pos - float(cl[0])) <= cluster_tol and abs(ang - float(cl[1])) <= angle_tol:
                cl_items = cl[2]
                assert isinstance(cl_items, list)
                cl_items.append(item)
                cl[0] = float(np.mean([it[0] for it in cl_items]))
                cl[1] = float(np.mean([it[1] for it in cl_items]))
                placed = True
                break
        if not placed:
            raw_clusters.append([pos, ang, [item]])

    clusters: List[Tuple[float, float, List[Tuple[float, float, float, Tuple[float, float, float, float]]]]] = []
    for cl in raw_clusters:
        cl_items = cl[2]
        assert isinstance(cl_items, list)
        total_len = float(sum(it[2] for it in cl_items))
        clusters.append((float(cl[0]), total_len, cl_items))
    clusters.sort(key=lambda c: c[0])

    if len(clusters) < 5:
        return [(0.0, clusters[0][2], clusters[-1][2])]

    centers = [c[0] for c in clusters]
    lengths = [c[1] for c in clusters]
    ks = np.linspace(0.0, 1.0, 9)
    ratios = np.geomspace(0.28, 3.6, 37)
    candidates: List[Tuple[float, int, int]] = []

    for i in range(len(centers)):
        for j in range(i + 4, len(centers)):
            c0, c8 = centers[i], centers[j]
            span = c8 - c0
            if span < 0.35 * dim:
                continue

            tol_k = max(5.0, span * 0.024)
            fragile_penalty = 0.0
            if (len(clusters[i][2]) == 1 and lengths[i] < span * 0.22) or (
                len(clusters[j][2]) == 1 and lengths[j] < span * 0.22
            ):
                fragile_penalty = 2.0

            best_pair_score: Optional[float] = None
            for r in ratios:
                ts = ks / (ks + r * (1.0 - ks))
                preds = c0 + span * ts
                inliers = 0
                err_sum = 0.0
                for p in preds:
                    md = min(abs(c - p) for c in centers)
                    if md <= tol_k:
                        inliers += 1
                        err_sum += md
                if inliers >= 5:
                    score = inliers * 10.0 - err_sum * 0.12 + (span / max(1, dim)) * 0.5 - fragile_penalty
                    if best_pair_score is None or score > best_pair_score:
                        best_pair_score = score

            if best_pair_score is not None:
                candidates.append((best_pair_score, i, j))

    if not candidates:
        return [(0.0, clusters[0][2], clusters[-1][2])]

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(score, clusters[i][2], clusters[j][2]) for score, i, j in candidates[:top_k]]


def _checkerboard_border_score(gray: np.ndarray, corners: List[Tuple[float, float]]) -> float:
    """Scores alternating checkerboard contrast, grid-line phase alignment, and outer rim alternation."""
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0.0, 0.0], [256.0, 0.0], [256.0, 256.0], [0.0, 256.0]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, h_mat, (256, 256)).astype(np.float32)

    row_alts: List[float] = []
    for r in range(8):
        vals = [warped[r * 32 + 8 : (r + 1) * 32 - 8, c * 32 + 8 : (c + 1) * 32 - 8].mean() for c in range(8)]
        center_diffs = [abs(vals[c] - vals[c + 1]) for c in range(2, 5)]
        all_diffs = sorted([abs(vals[c] - vals[c + 1]) for c in range(7)], reverse=True)[:4]
        row_alts.append(min(float(np.mean(center_diffs)), float(np.mean(all_diffs))))

    col_alts: List[float] = []
    for c in range(8):
        vals = [warped[r * 32 + 8 : (r + 1) * 32 - 8, c * 32 + 8 : (c + 1) * 32 - 8].mean() for r in range(8)]
        center_diffs = [abs(vals[r] - vals[r + 1]) for r in range(2, 5)]
        all_diffs = sorted([abs(vals[r] - vals[r + 1]) for r in range(7)], reverse=True)[:4]
        col_alts.append(min(float(np.mean(center_diffs)), float(np.mean(all_diffs))))

    # Measure outer strip dynamic range ONLY as a gentle tie-breaker against uniform outer bezels
    edge_strips = [
        warped[2:9, 16:240],
        warped[247:254, 16:240],
        warped[16:240, 2:9],
        warped[16:240, 247:254],
    ]
    min_rim_range = min(float(np.percentile(st, 90) - np.percentile(st, 10)) for st in edge_strips)
    rim_bonus = 1.5 if min_rim_range >= 18.0 else -1.5

    # Verify each of the 14 interior grid lines (x=32k, y=32k for k=1..7) individually so stretched
    # half-boards (missing k=1,3,5,7) never beat the true 8x8 checkerboard.
    gx = np.abs(cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3))
    mid_v = float(np.median([float(np.median(gx[16:240, 32 * k + 12 : 32 * k + 20])) for k in range(8)]))
    mid_h = float(np.median([float(np.median(gy[32 * k + 12 : 32 * k + 20, 16:240])) for k in range(8)]))
    aligned_v = sum(
        1 for k in range(1, 8)
        if float(np.percentile(gx[16:240, 32 * k - 2 : 32 * k + 3], 65)) > max(8.0, mid_v * 1.45)
    )
    aligned_h = sum(
        1 for k in range(1, 8)
        if float(np.percentile(gy[32 * k - 2 : 32 * k + 3, 16:240], 65)) > max(8.0, mid_h * 1.45)
    )

    center_alt = float(np.median(row_alts[2:6] + col_alts[2:6]))
    subgrid_penalty = -4.0 if (aligned_v < 5 or aligned_h < 5) else 0.0
    return rim_bonus + subgrid_penalty + 0.45 * float(aligned_v + aligned_h) + 0.02 * min(80.0, center_alt)


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
    2. Detects line segments with cv2.HoughLinesP scaled gently to ROI dimensions.
    3. Separates lines into horizontal-like and vertical-like bundles with signed angles.
    4. Clusters boundary lines by intercept + angle using 1D projective fitting & grid-phase scoring.
    5. Computes pairwise intersections of adjacent boundary lines.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h_img, w_img = gray.shape[:2]
    mid_x = w_img / 2.0
    mid_y = h_img / 2.0
    min_dim = min(h_img, w_img)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    eff_min_len = min(min_line_len, max(34, int(min_dim * 0.078)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=eff_min_len,
        maxLineGap=max_line_gap,
    )

    if lines is None or len(lines) < 8:
        return None

    h_lines: List[Tuple[float, float, float, Tuple[float, float, float, float]]] = []
    v_lines: List[Tuple[float, float, float, Tuple[float, float, float, float]]] = []

    for l in lines:
        x1, y1, x2, y2 = [float(v) for v in l.flatten()]
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx)) % 180.0
        if ang > 90.0:
            ang -= 180.0

        if abs(ang) <= 35.0 and abs(dx) > 1e-3:
            # y-intercept at mid_x
            y_mid = y1 + (mid_x - x1) * (dy / dx)
            h_lines.append((y_mid, float(ang), seg_len, (x1, y1, x2, y2)))
        elif abs(ang) > 35.0 and abs(dy) > 1e-3:
            # x-intercept at mid_y; signed tilt angle relative to +y axis
            x_mid = x1 + (mid_y - y1) * (dx / dy)
            vang = math.degrees(math.atan2(dx, dy) if dy >= 0.0 else math.atan2(-dx, -dy))
            v_lines.append((x_mid, vang, seg_len, (x1, y1, x2, y2)))

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    h_lines.sort(key=lambda it: it[0])
    v_lines.sort(key=lambda it: it[0])

    # Cluster boundary lines via 1D projective cross-ratio fitting + checkerboard verification
    eff_cluster_tol = max(cluster_tol, min_dim * 0.012)
    h_cands = _select_grid_boundary_candidates(h_lines, h_img, eff_cluster_tol, top_k=5)
    v_cands = _select_grid_boundary_candidates(v_lines, w_img, eff_cluster_tol, top_k=5)

    def extract_pts(items: List[Tuple[float, float, float, Tuple[float, float, float, float]]]) -> np.ndarray:
        pts = []
        for it in items:
            pts.append([it[3][0], it[3][1]])
            pts.append([it[3][2], it[3][3]])
        return np.array(pts, dtype=np.float32)

    best_corners: Optional[List[Tuple[float, float]]] = None
    best_total_score = float("-inf")
    skip_cb = len(h_cands) == 1 and len(v_cands) == 1

    for h_score, top_items, bot_items in h_cands:
        l_top = fit_line_cluster(extract_pts(top_items))
        l_bot = fit_line_cluster(extract_pts(bot_items))
        for v_score, left_items, right_items in v_cands:
            l_left = fit_line_cluster(extract_pts(left_items))
            l_right = fit_line_cluster(extract_pts(right_items))

            tl = intersect_lines_2d(l_top, l_left)
            tr = intersect_lines_2d(l_top, l_right)
            br = intersect_lines_2d(l_bot, l_right)
            bl = intersect_lines_2d(l_bot, l_left)

            if tl is None or tr is None or br is None or bl is None:
                continue

            corners = [tl, tr, br, bl]
            if any(
                x < -w_img * 0.2 or x > w_img * 1.2 or y < -h_img * 0.2 or y > h_img * 1.2
                for x, y in corners
            ):
                continue

            is_valid, _, _ = GeometryTransformer.validate_quadrilateral(
                corners,
                image_shape=(h_img, w_img),
                min_area_ratio=0.05,
            )
            if not is_valid:
                continue

            canonical = order_corners_canonical(corners)
            if skip_cb:
                return canonical

            total_score = h_score + v_score + _checkerboard_border_score(gray, canonical)
            if total_score > best_total_score:
                best_total_score = total_score
                best_corners = canonical

    return best_corners


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

        # Subpixel refinement anchored near analytical Hough intersection
        return refine_subpixel_corners(image, ordered, win_size=self.subpixel_win_size, max_shift=3.5)
