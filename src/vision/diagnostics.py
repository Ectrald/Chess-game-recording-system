"""Diagnostic visualizer for chessboard detection, homography grid, and cell ROIs.

Implements visual overlays according to:
- STAGE_1.md, Section 4.6
- Explanatory Note, Section 2.4.1
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.core.datatypes import BoardGeometry, BoardOrientation, CellROI, Frame
from src.vision.homography import GeometryTransformer


class BoardDiagnosticVisualizer:
    """Renders informative diagnostic overlays for computer vision inspection."""

    @staticmethod
    def draw_board_diagnostics(
        frame: Frame,
        geometry: Optional[BoardGeometry] = None,
        rois: Optional[Dict[str, CellROI]] = None,
        normalized_board: Optional[np.ndarray] = None,
        draw_labels: bool = True,
        draw_grid: bool = True,
        thumbnail_size: Tuple[int, int] = (240, 240),
    ) -> np.ndarray:
        """Renders comprehensive diagnostic overlay onto a copy of the source frame image.
        
        Overlays:
        1. 4 outer subpixel corners with labels [TL, TR, BR, BL] and boundary polygon.
        2. Projected 8x8 grid lines on the source frame.
        3. Algebraic square coordinates at cell centers.
        4. Information badge with orientation, confidence S_g, and reprojection error.
        5. Inset thumbnail of normalized board (if provided).
        
        Returns:
            New BGR image with diagnostic annotations.
        """
        vis = frame.image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        h_img, w_img = vis.shape[:2]

        if geometry is not None and geometry.is_valid:
            corners = geometry.to_numpy_corners()
            poly_pts = corners.astype(np.int32).reshape((-1, 1, 2))

            # 1. Outer quadrilateral polygon (bright green)
            cv2.polylines(vis, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

            # 2. 4 corner markers with colored circles and labels
            corner_info = [
                ("TL", (0, 0, 255)),    # Red
                ("TR", (0, 255, 255)),  # Yellow
                ("BR", (255, 0, 0)),    # Blue
                ("BL", (0, 200, 0)),    # Dark Green
            ]
            for (label, col), (x, y) in zip(corner_info, corners):
                ix, iy = int(round(x)), int(round(y))
                cv2.circle(vis, (ix, iy), 6, col, -1, lineType=cv2.LINE_AA)
                cv2.circle(vis, (ix, iy), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.putText(
                    vis,
                    label,
                    (ix + 10, iy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    label,
                    (ix + 10, iy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    col,
                    1,
                    cv2.LINE_AA,
                )

            # 3. Projected 8x8 grid lines
            h_inv = geometry.to_numpy_inverse_homography()
            if draw_grid and h_inv is not None:
                step = float(geometry.target_size) / 8.0
                grid_points_board: List[Tuple[float, float]] = []

                # Internal horizontal and vertical grid lines (ranks 1..7, files b..h)
                for i in range(1, 8):
                    coord = i * step
                    # Horizontal line (x: 0 -> target_size, y: coord)
                    grid_points_board.extend([(0.0, coord), (float(geometry.target_size), coord)])
                    # Vertical line (x: coord, y: 0 -> target_size)
                    grid_points_board.extend([(coord, 0.0), (coord, float(geometry.target_size))])

                if grid_points_board:
                    pts_np = np.array(grid_points_board, dtype=np.float32)
                    proj = GeometryTransformer.project_board_to_image(pts_np, h_inv)
                    for k in range(0, len(proj), 2):
                        p1 = (int(round(proj[k][0])), int(round(proj[k][1])))
                        p2 = (int(round(proj[k + 1][0])), int(round(proj[k + 1][1])))
                        cv2.line(vis, p1, p2, (0, 220, 100), 1, lineType=cv2.LINE_AA)

            # 4. Square labels
            if draw_labels and rois is not None:
                for sq, roi in rois.items():
                    # Draw text at projected center
                    cx, cy = int(round(roi.center_image[0])), int(round(roi.center_image[1]))
                    if 0 <= cx < w_img and 0 <= cy < h_img:
                        cv2.putText(
                            vis,
                            sq,
                            (cx - 10, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.38,
                            (240, 240, 240),
                            1,
                            cv2.LINE_AA,
                        )

            # 5. Diagnostic Information Badge
            badge_h = 100
            badge_w = 320
            overlay = vis.copy()
            cv2.rectangle(overlay, (20, 20), (20 + badge_w, 20 + badge_h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
            cv2.rectangle(vis, (20, 20), (20 + badge_w, 20 + badge_h), (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            orient_str = geometry.orientation.value.upper()
            method_str = geometry.detection_method.upper()
            cv2.putText(vis, f"METHOD: {method_str}", (30, 45), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"ORIENT: {orient_str}", (30, 68), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                vis,
                f"S_g: {geometry.confidence:.2f} | RMSE: {geometry.reprojection_error_px:.2f}px",
                (30, 92),
                font,
                0.48,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )

        # 6. Inset normalized board thumbnail
        if normalized_board is not None and thumbnail_size[0] > 0 and thumbnail_size[1] > 0:
            thumb_src = normalized_board
            if thumb_src.dtype != np.uint8:
                thumb_src = np.clip(thumb_src, 0, 255).astype(np.uint8)
            if thumb_src.ndim == 2:
                thumb_src = cv2.cvtColor(thumb_src, cv2.COLOR_GRAY2BGR)
            elif thumb_src.ndim == 3 and thumb_src.shape[2] == 4:
                thumb_src = cv2.cvtColor(thumb_src, cv2.COLOR_BGRA2BGR)
            thumb_w, thumb_h = thumbnail_size
            thumb = cv2.resize(thumb_src, (thumb_w, thumb_h))
            margin = 20
            x1 = w_img - thumb_w - margin
            y1 = margin
            x2 = x1 + thumb_w
            y2 = y1 + thumb_h

            if x1 >= 0 and y2 <= h_img:
                vis[y1:y2, x1:x2] = thumb
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    vis,
                    "NORMALIZED 640x640",
                    (x1 + 8, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        return vis


def draw_board_diagnostics(
    frame: Frame,
    geometry: Optional[BoardGeometry] = None,
    rois: Optional[Dict[str, CellROI]] = None,
    normalized_board: Optional[np.ndarray] = None,
    draw_labels: bool = True,
    draw_grid: bool = True,
) -> np.ndarray:
    """Convenience functional wrapper for BoardDiagnosticVisualizer.draw_board_diagnostics."""
    return BoardDiagnosticVisualizer.draw_board_diagnostics(
        frame=frame,
        geometry=geometry,
        rois=rois,
        normalized_board=normalized_board,
        draw_labels=draw_labels,
        draw_grid=draw_grid,
    )

