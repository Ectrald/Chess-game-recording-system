"""Chessboard localization subsystem implementing coarse-to-fine and classical strategies.

Implements localizer hierarchy according to:
- STAGE_1.md, Section 4.1
- Explanatory Note, Section 2.4.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.core.config import VisionConfig
from src.core.datatypes import BoardGeometry, BoardOrientation, Frame
from src.vision.corner_refinement import CornerRefiner, order_corners_canonical, refine_subpixel_corners
from src.vision.homography import GeometryTransformer


class BaseBoardLocalizer(ABC):
    """Abstract base strategy for chess board localization."""

    @abstractmethod
    def locate(self, frame: Frame) -> Optional[BoardGeometry]:
        """Locates the chess board in the frame and returns its geometry model, or None if lost."""
        pass


class PureHoughLocalizer(BaseBoardLocalizer):
    """Deterministic localizer using Canny edges and Hough line clustering without neural nets."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_thresh: int = 50,
        min_line_len: int = 50,
        max_line_gap: int = 20,
        subpixel_win_size: int = 5,
        target_size: int = 640,
        min_area_ratio: float = 0.15,
    ) -> None:
        self.refiner = CornerRefiner(
            canny_low=canny_low,
            canny_high=canny_high,
            hough_thresh=hough_thresh,
            min_line_len=min_line_len,
            max_line_gap=max_line_gap,
            subpixel_win_size=subpixel_win_size,
        )
        self.target_size = target_size
        self.min_area_ratio = min_area_ratio

    def locate(self, frame: Frame) -> Optional[BoardGeometry]:
        corners = self.refiner.find_board_corners(frame.image)
        if corners is None or len(corners) != 4:
            return None

        is_valid, _, _ = GeometryTransformer.validate_quadrilateral(
            corners,
            image_shape=(frame.height, frame.width),
            min_area_ratio=self.min_area_ratio,
        )
        if not is_valid:
            return None

        return GeometryTransformer.build_board_geometry(
            corners=corners,
            target_size=self.target_size,
            image_shape=(frame.height, frame.width),
            min_area_ratio=self.min_area_ratio,
            detection_method="pure_hough",
        )


class CoarseToFineLocalizer(BaseBoardLocalizer):
    """Primary two-stage localizer: Coarse BBox detection + Subpixel grid refinement."""

    def __init__(
        self,
        yolo_model_path: Optional[Union[str, Path]] = None,
        yolo_confidence: float = 0.40,
        config: Optional[VisionConfig] = None,
        target_size: int = 640,
        min_area_ratio: float = 0.15,
    ) -> None:
        self.yolo_model_path = Path(yolo_model_path) if yolo_model_path else None
        self.yolo_confidence = yolo_confidence
        self.target_size = target_size
        self.min_area_ratio = min_area_ratio
        self.yolo_model = None

        canny_low = 50
        canny_high = 150
        hough_thresh = 50
        min_line_len = 50
        max_line_gap = 20
        subpixel_win = 5


        if config is not None and hasattr(config, "localizer"):
            loc = config.localizer
            canny_low = loc.canny_low_threshold
            canny_high = loc.canny_high_threshold
            hough_thresh = loc.hough_threshold
            min_line_len = loc.hough_min_line_length
            max_line_gap = loc.hough_max_line_gap
            subpixel_win = loc.subpixel_win_size
            self.min_area_ratio = config.min_board_area_ratio

        self.refiner = CornerRefiner(
            canny_low=canny_low,
            canny_high=canny_high,
            hough_thresh=hough_thresh,
            min_line_len=min_line_len,
            max_line_gap=max_line_gap,
            subpixel_win_size=subpixel_win,
        )

        self._init_yolo_model()

    def _init_yolo_model(self) -> None:
        if self.yolo_model_path and self.yolo_model_path.exists():
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]
                self.yolo_model = YOLO(str(self.yolo_model_path))
            except Exception:
                self.yolo_model = None

    def _detect_coarse_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Stage 1: Detects coarse bounding box of the board with margin."""
        h, w = image.shape[:2]

        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    source=image,
                    conf=self.yolo_confidence,
                    verbose=False,
                )
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    if len(boxes) >= 4:
                        all_xyxy = boxes.xyxy.cpu().numpy()
                        x1 = int(np.min(all_xyxy[:, 0]))
                        y1 = int(np.min(all_xyxy[:, 1]))
                        x2 = int(np.max(all_xyxy[:, 2]))
                        y2 = int(np.max(all_xyxy[:, 3]))
                    else:
                        box = boxes[0]
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                    # Add 8% padding margin
                    pad_x = int((x2 - x1) * 0.08)
                    pad_y = int((y2 - y1) * 0.08)
                    return (
                        max(0, x1 - pad_x),
                        max(0, y1 - pad_y),
                        min(w, x2 + pad_x),
                        min(h, y2 + pad_y),
                    )
            except Exception:
                pass

        # Fallback coarse detector using grid-overlap saliency and Hough lines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)

        # 1. Try grid intersection saliency mask (simultaneous H and V line presence)
        min_len = max(20, int(min(w, h) * 0.08))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=min_len, maxLineGap=15)
        if lines is not None and len(lines) >= 6:
            h_mask = np.zeros((h, w), dtype=np.uint8)
            v_mask = np.zeros((h, w), dtype=np.uint8)
            for l in lines:
                x1, y1, x2, y2 = [float(v) for v in l.flatten()]
                dx = x2 - x1
                dy = y2 - y1
                ang = np.degrees(np.arctan2(dy, dx)) % 180.0
                if ang > 90.0:
                    ang -= 180.0
                if abs(ang) < 28.0:
                    cv2.line(h_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
                elif abs(ang) > 62.0:
                    cv2.line(v_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)

            k_size = max(15, int(min(w, h) * 0.06))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            h_dil = cv2.dilate(h_mask, kernel)
            v_dil = cv2.dilate(v_mask, kernel)
            grid_mask = cv2.bitwise_and(h_dil, v_dil)

            close_k = max(21, int(min(w, h) * 0.09))
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
            grid_closed = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, close_kernel)

            contours, _ = cv2.findContours(grid_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_boxes: List[Tuple[float, Tuple[int, int, int, int]]] = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < 0.05 * w * h:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh) if bh > 0 else 0.0
                if 0.5 <= aspect <= 2.0:
                    valid_boxes.append((area, (bx, by, bw, bh)))

            if valid_boxes:
                _, (bx, by, bw, bh) = max(valid_boxes, key=lambda it: it[0])
                pad_x = int(bw * 0.08)
                pad_y = int(bh * 0.08)
                return (
                    max(0, bx - pad_x),
                    max(0, by - pad_y),
                    min(w, bx + bw + pad_x),
                    min(h, by + bh + pad_y),
                )

        # 2. Secondary fallback: find external contours with chessboard-like aspect ratio
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            valid_contours: List[Tuple[float, Tuple[int, int, int, int]]] = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 0.10 * w * h:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = bw / float(bh) if bh > 0 else 0.0
                    # Board must be reasonably square, not spanning entire frame
                    if 0.6 <= aspect <= 1.8 and (bw < 0.95 * w or bh < 0.95 * h):
                        valid_contours.append((area, (bx, by, bw, bh)))
            if valid_contours:
                _, (bx, by, bw, bh) = max(valid_contours, key=lambda it: it[0])
                pad_x = int(bw * 0.06)
                pad_y = int(bh * 0.06)
                return (
                    max(0, bx - pad_x),
                    max(0, by - pad_y),
                    min(w, bx + bw + pad_x),
                    min(h, by + bh + pad_y),
                )
        return None

    def locate(self, frame: Frame) -> Optional[BoardGeometry]:
        # Stage 1: Coarse BBox
        roi_bbox = self._detect_coarse_bbox(frame.image)

        # Stage 2: Fine Corner Refinement inside ROI
        corners = self.refiner.find_board_corners(frame.image, roi_bbox=roi_bbox)
        if corners is None or len(corners) != 4:
            return None

        is_valid, _, _ = GeometryTransformer.validate_quadrilateral(
            corners,
            image_shape=(frame.height, frame.width),
            min_area_ratio=self.min_area_ratio,
        )
        if not is_valid:
            return None

        return GeometryTransformer.build_board_geometry(
            corners=corners,
            target_size=self.target_size,
            image_shape=(frame.height, frame.width),
            min_area_ratio=self.min_area_ratio,
            detection_method="yolo_hough",
        )


class ManualCornerLocalizer(BaseBoardLocalizer):
    """Fallback localizer calibrated manually by the human operator."""

    def __init__(
        self,
        corners: Sequence[Tuple[float, float]],
        target_size: int = 640,
        subpixel_refinement: bool = True,
    ) -> None:
        self.corners = order_corners_canonical(corners)
        self.target_size = target_size
        self.subpixel_refinement = subpixel_refinement

    def update_corners(self, corners: Sequence[Tuple[float, float]]) -> None:
        """Allows operator to update the 4 calibrated corners."""
        self.corners = order_corners_canonical(corners)

    def locate(self, frame: Frame) -> Optional[BoardGeometry]:
        pts = self.corners
        if self.subpixel_refinement:
            pts = refine_subpixel_corners(frame.image, pts)

        is_valid, _, _ = GeometryTransformer.validate_quadrilateral(pts)
        if not is_valid:
            return None

        return GeometryTransformer.build_board_geometry(
            corners=pts,
            target_size=self.target_size,
            image_shape=(frame.height, frame.width),
            min_area_ratio=0.05,
            detection_method="manual",
        )


class ArucoLocalizer(BaseBoardLocalizer):
    """Laboratory reference localizer using 4 corner ArUco markers for Ground Truth."""

    def __init__(
        self,
        marker_ids: Tuple[int, int, int, int] = (0, 1, 2, 3),  # TL, TR, BR, BL
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
        target_size: int = 640,
    ) -> None:
        self.marker_ids = marker_ids
        self.target_size = target_size
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def locate(self, frame: Frame) -> Optional[BoardGeometry]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY) if frame.image.ndim == 3 else frame.image
        marker_corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            return None

        id_map: Dict[int, np.ndarray] = {}
        for m_id, m_corners in zip(ids.flatten(), marker_corners):
            id_map[int(m_id)] = m_corners[0]

        # Verify all 4 required marker IDs are present
        if not all(req_id in id_map for req_id in self.marker_ids):
            return None

        # Extract center of each marker:
        # TL = marker_ids[0], TR = marker_ids[1], BR = marker_ids[2], BL = marker_ids[3]
        board_corners: List[Tuple[float, float]] = []
        for req_id in self.marker_ids:
            pts = id_map[req_id]
            center_x = float(np.mean(pts[:, 0]))
            center_y = float(np.mean(pts[:, 1]))
            board_corners.append((center_x, center_y))

        ordered = order_corners_canonical(board_corners)
        is_valid, _, _ = GeometryTransformer.validate_quadrilateral(ordered)
        if not is_valid:
            return None

        return GeometryTransformer.build_board_geometry(
            corners=ordered,
            target_size=self.target_size,
            image_shape=(frame.height, frame.width),
            min_area_ratio=0.05,
            detection_method="aruco",
        )


def create_board_localizer(
    config: Union[VisionConfig, Dict[str, Any]],
) -> BaseBoardLocalizer:
    """Factory creating localizer strategy from configuration settings."""
    if isinstance(config, VisionConfig):
        mode = config.localizer.mode
        yolo_path = config.localizer.yolo_model_path
        yolo_conf = config.localizer.yolo_confidence
    elif isinstance(config, dict):
        loc_cfg = config.get("localizer", {})
        mode = loc_cfg.get("mode", "coarse_to_fine")
        yolo_path = loc_cfg.get("yolo_model_path", "data/weights/yolov8n_board.pt")
        yolo_conf = loc_cfg.get("yolo_confidence", 0.40)
    else:
        mode = "coarse_to_fine"
        yolo_path = None
        yolo_conf = 0.40

    if mode == "pure_hough":
        return PureHoughLocalizer()
    elif mode == "manual":
        # Default placeholder corners for manual mode
        return ManualCornerLocalizer(
            corners=[(200.0, 100.0), (800.0, 100.0), (800.0, 700.0), (200.0, 700.0)]
        )
    elif mode == "aruco":
        return ArucoLocalizer()
    else:
        return CoarseToFineLocalizer(
            yolo_model_path=yolo_path,
            yolo_confidence=yolo_conf,
        )
