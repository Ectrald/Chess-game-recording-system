"""Hand detection and chess board occlusion masking module.

Implements hand tracking, polygon projection, and per-square occlusion calculation according to:
- STAGE_2.md, Section 4.2
- 04_VISION_PIPELINE.md, Section 4.2
- Explanatory Note, Section 2.4.2
"""

from __future__ import annotations

import gc
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.core.config import HandDetectorConfig
from src.core.datatypes import (
    BoardGeometry,
    CellROI,
    CHESS_SQUARES,
    Frame,
    HandDetection,
    HandLandmark,
)

logger = logging.getLogger(__name__)


def compute_occlusion_from_board_mask(
    board_mask: np.ndarray,
    rois: Optional[Dict[str, CellROI]] = None,
    occlusion_threshold: float = 0.15,
    board_size: int = 640,
) -> Tuple[List[str], Dict[str, float]]:
    """Computes occlusion ratio and occluded status for all 64 chess squares on 640x640 board.

    Args:
        board_mask: Binary uint8 mask (640x640) where non-zero indicates hand presence.
        rois: Optional pre-extracted 64-square CellROI dictionary.
        occlusion_threshold: Ratio threshold (0..1) above which a cell is marked occluded.
        board_size: Size of square board in pixels (default 640).

    Returns:
        Tuple of (occluded_squares_list, occlusion_ratios_dict).
    """
    occluded_squares: List[str] = []
    occlusion_ratios: Dict[str, float] = {}

    h, w = board_mask.shape[:2]

    if rois is not None:
        for sq, cell_roi in rois.items():
            xmin, ymin, xmax, ymax = cell_roi.board_bbox
            c_ymin = max(0, min(h, ymin))
            c_ymax = max(0, min(h, ymax))
            c_xmin = max(0, min(w, xmin))
            c_xmax = max(0, min(w, xmax))

            cell_area = max(1, (c_ymax - c_ymin) * (c_xmax - c_xmin))
            cell_patch = board_mask[c_ymin:c_ymax, c_xmin:c_xmax]
            occluded_px = int(np.count_nonzero(cell_patch))
            ratio = float(occluded_px / cell_area)
            occlusion_ratios[sq] = ratio
            if ratio >= occlusion_threshold:
                occluded_squares.append(sq)
    else:
        # Fallback to standard 8x8 uniform grid
        cell_size = board_size / 8.0
        for sq in CHESS_SQUARES:
            col = ord(sq[0]) - ord("a")
            row = 8 - int(sq[1])
            xmin = int(round(col * cell_size))
            xmax = int(round((col + 1) * cell_size))
            ymin = int(round(row * cell_size))
            ymax = int(round((row + 1) * cell_size))

            c_ymin = max(0, min(h, ymin))
            c_ymax = max(0, min(h, ymax))
            c_xmin = max(0, min(w, xmin))
            c_xmax = max(0, min(w, xmax))

            cell_area = max(1, (c_ymax - c_ymin) * (c_xmax - c_xmin))
            cell_patch = board_mask[c_ymin:c_ymax, c_xmin:c_xmax]
            occluded_px = int(np.count_nonzero(cell_patch))
            ratio = float(occluded_px / cell_area)
            occlusion_ratios[sq] = ratio
            if ratio >= occlusion_threshold:
                occluded_squares.append(sq)

    return occluded_squares, occlusion_ratios


class BaseHandDetector(ABC):
    """Abstract base class for hand detectors."""

    @abstractmethod
    def detect(
        self,
        frame: Frame,
        geometry: Optional[BoardGeometry] = None,
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> HandDetection:
        """Detects hand presence, keypoints, and computes board square occlusions."""
        pass


class DummyHandDetector(BaseHandDetector):
    """Null-object detector when hand detection is disabled in configuration."""

    def detect(
        self,
        frame: Frame,
        geometry: Optional[BoardGeometry] = None,
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> HandDetection:
        return HandDetection(
            is_hand_present=False,
            confidence=0.0,
            landmarks=[],
            bounding_box=None,
            occluded_squares=[],
            occlusion_ratios={sq: 0.0 for sq in CHESS_SQUARES},
            polygon_board=None,
        )


class MockHandDetector(BaseHandDetector):
    """Mock detector for programmatic simulation in unit tests and benchmarks."""

    def __init__(
        self,
        is_hand_present: bool = False,
        confidence: float = 0.95,
        occluded_squares: Optional[List[str]] = None,
        occlusion_ratios: Optional[Dict[str, float]] = None,
        landmarks: Optional[List[Tuple[float, float, float]]] = None,
        bounding_box: Optional[Tuple[int, int, int, int]] = None,
        polygon_board: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.is_hand_present = is_hand_present
        self.confidence = confidence
        self.occluded_squares = occluded_squares or []
        self.occlusion_ratios = occlusion_ratios or {}
        self.landmarks = landmarks or []
        self.bounding_box = bounding_box
        self.polygon_board = polygon_board

    def set_hand_state(
        self,
        is_present: bool,
        squares: Optional[List[str]] = None,
        ratios: Optional[Dict[str, float]] = None,
    ) -> None:
        """Convenience method to dynamically alter mock state during test runs."""
        self.is_hand_present = is_present
        self.occluded_squares = list(squares) if squares else []
        if ratios:
            self.occlusion_ratios = dict(ratios)
        elif squares:
            self.occlusion_ratios = {sq: 0.85 for sq in squares}
        else:
            self.occlusion_ratios = {}

    def detect(
        self,
        frame: Frame,
        geometry: Optional[BoardGeometry] = None,
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> HandDetection:
        full_ratios = {sq: 0.0 for sq in CHESS_SQUARES}
        full_ratios.update(self.occlusion_ratios)

        return HandDetection(
            is_hand_present=self.is_hand_present,
            confidence=self.confidence if self.is_hand_present else 0.0,
            landmarks=list(self.landmarks),
            bounding_box=self.bounding_box,
            occluded_squares=list(self.occluded_squares),
            occlusion_ratios=full_ratios,
            polygon_board=list(self.polygon_board) if self.polygon_board else None,
        )


class MediaPipeHandDetector(BaseHandDetector):
    """MediaPipe-based hand tracker with convex hull projection and occlusion masking."""

    _CACHE_LOCK = threading.Lock()
    _LANDMARKER_CACHE: Dict[Tuple[str, float, float], Tuple[Any, threading.Lock]] = {}

    def __init__(self, config: Optional[HandDetectorConfig] = None) -> None:
        self.config = config or HandDetectorConfig()
        self._mp = None
        self._hands = None
        self._landmarker = None
        self._landmarker_lock: Optional[threading.Lock] = None
        self._is_tasks_api = False
        self._available = False

        try:
            import mediapipe as mp
            self._mp = mp
            task_model_path = Path("data/weights/hand_landmarker.task")
            if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and task_model_path.is_file():
                try:
                    cache_key = (
                        str(task_model_path),
                        float(self.config.min_detection_confidence),
                        float(self.config.min_tracking_confidence),
                    )
                    with self._CACHE_LOCK:
                        cached = self._LANDMARKER_CACHE.get(cache_key)
                        if cached is None:
                            BaseOptions = mp.tasks.BaseOptions
                            HandLandmarker = mp.tasks.vision.HandLandmarker
                            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                            VisionRunningMode = mp.tasks.vision.RunningMode

                            options = HandLandmarkerOptions(
                                base_options=BaseOptions(model_asset_path=str(task_model_path)),
                                running_mode=VisionRunningMode.IMAGE,
                                num_hands=2,
                                min_hand_detection_confidence=self.config.min_detection_confidence,
                                min_hand_presence_confidence=self.config.min_tracking_confidence,
                                min_tracking_confidence=self.config.min_tracking_confidence,
                            )
                            # Guard MediaPipe SerialDispatcher init from re-entrant GC __del__ deadlock
                            gc_was_enabled = gc.isenabled()
                            gc.disable()
                            try:
                                landmarker = HandLandmarker.create_from_options(options)
                            finally:
                                if gc_was_enabled:
                                    gc.enable()
                            cached = (landmarker, threading.Lock())
                            self._LANDMARKER_CACHE[cache_key] = cached

                    self._landmarker, self._landmarker_lock = cached
                    self._is_tasks_api = True
                    self._available = True
                    logger.info("MediaPipe HandLandmarker (Tasks API) initialized successfully from %s", task_model_path)
                except Exception as t_err:
                    logger.debug("Tasks API initialization skipped (%s), trying Solutions API...", t_err)

            if not self._available and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=self.config.min_detection_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence,
                )
                self._is_tasks_api = False
                self._available = True
                logger.info("MediaPipe HandLandmarker (Solutions API) initialized successfully.")

            if not self._available:
                raise RuntimeError("No compatible MediaPipe HandLandmarker backend found.")
        except Exception as exc:
            logger.warning(
                "MediaPipe could not be initialized (%s). Operating in fallback mode.",
                exc,
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def close(self) -> None:
        """Releases local hand detector references without closing shared Tasks API handles."""
        # Shared _LANDMARKER_CACHE entries stay alive for process lifetime to avoid MediaPipe SerialDispatcher close() deadlocks
        self._landmarker = None
        self._landmarker_lock = None
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None
        self._available = False

    def detect(
        self,
        frame: Frame,
        geometry: Optional[BoardGeometry] = None,
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> HandDetection:
        if not self._available or (self._hands is None and self._landmarker is None):
            return HandDetection(
                is_hand_present=False,
                confidence=0.0,
                landmarks=[],
                bounding_box=None,
                occluded_squares=[],
                occlusion_ratios={sq: 0.0 for sq in CHESS_SQUARES},
                polygon_board=None,
            )

        img = frame.image
        h, w = img.shape[:2]
        if img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        all_landmarks: List[Tuple[float, float, float]] = []
        all_pts_image: List[Tuple[float, float]] = []

        if self._is_tasks_api and self._landmarker is not None:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            if self._landmarker_lock is not None:
                with self._landmarker_lock:
                    detection_result = self._landmarker.detect(mp_image)
            else:
                detection_result = self._landmarker.detect(mp_image)
            if not detection_result.hand_landmarks:
                return HandDetection(
                    is_hand_present=False,
                    confidence=0.0,
                    landmarks=[],
                    bounding_box=None,
                    occluded_squares=[],
                    occlusion_ratios={sq: 0.0 for sq in CHESS_SQUARES},
                    polygon_board=None,
                )
            for hand_lms in detection_result.hand_landmarks:
                for lm in hand_lms:
                    all_landmarks.append((float(lm.x), float(lm.y), float(lm.z)))
                    px = float(lm.x * w)
                    py = float(lm.y * h)
                    all_pts_image.append((px, py))
        else:
            results = self._hands.process(rgb)
            if not results.multi_hand_landmarks:
                return HandDetection(
                    is_hand_present=False,
                    confidence=0.0,
                    landmarks=[],
                    bounding_box=None,
                    occluded_squares=[],
                    occlusion_ratios={sq: 0.0 for sq in CHESS_SQUARES},
                    polygon_board=None,
                )
            for hand_lms in results.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    all_landmarks.append((float(lm.x), float(lm.y), float(lm.z)))
                    px = float(lm.x * w)
                    py = float(lm.y * h)
                    all_pts_image.append((px, py))

        # Calculate bounding box in original image
        xs = [p[0] for p in all_pts_image]
        ys = [p[1] for p in all_pts_image]
        bbox = (
            max(0, int(min(xs))),
            max(0, int(min(ys))),
            min(w - 1, int(max(xs))),
            min(h - 1, int(max(ys))),
        )

        # Compute 2D convex hull of hand landmarks
        pts_arr = np.array(all_pts_image, dtype=np.float32)
        hull = cv2.convexHull(pts_arr)

        board_size = geometry.target_size if geometry else 640
        board_mask = np.zeros((board_size, board_size), dtype=np.uint8)
        polygon_board: Optional[List[Tuple[float, float]]] = None

        if geometry is not None and geometry.homography_matrix is not None:
            H = np.array(geometry.homography_matrix, dtype=np.float32)
            pts_reshaped = hull.reshape(-1, 1, 2)
            pts_board = cv2.perspectiveTransform(pts_reshaped, H).reshape(-1, 2)
            polygon_board = [(float(pt[0]), float(pt[1])) for pt in pts_board]

            # Rasterize polygon onto 640x640 board mask
            pts_int = np.int32([pts_board])
            cv2.fillPoly(board_mask, pts_int, 255)

            # Dilate mask to account for finger volume and cast shadows
            margin = self.config.dilation_margin_px
            if margin > 0:
                ksize = margin * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                board_mask = cv2.dilate(board_mask, kernel)

            occluded_squares, occlusion_ratios = compute_occlusion_from_board_mask(
                board_mask=board_mask,
                rois=rois,
                occlusion_threshold=self.config.occlusion_threshold,
                board_size=board_size,
            )
            is_present = np.any(board_mask > 0)
        else:
            occluded_squares = []
            occlusion_ratios = {sq: 0.0 for sq in CHESS_SQUARES}
            is_present = True

        return HandDetection(
            is_hand_present=is_present,
            confidence=float(self.config.min_detection_confidence),
            landmarks=all_landmarks,
            bounding_box=bbox,
            occluded_squares=occluded_squares,
            occlusion_ratios=occlusion_ratios,
            polygon_board=polygon_board,
        )


class ColorSilhouetteHandDetector(BaseHandDetector):
    """Skin-tone color thresholding hand detector in YCrCb color space.

    Provides a fast, zero-dependency CV fallback when MediaPipe is unavailable.
    """

    def __init__(
        self,
        config: Optional[HandDetectorConfig] = None,
        min_hand_area_px: int = 400,
    ) -> None:
        self.config = config or HandDetectorConfig()
        self.min_hand_area_px = min_hand_area_px

    def detect(
        self,
        frame: Frame,
        geometry: Optional[BoardGeometry] = None,
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> HandDetection:
        img = frame.image
        h, w = img.shape[:2]

        if img.ndim != 3 or img.shape[2] != 3:
            # Grayscale images cannot support skin color segmentation
            return HandDetection(
                is_hand_present=False,
                confidence=0.0,
                landmarks=[],
                bounding_box=None,
                occluded_squares=[],
                occlusion_ratios={sq: 0.0 for sq in CHESS_SQUARES},
                polygon_board=None,
            )

        # Convert to YCrCb space for robust skin tone filtering
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # Standard human skin tone empirical range in YCrCb
        lower_skin = np.array([40, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) >= self.min_hand_area_px]

        if not significant_contours:
            return HandDetection(
                is_hand_present=False,
                confidence=0.0,
                landmarks=[],
                bounding_box=None,
                occluded_squares=[],
                occlusion_ratios={sq: 0.0 for sq in CHESS_SQUARES},
                polygon_board=None,
            )

        board_size = geometry.target_size if geometry else 640
        board_mask = np.zeros((board_size, board_size), dtype=np.uint8)
        polygon_board: Optional[List[Tuple[float, float]]] = None

        if geometry is not None and geometry.homography_matrix is not None:
            H = np.array(geometry.homography_matrix, dtype=np.float32)
            # Restrict skin silhouette analysis strictly inside the rectified board space
            # so warm wooden table regions outside the board do not hull across all 64 squares.
            warped_skin = cv2.warpPerspective(skin_mask, H, (board_size, board_size))
            board_contours, _ = cv2.findContours(warped_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_board_hand_area = max(300.0, (board_size * board_size) * 0.008)
            valid_board_contours = [c for c in board_contours if cv2.contourArea(c) >= min_board_hand_area]

            if not valid_board_contours:
                return HandDetection(
                    is_hand_present=False,
                    confidence=0.0,
                    landmarks=[],
                    bounding_box=None,
                    occluded_squares=[],
                    occlusion_ratios={sq: 0.0 for sq in CHESS_SQUARES},
                    polygon_board=None,
                )

            all_board_pts = np.vstack(valid_board_contours)
            board_hull = cv2.convexHull(all_board_pts)
            polygon_board = [(float(pt[0][0]), float(pt[0][1])) for pt in board_hull]
            cv2.fillPoly(board_mask, [np.int32(board_hull)], 255)

            margin = self.config.dilation_margin_px
            if margin > 0:
                ksize = margin * 2 + 1
                dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                board_mask = cv2.dilate(board_mask, dilate_k)

            occluded_squares, occlusion_ratios = compute_occlusion_from_board_mask(
                board_mask=board_mask,
                rois=rois,
                occlusion_threshold=self.config.occlusion_threshold,
                board_size=board_size,
            )
            is_present = bool(np.any(board_mask > 0))
            if geometry.inverse_homography is not None:
                H_inv = np.array(geometry.inverse_homography, dtype=np.float32)
                img_pts = cv2.perspectiveTransform(board_hull.astype(np.float32), H_inv)
                x, y, bw, bh = cv2.boundingRect(img_pts)
                bbox = (x, y, x + bw, y + bh)
            else:
                x, y, bw, bh = cv2.boundingRect(board_hull)
                bbox = (x, y, x + bw, y + bh)
        else:
            # No board geometry locked yet: take largest skin contour rather than spanning entire table
            largest = max(significant_contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            x, y, bw, bh = cv2.boundingRect(hull)
            bbox = (x, y, x + bw, y + bh)
            occluded_squares = []
            occlusion_ratios = {sq: 0.0 for sq in CHESS_SQUARES}
            is_present = True

        return HandDetection(
            is_hand_present=is_present,
            confidence=0.70,
            landmarks=[],
            bounding_box=bbox,
            occluded_squares=occluded_squares,
            occlusion_ratios=occlusion_ratios,
            polygon_board=polygon_board,
        )


class HandDetector(BaseHandDetector):
    """Unified HandDetector facade with automatic backend selection.

    Selects MediaPipe if available, gracefully falling back to ColorSilhouette
    or Mock without halting execution.
    """

    def __init__(
        self,
        config: Optional[HandDetectorConfig] = None,
        backend: str = "auto",
    ) -> None:
        self.config = config or HandDetectorConfig()
        self.backend_name = backend
        self._detector: BaseHandDetector

        if not self.config.enabled:
            self._detector = DummyHandDetector()
            self.backend_name = "disabled"
            return

        if backend == "mock":
            self._detector = MockHandDetector()
        elif backend == "color":
            self._detector = ColorSilhouetteHandDetector(self.config)
        elif backend == "mediapipe":
            mp_det = MediaPipeHandDetector(self.config)
            if mp_det.is_available:
                self._detector = mp_det
            else:
                logger.warning("MediaPipe requested but unavailable. Falling back to color silhouette.")
                self._detector = ColorSilhouetteHandDetector(self.config)
                self.backend_name = "color_fallback"
        else:  # "auto"
            mp_det = MediaPipeHandDetector(self.config)
            if mp_det.is_available:
                self._detector = mp_det
                self.backend_name = "mediapipe"
            else:
                self._detector = ColorSilhouetteHandDetector(self.config)
                self.backend_name = "color"

    @property
    def active_backend(self) -> str:
        return self.backend_name

    @property
    def underlying_detector(self) -> BaseHandDetector:
        return self._detector

    def close(self) -> None:
        """Closes the underlying hand detector backend if supported."""
        if hasattr(self._detector, "close"):
            self._detector.close()

    def detect(
        self,
        frame: Frame,
        geometry: Optional[BoardGeometry] = None,
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> HandDetection:
        return self._detector.detect(frame, geometry=geometry, rois=rois)


def create_hand_detector(
    config: Optional[HandDetectorConfig] = None,
    backend: str = "auto",
) -> BaseHandDetector:
    """Factory creating configured hand detector instance."""
    return HandDetector(config=config, backend=backend)
