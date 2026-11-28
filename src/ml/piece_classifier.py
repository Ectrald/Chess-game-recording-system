"""Piece classifier runtime interfaces and inference engines.

Implements ONNX Runtime batch-64 inference, Mock deterministic simulation,
and automatic backend selection according to:
- STAGE_3.md, Section 4.4
- 06_DATA_AND_MLOPS.md, Section 8
- Explanatory Note, Section 2.4.2
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.core.config import ClassifierConfig
from src.core.datatypes import (
    BoardOrientation,
    CellObservation,
    CellROI,
    CHESS_SQUARES,
    PIECE_CLASSES,
)
from src.ml.dataset import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


def preprocess_patches_to_batch(
    cell_patches: Dict[str, np.ndarray],
    target_size: Tuple[int, int] = (112, 112),
    ordered_squares: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Preprocesses a dictionary of 64 cell crops into a normalized batch tensor (64, 3, H, W).

    Args:
        cell_patches: Mapping from square name ('a1'..'h8') to crop image array.
        target_size: Target square dimensions (W, H).
        ordered_squares: Optional sequence defining square ordering (default: canonical CHESS_SQUARES).

    Returns:
        float32 array of shape (64, 3, target_size[1], target_size[0]) with ImageNet normalization.
    """
    squares = ordered_squares or CHESS_SQUARES
    batch_tensors: List[np.ndarray] = []

    for sq in squares:
        patch = cell_patches.get(sq)
        if patch is None or patch.size == 0:
            patch = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)
        elif (patch.shape[1], patch.shape[0]) != target_size:
            patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_LINEAR)

        # Convert to RGB float32 [0, 1]
        if patch.ndim == 2:
            rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        elif patch.shape[2] == 3:
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        elif patch.shape[2] == 4:
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGB)
        else:
            rgb = patch[:, :, :3]

        tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        # ImageNet standardization: (x - mean) / std
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        batch_tensors.append(tensor)

    return np.ascontiguousarray(np.stack(batch_tensors, axis=0), dtype=np.float32)


def probabilities_to_cell_observation(
    square: str,
    probs_13: np.ndarray,
    occupancy_threshold: float = 0.50,
) -> CellObservation:
    """Converts a 13-element probability array into a structured CellObservation.

    Args:
        square: Algebraic square coordinate ('a1'..'h8').
        probs_13: 1D array of 13 probabilities corresponding to PIECE_CLASSES.
        occupancy_threshold: Threshold tau_o above which a square is considered occupied.

    Returns:
        Populated CellObservation instance.
    """
    probs_arr = np.asarray(probs_13, dtype=np.float32)
    prob_sum = float(np.sum(probs_arr))
    if np.all(probs_arr >= 0.0) and np.all(probs_arr <= 1.0) and 0.90 <= prob_sum <= 1.10 and prob_sum > 0.0:
        if not np.isclose(prob_sum, 1.0, atol=1e-4):
            probs_arr = probs_arr / prob_sum
    else:
        exp_vals = np.exp(probs_arr - np.max(probs_arr))
        probs_arr = exp_vals / np.sum(exp_vals)

    probs_dict = {cls_name: float(probs_arr[i]) for i, cls_name in enumerate(PIECE_CLASSES)}

    # Occupancy decision based on configured threshold
    empty_prob = probs_dict["empty"]
    occupied_prob = 1.0 - empty_prob
    is_occ = occupied_prob >= occupancy_threshold

    if not is_occ:
        best_cls = "empty"
        confidence = empty_prob
    else:
        # Select highest-probability piece class (indices 1..12) so occupied never contradicts predicted_piece='empty'
        best_piece_idx = 1 + int(np.argmax(probs_arr[1:]))
        best_cls = PIECE_CLASSES[best_piece_idx]
        confidence = float(probs_arr[best_piece_idx])

    return CellObservation(
        square=square,
        probabilities=probs_dict,
        predicted_piece=best_cls,
        confidence=confidence,
        is_occupied=is_occ,
        is_occluded=False,
    )


class BasePieceClassifier(ABC):
    """Abstract base class for chessboard square piece classifiers."""

    @abstractmethod
    def classify_cell(self, cell_patch: np.ndarray, square: str) -> CellObservation:
        """Classifies a single cropped square patch."""
        pass

    @abstractmethod
    def classify_board(
        self,
        cell_patches: Dict[str, np.ndarray],
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> Dict[str, CellObservation]:
        """Classifies all 64 squares in a single batch-64 forward pass."""
        pass

    def classify_squares(
        self,
        cell_patches: Dict[str, np.ndarray],
        squares: Optional[Sequence[str]] = None,
    ) -> Dict[str, CellObservation]:
        """Classifies a specific subset of squares (sparse dynamic batching)."""
        target_squares = set(squares) if squares is not None else set(cell_patches.keys())
        board_obs = self.classify_board(cell_patches)
        return {sq: obs for sq, obs in board_obs.items() if sq in target_squares}

    @property
    def last_latency_ms(self) -> float:
        """Wall-clock latency of the most recent board inference call in milliseconds."""
        return 0.0


class MockPieceClassifier(BasePieceClassifier):
    """Deterministic piece classifier for testing, benchmarking and CI runs."""

    def __init__(
        self,
        default_piece: str = "empty",
        ground_truth_map: Optional[Dict[str, str]] = None,
        confidence: float = 0.95,
    ) -> None:
        self.default_piece = default_piece
        self.ground_truth_map: Dict[str, str] = dict(ground_truth_map or {})
        self.confidence = confidence
        self._last_latency_ms: float = 0.5

    def set_ground_truth(self, piece_map: Dict[str, str]) -> None:
        """Updates the ground-truth mapping for subsequent classifications."""
        self.ground_truth_map = dict(piece_map)

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def _build_observation(self, square: str) -> CellObservation:
        target_piece = self.ground_truth_map.get(square, self.default_piece)
        if target_piece not in PIECE_CLASSES:
            target_piece = "empty"

        is_occ = target_piece != "empty"
        probs = {}
        rem = (1.0 - self.confidence) / max(1, len(PIECE_CLASSES) - 1)

        for c in PIECE_CLASSES:
            probs[c] = self.confidence if c == target_piece else rem

        return CellObservation(
            square=square,
            probabilities=probs,
            predicted_piece=target_piece,
            confidence=self.confidence,
            is_occupied=is_occ,
            is_occluded=False,
        )

    def classify_cell(self, cell_patch: np.ndarray, square: str) -> CellObservation:
        return self._build_observation(square)

    def classify_board(
        self,
        cell_patches: Dict[str, np.ndarray],
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> Dict[str, CellObservation]:
        t0 = time.perf_counter()
        results = {sq: self._build_observation(sq) for sq in CHESS_SQUARES}
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return results


class ONNXPieceClassifier(BasePieceClassifier):
    """High-performance piece classifier utilizing ONNX Runtime."""

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        self.config = config or ClassifierConfig()
        self._session = None
        self._input_name = None
        self._output_name = None
        self._available = False
        self._last_latency_ms = 0.0

        model_path = Path(self.config.model_path)
        if not model_path.is_file():
            logger.warning("ONNX model file not found at %s. Operating in uninitialized mode.", model_path)
            return

        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if self.config.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            self._session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._available = True
            logger.info("ONNXPieceClassifier loaded model from %s with providers %s", model_path, providers)
        except Exception as exc:
            logger.warning("Could not initialize ONNX Runtime session: %s", exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def classify_cell(self, cell_patch: np.ndarray, square: str) -> CellObservation:
        board_obs = self.classify_board({square: cell_patch})
        return board_obs[square]

    def classify_squares(
        self,
        cell_patches: Dict[str, np.ndarray],
        squares: Optional[Sequence[str]] = None,
    ) -> Dict[str, CellObservation]:
        """Classifies only requested squares in a single dynamic-batch ONNX forward pass."""
        requested = list(squares) if squares is not None else list(cell_patches.keys())
        target_squares = [sq for sq in requested if sq in cell_patches]
        if not target_squares:
            return {}
        if not self._available or self._session is None:
            fallback = MockPieceClassifier()
            return {sq: fallback.classify_cell(cell_patches[sq], sq) for sq in target_squares}

        t0 = time.perf_counter()
        batch_input = preprocess_patches_to_batch(
            cell_patches=cell_patches,
            target_size=self.config.input_size,
            ordered_squares=target_squares,
        )
        outputs = self._session.run([self._output_name], {self._input_name: batch_input})
        probs_batch = outputs[0]
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            sq: probabilities_to_cell_observation(
                square=sq,
                probs_13=probs_batch[idx],
                occupancy_threshold=self.config.occupancy_threshold,
            )
            for idx, sq in enumerate(target_squares)
        }

    def classify_board(
        self,
        cell_patches: Dict[str, np.ndarray],
        rois: Optional[Dict[str, CellROI]] = None,
    ) -> Dict[str, CellObservation]:
        if not self._available or self._session is None:
            # Fallback mock observation if weights are uninitialized
            fallback = MockPieceClassifier()
            return fallback.classify_board(cell_patches, rois=rois)

        t0 = time.perf_counter()
        batch_input = preprocess_patches_to_batch(
            cell_patches=cell_patches,
            target_size=self.config.input_size,
            ordered_squares=CHESS_SQUARES,
        )

        outputs = self._session.run([self._output_name], {self._input_name: batch_input})
        probs_64 = outputs[0]  # shape: (64, 13)

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        results: Dict[str, CellObservation] = {}
        for idx, sq in enumerate(CHESS_SQUARES):
            probs_13 = probs_64[idx]
            obs = probabilities_to_cell_observation(
                square=sq,
                probs_13=probs_13,
                occupancy_threshold=self.config.occupancy_threshold,
            )
            results[sq] = obs

        return results


def create_piece_classifier(
    config: Optional[ClassifierConfig] = None,
    backend: str = "auto",
) -> BasePieceClassifier:
    """Factory creating configured piece classifier instance.

    Args:
        config: Optional ClassifierConfig instance.
        backend: Backend selection: 'auto', 'onnx', 'mock'.

    Returns:
        Instantiated BasePieceClassifier.
    """
    cfg = config or ClassifierConfig()

    if not cfg.enabled:
        return MockPieceClassifier(default_piece="empty")

    target_backend = cfg.backend if backend == "auto" else backend

    if target_backend == "mock":
        return MockPieceClassifier()
    elif target_backend == "onnx":
        onnx_cls = ONNXPieceClassifier(config=cfg)
        if onnx_cls.is_available:
            return onnx_cls
        logger.warning("ONNXPieceClassifier unavailable, falling back to Mock.")
        return MockPieceClassifier()
    else:  # "auto"
        onnx_cls = ONNXPieceClassifier(config=cfg)
        if onnx_cls.is_available:
            return onnx_cls
        return MockPieceClassifier()


class ONNXOccupancyClassifier:
    """Standalone chesscog ResNet occupancy classifier predicting P(occupied) on 100x100 square crops."""

    def __init__(self, model_path: Union[str, Path] = "models/occupancy_classifier.onnx", device: str = "cpu") -> None:
        self._session = None
        self._input_name = None
        self._output_name = None
        self._available = False

        path = Path(model_path)
        if not path.is_file():
            return
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            self._session = ort.InferenceSession(str(path), sess_options, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._available = True
            logger.info("ONNXOccupancyClassifier loaded model from %s", path)
        except Exception as exc:
            logger.debug("Could not initialize ONNXOccupancyClassifier: %s", exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def predict_squares(
        self,
        cell_crops: Dict[str, np.ndarray],
        squares: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Returns P(occupied) in [0, 1] for each requested square."""
        if not self._available or self._session is None:
            return {}
        requested = list(squares) if squares is not None else list(CHESS_SQUARES)
        target = [sq for sq in requested if sq in cell_crops]
        if not target:
            return {}
        batch = preprocess_patches_to_batch(
            cell_patches=cell_crops,
            target_size=(100, 100),
            ordered_squares=target,
        )
        logits = self._session.run([self._output_name], {self._input_name: batch})[0]
        logits_arr = np.asarray(logits, dtype=np.float32)
        exp_vals = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
        probs = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        return {sq: float(probs[idx, 1]) for idx, sq in enumerate(target)}


CHESSQ_CLASS_SYMBOLS: Tuple[str, ...] = (".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
CHESSQ_TO_CANONICAL: Dict[str, str] = {
    ".": "empty",
    "P": "wP", "N": "wN", "B": "wB", "R": "wR", "Q": "wQ", "K": "wK",
    "p": "bP", "n": "bN", "b": "bB", "r": "bR", "q": "bQ", "k": "bK",
}
_CANONICAL_TO_CHESSQ_IDX: List[int] = [
    CHESSQ_CLASS_SYMBOLS.index(k)
    for k in (".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
]


class WholeBoardChessQClassifier:
    """Holistic ViT-S14 INT8 ONNX classifier predicting all 64 squares & board orientation in one pass."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/chessq_vit_int8.onnx",
        device: str = "cpu",
        occupancy_threshold: float = 0.50,
    ) -> None:
        self._session = None
        self._input_name: Optional[str] = None
        self._output_names: List[str] = []
        self._available = False
        self._last_latency_ms = 0.0
        self.occupancy_threshold = occupancy_threshold

        path = Path(model_path)
        if not path.is_file():
            return
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            self._session = ort.InferenceSession(str(path), sess_options, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            self._available = True
            logger.info("WholeBoardChessQClassifier loaded %s (%s)", path, providers)
        except Exception as exc:
            logger.warning("Could not load WholeBoardChessQClassifier from %s: %s", path, exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @staticmethod
    def _rotate_to_white_bottom(img_bgr: np.ndarray, orientation: BoardOrientation) -> np.ndarray:
        if orientation == BoardOrientation.WHITE_RIGHT:
            return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        if orientation == BoardOrientation.WHITE_LEFT:
            return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if orientation == BoardOrientation.BLACK_BOTTOM:
            return cv2.rotate(img_bgr, cv2.ROTATE_180)
        return img_bgr

    def _run_raw(self, board_bgr: np.ndarray, add_padding: int = 60) -> Tuple[np.ndarray, float]:
        if not self._available or self._session is None or self._input_name is None:
            raise RuntimeError("WholeBoardChessQClassifier is not initialized")

        img = board_bgr
        if add_padding > 0:
            img = cv2.copyMakeBorder(
                img,
                add_padding, add_padding, add_padding, add_padding,
                cv2.BORDER_CONSTANT,
                value=(20, 20, 20),
            )
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rz = cv2.resize(rgb, (644, 644), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        ten = np.ascontiguousarray(((rz - mean) / std).transpose(2, 0, 1)[None, ...], dtype=np.float32)

        t0 = time.perf_counter()
        outputs = self._session.run(self._output_names, {self._input_name: ten})
        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = np.asarray(outputs[0][0], dtype=np.float32)  # (64, 13)
        exp_v = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_v / np.sum(exp_v, axis=-1, keepdims=True)
        direction = outputs[1][0] if len(outputs) > 1 else np.array([1.0, 0.0], dtype=np.float32)
        angle_deg = float(np.degrees(np.arctan2(float(direction[1]), float(direction[0]))))
        return probs, angle_deg

    def detect_orientation(self, board_bgr: np.ndarray, add_padding: int = 0) -> Tuple[BoardOrientation, float]:
        """Infers viewing orientation (WHITE_BOTTOM / BLACK_BOTTOM / WHITE_RIGHT / WHITE_LEFT)."""
        if not self._available:
            return BoardOrientation.UNKNOWN, 0.0
        _, ang = self._run_raw(board_bgr, add_padding=add_padding)
        if not math.isfinite(ang):
            return BoardOrientation.UNKNOWN, 0.0
        if -45.0 <= ang <= 45.0:
            ori = BoardOrientation.WHITE_BOTTOM
        elif -135.0 <= ang < -45.0:
            ori = BoardOrientation.WHITE_RIGHT
        elif 45.0 < ang <= 135.0:
            ori = BoardOrientation.WHITE_LEFT
        else:
            ori = BoardOrientation.BLACK_BOTTOM
        return ori, ang

    def classify_board_image(
        self,
        board_bgr: np.ndarray,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
        add_padding: int = 60,
    ) -> Tuple[Dict[str, CellObservation], float]:
        """Runs whole-board inference and maps logits onto algebraic squares a1..h8."""
        oriented = self._rotate_to_white_bottom(board_bgr, orientation)
        pad = add_padding if min(oriented.shape[:2]) <= 660 else 0
        probs_64, ang_deg = self._run_raw(oriented, add_padding=pad)

        # ChessQ output ordering is a8..h8 (row 0), a7..h7 (row 1), ..., a1..h1 (row 7)
        cells: Dict[str, CellObservation] = {}
        files = "abcdefgh"
        for r in range(8):
            rank = 8 - r
            for c in range(8):
                sq = f"{files[c]}{rank}"
                raw_p = probs_64[r * 8 + c]
                # Map directly into PIECE_CLASSES order
                mapped = np.array([float(raw_p[idx]) for idx in _CANONICAL_TO_CHESSQ_IDX], dtype=np.float32)
                cells[sq] = probabilities_to_cell_observation(
                    square=sq,
                    probs_13=mapped,
                    occupancy_threshold=self.occupancy_threshold,
                )
        return cells, ang_deg


class DINOv2PatchChangeDetector:
    """Computes semantic cosine change across 64 chessboard squares using DINOv2-Small INT8."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/dinov2_small_int8.onnx",
        device: str = "cpu",
    ) -> None:
        self._session = None
        self._input_name: Optional[str] = None
        self._available = False
        self._last_latency_ms = 0.0
        self._lock = threading.Lock()
        self._cached_before_sig: Optional[Tuple[int, int, str]] = None
        self._cached_before_grid: Optional[np.ndarray] = None

        path = Path(model_path)
        if not path.is_file():
            return
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            self._session = ort.InferenceSession(str(path), sess_options, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._available = True
            logger.info("DINOv2PatchChangeDetector loaded %s (%s)", path, providers)
        except Exception as exc:
            logger.warning("Could not load DINOv2PatchChangeDetector from %s: %s", path, exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    def reset(self) -> None:
        """Clears cached reference board embedding (e.g. on recalibrate or board lost)."""
        with self._lock:
            self._cached_before_sig = None
            self._cached_before_grid = None

    @staticmethod
    def _image_sig(img: np.ndarray) -> Tuple[int, int, str]:
        thumb = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
        return (int(img.shape[0]), int(img.shape[1]), hashlib.sha1(thumb.tobytes()).hexdigest())

    def _embed_8x8(self, board_bgr: np.ndarray) -> np.ndarray:
        if not self._available or self._session is None or self._input_name is None:
            raise RuntimeError("DINOv2PatchChangeDetector is not initialized")
        img = board_bgr
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            rgb = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rz = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        ten = np.ascontiguousarray(((rz - mean) / std).transpose(2, 0, 1)[None, ...], dtype=np.float32)
        hs = self._session.run(None, {self._input_name: ten})[0][0, 1:, :]  # (256, 384)
        grid = hs.reshape(16, 16, 384)
        cell_e = grid.reshape(8, 2, 8, 2, 384).mean(axis=(1, 3))
        norms = np.linalg.norm(cell_e, axis=-1, keepdims=True) + 1e-8
        return cell_e / norms

    def compute_square_change_map(
        self,
        warped_after: np.ndarray,
        warped_before: np.ndarray,
        orientation: BoardOrientation,
        roi_extractor: Any,
    ) -> Dict[str, float]:
        """Returns cosine distance in [0, 2] per square mapped for current board orientation."""
        if not self._available:
            return {sq: 0.0 for sq in CHESS_SQUARES}
        t0 = time.perf_counter()
        sig_b = self._image_sig(warped_before)
        with self._lock:
            if self._cached_before_sig == sig_b and self._cached_before_grid is not None:
                emb_b = self._cached_before_grid
            else:
                emb_b = self._embed_8x8(warped_before)
                self._cached_before_sig = sig_b
                self._cached_before_grid = emb_b

            emb_a = self._embed_8x8(warped_after)
            self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        cos_dist = np.clip(1.0 - np.sum(emb_b * emb_a, axis=-1), 0.0, 2.0)  # (8, 8)
        board_dim = float(getattr(roi_extractor, "board_size", warped_after.shape[0]) or warped_after.shape[0])
        cell_dim = max(1.0, board_dim / 8.0)
        out: Dict[str, float] = {}
        for sq in CHESS_SQUARES:
            x0, y0, x1, y1 = roi_extractor.get_square_coordinates(sq, orientation)
            r = min(7, max(0, int(((y0 + y1) * 0.5) / cell_dim)))
            c = min(7, max(0, int(((x0 + x1) * 0.5) / cell_dim)))
            out[sq] = float(cos_dist[r, c])
        return out

