"""Piece classifier runtime interfaces and inference engines.

Implements ONNX Runtime batch-64 inference, Mock deterministic simulation,
and automatic backend selection according to:
- STAGE_3.md, Section 4.4
- 06_DATA_AND_MLOPS.md, Section 8
- Explanatory Note, Section 2.4.2
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.config import ClassifierConfig
from src.core.datatypes import (
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
    # Apply stable softmax if raw logits were passed
    if np.any(probs_arr < 0.0) or np.any(probs_arr > 1.0) or not np.isclose(float(np.sum(probs_arr)), 1.0, atol=0.05):
        exp_vals = np.exp(probs_arr - np.max(probs_arr))
        probs_arr = exp_vals / np.sum(exp_vals)

    probs_dict = {cls_name: float(probs_arr[i]) for i, cls_name in enumerate(PIECE_CLASSES)}

    best_idx = int(np.argmax(probs_arr))
    best_cls = PIECE_CLASSES[best_idx]
    confidence = float(probs_arr[best_idx])

    # Occupancy decision based on configured threshold
    empty_prob = probs_dict["empty"]
    occupied_prob = 1.0 - empty_prob
    is_occ = occupied_prob >= occupancy_threshold

    if not is_occ:
        best_cls = "empty"
        confidence = empty_prob

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

