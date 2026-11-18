"""Unit tests for piece classifier inference engines and facades in src/ml/piece_classifier.py.

Covers:
- STAGE_3.md Section 6: test_piece_classifier.py specifications
- preprocess_patches_to_batch tensor shaping and normalization
- probabilities_to_cell_observation thresholding and piece selection
- MockPieceClassifier ground-truth simulation and latency tracking
- ONNXPieceClassifier graceful uninitialized handling
- Factory create_piece_classifier backend resolution
"""

import numpy as np
import pytest

from src.core.config import ClassifierConfig
from src.core.datatypes import CHESS_SQUARES, CellObservation, PIECE_CLASSES
from src.ml.piece_classifier import (
    BasePieceClassifier,
    MockPieceClassifier,
    ONNXPieceClassifier,
    create_piece_classifier,
    preprocess_patches_to_batch,
    probabilities_to_cell_observation,
)


def test_preprocess_patches_to_batch():
    """Verifies that 64 cell patches are formatted into a contiguous (64, 3, 112, 112) tensor."""
    patches = {
        sq: np.full((112, 112, 3), 120, dtype=np.uint8)
        for sq in CHESS_SQUARES
    }
    # Test handling of missing squares
    del patches["a1"]

    batch_tensor = preprocess_patches_to_batch(patches, target_size=(112, 112))
    assert isinstance(batch_tensor, np.ndarray)
    assert batch_tensor.shape == (64, 3, 112, 112)
    assert batch_tensor.dtype == np.float32
    assert batch_tensor.flags["C_CONTIGUOUS"]


def test_probabilities_to_cell_observation():
    """Verifies probability array mapping to CellObservation with occupancy thresholding."""
    probs = np.zeros(13, dtype=np.float32)
    # White Knight on b1 with 0.85 confidence
    # PIECE_CLASSES: ['empty', 'wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', ...]
    probs[2] = 0.85
    probs[0] = 0.10
    probs[1] = 0.05

    obs = probabilities_to_cell_observation("b1", probs, occupancy_threshold=0.50)
    assert isinstance(obs, CellObservation)
    assert obs.square == "b1"
    assert obs.predicted_piece == "wN"
    assert obs.is_occupied is True
    assert pytest.approx(obs.confidence, rel=1e-3) == 0.85
    assert pytest.approx(obs.probabilities["wN"], rel=1e-3) == 0.85

    # Test occupancy thresholding: if occupied prob < threshold -> empty
    probs_weak = np.zeros(13, dtype=np.float32)
    probs_weak[0] = 0.60  # empty prob = 0.60, occupied = 0.40
    probs_weak[1] = 0.40  # wP prob = 0.40
    obs_weak = probabilities_to_cell_observation("e4", probs_weak, occupancy_threshold=0.50)
    assert obs_weak.is_occupied is False
    assert obs_weak.predicted_piece == "empty"
    assert pytest.approx(obs_weak.confidence, rel=1e-3) == 0.60


def test_mock_piece_classifier():
    """Verifies MockPieceClassifier generates valid 64-cell board observations."""
    classifier = MockPieceClassifier(default_piece="empty")
    classifier.set_ground_truth({"e4": "wP", "e8": "bK", "d1": "wQ"})

    dummy_patches = {sq: np.zeros((112, 112, 3), dtype=np.uint8) for sq in CHESS_SQUARES}
    board_obs = classifier.classify_board(dummy_patches)

    assert len(board_obs) == 64
    assert board_obs["e4"].predicted_piece == "wP"
    assert board_obs["e4"].is_occupied is True

    assert board_obs["e8"].predicted_piece == "bK"
    assert board_obs["e8"].is_occupied is True

    assert board_obs["d1"].predicted_piece == "wQ"
    assert board_obs["d1"].is_occupied is True

    assert board_obs["a1"].predicted_piece == "empty"
    assert board_obs["a1"].is_occupied is False

    assert classifier.last_latency_ms >= 0.0


def test_onnx_piece_classifier_missing_model():
    """Verifies ONNXPieceClassifier handles missing model file gracefully without crashing."""
    cfg = ClassifierConfig(model_path="non_existent_weights.onnx")
    onnx_cls = ONNXPieceClassifier(config=cfg)
    assert not onnx_cls.is_available

    dummy_patches = {sq: np.zeros((112, 112, 3), dtype=np.uint8) for sq in CHESS_SQUARES}
    # Must fallback gracefully to mock classification
    results = onnx_cls.classify_board(dummy_patches)
    assert len(results) == 64


def test_classifier_factory():
    """Verifies create_piece_classifier factory returns working classifier instances."""
    # Disabled
    cls_disabled = create_piece_classifier(ClassifierConfig(enabled=False))
    assert isinstance(cls_disabled, BasePieceClassifier)

    # Mock
    cls_mock = create_piece_classifier(backend="mock")
    assert isinstance(cls_mock, MockPieceClassifier)

    # Auto fallback
    cls_auto = create_piece_classifier(backend="auto")
    assert isinstance(cls_auto, BasePieceClassifier)

