"""Unit tests for training metrics, evaluation report, and probability synthesis.

Covers:
- STAGE_3.md Section 6: test_training_pipeline.py specifications
- compute_classification_metrics accuracy, macro F1, occupancy/color accuracy
- 13x13 confusion matrix construction
- Expected Calibration Error (ECE) calculation
- synthesize_13_class_probabilities mathematical normalization
- PieceClassifierTrainer local logging and evaluation report generation
"""

from pathlib import Path
import numpy as np
import pytest

from src.core.config import TrainingConfig
from src.core.datatypes import EvaluationReport, PIECE_CLASSES
from src.ml.models import synthesize_13_class_probabilities
from src.ml.trainer import PieceClassifierTrainer, compute_classification_metrics


def test_synthesize_13_class_probabilities_identity():
    """Verifies that multi-head probability decomposition sums strictly to 1.0."""
    # Batch size = 3
    # Sample 0: confident empty
    # Sample 1: confident White Pawn (wP)
    # Sample 2: confident Black King (bK)
    p_occ = np.array([
        [0.98, 0.02],
        [0.05, 0.95],
        [0.01, 0.99],
    ], dtype=np.float32)

    p_col = np.array([
        [0.50, 0.50],
        [0.96, 0.04],  # White
        [0.02, 0.98],  # Black
    ], dtype=np.float32)

    # Types: [P, N, B, R, Q, K]
    p_typ = np.array([
        [1/6]*6,
        [0.90, 0.02, 0.02, 0.02, 0.02, 0.02],  # Pawn
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.95],  # King
    ], dtype=np.float32)

    probs_13 = synthesize_13_class_probabilities(p_occ, p_col, p_typ)

    assert probs_13.shape == (3, 13)
    # Every row must sum to 1.0 within float numerical tolerance
    for row in probs_13:
        assert pytest.approx(float(np.sum(row)), rel=1e-5) == 1.0

    # Sample 0: empty is index 0
    assert np.argmax(probs_13[0]) == 0
    assert pytest.approx(probs_13[0, 0], rel=1e-3) == 0.98

    # Sample 1: wP is index 1
    assert np.argmax(probs_13[1]) == 1
    assert probs_13[1, 1] > 0.80

    # Sample 2: bK is index 12 (last)
    assert np.argmax(probs_13[2]) == 12
    assert probs_13[2, 12] > 0.90


def test_compute_classification_metrics_perfect():
    """Verifies metrics report when all predictions are 100% correct."""
    y_true = np.array([0, 1, 2, 7, 12], dtype=np.int32)
    y_pred = np.array([0, 1, 2, 7, 12], dtype=np.int32)

    probs = np.zeros((5, 13), dtype=np.float32)
    for i, c in enumerate(y_pred):
        probs[i, c] = 1.0

    report = compute_classification_metrics(y_true, y_pred, y_prob=probs)
    assert report.overall_accuracy == 1.0
    assert report.occupancy_accuracy == 1.0
    assert report.color_accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.expected_calibration_error == 0.0
    assert np.array(report.confusion_matrix).shape == (13, 13)


def test_compute_classification_metrics_with_errors():
    """Verifies metrics calculations with known mistake patterns."""
    # 20 samples:
    # 10 empty (0)
    # 5 White Pawn (1)
    # 5 Black Knight (8)
    y_true = np.array([0]*10 + [1]*5 + [8]*5, dtype=np.int32)
    # 1 empty classified as wP, 1 wP classified as wN
    y_pred = np.array([0]*9 + [1]*1 + [1]*4 + [2]*1 + [8]*5, dtype=np.int32)

    probs = np.zeros((20, 13), dtype=np.float32)
    for i, c in enumerate(y_pred):
        probs[i, c] = 0.80
        probs[i, (c + 1) % 13] = 0.20

    report = compute_classification_metrics(y_true, y_pred, y_prob=probs)
    # 18 out of 20 correct -> 0.90 accuracy
    assert pytest.approx(report.overall_accuracy, rel=1e-3) == 0.90
    assert report.macro_f1 > 0.65
    assert report.per_class_f1["bN"] == 1.0  # All black knights were correct
    assert report.expected_calibration_error >= 0.0


def test_compute_classification_metrics_empty():
    """Verifies graceful handling of empty sample lists."""
    report = compute_classification_metrics([], [])
    assert report.overall_accuracy == 0.0
    assert report.macro_f1 == 0.0


def test_trainer_evaluate_and_log(tmp_path: Path):
    """Verifies that PieceClassifierTrainer logs hyperparameters and writes evaluation_report.json."""
    cfg = TrainingConfig(epochs=10, learning_rate=1e-3, output_dir=str(tmp_path))
    trainer = PieceClassifierTrainer(config=cfg, output_dir=tmp_path)

    trainer.log_hyperparameters(extra_params={"backbone_frozen": True})
    params_file = tmp_path / "hyperparameters.json"
    assert params_file.is_file()

    y_true = [0, 1, 2, 0, 7]
    y_pred = [0, 1, 2, 0, 7]
    report = trainer.evaluate(y_true, y_pred)

    assert report.overall_accuracy == 1.0
    report_file = tmp_path / "evaluation_report.json"
    assert report_file.is_file()
