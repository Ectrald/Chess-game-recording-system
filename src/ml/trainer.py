"""Training loop, evaluation metrics and MLOps tracking for piece classifiers.

Implements accuracy, per-class F1, 13x13 confusion matrix, ECE calibration,
and MLflow experiment tracking according to:
- STAGE_3.md, Section 4.5
- 06_DATA_AND_MLOPS.md, Section 6, 7
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.core.config import TrainingConfig
from src.core.datatypes import EvaluationReport, ModelManifest, PIECE_CLASSES
from src.ml.dataset import ChessCropDataset, DatasetSample

logger = logging.getLogger(__name__)

# Check MLflow availability
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 13,
) -> EvaluationReport:
    """Computes comprehensive classification metrics across the 13 canonical piece classes.

    Args:
        y_true: Array of ground-truth class indices (0..12).
        y_pred: Array of predicted class indices (0..12).
        y_prob: Optional predicted class probabilities array (N, 13) for ECE calculation.
        num_classes: Total number of classes (default 13).

    Returns:
        EvaluationReport instance.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_pred, dtype=np.int32)

    total = len(y_t)
    if total == 0:
        return EvaluationReport()

    # 1. 13x13 Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # 2. Overall Accuracy
    overall_acc = float(np.mean(y_t == y_p))

    # 3. Occupancy Accuracy (empty = 0 vs occupied = 1..12)
    occ_t = y_t > 0
    occ_p = y_p > 0
    occ_acc = float(np.mean(occ_t == occ_p))

    # 4. Color Accuracy (white = 1..6 vs black = 7..12 on occupied true pieces)
    occupied_mask = occ_t & occ_p
    if np.any(occupied_mask):
        col_t = y_t[occupied_mask] > 6  # False for white (1..6), True for black (7..12)
        col_p = y_p[occupied_mask] > 6
        color_acc = float(np.mean(col_t == col_p))
    else:
        color_acc = 1.0

    # 5. Per-class and Macro Precision, Recall, F1
    precisions = []
    recalls = []
    f1s = []
    per_class_f1: Dict[str, float] = {}

    for i in range(num_classes):
        cls_name = PIECE_CLASSES[i]
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_f1[cls_name] = f1

        # Only average classes present in true or predicted distributions
        if (tp + fn) > 0 or (tp + fp) > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # 6. Expected Calibration Error (ECE)
    ece = 0.0
    if y_prob is not None and len(y_prob) == total:
        y_prob_arr = np.asarray(y_prob, dtype=np.float32)
        confidences = np.max(y_prob_arr, axis=-1)
        predictions = np.argmax(y_prob_arr, axis=-1)
        accuracies = (predictions == y_t).astype(np.float32)

        n_bins = 10
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for b_low, b_up in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > b_low) & (confidences <= b_up)
            prop_in_bin = np.mean(in_bin.astype(np.float32))
            if prop_in_bin > 0:
                acc_in_bin = np.mean(accuracies[in_bin])
                conf_in_bin = np.mean(confidences[in_bin])
                ece += float(np.abs(acc_in_bin - conf_in_bin) * prop_in_bin)

    return EvaluationReport(
        overall_accuracy=overall_acc,
        occupancy_accuracy=occ_acc,
        color_accuracy=color_acc,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        confusion_matrix=cm.tolist(),
        expected_calibration_error=ece,
    )


class PieceClassifierTrainer:
    """Trainer orchestrating dataset iteration, model optimization and MLOps logging."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_enabled = MLFLOW_AVAILABLE and self.config.mlflow_tracking_uri is not None

        if self.mlflow_enabled:
            try:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.set_experiment(self.config.experiment_name)
                logger.info("MLflow tracking initialized at %s", self.config.mlflow_tracking_uri)
            except Exception as exc:
                logger.warning("Could not connect to MLflow server (%s). Logging locally.", exc)
                self.mlflow_enabled = False

    def evaluate(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[np.ndarray] = None,
    ) -> EvaluationReport:
        """Computes evaluation report and optionally logs to MLflow or local file."""
        report = compute_classification_metrics(y_true, y_pred, y_prob=y_prob)

        report_file = self.output_dir / "evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        if self.mlflow_enabled:
            try:
                mlflow.log_metrics({
                    "val_accuracy": report.overall_accuracy,
                    "val_macro_f1": report.macro_f1,
                    "val_occ_accuracy": report.occupancy_accuracy,
                    "val_color_accuracy": report.color_accuracy,
                    "val_ece": report.expected_calibration_error,
                })
                mlflow.log_artifact(str(report_file))
            except Exception as exc:
                logger.warning("Failed to log metrics to MLflow: %s", exc)

        return report

    def log_hyperparameters(self, extra_params: Optional[Dict[str, Any]] = None) -> None:
        """Logs training hyperparameters to MLflow and local JSON."""
        params = self.config.model_dump()
        if extra_params:
            params.update(extra_params)

        params_file = self.output_dir / "hyperparameters.json"
        with open(params_file, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        if self.mlflow_enabled:
            try:
                # MLflow accepts str, int, float values
                mlflow.log_params({k: str(v) for k, v in params.items()})
            except Exception as exc:
                logger.warning("Failed to log parameters to MLflow: %s", exc)

