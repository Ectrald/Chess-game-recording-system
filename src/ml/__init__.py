"""Machine learning module: automated dataset harvesting, piece classification and MLOps."""

from src.ml.auto_labeller import AutoLabeller, AutoLabellerReport
from src.ml.dataset import (
    ChessCropDataset,
    DatasetSample,
    compute_class_weights,
    decompose_piece_class,
    grouped_split_dataset,
    load_dataset_from_manifest,
)
from src.ml.export import export_model_to_onnx, verify_onnx_numerical_parity
from src.ml.models import synthesize_13_class_probabilities
from src.ml.piece_classifier import (
    BasePieceClassifier,
    MockPieceClassifier,
    ONNXPieceClassifier,
    create_piece_classifier,
    preprocess_patches_to_batch,
    probabilities_to_cell_observation,
)
from src.ml.trainer import PieceClassifierTrainer, compute_classification_metrics
from src.ml.transforms import (
    AnchoredScale,
    AnchoredShear,
    ColorJitter,
    Compose,
    Resize,
    SensorNoiseAndBlur,
    ShadowInjection,
    get_chess_train_transforms,
    get_chess_val_transforms,
)

__all__ = [
    "AutoLabeller",
    "AutoLabellerReport",
    "ChessCropDataset",
    "DatasetSample",
    "compute_class_weights",
    "decompose_piece_class",
    "grouped_split_dataset",
    "load_dataset_from_manifest",
    "export_model_to_onnx",
    "verify_onnx_numerical_parity",
    "synthesize_13_class_probabilities",
    "BasePieceClassifier",
    "MockPieceClassifier",
    "ONNXPieceClassifier",
    "create_piece_classifier",
    "preprocess_patches_to_batch",
    "probabilities_to_cell_observation",
    "PieceClassifierTrainer",
    "compute_classification_metrics",
    "AnchoredScale",
    "AnchoredShear",
    "ColorJitter",
    "Compose",
    "Resize",
    "SensorNoiseAndBlur",
    "ShadowInjection",
    "get_chess_train_transforms",
    "get_chess_val_transforms",
]

