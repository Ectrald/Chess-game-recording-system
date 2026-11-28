"""Configuration models and loader for the Chess Game Recording System.

Defines Pydantic v2 configuration schemas reflecting the mathematical
parameters of the Explanatory Note (Sections 2.4.1, 2.4.2, 2.4.4, 2.4.5, 2.4.8).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class LocalizerConfig(BaseModel):
    """Configuration for board detection and corner refinement."""
    mode: str = Field(
        default="coarse_to_fine",
        description="Localizer strategy: 'coarse_to_fine', 'pure_hough', 'manual', 'aruco'",
    )
    yolo_model_path: str = Field(
        default="data/weights/yolov8n_board.pt",
        description="Path to pretrained YOLOv8 board detector weights",
    )
    yolo_confidence: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="Confidence threshold for YOLO bounding box detection",
    )
    canny_low_threshold: int = Field(default=50, ge=1, le=255)
    canny_high_threshold: int = Field(default=150, ge=1, le=255)
    hough_threshold: int = Field(default=80, ge=1)
    hough_min_line_length: int = Field(default=100, ge=1)
    hough_max_line_gap: int = Field(default=20, ge=0)
    subpixel_win_size: int = Field(default=11, ge=3)
    max_reprojection_error_px: float = Field(default=2.5, ge=0.1)


class OrientationConfig(BaseModel):
    """Configuration for board perspective orientation resolution."""
    color_check_enabled: bool = Field(default=True)
    mass_distribution_enabled: bool = Field(default=True)
    allow_sideways: bool = Field(default=True, description="Allow 90°/270° sideways board orientations")
    fallback_orientation: str = Field(default="unknown")


class HandDetectorConfig(BaseModel):
    """Configuration for hand detection and occlusion polygon generation."""
    enabled: bool = Field(default=True, description="Enable MediaPipe hand detector")
    min_detection_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    dilation_margin_px: int = Field(default=18, ge=0, le=100, description="Dilation margin for hand shadow")
    occlusion_threshold: float = Field(default=0.15, ge=0.01, le=1.0, description="Cell overlap ratio to flag occluded")


class LabellerConfig(BaseModel):
    """Configuration for automated dataset collection and labelling (AutoLabeller)."""
    output_dir: str = Field(default="data/dataset", description="Root directory to store extracted crops")
    target_crop_size: Tuple[int, int] = Field(default=(112, 112), description="Output crop resolution (W, H)")
    max_samples_per_move: int = Field(default=64, ge=1, le=64, description="Maximum squares saved per move")
    save_format: str = Field(default="png", description="Image save format: 'png' or 'jpg'")


class ClassifierConfig(BaseModel):
    """Configuration for neural piece classifier runtime inference."""
    enabled: bool = Field(default=True, description="Enable neural piece classification")
    model_path: str = Field(
        default="models/piece_classifier.onnx",
        description="Path to ONNX weights model file",
    )
    chessq_model_path: str = Field(
        default="models/chessq_vit_int8.onnx",
        description="Path to holistic ViT whole-board ChessQ-Lite ONNX weights",
    )
    dinov2_model_path: str = Field(
        default="models/dinov2_small_int8.onnx",
        description="Path to DINOv2-Small INT8 ONNX model for semantic per-square change verification",
    )
    use_chessq_whole_board: bool = Field(
        default=True,
        description="Use whole-board ViT classifier (ChessQ-Lite) when weights are present",
    )
    use_dinov2_change_detector: bool = Field(
        default=True,
        description="Use DINOv2-Small patch cosine change map to disambiguate same-color translucent pieces",
    )
    architecture: str = Field(
        default="chesscog",
        description="Backbone architecture: 'chesscog', 'efficientnet_b0', 'resnet18', or 'mock'",
    )
    backend: str = Field(
        default="auto",
        description="Inference engine backend: 'onnx', 'torch', 'mock', or 'auto'",
    )
    batch_size: int = Field(default=64, ge=1, le=64, description="Inference batch size")
    input_size: Tuple[int, int] = Field(default=(112, 112), description="Square crop input size (W, H)")
    occupancy_threshold: float = Field(
        default=0.50, ge=0.01, le=0.99,
        description="Occupancy probability threshold tau_o",
    )
    confidence_threshold: float = Field(
        default=0.60, ge=0.01, le=0.99,
        description="Piece type confidence threshold",
    )
    device: str = Field(default="cpu", description="Execution device: 'cpu' or 'cuda'")


class TrainingConfig(BaseModel):
    """Configuration for model training loop, dataset splits and MLOps logging."""
    dataset_dir: str = Field(default="data/dataset", description="Root dataset directory")
    output_dir: str = Field(default="models/checkpoints", description="Model checkpoint export directory")
    architecture: str = Field(default="efficientnet_b0", description="Target backbone: 'efficientnet_b0' or 'resnet18'")
    batch_size: int = Field(default=32, ge=1, le=256, description="Training batch size")
    epochs: int = Field(default=30, ge=1, le=200, description="Maximum training epochs")
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1e-1, description="Initial learning rate")
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1e-2, description="L2 weight decay regularization")
    patience: int = Field(default=5, ge=1, le=20, description="Early stopping patience epochs")
    val_split_ratio: float = Field(default=0.15, ge=0.05, le=0.30, description="Validation split ratio")
    test_split_ratio: float = Field(default=0.15, ge=0.05, le=0.30, description="Test split ratio for unseen sets")
    seed: int = Field(default=42, description="Random generator seed for reproducibility")
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking server URI")
    experiment_name: str = Field(default="chess_piece_classification", description="MLflow experiment name")


class VisionConfig(BaseModel):
    """Configuration for visual perception, board normalization and piece classification."""
    target_board_size: int = Field(
        default=640,
        ge=256,
        le=1920,
        description="Target normalized board resolution in pixels (NxN, default 640x640)",
    )
    min_camera_angle: float = Field(
        default=50.0,
        ge=30.0,
        le=90.0,
        description="Minimum permitted camera viewing angle in degrees relative to board plane",
    )
    parallax_height_alpha: float = Field(
        default=1.25,
        ge=1.0,
        le=2.0,
        description="Cell ROI vertical expansion factor alpha to compensate piece height parallax",
    )
    min_board_area_ratio: float = Field(
        default=0.15,
        ge=0.01,
        le=0.90,
        description="Minimum permitted board polygon area as fraction of frame area",
    )
    localizer: LocalizerConfig = Field(default_factory=LocalizerConfig)
    orientation: OrientationConfig = Field(default_factory=OrientationConfig)
    hand_detector: HandDetectorConfig = Field(default_factory=HandDetectorConfig)
    yolo_board_confidence: float = Field(
        default=0.50,
        ge=0.1,
        le=1.0,
        description="Minimum detection confidence for YOLO board corners detector",
    )
    classifier_batch_size: int = Field(
        default=64,
        ge=1,
        le=128,
        description="Batch size for parallel piece classification (default 64 cells)",
    )
    threshold_occupied: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="tau_o: Occupancy probability threshold for piece presence",
    )
    threshold_color: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="tau_c: Piece color classification confidence threshold",
    )
    threshold_piece: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="tau_t: Piece type classification confidence threshold",
    )


class TemporalConfig(BaseModel):
    """Configuration for temporal state machine, debounce window and motion detection."""
    settling_time_sec: float = Field(
        default=0.8,
        gt=0.0,
        le=5.0,
        description="Delta t_settle: Debouncing settling window in seconds after motion stops",
    )
    motion_diff_threshold: float = Field(
        default=0.04,
        gt=0.0,
        le=1.0,
        description="Differential optical change threshold to enter MOVING state",
    )
    verifying_stable_frames: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Number of consecutive stable frames required to confirm VERIFYING state",
    )
    pixel_diff_threshold: int = Field(
        default=25,
        ge=1,
        le=255,
        description="Sensitivity threshold tau_diff for pixel luminance delta (0..255)",
    )
    motion_ema_alpha: float = Field(
        default=0.35,
        ge=0.01,
        le=1.0,
        description="Exponential moving average alpha factor for temporal motion smoothing",
    )
    recovery_grace_sec: float = Field(
        default=2.0,
        ge=0.0,
        le=15.0,
        description="Grace window before escalating large disturbances (knocked pieces) to UNCERTAIN UI",
    )
    auto_takeback_enabled: bool = Field(
        default=True,
        description="Allow automatic rollback when last committed move is physically taken back",
    )
    drift_check_interval_frames: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Frame interval between lightweight STABLE board drift checks",
    )
    hand_idle_skip_stride: int = Field(
        default=3,
        ge=1,
        le=15,
        description="Run hand detector every N frames when board is STABLE with minimal motion",
    )



class DecisionConfig(BaseModel):
    """Weights and thresholds for multi-criteria candidate move scoring S(m).
    
    Formula: S(m) = w_v * S_v(m) + w_t * S_t(m) + w_g * S_g(m) - w_o * P_o(m)
    Constraint: w_v + w_t + w_g + w_o == 1.0 (normalized weights)
    """
    weight_visual: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="w_v: Weight of visual piece classification likelihood",
    )
    weight_temporal: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="w_t: Weight of temporal stability across settling frames",
    )
    weight_geometry: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="w_g: Weight of board quadrilateral homography stability",
    )
    weight_occlusion: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="w_o: Weight of occlusion penalty P_o",
    )
    min_confidence_margin: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Delta: Minimum score margin between top-1 and top-2 candidate moves",
    )
    min_acceptance_score: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="tau_accept: Minimum absolute composite score required for candidate acceptance",
    )
    max_occlusion_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="tau_occ_crit: Maximum permitted occlusion ratio on affected squares",
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> DecisionConfig:
        total = self.weight_visual + self.weight_temporal + self.weight_geometry + self.weight_occlusion
        if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(
                f"Sum of decision scoring weights (w_v + w_t + w_g + w_o) must equal 1.0, "
                f"got {total:.4f} (v={self.weight_visual}, t={self.weight_temporal}, "
                f"g={self.weight_geometry}, o={self.weight_occlusion})"
            )
        return self


class AnalysisConfig(BaseModel):
    """Configuration for Stockfish engine integration and blunder detection."""
    stockfish_enabled: bool = Field(
        default=True,
        description="Whether to run background position evaluation with Stockfish",
    )
    stockfish_depth: int = Field(
        default=15,
        ge=1,
        le=30,
        description="Search depth for Stockfish evaluation",
    )
    stockfish_path: Optional[str] = Field(
        default=None,
        description="Explicit filesystem path to Stockfish binary; searches PATH if None",
    )
    blunder_threshold_cp: int = Field(
        default=150,
        ge=50,
        le=1000,
        description="Evaluation drop in centipawns classified as a blunder (e.g. 150 = 1.5 pawns)",
    )
    timeout_sec: float = Field(
        default=2.5,
        gt=0.1,
        le=30.0,
        description="Maximum allowed analysis calculation time per position in seconds",
    )
    max_queue_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum bounded capacity of the background analysis queue",
    )
    threads: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of CPU threads allocated to Stockfish",
    )
    hash_mb: int = Field(
        default=32,
        ge=8,
        le=1024,
        description="Size of Stockfish transposition table memory in MB",
    )


class PersistenceConfig(BaseModel):
    """Configuration for event journal persistence and evidence artifact retention."""
    base_dir: str = Field(
        default="data/sessions",
        description="Root directory for saving game session journals and evidence",
    )
    enable_checksum: bool = Field(
        default=True,
        description="Whether to maintain cryptographic SHA-256 integrity hash chaining",
    )
    evidence_format: str = Field(
        default="jpg",
        description="Format for saving board evidence frames: 'jpg' or 'png'",
    )
    evidence_retention: str = Field(
        default="selective",
        description="Evidence retention policy: 'all', 'selective', or 'none'",
    )
    save_full_frame_on_uncertain: bool = Field(
        default=True,
        description="Whether to preserve uncompressed full frames for UNCERTAIN moves",
    )
    sync_to_disk: bool = Field(
        default=True,
        description="Whether to perform os.fsync after every atomic move event write",
    )


class CameraConfig(BaseModel):
    """Configuration for physical or synthetic video source."""
    source_type: str = Field(
        default="synthetic",
        description="Source type: 'synthetic', 'file', 'device', or 'rtsp'",
    )
    source_path: str = Field(
        default="",
        description="Path to video file, device index as string ('0'), RTSP URL, or PGN file for synthetic",
    )
    fps: float = Field(
        default=30.0,
        gt=0.0,
        le=120.0,
        description="Nominal video frame rate in frames per second",
    )
    width: int = Field(
        default=1920,
        gt=0,
        description="Capture resolution width in pixels",
    )
    height: int = Field(
        default=1080,
        gt=0,
        description="Capture resolution height in pixels",
    )
    rotate: int = Field(
        default=0,
        description="Rotation in degrees (0, 90, 180, 270)",
    )

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        valid = {"synthetic", "file", "device", "rtsp"}
        if v.lower() not in valid:
            raise ValueError(f"source_type must be one of {valid}, got '{v}'")
        return v.lower()

    @field_validator("rotate")
    @classmethod
    def validate_rotation(cls, v: int) -> int:
        if v not in (0, 90, 180, 270):
            raise ValueError(f"Rotation must be one of 0, 90, 180, 270 degrees, got {v}")
        return v


class UIConfig(BaseModel):
    """Configuration for operator visual dashboard and GUI display."""
    headless: bool = Field(
        default=False,
        description="Whether to run in headless mode without X11/GUI display",
    )
    window_title: str = Field(
        default="Chess Recording System HUD",
        description="Title of the HighGUI display window",
    )
    display_width: int = Field(
        default=1280,
        gt=320,
        le=3840,
        description="Display window width in pixels",
    )
    display_height: int = Field(
        default=720,
        gt=240,
        le=2160,
        description="Display window height in pixels",
    )
    show_virtual_board: bool = Field(
        default=True,
        description="Whether to render virtual 2D board overlay",
    )
    show_move_list: bool = Field(
        default=True,
        description="Whether to display recent committed moves list",
    )
    show_eval_bar: bool = Field(
        default=True,
        description="Whether to display Stockfish evaluation indicator",
    )
    show_hand_landmarks: bool = Field(
        default=True,
        description="Whether to draw hand landmarks and occlusion masks",
    )
    font_scale: float = Field(
        default=0.6,
        ge=0.2,
        le=2.0,
        description="Font scale factor for HUD overlay text",
    )


class WebUIConfig(BaseModel):
    """Configuration for web-based operator dashboard."""
    enabled: bool = Field(default=False, description="Whether web UI server is enabled")
    host: str = Field(default="0.0.0.0", description="Host address to bind Web UI")
    port: int = Field(default=8000, ge=1024, le=65535, description="HTTP/WebSocket port")
    stream_fps: int = Field(default=25, ge=1, le=60, description="Target streaming frame rate for MJPEG")
    jpeg_quality: int = Field(default=80, ge=10, le=100, description="JPEG encoding compression quality (0-100)")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origin domains")


class AppConfig(BaseModel):
    """Top-level unified system configuration."""
    vision: VisionConfig = Field(default_factory=VisionConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    web_ui: WebUIConfig = Field(default_factory=WebUIConfig)
    labeller: LabellerConfig = Field(default_factory=LabellerConfig)
    piece_classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)



def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges source dict into target dict."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Loads configuration from YAML file and applies optional programmatic overrides.
    
    Args:
        config_path: Optional path to YAML configuration file. If not provided or file
                     does not exist, built-in defaults are used.
        overrides: Optional dictionary of section/parameter overrides.
        
    Returns:
        Validated AppConfig instance.
        
    Raises:
        ValueError: If configuration values fail schema or weight sum validation.
        FileNotFoundError: If an explicitly provided config_path does not exist.
    """
    raw_config: Dict[str, Any] = {}

    if config_path is not None:
        p = Path(config_path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p.resolve()}")
        with open(p, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw_config = loaded

    if overrides:
        _deep_merge(raw_config, overrides)

    # Allow environment variable override for stockfish path
    sf_env = os.environ.get("STOCKFISH_PATH")
    if sf_env:
        analysis_cfg = raw_config.get("analysis")
        if not isinstance(analysis_cfg, dict):
            analysis_cfg = {}
        analysis_cfg["stockfish_path"] = sf_env
        raw_config["analysis"] = analysis_cfg

    return AppConfig.model_validate(raw_config)


def save_config(config: AppConfig, path: str | Path) -> None:
    """Saves current configuration to a YAML file.
    
    Args:
        config: AppConfig instance to serialize.
        path: Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
