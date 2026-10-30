"""Configuration models and loader for the Chess Game Recording System.

Defines Pydantic v2 configuration schemas reflecting the mathematical
parameters of the Explanatory Note (Sections 2.4.1, 2.4.2, 2.4.4, 2.4.5, 2.4.8).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class VisionConfig(BaseModel):
    """Configuration for visual perception, board normalization and piece classification."""
    target_board_size: int = Field(
        default=640,
        ge=256,
        le=1920,
        description="Target normalized board resolution in pixels (NxN, default 640x640)",
    )
    min_camera_angle: float = Field(
        default=60.0,
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
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Differential optical change threshold to enter MOVING state",
    )
    verifying_stable_frames: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Number of consecutive stable frames required to confirm VERIFYING state",
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


class AppConfig(BaseModel):
    """Top-level unified system configuration."""
    vision: VisionConfig = Field(default_factory=VisionConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)


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
        raw_config.setdefault("analysis", {})["stockfish_path"] = sf_env

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
