"""Strict data contracts for the Chess Game Recording System.

Defines Pydantic v2 data models mapped directly to:
- Explanatory Note, Section 2.4.8, Table 2.2 ('Основные структуры данных')
- ROADMAP.md, Section 1 ('Контракты данных')
- STAGE_0.md, Section 3
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Canonical 64 chess squares from a1 to h8
CHESS_SQUARES: List[str] = [
    f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh"
]

# Canonical 13 piece classes for classifier
PIECE_CLASSES: List[str] = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]


class Frame(BaseModel):
    """Raw video frame with metadata and numpy image array.
    
    Mapped to Table 2.2 'Frame': frame_id, timestamp, image.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: int = Field(ge=0, description="Monotonically increasing frame index")
    timestamp: float = Field(ge=0.0, description="Time in seconds from session start")
    image: np.ndarray = Field(description="Raw image data (BGR/RGB ndarray)")
    width: int = Field(gt=0, description="Frame width in pixels")
    height: int = Field(gt=0, description="Frame height in pixels")

    @field_validator("image")
    @classmethod
    def validate_image_array(cls, v: Any) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError("Frame image must be a numpy.ndarray")
        if v.ndim not in (2, 3):
            raise ValueError(f"Image ndim must be 2 or 3, got {v.ndim}")
        return v

    @model_validator(mode="before")
    @classmethod
    def auto_dimensions(cls, values: Any) -> Any:
        if isinstance(values, dict):
            image = values.get("image")
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
                values.setdefault("width", w)
                values.setdefault("height", h)
        return values

    @property
    def channels(self) -> int:
        return self.image.shape[2] if self.image.ndim == 3 else 1

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.image.shape


class BoardGeometry(BaseModel):
    """Geometric representation of the detected chess board in the frame.
    
    Mapped to Table 2.2 'BoardGeometry': corners[4], homography matrix, confidence.
    """
    corners: List[Tuple[float, float]] = Field(
        min_length=4,
        max_length=4,
        description="4 corners [(x,y)] ordered as Top-Left, Top-Right, Bottom-Right, Bottom-Left",
    )
    homography_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="3x3 projective transformation matrix to normalized 640x640 space",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Geometric quality measure S_g (convexity, parallelism, area)",
    )
    is_valid: bool = Field(
        description="True if geometry passes non-crossing and positive area constraints",
    )
    orientation_calibrated: bool = Field(
        default=False,
        description="True if a1-h8 orientation has been verified against FIDE rules",
    )

    @field_validator("homography_matrix")
    @classmethod
    def validate_homography_shape(cls, v: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if v is not None:
            if len(v) != 3 or any(len(row) != 3 for row in v):
                raise ValueError("Homography matrix must be exactly 3x3")
        return v

    def to_numpy_corners(self) -> np.ndarray:
        """Returns corners as float32 np.ndarray of shape (4, 2)."""
        return np.array(self.corners, dtype=np.float32)

    def to_numpy_homography(self) -> Optional[np.ndarray]:
        """Returns 3x3 homography matrix as float64 np.ndarray."""
        if self.homography_matrix is None:
            return None
        return np.array(self.homography_matrix, dtype=np.float64)


class CellObservation(BaseModel):
    """Visual observation of a single chess board cell.
    
    Mapped to Table 2.2 'CellObservation': square, probabilities, predicted, confidence, occluded.
    """
    square: str = Field(
        pattern=r"^[a-h][1-8]$",
        description="Cell coordinate in standard algebraic notation (e.g. 'e4')",
    )
    probabilities: Dict[str, float] = Field(
        description="Probability distribution over the 13 piece classes",
    )
    predicted_piece: Optional[str] = Field(
        default=None,
        description="Most probable piece class or 'empty'",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of predicted piece class",
    )
    is_occupied: bool = Field(
        description="Binary state: True if square contains a piece",
    )
    is_occluded: bool = Field(
        default=False,
        description="True if square is overlapped by MediaPipe hand mask",
    )

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, p in v.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability for {k} must be in [0, 1], got {p}")
        return v

    @model_validator(mode="before")
    @classmethod
    def set_prediction_if_missing(cls, values: Any) -> Any:
        if isinstance(values, dict):
            probs = values.get("probabilities", {})
            if probs and ("predicted_piece" not in values or values["predicted_piece"] is None):
                best_cls = max(probs, key=probs.get)
                values["predicted_piece"] = best_cls
                values.setdefault("confidence", probs[best_cls])
                values.setdefault("is_occupied", best_cls != "empty")
        return values


class BoardObservation(BaseModel):
    """Observation of the full 8x8 chess board at a specific timestamp.
    
    Mapped to Table 2.2 'BoardObservation': cells[64], motion, occlusion, geometry confidence.
    """
    timestamp: float = Field(ge=0.0, description="Timestamp of observation in seconds")
    frame_id: int = Field(ge=0, description="Source frame index")
    cells: Dict[str, CellObservation] = Field(
        description="Mapping of all 64 squares ('a1'..'h8') to their observations",
    )
    motion_detected: bool = Field(
        description="True if differential optical motion is detected above threshold",
    )
    motion_level: float = Field(
        ge=0.0,
        description="Normalized pixel change measure across stabilized board",
    )
    hand_present: bool = Field(
        description="True if MediaPipe Hand Landmarker detected a hand in the board ROI",
    )
    geometry_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Current geometry stability score S_g",
    )

    @field_validator("cells")
    @classmethod
    def validate_cells_count(cls, v: Dict[str, CellObservation]) -> Dict[str, CellObservation]:
        missing = set(CHESS_SQUARES) - set(v.keys())
        extra = set(v.keys()) - set(CHESS_SQUARES)
        if missing or extra:
            raise ValueError(
                f"BoardObservation must contain exactly 64 cells. "
                f"Missing: {missing}, Extra: {extra}"
            )
        return v

    def get_occupied_squares(self) -> Set[str]:
        """Returns set of square names currently occupied by pieces."""
        return {sq for sq, cell in self.cells.items() if cell.is_occupied}

    def get_piece_map(self) -> Dict[str, str]:
        """Returns mapping of occupied squares to piece symbols (e.g. {'e4': 'wP'})."""
        return {
            sq: cell.predicted_piece
            for sq, cell in self.cells.items()
            if cell.is_occupied and cell.predicted_piece and cell.predicted_piece != "empty"
        }

    def diff_squares(self, other: BoardObservation, threshold: float = 0.5) -> List[str]:
        """Returns list of square coordinates where occupancy or piece type differed."""
        changed = []
        for sq in CHESS_SQUARES:
            c1 = self.cells[sq]
            c2 = other.cells[sq]
            if c1.is_occupied != c2.is_occupied or c1.predicted_piece != c2.predicted_piece:
                changed.append(sq)
        return changed


class MoveHypothesis(BaseModel):
    """Candidate move evaluated against visual and temporal evidence.
    
    Mapped to Table 2.2 'MoveHypothesis': move, changed_squares, visual/temporal scores, total score.
    Formula: S(m) = w_v * S_v(m) + w_t * S_t(m) + w_g * S_g(m) - w_o * P_o(m)
    """
    move_uci: str = Field(description="Move representation in UCI notation, e.g. 'e2e4'")
    move_san: Optional[str] = Field(default=None, description="Standard Algebraic Notation, e.g. 'e4'")
    changed_squares: List[str] = Field(description="Squares affected by candidate move (delta S)")
    score_visual: float = Field(description="Log-likelihood visual consistency S_v(m)")
    score_temporal: float = Field(ge=0.0, le=1.0, description="Temporal consistency S_t(m)")
    score_geometry: float = Field(ge=0.0, le=1.0, description="Geometry quality score S_g(m)")
    penalty_occlusion: float = Field(ge=0.0, description="Occlusion penalty P_o(m)")
    total_score: float = Field(description="Aggregated score S(m)")


class MoveEvent(BaseModel):
    """Confirmed move event committed to the official game state.
    
    Mapped to Table 2.2 'MoveEvent': ply, FEN before/after, SAN/UCI, confidence, evidence frames.
    """
    ply: int = Field(ge=1, description="Half-move index (1 = White move 1, 2 = Black move 1, ...)")
    fen_before: str = Field(description="FEN position string prior to move")
    move_uci: str = Field(pattern=r"^[a-h][1-8][a-h][1-8][qrbn]?$", description="UCI move (e.g. 'e7e8q')")
    move_san: str = Field(description="SAN move string (e.g. 'e8=Q+', 'O-O')")
    fen_after: str = Field(description="FEN position string after move execution")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence margin Delta: S(m1) - S(m2)")
    timestamp: float = Field(ge=0.0, description="Session timestamp of move confirmation")
    evidence_frame_id: int = Field(ge=0, description="Reference frame_id documenting move completion")
    eval_centipawns: Optional[int] = Field(default=None, description="Stockfish evaluation in centipawns")
    is_blunder: Optional[bool] = Field(default=None, description="True if move significantly reduced eval")

    @field_validator("fen_before", "fen_after")
    @classmethod
    def validate_fen_structure(cls, v: str) -> str:
        parts = v.strip().split()
        if len(parts) != 6:
            raise ValueError(f"FEN must have 6 fields, got {len(parts)}: '{v}'")
        ranks = parts[0].split("/")
        if len(ranks) != 8:
            raise ValueError(f"FEN board must have 8 ranks, got {len(ranks)}: '{parts[0]}'")
        return v
