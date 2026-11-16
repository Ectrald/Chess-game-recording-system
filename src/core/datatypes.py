"""Strict data contracts for the Chess Game Recording System.

Defines Pydantic v2 data models mapped directly to:
- Explanatory Note, Section 2.4.8, Table 2.2 ('Основные структуры данных')
- ROADMAP.md, Section 1 ('Контракты данных')
- STAGE_0.md, Section 3
"""

from __future__ import annotations

from enum import Enum
import math
import re
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Set, Tuple
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


class BoardOrientation(str, Enum):
    """Board perspective orientation relative to player camps."""
    UNKNOWN = "unknown"
    WHITE_BOTTOM = "white_bottom"  # White camp at bottom (ranks 1-2 at bottom, ranks 7-8 at top)
    BLACK_BOTTOM = "black_bottom"  # Black camp at bottom (ranks 7-8 at bottom, ranks 1-2 at top)


class Point2D(NamedTuple):
    """Immutable 2D point with subpixel float precision."""
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (float(self.x), float(self.y))


class Quadrilateral(NamedTuple):
    """4-vertex polygon representing the detected board boundary."""
    top_left: Point2D
    top_right: Point2D
    bottom_right: Point2D
    bottom_left: Point2D

    def as_list(self) -> List[Tuple[float, float]]:
        return [
            self.top_left.as_tuple(),
            self.top_right.as_tuple(),
            self.bottom_right.as_tuple(),
            self.bottom_left.as_tuple(),
        ]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.as_list(), dtype=np.float32)



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
    
    Mapped to Table 2.2 'BoardGeometry': corners[4], homography matrix, confidence,
    and STAGE_1.md Section 3.2.
    """
    corners: List[Tuple[float, float]] = Field(
        min_length=4,
        max_length=4,
        description="4 corners [(x,y)] ordered as Top-Left, Top-Right, Bottom-Right, Bottom-Left",
    )
    corners_image: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="4 subpixel corners [TL, TR, BR, BL] in source frame coordinate space",
    )
    homography_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="3x3 projective transformation matrix to normalized 640x640 space (image -> board)",
    )
    inverse_homography: Optional[List[List[float]]] = Field(
        default=None,
        description="Inverse 3x3 homography matrix (board -> image)",
    )
    target_size: int = Field(
        default=640,
        ge=64,
        description="Canonical edge resolution of normalized square board (default 640)",
    )
    orientation: BoardOrientation = Field(
        default=BoardOrientation.UNKNOWN,
        description="Resolved board orientation (white_bottom, black_bottom, unknown)",
    )
    detection_method: Literal["yolo_hough", "pure_hough", "manual", "aruco", "synthetic", "unknown"] = Field(
        default="unknown",
        description="Detection method used to obtain current geometry",
    )
    reprojection_error_px: float = Field(
        default=0.0,
        ge=0.0,
        description="Root mean square reprojection error of grid inliers (px)",
    )
    polygon_area_px2: float = Field(
        default=1.0,
        gt=0.0,
        description="Area of board quadrilateral in image frame space (px^2)",
    )
    aspect_ratio: float = Field(
        default=1.0,
        gt=0.0,
        description="Aspect ratio (average width / average height) of detected quadrilateral",
    )
    geometry_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Integral geometric reliability metric S_g in [0, 1]",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Geometric quality measure S_g (convexity, parallelism, area)",
    )
    is_valid: bool = Field(
        default=True,
        description="True if geometry passes non-crossing, convexity and positive area constraints",
    )
    orientation_calibrated: bool = Field(
        default=False,
        description="True if a1-h8 orientation has been verified against FIDE rules",
    )

    @field_validator("homography_matrix", "inverse_homography")
    @classmethod
    def validate_homography_shape(cls, v: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if v is not None:
            if len(v) != 3 or any(len(row) != 3 for row in v):
                raise ValueError("Homography matrix must be exactly 3x3")
        return v

    @model_validator(mode="before")
    @classmethod
    def sync_and_compute_geometry(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # Sync corners and corners_image
            if "corners" in values and "corners_image" not in values:
                values["corners_image"] = values["corners"]
            elif "corners_image" in values and "corners" not in values:
                values["corners"] = values["corners_image"]

            # Sync confidence and geometry_quality
            if "confidence" in values and "geometry_quality" not in values:
                values["geometry_quality"] = values["confidence"]
            elif "geometry_quality" in values and "confidence" not in values:
                values["confidence"] = values["geometry_quality"]

            corners = values.get("corners")
            if corners and len(corners) == 4:
                # Compute Shoelace polygon area
                area = 0.5 * abs(
                    sum(
                        corners[i][0] * corners[(i + 1) % 4][1]
                        - corners[(i + 1) % 4][0] * corners[i][1]
                        for i in range(4)
                    )
                )
                if "polygon_area_px2" not in values:
                    values["polygon_area_px2"] = max(float(area), 1e-4)

                # Compute approximate aspect ratio
                if "aspect_ratio" not in values:
                    w1 = math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
                    w2 = math.hypot(corners[2][0] - corners[3][0], corners[2][1] - corners[3][1])
                    h1 = math.hypot(corners[3][0] - corners[0][0], corners[3][1] - corners[0][1])
                    h2 = math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
                    avg_w = (w1 + w2) / 2.0
                    avg_h = max((h1 + h2) / 2.0, 1e-4)
                    values["aspect_ratio"] = max(float(avg_w / avg_h), 1e-4)
        return values

    @model_validator(mode="after")
    def validate_convexity_and_validity(self) -> BoardGeometry:
        if self.is_valid and self.corners and len(self.corners) == 4:
            pts = self.corners
            cross_products: List[float] = []
            for i in range(4):
                p0 = pts[i]
                p1 = pts[(i + 1) % 4]
                p2 = pts[(i + 2) % 4]
                dx1 = p1[0] - p0[0]
                dy1 = p1[1] - p0[1]
                dx2 = p2[0] - p1[0]
                dy2 = p2[1] - p1[1]
                cross = dx1 * dy2 - dy1 * dx2
                cross_products.append(cross)

            all_pos = all(cp > 1e-5 for cp in cross_products)
            all_neg = all(cp < -1e-5 for cp in cross_products)
            if not (all_pos or all_neg):
                raise ValueError(
                    f"Board corners must form a strictly convex non-self-intersecting quadrilateral. "
                    f"Cross products: {cross_products}"
                )
        return self

    def to_numpy_corners(self) -> np.ndarray:
        """Returns corners as float32 np.ndarray of shape (4, 2)."""
        return np.array(self.corners, dtype=np.float32)

    def to_numpy_homography(self) -> Optional[np.ndarray]:
        """Returns 3x3 homography matrix as float64 np.ndarray."""
        if self.homography_matrix is None:
            return None
        return np.array(self.homography_matrix, dtype=np.float64)

    def to_numpy_inverse_homography(self) -> Optional[np.ndarray]:
        """Returns inverse 3x3 homography matrix as float64 np.ndarray."""
        if self.inverse_homography is None:
            return None
        return np.array(self.inverse_homography, dtype=np.float64)

    def to_quadrilateral(self) -> Quadrilateral:
        """Converts corners to typed Quadrilateral model."""
        return Quadrilateral(
            top_left=Point2D(x=self.corners[0][0], y=self.corners[0][1]),
            top_right=Point2D(x=self.corners[1][0], y=self.corners[1][1]),
            bottom_right=Point2D(x=self.corners[2][0], y=self.corners[2][1]),
            bottom_left=Point2D(x=self.corners[3][0], y=self.corners[3][1]),
        )


class CellROI(BaseModel):
    """Region of interest for an individual chess square on the board."""
    square: str = Field(
        pattern=r"^[a-h][1-8]$",
        description="Cell coordinate in standard algebraic notation (e.g. 'e4')",
    )
    index: int = Field(
        ge=0,
        le=63,
        description="Square index 0..63 in canonical order (a1=0, b1=1, ..., h8=63)",
    )
    board_bbox: Tuple[int, int, int, int] = Field(
        description="Bounding box (x_min, y_min, x_max, y_max) in normalized 640x640 space",
    )
    extended_bbox: Tuple[int, int, int, int] = Field(
        description="Parallax-compensated bounding box with alpha_height expansion",
    )
    center_image: Tuple[float, float] = Field(
        description="Square center projected into source frame coordinates (px)",
    )
    center_board: Tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Square center in normalized 640x640 coordinates (px)",
    )


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
