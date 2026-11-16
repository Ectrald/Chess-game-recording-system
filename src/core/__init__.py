"""Core data contracts, configurations, benchmark and FSM definitions."""

from src.core.exceptions import (
    ChessRecordingError,
    BoardNotFoundError,
    BoardGeometryError,
    BoardLostError,
    AmbiguousMoveError,
    InvalidObservationError,
    VideoSourceError,
)
from src.core.datatypes import (
    AutoLabellerSample,
    BoardGeometry,
    BoardObservation,
    BoardOrientation,
    CellObservation,
    CellROI,
    Frame,
    FSMState,
    HandDetection,
    HandLandmark,
    MotionMetrics,
    MoveEvent,
    MoveHypothesis,
    Point2D,
    Quadrilateral,
    CHESS_SQUARES,
    PIECE_CLASSES,
)

__all__ = [
    "ChessRecordingError",
    "BoardNotFoundError",
    "BoardGeometryError",
    "BoardLostError",
    "AmbiguousMoveError",
    "InvalidObservationError",
    "VideoSourceError",
    "AutoLabellerSample",
    "BoardGeometry",
    "BoardObservation",
    "BoardOrientation",
    "CellObservation",
    "CellROI",
    "Frame",
    "FSMState",
    "HandDetection",
    "HandLandmark",
    "MotionMetrics",
    "MoveEvent",
    "MoveHypothesis",
    "Point2D",
    "Quadrilateral",
    "CHESS_SQUARES",
    "PIECE_CLASSES",
]


