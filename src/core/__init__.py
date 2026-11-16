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
    BoardGeometry,
    BoardObservation,
    BoardOrientation,
    CellObservation,
    CellROI,
    Frame,
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
    "BoardGeometry",
    "BoardObservation",
    "BoardOrientation",
    "CellObservation",
    "CellROI",
    "Frame",
    "MoveEvent",
    "MoveHypothesis",
    "Point2D",
    "Quadrilateral",
    "CHESS_SQUARES",
    "PIECE_CLASSES",
]

