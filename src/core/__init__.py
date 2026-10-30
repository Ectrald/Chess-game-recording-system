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

__all__ = [
    "ChessRecordingError",
    "BoardNotFoundError",
    "BoardGeometryError",
    "BoardLostError",
    "AmbiguousMoveError",
    "InvalidObservationError",
    "VideoSourceError",
]
