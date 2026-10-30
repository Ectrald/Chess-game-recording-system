"""Domain exceptions for the Chess Game Recording System.

Defines typed errors mapped directly to system failure modes from
Explanatory Note, Table 2.3 ('Обработка отказов и неоднозначных состояний').
"""

from typing import Optional


class ChessRecordingError(Exception):
    """Base exception for all domain-specific errors in the system."""
    pass


class VideoSourceError(ChessRecordingError):
    """Raised when the video source fails to open, disconnects or encounters EOF."""
    pass


class BoardNotFoundError(ChessRecordingError):
    """Raised when the board detector fails to locate a chess board in the frame."""
    pass


class BoardGeometryError(ChessRecordingError):
    """Raised when the localized board fails quadrilateral convexity, area or aspect checks."""
    pass


class BoardLostError(ChessRecordingError):
    """Raised when board tracking is lost during a recording session."""
    pass


class InvalidObservationError(ChessRecordingError):
    """Raised when an observation does not conform to data integrity rules (e.g. != 64 cells)."""
    pass


class AmbiguousMoveError(ChessRecordingError):
    """Raised when the difference S(m1) - S(m2) is below Delta threshold, requiring human verification."""

    def __init__(self, message: str, top_candidates: Optional[list] = None):
        super().__init__(message)
        self.top_candidates = top_candidates or []


class IllegalMoveTransitionError(ChessRecordingError):
    """Raised when no legal chess move explains the observed state transition."""
    pass
