"""Candidate move generation constrained by authoritative FIDE chess rules.

Restricts move evaluation to legal moves on the current chess board,
mapping each candidate move to its topological affected square signature:
- Regular move (quiet or capture): 2 squares (from, to)
- Castling: 4 squares (king from/to, rook from/to)
- En passant: 3 squares (pawn from/to, captured pawn square)
- Pawn promotion: 2 squares with 4 piece choices (q, r, b, n)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
import chess

from src.core.datatypes import (
    CHESS_SQUARES,
    BoardObservation,
    CellObservation,
)

logger = logging.getLogger(__name__)


def board_square_to_piece_class(board: chess.Board, square_idx: int) -> str:
    """Converts a board square occupant to 13-class canonical piece string.

    Returns:
        'empty' or e.g. 'wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', etc.
    """
    piece = board.piece_at(square_idx)
    if piece is None:
        return "empty"
    prefix = "w" if piece.color == chess.WHITE else "b"
    return f"{prefix}{piece.symbol().upper()}"


class LegalMoveCandidateGenerator:
    """Generates and prioritizes legal candidate moves against visual board changes."""

    def __init__(self) -> None:
        pass

    def detect_changed_squares(
        self,
        board: chess.Board,
        observation: BoardObservation,
    ) -> List[str]:
        """Identifies squares whose visual observation differs from authoritative state.

        Squares marked as occluded by hands/fingers are excluded to prevent false positives.

        Args:
            board: Authoritative chess.Board state prior to move.
            observation: Settled BoardObservation containing 64 cell predictions.

        Returns:
            Sorted list of square names (e.g. ['e2', 'e4']) showing visual divergence.
        """
        changed: List[str] = []
        for sq_name in CHESS_SQUARES:
            sq_idx = chess.parse_square(sq_name)
            expected_class = board_square_to_piece_class(board, sq_idx)

            cell_obs: Optional[CellObservation] = observation.cells.get(sq_name)
            if cell_obs is None:
                continue

            # Occluded squares cannot be reliably assessed for change
            if cell_obs.is_occluded:
                continue

            observed_class = cell_obs.predicted_piece
            if observed_class != expected_class:
                changed.append(sq_name)

        return sorted(changed)

    def get_affected_squares(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> List[str]:
        """Calculates topological affected squares signature A(m) for a legal move.

        Signatures:
            - Regular move: [from_square, to_square] (2 squares)
            - Castling (White O-O): ['e1', 'g1', 'f1', 'h1'] (4 squares)
            - Castling (White O-O-O): ['e1', 'c1', 'd1', 'a1'] (4 squares)
            - Castling (Black O-O): ['e8', 'g8', 'f8', 'h8'] (4 squares)
            - Castling (Black O-O-O): ['e8', 'c8', 'd8', 'a8'] (4 squares)
            - En passant: [from_square, to_square, captured_square] (3 squares)
            - Promotion: [from_square, to_square] (2 squares)

        Args:
            board: Authoritative board state on which move is evaluated.
            move: chess.Move candidate.

        Returns:
            List of square names affected by the execution of move m.
        """
        from_sq_name = chess.square_name(move.from_square)
        to_sq_name = chess.square_name(move.to_square)

        # Castling signature: 4 squares
        if board.is_castling(move):
            if board.is_kingside_castling(move):
                if board.turn == chess.WHITE:
                    return ["e1", "g1", "f1", "h1"]
                else:
                    return ["e8", "g8", "f8", "h8"]
            elif board.is_queenside_castling(move):
                if board.turn == chess.WHITE:
                    return ["e1", "c1", "d1", "a1"]
                else:
                    return ["e8", "c8", "d8", "a8"]

        # En passant signature: 3 squares
        if board.is_en_passant(move):
            # Captured pawn is on destination square's file and source square's rank
            captured_sq = chess.square(
                chess.square_file(move.to_square),
                chess.square_rank(move.from_square),
            )
            return [from_sq_name, to_sq_name, chess.square_name(captured_sq)]

        # Regular move & promotion signature: 2 squares
        return [from_sq_name, to_sq_name]

    def generate_candidates(
        self,
        board: chess.Board,
        observation: Optional[BoardObservation] = None,
    ) -> List[Tuple[chess.Move, List[str]]]:
        """Generates all legal candidate moves with their affected squares signature.

        If observation is provided, candidates whose affected squares overlap with
        detected visual changes are prioritized first.

        Args:
            board: Authoritative board state.
            observation: Optional current BoardObservation for change filtering.

        Returns:
            List of tuples (candidate_move, affected_squares), ordered by relevance.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        candidates = [
            (move, self.get_affected_squares(board, move))
            for move in legal_moves
        ]

        if observation is not None:
            changed_squares = set(self.detect_changed_squares(board, observation))
            if changed_squares:
                # Rank candidates with higher overlap with changed_squares first
                candidates.sort(
                    key=lambda item: (
                        len(set(item[1]).intersection(changed_squares)),
                        -len(item[1]),  # tie-breaker
                    ),
                    reverse=True,
                )

        return candidates

