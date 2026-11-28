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

    def find_two_ply_sequence(
        self,
        board: chess.Board,
        changed_squares: Set[str],
        square_deltas: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[chess.Move, chess.Move]]:
        """Bounded 2-ply move path search when two rapid moves occur inside one debounce window.

        Triggered for 3..8 changed squares when no single legal move explains them cleanly.
        """
        if not (3 <= len(changed_squares) <= 8):
            return []

        deltas = square_deltas or {}
        scored_pairs: List[Tuple[float, chess.Move, chess.Move]] = []
        for m1 in board.legal_moves:
            f1 = chess.square_name(m1.from_square)
            t1 = chess.square_name(m1.to_square)
            if f1 not in changed_squares and t1 not in changed_squares:
                continue
            aff1 = set(self.get_affected_squares(board, m1))
            if not aff1.issubset(changed_squares):
                continue

            b2 = board.copy()
            b2.push(m1)
            for m2 in b2.legal_moves:
                f2 = chess.square_name(m2.from_square)
                t2 = chess.square_name(m2.to_square)
                if f2 not in changed_squares and t2 not in changed_squares:
                    continue
                aff2 = set(self.get_affected_squares(b2, m2))
                # Captures on the same target square (aff1 & aff2) overlap on 1 square -> union has 3..4 squares
                sym_diff = aff1.symmetric_difference(aff2)
                union = aff1.union(aff2)
                if union == changed_squares or sym_diff == changed_squares:
                    exact_bonus = 10.0 if union == changed_squares else 5.0
                    delta_sum = sum(deltas.get(sq, 0.0) for sq in union)
                    scored_pairs.append((exact_bonus + delta_sum, m1, m2))
                elif union.issubset(changed_squares) and len(changed_squares - union) <= 4:
                    residual = sum(deltas.get(sq, 0.0) for sq in (changed_squares - union))
                    delta_sum = sum(deltas.get(sq, 0.0) for sq in union)
                    if delta_sum > residual * 1.5:
                        scored_pairs.append((delta_sum - 0.5 * residual, m1, m2))

        scored_pairs.sort(key=lambda it: it[0], reverse=True)
        return [(m1, m2) for _, m1, m2 in scored_pairs]

