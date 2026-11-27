"""Multi-criteria candidate move scoring and likelihood ranking.

Evaluates candidate moves against visual, temporal, geometric, and occlusion evidence:
Formula: S(m) = w_v * S_v(m) + w_t * S_t(m) + w_g * S_g(m) - w_o * P_o(m)
Reference:
- Explanatory Note, Section 2.4.4, Formula 2.1
- STAGE_4.md, Section 4.2
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple
import chess

from src.core.config import DecisionConfig
from src.core.datatypes import (
    CHESS_SQUARES,
    BoardObservation,
    CellObservation,
    MoveHypothesis,
)
from src.reasoning.candidate_generator import board_square_to_piece_class

logger = logging.getLogger(__name__)


class MoveScorer:
    """Computes composite likelihood scores S(m) for legal candidate moves."""

    def __init__(
        self,
        config: Optional[DecisionConfig] = None,
        alpha_affected: float = 0.75,
    ) -> None:
        """Initializes MoveScorer with weights and parameters.

        Args:
            config: DecisionConfig containing weights w_v, w_t, w_g, w_o and margin thresholds.
            alpha_affected: Weight between affected squares likelihood and global consistency.
        """
        self.config = config or DecisionConfig()
        self.alpha_affected = alpha_affected

    def _compute_visual_score(
        self,
        board: chess.Board,
        move: chess.Move,
        affected_squares: List[str],
        observation: BoardObservation,
    ) -> float:
        """Computes visual likelihood S_v(m) comparing hypothetical board to observation probabilities.

        Args:
            board: Authoritative board prior to move.
            move: Candidate move.
            affected_squares: Squares in A(m).
            observation: Settled observation with cell class probabilities.

        Returns:
            Visual likelihood score in [0.0, 1.0].
        """
        # Create hypothetical resulting board state
        board_hypo = board.copy()
        board_hypo.push(move)

        # 1. Affected squares likelihood S_affected
        affected_probs: List[float] = []
        for sq_name in affected_squares:
            cell_obs: Optional[CellObservation] = observation.cells.get(sq_name)
            if cell_obs is None:
                continue

            sq_idx = chess.parse_square(sq_name)
            expected_class = board_square_to_piece_class(board_hypo, sq_idx)

            if cell_obs.is_occluded:
                # Occluded cell is neutral (0.5) to avoid penalizing hidden squares
                affected_probs.append(0.5)
            else:
                prob = max(0.0, min(1.0, cell_obs.probabilities.get(expected_class, 0.0)))
                if expected_class != "empty" and move.promotion is None:
                    # ADR-020: Factorize (occupancy + color) from uncertain fine subtype in zero-shot / replacement figurines
                    prefix = expected_class[0]
                    within_color = [
                        max(0.0, float(p))
                        for cls, p in cell_obs.probabilities.items()
                        if cls.startswith(prefix)
                    ]
                    color_prob = sum(within_color)
                    max_within = max(within_color) if within_color else 0.0
                    if color_prob >= 0.50 and max_within <= 0.48 * max(color_prob, 1e-6):
                        prob = max(prob, min(1.0, 0.90 * color_prob))
                    elif color_prob >= 0.60:
                        prob = max(prob, min(1.0, 0.88 * color_prob))
                affected_probs.append(prob)

        if affected_probs:
            arith_mean = sum(affected_probs) / len(affected_probs)
            geom_prod = 1.0
            for p in affected_probs:
                geom_prod *= max(1e-3, p)
            geom_mean = geom_prod ** (1.0 / len(affected_probs))
            s_affected = 0.20 * arith_mean + 0.80 * geom_mean
        else:
            s_affected = 0.5

        # 2. Global consistency across non-affected squares S_global
        global_probs: List[float] = []
        affected_set = set(affected_squares)
        for sq_name in CHESS_SQUARES:
            if sq_name in affected_set:
                continue
            cell_obs = observation.cells.get(sq_name)
            if cell_obs is None or cell_obs.is_occluded:
                continue

            sq_idx = chess.parse_square(sq_name)
            expected_class = board_square_to_piece_class(board_hypo, sq_idx)
            prob = cell_obs.probabilities.get(expected_class, 0.0)
            global_probs.append(max(0.0, min(1.0, prob)))

        s_global = sum(global_probs) / len(global_probs) if global_probs else 1.0

        # Weighted blend between affected squares and global board
        base_s_v = self.alpha_affected * s_affected + (1.0 - self.alpha_affected) * s_global

        # Coverage penalty for leaving visibly changed squares unexplained
        changed_squares: List[str] = []
        for sq_name in CHESS_SQUARES:
            cell = observation.cells.get(sq_name)
            if cell is not None and not cell.is_occluded:
                sq_idx = chess.parse_square(sq_name)
                exp = board_square_to_piece_class(board, sq_idx)
                if cell.predicted_piece != exp:
                    changed_squares.append(sq_name)

        if changed_squares:
            unexplained = set(changed_squares) - set(affected_squares)
            coverage_factor = max(0.0, 1.0 - 0.75 * (len(unexplained) / len(changed_squares)))
        else:
            coverage_factor = 1.0

        s_v = base_s_v * coverage_factor
        return max(0.0, min(1.0, s_v))

    def _compute_temporal_score(
        self,
        affected_squares: List[str],
        observation: BoardObservation,
    ) -> float:
        """Computes temporal stability score S_t(m) = 1.0 - max(motion_level on A(m))."""
        max_motion = 0.0
        for sq_name in affected_squares:
            cell_obs = observation.cells.get(sq_name)
            if cell_obs is not None:
                motion = getattr(cell_obs, "motion_level", 0.0)
                if motion == 0.0 and observation.motion_level > 0.0:
                    motion = observation.motion_level
                max_motion = max(max_motion, motion)
            elif observation.motion_level > 0.0:
                max_motion = max(max_motion, observation.motion_level)

        return max(0.0, min(1.0, 1.0 - max_motion))

    def _compute_occlusion_penalty(
        self,
        affected_squares: List[str],
        observation: BoardObservation,
    ) -> float:
        """Computes occlusion penalty P_o(m) = mean(occlusion_ratio on A(m))."""
        ratios: List[float] = []
        for sq_name in affected_squares:
            cell_obs = observation.cells.get(sq_name)
            if cell_obs is not None:
                ratio = getattr(cell_obs, "occlusion_ratio", 0.0)
                if ratio == 0.0 and cell_obs.is_occluded:
                    ratio = 1.0
                ratios.append(ratio)

        return sum(ratios) / len(ratios) if ratios else 0.0

    def score_hypothesis(
        self,
        board: chess.Board,
        move: chess.Move,
        affected_squares: List[str],
        observation: BoardObservation,
    ) -> MoveHypothesis:
        """Evaluates a single legal move candidate against observation evidence.

        Args:
            board: Authoritative board state.
            move: Candidate move.
            affected_squares: Squares A(m).
            observation: BoardObservation.

        Returns:
            MoveHypothesis populated with component and total scores.
        """
        # Standard Algebraic Notation
        try:
            move_san = board.san(move)
        except Exception:
            move_san = None

        s_v = self._compute_visual_score(board, move, affected_squares, observation)
        s_t = self._compute_temporal_score(affected_squares, observation)
        s_g = max(0.0, min(1.0, observation.geometry_confidence))
        p_o = self._compute_occlusion_penalty(affected_squares, observation)

        # Formula: S(m) = w_v * S_v(m) + w_t * S_t(m) + w_g * S_g(m) - w_o * P_o(m)
        raw_total = (
            self.config.weight_visual * s_v
            + self.config.weight_temporal * s_t
            + self.config.weight_geometry * s_g
            - self.config.weight_occlusion * p_o
        )
        total_score = max(0.0, min(1.0, raw_total))

        return MoveHypothesis(
            move_uci=move.uci(),
            move_san=move_san,
            changed_squares=affected_squares,
            score_visual=round(s_v, 4),
            score_temporal=round(s_t, 4),
            score_geometry=round(s_g, 4),
            penalty_occlusion=round(p_o, 4),
            total_score=round(total_score, 4),
            is_legal=True,
        )

    def rank_candidates(
        self,
        board: chess.Board,
        candidates: List[Tuple[chess.Move, List[str]]],
        observation: BoardObservation,
    ) -> List[MoveHypothesis]:
        """Scores and ranks all candidate moves by composite likelihood S(m).

        Args:
            board: Authoritative board state.
            candidates: List of (candidate_move, affected_squares) from CandidateGenerator.
            observation: Current BoardObservation.

        Returns:
            Sorted list of MoveHypothesis in descending order of total_score.
        """
        if not candidates:
            return []

        hypotheses = [
            self.score_hypothesis(board, move, affected, observation)
            for move, affected in candidates
        ]

        # Sort descending by total score, then visual score, then UCI
        hypotheses.sort(
            key=lambda h: (h.total_score, h.score_visual, h.move_uci),
            reverse=True,
        )
        return hypotheses
