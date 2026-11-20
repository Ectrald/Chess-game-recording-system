"""Move quality evaluator with side-to-move perspective normalization.

Translates absolute Stockfish centipawn assessments into relative mover evaluations,
calculating centipawn loss and assigning standard chess quality labels:
- Best (matches engine top-1 PV or delta >= -10 cp)
- Excellent (delta >= -30 cp)
- Good (delta >= -60 cp)
- Inaccuracy (delta >= -100 cp)
- Mistake (delta >= -250 cp)
- Blunder (delta < -250 cp or missed mate)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.datatypes import (
    AnalysisResult,
    EngineEvaluation,
    EngineTaskStatus,
    MoveEvent,
    MoveQuality,
)

logger = logging.getLogger(__name__)


def score_to_numerical(eval_obj: Optional[EngineEvaluation]) -> int:
    """Converts EngineEvaluation centipawn or mate_in to continuous numerical value from White's perspective."""
    if eval_obj is None:
        return 0

    if eval_obj.mate_in is not None:
        if eval_obj.mate_in > 0:
            return 10000 - eval_obj.mate_in * 10
        else:
            return -10000 - eval_obj.mate_in * 10

    return eval_obj.eval_cp if eval_obj.eval_cp is not None else 0


class MoveEvaluator:
    """Evaluates played moves by comparing pre- and post-move engine assessments."""

    def __init__(self, blunder_threshold_cp: int = 150) -> None:
        """Initializes MoveEvaluator with blunder threshold.

        Args:
            blunder_threshold_cp: Loss in centipawns qualifying as a blunder.
        """
        self.blunder_threshold_cp = blunder_threshold_cp

    def evaluate_move(
        self,
        event: MoveEvent,
        eval_before: EngineEvaluation,
        eval_after: EngineEvaluation,
    ) -> AnalysisResult:
        """Evaluates a confirmed MoveEvent against before/after engine evaluations.

        Args:
            event: Confirmed MoveEvent.
            eval_before: EngineEvaluation of position prior to move.
            eval_after: EngineEvaluation of position after move.

        Returns:
            AnalysisResult with normalized eval loss and move classification.
        """
        # Determine who made the move from fen_before ('w' or 'b')
        fen_parts = event.fen_before.split()
        is_white_mover = (len(fen_parts) > 1 and fen_parts[1] == "w")

        raw_before = score_to_numerical(eval_before)
        raw_after = score_to_numerical(eval_after)

        # Normalize to mover's perspective
        mover_before = raw_before if is_white_mover else -raw_before
        mover_after = raw_after if is_white_mover else -raw_after

        # Centipawn loss (positive integer = lost advantage)
        delta = mover_after - mover_before
        eval_loss_cp = max(0, -delta)

        # Quality classification
        if eval_before.best_move_uci == event.move_uci:
            quality = MoveQuality.BEST
        elif eval_loss_cp <= 10:
            quality = MoveQuality.EXCELLENT
        elif eval_loss_cp <= 30:
            quality = MoveQuality.GOOD
        elif eval_loss_cp <= 90:
            quality = MoveQuality.INACCURACY
        elif eval_loss_cp <= 200:
            quality = MoveQuality.MISTAKE
        else:
            quality = MoveQuality.BLUNDER

        is_blunder = (eval_loss_cp >= self.blunder_threshold_cp) or (quality == MoveQuality.BLUNDER)

        return AnalysisResult(
            event_id=event.event_id,
            ply=event.ply,
            move_uci=event.move_uci,
            status=EngineTaskStatus.COMPLETED,
            eval_before=eval_before,
            eval_after=eval_after,
            eval_loss_cp=eval_loss_cp,
            quality=quality,
            is_blunder=is_blunder,
        )

