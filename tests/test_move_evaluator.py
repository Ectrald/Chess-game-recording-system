"""Tests for MoveEvaluator: perspective normalization and move quality classification."""

import pytest

from src.core.datatypes import (
    EngineEvaluation,
    MoveEvent,
    MoveQuality,
)
from src.analysis.move_evaluator import MoveEvaluator, score_to_numerical


def make_test_event(ply: int, fen_before: str, move_uci: str) -> MoveEvent:
    return MoveEvent(
        ply=ply,
        fen_before=fen_before,
        move_uci=move_uci,
        move_san=move_uci,
        fen_after="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        confidence=0.9,
        timestamp=1.0,
        evidence_frame_id=1,
    )


def test_score_to_numerical():
    # Regular CP
    assert score_to_numerical(EngineEvaluation(eval_cp=150)) == 150
    # Mate in 2 for White
    assert score_to_numerical(EngineEvaluation(mate_in=2)) == 10000 - 20
    # Mate in 1 for Black (-1)
    assert score_to_numerical(EngineEvaluation(mate_in=-1)) == -10000 + 10
    # None
    assert score_to_numerical(None) == 0


def test_evaluator_white_best_move():
    evaluator = MoveEvaluator()
    # White played e2e4
    event = make_test_event(
        1,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="e2e4",
    )
    eval_before = EngineEvaluation(eval_cp=30, best_move_uci="e2e4")
    eval_after = EngineEvaluation(eval_cp=30, best_move_uci="e7e5")

    res = evaluator.evaluate_move(event, eval_before, eval_after)
    assert res.quality == MoveQuality.BEST
    assert res.eval_loss_cp == 0
    assert res.is_blunder is False


def test_evaluator_white_blunder():
    evaluator = MoveEvaluator(blunder_threshold_cp=150)
    # White played f2f3, losing 300 centipawns
    event = make_test_event(
        1,
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="f2f3",
    )
    eval_before = EngineEvaluation(eval_cp=30, best_move_uci="e2e4")
    eval_after = EngineEvaluation(eval_cp=-270, best_move_uci="e7e5")

    res = evaluator.evaluate_move(event, eval_before, eval_after)
    assert res.quality == MoveQuality.BLUNDER
    assert res.eval_loss_cp == 300
    assert res.is_blunder is True


def test_evaluator_black_perspective_normalization():
    evaluator = MoveEvaluator()
    # Black played e7e5 from position after 1. e4
    # fen_before has active color 'b'
    event = make_test_event(
        2,
        fen_before="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        move_uci="e7e5",
    )
    # Before move: White was +35 (so Black was -35)
    eval_before = EngineEvaluation(eval_cp=35, best_move_uci="e7e5")
    # After move: White is still +35 (so Black is still -35, delta = 0)
    eval_after = EngineEvaluation(eval_cp=35, best_move_uci="g1f3")

    res = evaluator.evaluate_move(event, eval_before, eval_after)
    assert res.quality == MoveQuality.BEST
    assert res.eval_loss_cp == 0

    # Now simulate Black blunder: Black played f7f6 instead of best move e7e5
    event_blunder = make_test_event(
        2,
        fen_before="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        move_uci="f7f6",
    )
    # White becomes +350 (Black drops from -35 to -350 = loss 315)
    eval_blunder = EngineEvaluation(eval_cp=350, best_move_uci="g1f3")
    res_blunder = evaluator.evaluate_move(event_blunder, eval_before, eval_blunder)
    assert res_blunder.eval_loss_cp == 315
    assert res_blunder.quality == MoveQuality.BLUNDER
    assert res_blunder.is_blunder is True
