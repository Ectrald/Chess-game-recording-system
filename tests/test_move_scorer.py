"""Tests for MoveScorer and composite likelihood score calculations."""

import pytest
import chess

from src.core.config import DecisionConfig
from src.core.datatypes import (
    CHESS_SQUARES,
    PIECE_CLASSES,
    BoardObservation,
    CellObservation,
    FSMState,
)
from src.reasoning.candidate_generator import (
    LegalMoveCandidateGenerator,
    board_square_to_piece_class,
)
from src.reasoning.scorer import MoveScorer


def create_observation_for_board(board: chess.Board) -> BoardObservation:
    """Creates a synthetic BoardObservation with 1.0 probability on current board positions."""
    cells = {}
    for sq_name in CHESS_SQUARES:
        sq_idx = chess.parse_square(sq_name)
        p_class = board_square_to_piece_class(board, sq_idx)
        probs = {c: 0.0 for c in PIECE_CLASSES}
        probs[p_class] = 1.0
        cells[sq_name] = CellObservation(
            square=sq_name,
            probabilities=probs,
            predicted_piece=p_class,
            is_occupied=(p_class != "empty"),
            confidence=1.0,
            is_occluded=False,
            occlusion_ratio=0.0,
            motion_level=0.0,
        )

    return BoardObservation(
        frame_id=10,
        timestamp=1.0,
        cells=cells,
        fsm_state=FSMState.STABLE,
        motion_detected=False,
        motion_level=0.0,
        hand_present=False,
        geometry_confidence=1.0,
    )


def test_scorer_perfect_visual_match():
    scorer = MoveScorer()
    board = chess.Board()

    # Move e2e4
    move = chess.Move.from_uci("e2e4")
    # Simulate board after e2e4
    board_after = board.copy()
    board_after.push(move)
    obs = create_observation_for_board(board_after)

    hypo = scorer.score_hypothesis(board, move, ["e2", "e4"], obs)

    assert hypo.move_uci == "e2e4"
    assert hypo.move_san == "e4"
    assert hypo.score_visual == 1.0
    assert hypo.score_temporal == 1.0
    assert hypo.score_geometry == 1.0
    assert hypo.penalty_occlusion == 0.0
    # Formula: w_v(0.5) + w_t(0.25) + w_g(0.15) - w_o(0.1)*0 = 0.90
    assert hypo.total_score == 0.90
    assert hypo.is_legal is True


def test_scorer_occlusion_and_motion_penalties():
    config = DecisionConfig(
        weight_visual=0.50,
        weight_temporal=0.25,
        weight_geometry=0.15,
        weight_occlusion=0.10,
    )
    scorer = MoveScorer(config=config)
    board = chess.Board()

    move = chess.Move.from_uci("e2e4")
    board_after = board.copy()
    board_after.push(move)
    obs = create_observation_for_board(board_after)

    # Introduce motion on e4
    obs.cells["e4"].motion_level = 0.4
    # Introduce occlusion on e2
    obs.cells["e2"].occlusion_ratio = 0.6

    hypo = scorer.score_hypothesis(board, move, ["e2", "e4"], obs)

    # s_t = 1.0 - 0.4 = 0.6
    assert abs(hypo.score_temporal - 0.6) < 1e-3
    # p_o = (0.6 + 0.0) / 2 = 0.3
    assert abs(hypo.penalty_occlusion - 0.3) < 1e-3
    # raw_total = 0.50*1.0 + 0.25*0.6 + 0.15*1.0 - 0.10*0.3 = 0.50 + 0.15 + 0.15 - 0.03 = 0.77
    assert abs(hypo.total_score - 0.77) < 1e-3


def test_rank_candidates_correct_move_top1():
    generator = LegalMoveCandidateGenerator()
    scorer = MoveScorer()
    board = chess.Board()

    # Apply e2e4 to create observation
    board_after = board.copy()
    board_after.push(chess.Move.from_uci("e2e4"))
    obs = create_observation_for_board(board_after)

    candidates = generator.generate_candidates(board, observation=obs)
    ranked = scorer.rank_candidates(board, candidates, obs)

    assert len(ranked) == 20
    top_hypo = ranked[0]
    assert top_hypo.move_uci == "e2e4"
    assert top_hypo.move_san == "e4"
    assert top_hypo.total_score >= 0.85

    # Second candidate (e.g. e2e3 or other) should have significantly lower score
    second_hypo = ranked[1]
    margin = top_hypo.total_score - second_hypo.total_score
    assert margin > 0.15, f"Margin {margin} is smaller than expected 0.15"


def test_rank_candidates_empty_input():
    scorer = MoveScorer()
    board = chess.Board()
    obs = create_observation_for_board(board)
    ranked = scorer.rank_candidates(board, [], obs)
    assert ranked == []


def test_scorer_factorized_zeroshot_and_figurine_margin():
    """Zero-shot uniform subtype distribution or replacement figurine must not produce false UNCERTAIN (~0.7 vs 0.6)."""
    generator = LegalMoveCandidateGenerator()
    scorer = MoveScorer()
    board = chess.Board()
    obs = create_observation_for_board(board)

    # e2 becomes empty; e4 receives a translucent/unknown-subtype white piece (diffuse across 6 w* classes)
    obs.cells["e2"].probabilities = {c: (0.92 if c == "empty" else 0.005) for c in PIECE_CLASSES}
    obs.cells["e2"].predicted_piece = "empty"
    obs.cells["e2"].is_occupied = False

    diffuse_probs = {c: 0.005 for c in PIECE_CLASSES}
    diffuse_probs["empty"] = 0.04
    for w_cls in ("wP", "wN", "wB", "wR", "wQ", "wK"):
        diffuse_probs[w_cls] = 0.15
    obs.cells["e4"].probabilities = diffuse_probs
    obs.cells["e4"].predicted_piece = "wN"
    obs.cells["e4"].is_occupied = True

    ranked = scorer.rank_candidates(board, generator.generate_candidates(board, obs), obs)
    assert ranked[0].move_uci == "e2e4"
    assert ranked[0].total_score >= 0.80
    assert (ranked[0].total_score - ranked[1].total_score) >= 0.15
