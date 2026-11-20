"""Tests for LegalMoveCandidateGenerator and topological move signatures."""

import pytest
import chess

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


def create_mock_observation(board: chess.Board) -> BoardObservation:
    """Creates a synthetic BoardObservation perfectly matching the given board."""
    cells = {}
    for sq_name in CHESS_SQUARES:
        sq_idx = chess.parse_square(sq_name)
        p_class = board_square_to_piece_class(board, sq_idx)
        probs = {cls_name: 0.0 for cls_name in PIECE_CLASSES}
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
        frame_id=1,
        timestamp=0.1,
        cells=cells,
        fsm_state=FSMState.STABLE,
        motion_detected=False,
        motion_level=0.0,
        hand_present=False,
        geometry_confidence=0.95,
    )


def test_initial_board_candidate_count():
    generator = LegalMoveCandidateGenerator()
    board = chess.Board()
    candidates = generator.generate_candidates(board)
    assert len(candidates) == 20
    # Check that all moves are legal
    for move, affected in candidates:
        assert board.is_legal(move)
        assert len(affected) == 2


def test_regular_move_topological_signature():
    generator = LegalMoveCandidateGenerator()
    board = chess.Board()

    # e2e4
    move = chess.Move.from_uci("e2e4")
    affected = generator.get_affected_squares(board, move)
    assert affected == ["e2", "e4"]

    # Knight g1f3
    move_n = chess.Move.from_uci("g1f3")
    affected_n = generator.get_affected_squares(board, move_n)
    assert affected_n == ["g1", "f3"]


def test_castling_topological_signatures():
    generator = LegalMoveCandidateGenerator()

    # White Kingside & Queenside castling board
    board_w = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    oo_w = chess.Move.from_uci("e1g1")
    ooo_w = chess.Move.from_uci("e1c1")

    assert generator.get_affected_squares(board_w, oo_w) == ["e1", "g1", "f1", "h1"]
    assert generator.get_affected_squares(board_w, ooo_w) == ["e1", "c1", "d1", "a1"]

    # Black Kingside & Queenside castling board
    board_b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    oo_b = chess.Move.from_uci("e8g8")
    ooo_b = chess.Move.from_uci("e8c8")

    assert generator.get_affected_squares(board_b, oo_b) == ["e8", "g8", "f8", "h8"]
    assert generator.get_affected_squares(board_b, ooo_b) == ["e8", "c8", "d8", "a8"]


def test_en_passant_topological_signature():
    generator = LegalMoveCandidateGenerator()

    # White en passant on f6 capturing f5 pawn
    board_ep_w = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    m_ep_w = chess.Move.from_uci("e5f6")
    assert board_ep_w.is_en_passant(m_ep_w)
    affected_w = generator.get_affected_squares(board_ep_w, m_ep_w)
    assert affected_w == ["e5", "f6", "f5"]

    # Black en passant on c3 capturing c4 pawn
    board_ep_b = chess.Board("rnbqkbnr/pppp1ppp/8/8/2P1p3/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 2")
    # Let White push c2c4 to enable en passant
    board_ep_b = chess.Board("rnbqkbnr/pp1ppppp/8/8/2pP4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3")
    m_ep_b = chess.Move.from_uci("c4d3")
    assert board_ep_b.is_en_passant(m_ep_b)
    affected_b = generator.get_affected_squares(board_ep_b, m_ep_b)
    assert affected_b == ["c4", "d3", "d4"]


def test_promotion_topological_signature():
    generator = LegalMoveCandidateGenerator()
    board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")

    for promo_piece in ["q", "r", "b", "n"]:
        move = chess.Move.from_uci(f"e7e8{promo_piece}")
        affected = generator.get_affected_squares(board, move)
        assert affected == ["e7", "e8"]


def test_detect_changed_squares():
    generator = LegalMoveCandidateGenerator()
    board = chess.Board()
    obs = create_mock_observation(board)

    # Identical state: 0 changes
    changed = generator.detect_changed_squares(board, obs)
    assert changed == []

    # Modify e2 (was wP, now empty) and e4 (was empty, now wP)
    obs.cells["e2"].predicted_piece = "empty"
    obs.cells["e4"].predicted_piece = "wP"

    changed = generator.detect_changed_squares(board, obs)
    assert changed == ["e2", "e4"]

    # If e4 is occluded, it should not be considered changed
    obs.cells["e4"].is_occluded = True
    changed_with_occ = generator.detect_changed_squares(board, obs)
    assert changed_with_occ == ["e2"]


def test_generate_candidates_with_observation_prioritization():
    generator = LegalMoveCandidateGenerator()
    board = chess.Board()
    obs = create_mock_observation(board)

    # Simulate visual move e2e4
    obs.cells["e2"].predicted_piece = "empty"
    obs.cells["e4"].predicted_piece = "wP"

    candidates = generator.generate_candidates(board, observation=obs)
    assert len(candidates) == 20
    # First candidate should be e2e4 because it has 2/2 overlap with changed squares
    top_move, top_affected = candidates[0]
    assert top_move.uci() == "e2e4"
    assert top_affected == ["e2", "e4"]


def test_terminal_position_candidates():
    generator = LegalMoveCandidateGenerator()
    # Checkmate position (Fool's Mate)
    board = chess.Board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    candidates = generator.generate_candidates(board)
    assert candidates == []
