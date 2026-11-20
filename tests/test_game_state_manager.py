"""Tests for GameStateManager: atomic commit, idempotency, PGN export, rollback and replay."""

import io
import pytest
import chess
import chess.pgn

from src.core.datatypes import (
    CHESS_SQUARES,
    BoardObservation,
    CellObservation,
    FSMState,
    MoveEvent,
    MoveHypothesis,
)
from src.reasoning.game_state_manager import GameStateManager


def make_dummy_obs(frame_id: int = 1, timestamp: float = 0.5) -> BoardObservation:
    cells = {
        sq: CellObservation(
            square=sq,
            probabilities={"empty": 1.0},
            predicted_piece="empty",
            is_occupied=False,
            confidence=1.0,
        )
        for sq in CHESS_SQUARES
    }
    return BoardObservation(
        frame_id=frame_id,
        timestamp=timestamp,
        cells=cells,
        fsm_state=FSMState.STABLE,
        motion_detected=False,
        motion_level=0.0,
        hand_present=False,
        geometry_confidence=0.95,
    )


def make_hypo(move_uci: str, total_score: float = 0.88) -> MoveHypothesis:
    return MoveHypothesis(
        move_uci=move_uci,
        move_san=None,
        changed_squares=[],
        score_visual=total_score,
        score_temporal=1.0,
        score_geometry=1.0,
        penalty_occlusion=0.0,
        total_score=total_score,
        is_legal=True,
    )


def test_game_state_manager_commit():
    committed_events = []
    mgr = GameStateManager(on_move_committed=lambda ev: committed_events.append(ev))

    assert mgr.ply == 0
    assert mgr.current_fen == chess.STARTING_FEN

    # Move 1: 1. e4
    obs1 = make_dummy_obs(frame_id=10, timestamp=1.0)
    ev1 = mgr.commit(make_hypo("e2e4"), obs1)

    assert ev1.ply == 1
    assert ev1.move_san == "e4"
    assert ev1.move_uci == "e2e4"
    assert ev1.fen_before == chess.STARTING_FEN
    assert ev1.fen_after == mgr.current_fen
    assert mgr.ply == 1
    assert len(committed_events) == 1

    # Move 2: 1... e5
    obs2 = make_dummy_obs(frame_id=25, timestamp=2.5)
    ev2 = mgr.commit(make_hypo("e7e5"), obs2)

    assert ev2.ply == 2
    assert ev2.move_san == "e5"
    assert mgr.ply == 2
    assert len(committed_events) == 2


def test_game_state_manager_illegal_move():
    mgr = GameStateManager()
    obs = make_dummy_obs()

    # Move e2e5 is illegal from starting position
    with pytest.raises(ValueError, match="is not legal"):
        mgr.commit(make_hypo("e2e5"), obs)

    # State must remain untouched
    assert mgr.ply == 0
    assert mgr.current_fen == chess.STARTING_FEN


def test_game_state_manager_idempotency():
    mgr = GameStateManager()
    obs = make_dummy_obs(frame_id=100)
    hypo = make_hypo("e2e4")

    ev1 = mgr.commit(hypo, obs)
    assert mgr.ply == 1

    # Re-committing with identical parameters should return ev1 without advancing state
    ev2 = mgr.commit(hypo, obs)
    assert mgr.ply == 1
    assert ev1.event_id == ev2.event_id
    assert len(mgr.history) == 1


def test_game_state_manager_rollback():
    mgr = GameStateManager()
    mgr.commit(make_hypo("e2e4"), make_dummy_obs(frame_id=1))
    mgr.commit(make_hypo("e7e5"), make_dummy_obs(frame_id=2))
    assert mgr.ply == 2

    # Rollback ply 2
    popped = mgr.rollback()
    assert popped is not None
    assert popped.move_san == "e5"
    assert mgr.ply == 1

    # Rollback ply 1
    popped1 = mgr.rollback()
    assert popped1 is not None
    assert popped1.move_san == "e4"
    assert mgr.ply == 0
    assert mgr.current_fen == chess.STARTING_FEN

    # Rollback on empty history returns None
    assert mgr.rollback() is None


def test_game_state_manager_manual_commit():
    mgr = GameStateManager()
    ev = mgr.manual_commit("d2d4")
    assert ev.ply == 1
    assert ev.move_san == "d4"
    assert mgr.ply == 1

    with pytest.raises(ValueError, match="Invalid UCI move string"):
        mgr.manual_commit("invalid_uci")


def test_game_state_manager_pgn_export():
    mgr = GameStateManager()
    mgr.commit(make_hypo("e2e4"), make_dummy_obs(frame_id=1))
    mgr.commit(make_hypo("e7e5"), make_dummy_obs(frame_id=2))
    mgr.commit(make_hypo("g1f3"), make_dummy_obs(frame_id=3))
    mgr.commit(make_hypo("b8c6"), make_dummy_obs(frame_id=4))

    pgn_str = mgr.export_pgn(headers={"Event": "World Championship", "White": "Magnus", "Black": "Ian"})
    assert "World Championship" in pgn_str
    assert "Magnus" in pgn_str
    assert "1. e4 e5 2. Nf3 Nc6" in pgn_str

    # Validate that python-chess can parse the exported PGN without errors
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    assert game is not None
    assert game.headers["Event"] == "World Championship"
    moves = [m.uci() for m in game.mainline_moves()]
    assert moves == ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_game_state_manager_replay():
    mgr = GameStateManager()
    mgr.commit(make_hypo("e2e4"), make_dummy_obs(frame_id=1))
    mgr.commit(make_hypo("e7e5"), make_dummy_obs(frame_id=2))
    mgr.commit(make_hypo("g1f3"), make_dummy_obs(frame_id=3))

    replayed_board = mgr.replay(mgr.history)
    assert replayed_board.fen() == mgr.current_fen

    # Replay divergence check
    corrupted_history = list(mgr.history)
    corrupted_event = corrupted_history[0].model_copy(update={"fen_before": "8/8/8/8/8/8/8/8 w - - 0 1"})
    corrupted_history[0] = corrupted_event

    with pytest.raises(ValueError, match="Replay divergence"):
        mgr.replay(corrupted_history)
