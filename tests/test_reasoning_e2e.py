"""End-to-End Reasoning Integration Tests.

Validates complete reasoning pipeline:
GameFSM + LegalMoveCandidateGenerator + MoveScorer + DecisionGate + GameStateManager
across:
1. Complete game simulation over test_short_game.pgn (Scholar's Mate / 8 plies)
2. Comprehensive special moves validation (En Passant, Kingside & Queenside Castling, Promotion)
3. Ambiguity handling (UNCERTAIN status, confidence margin deficit, manual operator commit)
4. Disturbance false alarms (piece touched and returned, zero changes -> STABLE without commit)
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple
import pytest
import chess
import chess.pgn

from src.core.config import DecisionConfig, TemporalConfig
from src.core.datatypes import (
    CHESS_SQUARES,
    PIECE_CLASSES,
    BoardObservation,
    CellObservation,
    DecisionStatus,
    FSMState,
)
from src.reasoning.candidate_generator import (
    LegalMoveCandidateGenerator,
    board_square_to_piece_class,
)
from src.reasoning.decision_gate import DecisionGate
from src.reasoning.fsm import GameFSM
from src.reasoning.game_state_manager import GameStateManager
from src.reasoning.scorer import MoveScorer


def create_perfect_observation(
    board: chess.Board,
    frame_id: int,
    timestamp: float,
    motion_detected: bool = False,
    hand_present: bool = False,
    geometry_confidence: float = 0.95,
) -> BoardObservation:
    """Builds a BoardObservation reflecting exact piece layout of board."""
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
        frame_id=frame_id,
        timestamp=timestamp,
        cells=cells,
        motion_detected=motion_detected,
        motion_level=0.15 if motion_detected else 0.0,
        hand_present=hand_present,
        geometry_confidence=geometry_confidence,
    )


def simulate_move_execution(
    fsm: GameFSM,
    resulting_board: chess.Board,
    start_frame_id: int,
    start_time: float,
    fps: float = 10.0,
) -> Tuple[int, float]:
    """Simulates realistic camera stream for a physical move execution:
    1. Moving phase: hand enters and moves piece (disturbance)
    2. Settling phase: hand leaves, board stabilizes
    3. Verifying phase: stable frames observed until FSM confirms decision
    """
    frame_id = start_frame_id
    t = start_time
    dt = 1.0 / fps

    # 1. Moving state (3 disturbance frames)
    for _ in range(3):
        obs = create_perfect_observation(
            resulting_board,
            frame_id=frame_id,
            timestamp=t,
            motion_detected=True,
            hand_present=True,
        )
        fsm.update(obs)
        frame_id += 1
        t += dt

    # 2. Settling debounce (wait > settling_time_sec, e.g. 1.0s at 10fps = 10 frames)
    for _ in range(10):
        obs = create_perfect_observation(
            resulting_board,
            frame_id=frame_id,
            timestamp=t,
            motion_detected=False,
            hand_present=False,
        )
        fsm.update(obs)
        frame_id += 1
        t += dt

    # 3. Verifying frames (wait verifying_stable_frames to trigger gate evaluation)
    for _ in range(fsm.config.verifying_stable_frames + 2):
        obs = create_perfect_observation(
            resulting_board,
            frame_id=frame_id,
            timestamp=t,
            motion_detected=False,
            hand_present=False,
        )
        fsm.update(obs)
        frame_id += 1
        t += dt
        if fsm.current_state == FSMState.STABLE:
            break

    return frame_id, t


def test_e2e_short_game_simulation():
    """Simulates entire Scholar's Mate game from data/ground_truth/test_short_game.pgn."""
    with open("data/ground_truth/test_short_game.pgn") as f:
        ground_truth_game = chess.pgn.read_game(f)
    assert ground_truth_game is not None

    expected_moves = list(ground_truth_game.mainline_moves())
    assert len(expected_moves) == 7  # 7 plies to checkmate (Scholar's Mate)

    temporal_cfg = TemporalConfig(
        settling_time_sec=0.5,
        verifying_stable_frames=4,
    )
    decision_cfg = DecisionConfig()

    cand_gen = LegalMoveCandidateGenerator()
    scorer = MoveScorer(config=decision_cfg)
    gate = DecisionGate(config=decision_cfg)
    state_mgr = GameStateManager()

    decisions = []
    fsm = GameFSM(
        config=temporal_cfg,
        candidate_generator=cand_gen,
        scorer=scorer,
        decision_gate=gate,
        game_state_manager=state_mgr,
        on_move_decision=lambda d, obs: decisions.append(d),
    )

    # Initial calibration
    calib_obs = create_perfect_observation(state_mgr.current_board, frame_id=1, timestamp=0.1)
    fsm.update(calib_obs)  # IDLE -> CALIBRATING
    fsm.update(calib_obs)  # CALIBRATING -> STABLE
    assert fsm.current_state == FSMState.STABLE

    frame_id = 10
    sim_time = 1.0
    sim_board = chess.Board()

    for move_idx, expected_move in enumerate(expected_moves):
        sim_board.push(expected_move)

        frame_id, sim_time = simulate_move_execution(
            fsm=fsm,
            resulting_board=sim_board,
            start_frame_id=frame_id,
            start_time=sim_time,
        )

        assert fsm.current_state == FSMState.STABLE
        assert state_mgr.ply == move_idx + 1

        last_event = state_mgr.history[-1]
        assert last_event.move_uci == expected_move.uci()

    # Game should be concluded with checkmate
    assert state_mgr.current_board.is_checkmate()
    assert state_mgr.ply == 7

    # Validate PGN export
    pgn_exported = state_mgr.export_pgn({"Event": "Scholar's Mate Test Game"})
    assert "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#" in pgn_exported

    # Validate replay determinism
    replayed = state_mgr.replay(state_mgr.history)
    assert replayed.fen() == state_mgr.current_fen
    assert replayed.is_checkmate()


def test_e2e_special_moves_simulation():
    """Simulates key special moves: Castling (O-O and O-O-O), En Passant, and Promotion."""
    temporal_cfg = TemporalConfig(
        settling_time_sec=0.4,
        verifying_stable_frames=3,
    )
    decision_cfg = DecisionConfig()

    cand_gen = LegalMoveCandidateGenerator()
    scorer = MoveScorer(config=decision_cfg)
    gate = DecisionGate(config=decision_cfg)

    # 1. Test White En Passant: 1. e4 Nf6 2. e5 d5 3. exd6
    state_mgr_ep = GameStateManager()
    fsm_ep = GameFSM(
        config=temporal_cfg,
        candidate_generator=cand_gen,
        scorer=scorer,
        decision_gate=gate,
        game_state_manager=state_mgr_ep,
    )

    # Calibrate
    init_obs = create_perfect_observation(state_mgr_ep.current_board, 1, 0.1)
    fsm_ep.update(init_obs)
    fsm_ep.update(init_obs)

    moves_ep = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("g8f6"),
        chess.Move.from_uci("e4e5"),
        chess.Move.from_uci("d7d5"),
        chess.Move.from_uci("e5d6"),  # En passant capture
    ]

    board_tracker = chess.Board()
    fid, t = 10, 1.0
    for m in moves_ep:
        board_tracker.push(m)
        fid, t = simulate_move_execution(fsm_ep, board_tracker, fid, t)

    assert state_mgr_ep.ply == 5
    ep_event = state_mgr_ep.history[-1]
    assert ep_event.move_uci == "e5d6"
    assert ep_event.move_san == "exd6"
    assert state_mgr_ep.replay(state_mgr_ep.history).fen() == state_mgr_ep.current_fen

    # 2. Test White Queenside Castling & Black Kingside Castling
    # Setup custom position where castling is immediately legal
    castling_fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    state_mgr_c = GameStateManager(initial_fen=castling_fen)
    fsm_c = GameFSM(
        config=temporal_cfg,
        candidate_generator=cand_gen,
        scorer=scorer,
        decision_gate=gate,
        game_state_manager=state_mgr_c,
    )
    c_obs = create_perfect_observation(state_mgr_c.current_board, 1, 0.1)
    fsm_c.update(c_obs)
    fsm_c.update(c_obs)

    # White Queenside Castling (O-O-O: e1c1)
    b_cast = chess.Board(castling_fen)
    b_cast.push(chess.Move.from_uci("e1c1"))
    fid, t = simulate_move_execution(fsm_c, b_cast, 10, 1.0)
    assert state_mgr_c.history[-1].move_san == "O-O-O"

    # Black Kingside Castling (O-O: e8g8)
    b_cast.push(chess.Move.from_uci("e8g8"))
    fid, t = simulate_move_execution(fsm_c, b_cast, fid, t)
    assert state_mgr_c.history[-1].move_san == "O-O"
    assert state_mgr_c.replay(state_mgr_c.history).fen() == state_mgr_c.current_fen

    # 3. Test Pawn Promotion (e7e8q)
    promo_fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
    state_mgr_p = GameStateManager(initial_fen=promo_fen)
    fsm_p = GameFSM(
        config=temporal_cfg,
        candidate_generator=cand_gen,
        scorer=scorer,
        decision_gate=gate,
        game_state_manager=state_mgr_p,
    )
    p_obs = create_perfect_observation(state_mgr_p.current_board, 1, 0.1)
    fsm_p.update(p_obs)
    fsm_p.update(p_obs)

    b_promo = chess.Board(promo_fen)
    b_promo.push(chess.Move.from_uci("e7e8q"))
    fid, t = simulate_move_execution(fsm_p, b_promo, 10, 1.0)
    assert state_mgr_p.history[-1].move_uci == "e7e8q"
    assert "Q" in state_mgr_p.history[-1].move_san
    assert state_mgr_p.replay(state_mgr_p.history).fen() == state_mgr_p.current_fen


def test_e2e_ambiguity_uncertain_and_manual_recovery():
    """Tests UNCERTAIN state triggered by close candidate margin, and manual operator commit."""
    temporal_cfg = TemporalConfig(
        settling_time_sec=0.3,
        verifying_stable_frames=3,
    )
    decision_cfg = DecisionConfig(min_confidence_margin=0.20)

    cand_gen = LegalMoveCandidateGenerator()
    scorer = MoveScorer(config=decision_cfg)
    gate = DecisionGate(config=decision_cfg)
    state_mgr = GameStateManager()

    fsm = GameFSM(
        config=temporal_cfg,
        candidate_generator=cand_gen,
        scorer=scorer,
        decision_gate=gate,
        game_state_manager=state_mgr,
    )

    init_obs = create_perfect_observation(state_mgr.current_board, 1, 0.1)
    fsm.update(init_obs)
    fsm.update(init_obs)

    # Simulate ambiguous move: e2 is empty, but e3 and e4 both look 50% likely as wP
    ambig_board = chess.Board()
    obs_ambig = create_perfect_observation(ambig_board, frame_id=10, timestamp=1.0)
    obs_ambig.cells["e2"].probabilities = {"empty": 1.0, "wP": 0.0}
    obs_ambig.cells["e2"].predicted_piece = "empty"
    obs_ambig.cells["e3"].probabilities = {"empty": 0.5, "wP": 0.5}
    obs_ambig.cells["e3"].predicted_piece = "wP"
    obs_ambig.cells["e4"].probabilities = {"empty": 0.5, "wP": 0.5}
    obs_ambig.cells["e4"].predicted_piece = "wP"

    # Moving
    fsm.update(create_perfect_observation(ambig_board, 11, 1.1, motion_detected=True, hand_present=True))
    # Settling
    for i in range(5):
        fsm.update(create_perfect_observation(ambig_board, 12 + i, 1.3 + i * 0.1))
    # Verifying with ambiguous observation
    for i in range(5):
        obs_ambig.frame_id = 20 + i
        obs_ambig.timestamp = 2.0 + i * 0.1
        fsm.update(obs_ambig)

    # Should transition to UNCERTAIN because margin is small
    assert fsm.current_state == FSMState.UNCERTAIN
    assert state_mgr.ply == 0  # No move committed!

    # Operator resolves ambiguity manually
    state_mgr.manual_commit("e2e4")
    assert state_mgr.ply == 1
    assert state_mgr.history[-1].move_san == "e4"

    # FSM resumes STABLE
    fsm.resume()
    assert fsm.current_state == FSMState.STABLE


def test_e2e_false_alarm_no_changes():
    """Tests disturbance without square changes (hand hovered or touched piece then left)."""
    temporal_cfg = TemporalConfig(
        settling_time_sec=0.3,
        verifying_stable_frames=3,
    )
    cand_gen = LegalMoveCandidateGenerator()
    scorer = MoveScorer()
    gate = DecisionGate()
    state_mgr = GameStateManager()

    fsm = GameFSM(
        config=temporal_cfg,
        candidate_generator=cand_gen,
        scorer=scorer,
        decision_gate=gate,
        game_state_manager=state_mgr,
    )

    init_obs = create_perfect_observation(state_mgr.current_board, 1, 0.1)
    fsm.update(init_obs)
    fsm.update(init_obs)

    # Trigger disturbance on identical board state
    frame_id, sim_time = simulate_move_execution(
        fsm=fsm,
        resulting_board=state_mgr.current_board,
        start_frame_id=10,
        start_time=1.0,
    )

    # Must return to STABLE without incrementing ply
    assert fsm.current_state == FSMState.STABLE
    assert state_mgr.ply == 0
    assert len(state_mgr.history) == 0
