"""Unit tests for the Game Finite State Machine (GameFSM).

Covers:
- STAGE_2.md Section 6: test_fsm.py specifications
- Full lifecycle transitions (STABLE -> MOVING -> SETTLING -> VERIFYING -> STABLE)
- Settling debounce cancellation on resumed disturbance
- Timestamp-based elapsed timing invariant across variable FPS
- Geometry loss and recovery handling
- Pause, resume, stop, and reset operator controls
"""

import pytest

from src.core.config import TemporalConfig
from src.core.datatypes import (
    BoardGeometry,
    BoardObservation,
    FSMState,
    HandDetection,
    MotionMetrics,
)
from src.reasoning.fsm import GameFSM


@pytest.fixture
def fsm_config() -> TemporalConfig:
    return TemporalConfig(
        settling_time_sec=0.8,
        motion_diff_threshold=0.04,
        verifying_stable_frames=4,
    )


@pytest.fixture
def valid_geometry() -> BoardGeometry:
    return BoardGeometry(
        corners=[(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)],
        is_valid=True,
        orientation_calibrated=True,
    )


def test_fsm_initialization_and_calibration(fsm_config: TemporalConfig, valid_geometry: BoardGeometry):
    """Verifies transition from IDLE through CALIBRATING into STABLE."""
    fsm = GameFSM(config=fsm_config)
    assert fsm.current_state == FSMState.IDLE

    # Frame 1: Valid calibrated geometry provided -> CALIBRATING -> STABLE
    obs = BoardObservation.create_default(
        timestamp=0.0,
        frame_id=1,
        geometry=valid_geometry,
        geometry_confidence=0.95,
    )
    state = fsm.update(obs)
    # Calibrated geometry transitions through CALIBRATING to STABLE on next frame
    assert state in (FSMState.CALIBRATING, FSMState.STABLE)

    # Frame 2: Remains STABLE
    obs2 = BoardObservation.create_default(
        timestamp=0.033,
        frame_id=2,
        geometry=valid_geometry,
        geometry_confidence=0.95,
    )
    state2 = fsm.update(obs2)
    assert state2 == FSMState.STABLE


def test_fsm_full_move_lifecycle(fsm_config: TemporalConfig, valid_geometry: BoardGeometry):
    """Verifies complete cycle: STABLE -> MOVING -> SETTLING -> VERIFYING -> STABLE with callback."""
    settled_observations = []

    def handle_settled(obs: BoardObservation):
        settled_observations.append(obs)

    fsm = GameFSM(config=fsm_config, on_settled_observation=handle_settled)

    # 1. Bring to STABLE
    fsm.update(BoardObservation.create_default(timestamp=0.0, frame_id=1, geometry=valid_geometry, geometry_confidence=1.0))
    fsm.update(BoardObservation.create_default(timestamp=0.033, frame_id=2, geometry=valid_geometry, geometry_confidence=1.0))
    assert fsm.current_state == FSMState.STABLE

    # 2. Player hand enters and moves piece -> MOVING
    fsm.update(BoardObservation.create_default(
        timestamp=0.1,
        frame_id=3,
        geometry=valid_geometry,
        geometry_confidence=1.0,
        hand_present=True,
        motion_detected=True,
        motion_level=0.12,
    ))
    assert fsm.current_state == FSMState.MOVING

    # 3. Hand leaves, motion ceases -> SETTLING initiated
    fsm.update(BoardObservation.create_default(
        timestamp=0.5,
        frame_id=4,
        geometry=valid_geometry,
        geometry_confidence=1.0,
        hand_present=False,
        motion_detected=False,
        motion_level=0.005,
    ))
    assert fsm.current_state == FSMState.SETTLING

    # 4. Debounce time is 0.8s. Advance time to 0.5 + 0.5 = 1.0s (elapsed 0.5s < 0.8s) -> still SETTLING
    fsm.update(BoardObservation.create_default(
        timestamp=1.0,
        frame_id=5,
        geometry=valid_geometry,
        geometry_confidence=1.0,
    ))
    assert fsm.current_state == FSMState.SETTLING

    # 5. Advance time to 0.5 + 0.85 = 1.35s (elapsed 0.85s >= 0.8s) -> transitions to VERIFYING
    fsm.update(BoardObservation.create_default(
        timestamp=1.35,
        frame_id=6,
        geometry=valid_geometry,
        geometry_confidence=1.0,
    ))
    assert fsm.current_state == FSMState.VERIFYING

    # 6. Accumulate remaining stable frames (verifying_stable_frames = 4)
    # Frame 6 was #1 in verify buffer. Feed 3 more stable frames:
    fsm.update(BoardObservation.create_default(timestamp=1.40, frame_id=7, geometry=valid_geometry, geometry_confidence=1.0))
    fsm.update(BoardObservation.create_default(timestamp=1.45, frame_id=8, geometry=valid_geometry, geometry_confidence=1.0))
    fsm.update(BoardObservation.create_default(timestamp=1.50, frame_id=9, geometry=valid_geometry, geometry_confidence=1.0))

    # All 4 verified: transitions back to STABLE and triggers callback!
    assert fsm.current_state == FSMState.STABLE
    assert len(settled_observations) == 1
    assert settled_observations[0].frame_id == 9


def test_fsm_debounce_cancellation_on_resumed_motion(fsm_config: TemporalConfig, valid_geometry: BoardGeometry):
    """Verifies that resumed motion resets settling timer and cancels transition to VERIFYING."""
    fsm = GameFSM(config=fsm_config)
    fsm.force_state(FSMState.MOVING)

    # Motion stops at t=1.0 -> SETTLING
    fsm.update(BoardObservation.create_default(timestamp=1.0, frame_id=10, geometry=valid_geometry))
    assert fsm.current_state == FSMState.SETTLING

    # Resumed motion at t=1.4 (before 1.8s) -> cancels debounce and reverts to MOVING
    fsm.update(BoardObservation.create_default(
        timestamp=1.4,
        frame_id=11,
        geometry=valid_geometry,
        motion_detected=True,
        motion_level=0.08,
    ))
    assert fsm.current_state == FSMState.MOVING

    # Settling restarts only when motion ceases again at t=2.0
    fsm.update(BoardObservation.create_default(timestamp=2.0, frame_id=12, geometry=valid_geometry))
    assert fsm.current_state == FSMState.SETTLING

    # Must wait until t >= 2.8s
    fsm.update(BoardObservation.create_default(timestamp=2.5, frame_id=13, geometry=valid_geometry))
    assert fsm.current_state == FSMState.SETTLING

    fsm.update(BoardObservation.create_default(timestamp=2.85, frame_id=14, geometry=valid_geometry))
    assert fsm.current_state == FSMState.VERIFYING


def test_fsm_geometry_loss_and_recovery(fsm_config: TemporalConfig, valid_geometry: BoardGeometry):
    """Verifies transition to BOARD_LOST on lost geometry and recovery when restored."""
    fsm = GameFSM(config=fsm_config)
    fsm.force_state(FSMState.STABLE)

    # Frame with missing / invalid geometry
    invalid_obs = BoardObservation.create_default(
        timestamp=1.0,
        frame_id=20,
        geometry=None,
        geometry_confidence=0.0,
    )

    # Tolerate momentary frame drops up to max_lost_frames (3)
    fsm.update(invalid_obs)
    fsm.update(invalid_obs)
    fsm.update(invalid_obs)
    assert fsm.current_state == FSMState.BOARD_LOST

    # Restore valid geometry
    restored_obs = BoardObservation.create_default(
        timestamp=2.0,
        frame_id=23,
        geometry=valid_geometry,
        geometry_confidence=0.9,
    )
    fsm.update(restored_obs)
    assert fsm.current_state == FSMState.LOCATING_BOARD


def test_fsm_operator_controls():
    """Verifies pause, resume, reset, and stop operator controls."""
    fsm = GameFSM()
    fsm.force_state(FSMState.STABLE)

    fsm.pause()
    assert fsm.current_state == FSMState.PAUSED

    fsm.resume()
    assert fsm.current_state == FSMState.STABLE

    fsm.stop()
    assert fsm.current_state == FSMState.STOPPED

    fsm.reset()
    assert fsm.current_state == FSMState.IDLE


def test_fsm_resume_without_board_enters_locating_board():
    """Resuming from PAUSED without locked board geometry must enter LOCATING_BOARD, not STABLE."""
    fsm = GameFSM()
    fsm.force_state(FSMState.LOCATING_BOARD)
    fsm.pause()
    assert fsm.current_state == FSMState.PAUSED

    fsm.resume(has_geometry=False, orientation_ready=False)
    assert fsm.current_state == FSMState.LOCATING_BOARD

    fsm.pause()
    fsm.resume(has_geometry=True, orientation_ready=False)
    assert fsm.current_state == FSMState.CALIBRATING


def test_calibrating_blocks_stable_until_orientation_calibrated(fsm_config: TemporalConfig):
    """Explicit BoardGeometry with orientation_calibrated=False must hold FSM in CALIBRATING."""
    fsm = GameFSM(config=fsm_config)
    uncalibrated_geom = BoardGeometry(
        corners=[(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)],
        is_valid=True,
        orientation_calibrated=False,
    )
    obs = BoardObservation.create_default(
        frame_id=1,
        timestamp=0.1,
        geometry_confidence=1.0,
        geometry=uncalibrated_geom,
    )
    assert fsm.update(obs) == FSMState.CALIBRATING
    # Must stay in CALIBRATING even on subsequent high-confidence frames until calibrated
    assert fsm.update(obs) == FSMState.CALIBRATING

    calibrated_geom = uncalibrated_geom.model_copy(update={"orientation_calibrated": True})
    obs_ok = obs.model_copy(update={"frame_id": 3, "timestamp": 0.3, "geometry": calibrated_geom})
    assert fsm.update(obs_ok) == FSMState.STABLE


def test_illegal_move_no_candidate_transitions_to_uncertain():
    """Physical board change with zero legal candidates must escalate to UNCERTAIN, not STABLE."""
    from src.reasoning.candidate_generator import LegalMoveCandidateGenerator
    from src.reasoning.decision_gate import DecisionGate
    from src.reasoning.game_state_manager import GameStateManager
    from src.reasoning.scorer import MoveScorer
    from src.core.datatypes import CellObservation, CHESS_SQUARES, PIECE_CLASSES
    from src.reasoning.candidate_generator import board_square_to_piece_class
    import chess

    cfg = TemporalConfig(settling_time_sec=0.05, verifying_stable_frames=1, debounce_frames=1)
    state_mgr = GameStateManager()
    cand_gen = LegalMoveCandidateGenerator()
    # Force empty candidate list to verify FSM handles DecisionStatus.NO_CANDIDATE cleanly
    cand_gen.generate_candidates = lambda board, obs: []  # type: ignore[method-assign]
    scorer = MoveScorer()
    gate = DecisionGate()
    fsm = GameFSM(
        config=cfg,
        candidate_generator=cand_gen,
        scorer=scorer,
        decision_gate=gate,
        game_state_manager=state_mgr,
    )
    calib_geom = BoardGeometry(
        corners=[(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)],
        is_valid=True,
        orientation_calibrated=True,
    )
    # Build illegal board observation: white queen moved straight from d1 to d5 illegally
    illegal_board = chess.Board()
    illegal_board.remove_piece_at(chess.D1)
    illegal_board.set_piece_at(chess.D5, chess.Piece(chess.QUEEN, chess.WHITE))
    cells = {}
    for sq in CHESS_SQUARES:
        cls_name = board_square_to_piece_class(illegal_board, chess.parse_square(sq))
        probs = {c: (0.99 if c == cls_name else 0.01 / (len(PIECE_CLASSES) - 1)) for c in PIECE_CLASSES}
        cells[sq] = CellObservation(
            square=sq,
            probabilities=probs,
            predicted_piece=cls_name,
            confidence=0.99,
            is_occupied=(cls_name != "empty"),
        )
    obs_calib = BoardObservation.create_default(frame_id=1, timestamp=0.1, geometry=calib_geom)
    assert fsm.update(obs_calib) == FSMState.CALIBRATING
    assert fsm.update(obs_calib) == FSMState.STABLE

    obs_move = BoardObservation(
        frame_id=2, timestamp=0.2, cells=cells, motion_detected=True, motion_level=0.2,
        hand_present=False, geometry_confidence=1.0, geometry=calib_geom,
    )
    assert fsm.update(obs_move) == FSMState.MOVING
    obs_settled = obs_move.model_copy(update={"frame_id": 3, "timestamp": 1.5, "motion_detected": False, "motion_level": 0.0})
    assert fsm.update(obs_settled) == FSMState.SETTLING
    obs_verify = obs_settled.model_copy(update={"frame_id": 4, "timestamp": 2.5})
    assert fsm.update(obs_verify) == FSMState.VERIFYING
    obs_commit = obs_settled.model_copy(update={"frame_id": 5, "timestamp": 2.6})
    state = fsm.update(obs_commit)
    assert state == FSMState.UNCERTAIN
    assert fsm.last_decision is not None
    assert "d1" in fsm.last_decision.uncertain_squares and "d5" in fsm.last_decision.uncertain_squares


def test_uncertain_freezes_without_motion_and_retriggers_on_motion():
    """Static frames during UNCERTAIN must not re-score/mutate decision; only physical motion re-verifies."""
    from src.reasoning.candidate_generator import LegalMoveCandidateGenerator
    from src.reasoning.decision_gate import DecisionGate
    from src.reasoning.game_state_manager import GameStateManager
    from src.reasoning.scorer import MoveScorer

    cfg = TemporalConfig(settling_time_sec=0.05, verifying_stable_frames=2, debounce_frames=1)
    score_calls = 0

    class CountingGate(DecisionGate):
        def evaluate(self, ranked, observation):
            nonlocal score_calls
            score_calls += 1
            return super().evaluate(ranked, observation)

    fsm = GameFSM(
        config=cfg,
        candidate_generator=LegalMoveCandidateGenerator(),
        scorer=MoveScorer(),
        decision_gate=CountingGate(),
        game_state_manager=GameStateManager(),
    )
    fsm.force_state(FSMState.UNCERTAIN)
    geom = BoardGeometry(
        corners=[(0.0, 0.0), (640.0, 0.0), (640.0, 640.0), (0.0, 640.0)],
        is_valid=True,
        orientation_calibrated=True,
    )
    for fid in range(1, 8):
        assert (
            fsm.update(
                BoardObservation.create_default(
                    frame_id=fid,
                    timestamp=fid * 0.2,
                    geometry=geom,
                    motion_detected=False,
                    hand_present=False,
                )
            )
            == FSMState.UNCERTAIN
        )
    assert score_calls == 0

    # Physical hand/piece motion re-enters MOVING cleanly for takeback/correction
    assert (
        fsm.update(
            BoardObservation.create_default(
                frame_id=8,
                timestamp=2.0,
                geometry=geom,
                motion_detected=True,
                motion_level=0.2,
            )
        )
        == FSMState.MOVING
    )
