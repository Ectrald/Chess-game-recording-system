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
