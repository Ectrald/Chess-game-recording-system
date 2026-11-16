"""Finite State Machine (FSM) for chess game lifecycle and move debounce.

Implements game recording state transitions, settling debounce timers, and verification:
- STAGE_2.md, Section 4.3
- Explanatory Note, Section 2.4.3
- Figure 2.9 (Overall operational algorithm)
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional
from collections import deque

from src.core.config import TemporalConfig
from src.core.datatypes import (
    BoardObservation,
    FSMState,
)

logger = logging.getLogger(__name__)


class GameFSM:
    """State machine governing the chess game recording lifecycle.

    Maintains temporal debounce guarantees, preventing premature move hypothesis
    evaluations during manual piece displacement and optical occlusion.
    """

    def __init__(
        self,
        config: Optional[TemporalConfig] = None,
        min_geometry_confidence: float = 0.5,
        on_transition: Optional[Callable[[FSMState, FSMState, BoardObservation], None]] = None,
        on_settled_observation: Optional[Callable[[BoardObservation], None]] = None,
    ) -> None:
        self.config = config or TemporalConfig()
        self.min_geometry_confidence = min_geometry_confidence
        self.on_transition = on_transition
        self.on_settled_observation = on_settled_observation

        self._state: FSMState = FSMState.IDLE
        self._settle_start_time: Optional[float] = None
        self._verify_buffer: deque[BoardObservation] = deque(
            maxlen=max(self.config.verifying_stable_frames * 2, 16)
        )
        self._lost_frame_count: int = 0
        self._max_lost_frames: int = 3  # Tolerate up to 3 momentary frame drops

    @property
    def current_state(self) -> FSMState:
        """Returns the current FSM operational state."""
        return self._state

    @property
    def settle_elapsed_time(self) -> float:
        """Elapsed time in seconds since settling debounce initiated."""
        if self._settle_start_time is None:
            return 0.0
        return self._settle_start_time

    @property
    def verifying_count(self) -> int:
        """Number of consecutive stable observations currently in the verification buffer."""
        return len(self._verify_buffer)

    def reset(self) -> None:
        """Resets FSM to IDLE initial state and clears all internal buffers."""
        self._state = FSMState.IDLE
        self._settle_start_time = None
        self._verify_buffer.clear()
        self._lost_frame_count = 0

    def pause(self) -> None:
        """Pauses processing if in active recording state."""
        if self._state not in (FSMState.STOPPED, FSMState.PAUSED):
            self._set_state(FSMState.PAUSED, observation=None)

    def resume(self) -> None:
        """Resumes processing from paused state."""
        if self._state == FSMState.PAUSED:
            self._set_state(FSMState.STABLE, observation=None)

    def stop(self) -> None:
        """Permanently transitions FSM to STOPPED state."""
        self._set_state(FSMState.STOPPED, observation=None)

    def force_state(self, state: FSMState) -> None:
        """Forces state transition for administrative overrides and testing."""
        self._set_state(state, observation=None)

    def _set_state(self, new_state: FSMState, observation: Optional[BoardObservation]) -> None:
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            logger.info("FSM transition: %s -> %s", old_state.value, new_state.value)
            if self.on_transition and observation is not None:
                self.on_transition(old_state, new_state, observation)

    def _is_motion_present(self, observation: BoardObservation) -> bool:
        if observation.motion_detected:
            return True
        if observation.motion_level >= self.config.motion_diff_threshold:
            return True
        if observation.motion_metrics and observation.motion_metrics.is_motion_detected:
            return True
        return False

    def _is_hand_present(self, observation: BoardObservation) -> bool:
        if observation.hand_present:
            return True
        if observation.hand_detection and observation.hand_detection.is_hand_present:
            return True
        return False

    def _is_geometry_valid(self, observation: BoardObservation) -> bool:
        if observation.geometry_confidence < self.min_geometry_confidence:
            return False
        if observation.geometry is not None and not observation.geometry.is_valid:
            return False
        return True

    def update(self, observation: BoardObservation) -> FSMState:
        """Processes a new board observation and advances the state machine.

        Args:
            observation: BoardObservation containing motion, hand, and geometry metrics.

        Returns:
            Updated FSMState after processing the observation.
        """
        # Stamp current state on observation
        observation.fsm_state = self._state

        if self._state == FSMState.STOPPED or self._state == FSMState.PAUSED:
            return self._state

        # Check geometry validity across active tracking states
        geom_valid = self._is_geometry_valid(observation)
        if not geom_valid:
            if self._state not in (FSMState.IDLE, FSMState.LOCATING_BOARD, FSMState.BOARD_LOST):
                self._lost_frame_count += 1
                if self._lost_frame_count >= self._max_lost_frames:
                    self._set_state(FSMState.BOARD_LOST, observation)
                    self._settle_start_time = None
                    self._verify_buffer.clear()
                    observation.fsm_state = self._state
                    return self._state
            elif self._state == FSMState.IDLE:
                self._set_state(FSMState.LOCATING_BOARD, observation)
                observation.fsm_state = self._state
                return self._state
        else:
            self._lost_frame_count = 0

        # Motion & Hand status
        motion = self._is_motion_present(observation)
        hand = self._is_hand_present(observation)
        active_disturbance = motion or hand

        # FSM State Transition Logic
        if self._state == FSMState.IDLE:
            if geom_valid:
                self._set_state(FSMState.CALIBRATING, observation)
            else:
                self._set_state(FSMState.LOCATING_BOARD, observation)

        elif self._state == FSMState.LOCATING_BOARD:
            if geom_valid:
                self._set_state(FSMState.CALIBRATING, observation)

        elif self._state == FSMState.CALIBRATING:
            # Check orientation confirmation
            orientation_ready = (
                observation.geometry is not None and observation.geometry.orientation_calibrated
            ) or (observation.geometry_confidence >= 0.70)

            if orientation_ready:
                self._set_state(FSMState.STABLE, observation)

        elif self._state == FSMState.BOARD_LOST:
            if geom_valid:
                self._set_state(FSMState.LOCATING_BOARD, observation)

        elif self._state == FSMState.STABLE:
            if active_disturbance:
                self._set_state(FSMState.MOVING, observation)

        elif self._state == FSMState.MOVING:
            if not active_disturbance:
                self._settle_start_time = observation.timestamp
                self._set_state(FSMState.SETTLING, observation)

        elif self._state == FSMState.SETTLING:
            if active_disturbance:
                # Disturbance resumed before debounce completed; return to MOVING
                self._settle_start_time = None
                self._set_state(FSMState.MOVING, observation)
            else:
                # Check debounce elapsed time
                elapsed = observation.timestamp - (self._settle_start_time or observation.timestamp)
                if elapsed >= self.config.settling_time_sec:
                    self._verify_buffer.clear()
                    self._verify_buffer.append(observation)
                    self._set_state(FSMState.VERIFYING, observation)

        elif self._state == FSMState.VERIFYING:
            if active_disturbance:
                # Disturbance during verification invalidates candidate move
                self._settle_start_time = None
                self._verify_buffer.clear()
                self._set_state(FSMState.MOVING, observation)
            else:
                self._verify_buffer.append(observation)
                if len(self._verify_buffer) >= self.config.verifying_stable_frames:
                    # Verified consecutive stable sequence
                    self._set_state(FSMState.STABLE, observation)
                    if self.on_settled_observation:
                        self.on_settled_observation(observation)
                    self._verify_buffer.clear()
                    self._settle_start_time = None

        elif self._state == FSMState.UNCERTAIN:
            if not active_disturbance and geom_valid:
                self._verify_buffer.append(observation)
                if len(self._verify_buffer) >= self.config.verifying_stable_frames:
                    self._set_state(FSMState.STABLE, observation)
                    self._verify_buffer.clear()

        observation.fsm_state = self._state
        return self._state
