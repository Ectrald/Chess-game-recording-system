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
    DecisionResult,
    DecisionStatus,
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
        candidate_generator: Optional[Any] = None,
        scorer: Optional[Any] = None,
        decision_gate: Optional[Any] = None,
        game_state_manager: Optional[Any] = None,
        on_move_decision: Optional[Callable[[DecisionResult, BoardObservation], None]] = None,
    ) -> None:
        self.config = config or TemporalConfig()
        self.min_geometry_confidence = min_geometry_confidence
        self.on_transition = on_transition
        self.on_settled_observation = on_settled_observation
        self.candidate_generator = candidate_generator
        self.scorer = scorer
        self.decision_gate = decision_gate
        self.game_state_manager = game_state_manager
        self.on_move_decision = on_move_decision

        self._state: FSMState = FSMState.IDLE
        self._settle_start_time: Optional[float] = None
        self._verify_buffer: deque[BoardObservation] = deque(
            maxlen=max(self.config.verifying_stable_frames * 2, 16)
        )
        self._lost_frame_count: int = 0
        self._max_lost_frames: int = 3  # Tolerate up to 3 momentary frame drops
        self._uncertain_motion_streak: int = 0
        self._last_decision: Optional[DecisionResult] = None
        self._last_timestamp: float = 0.0

    @property
    def current_state(self) -> FSMState:
        """Returns the current FSM operational state."""
        return self._state

    @property
    def last_decision(self) -> Optional[DecisionResult]:
        """Returns the most recent decision result evaluated during verification."""
        return self._last_decision

    @property
    def settle_elapsed_time(self) -> float:
        """Elapsed time in seconds since settling debounce initiated."""
        if self._settle_start_time is None:
            return 0.0
        return max(0.0, self._last_timestamp - self._settle_start_time)

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
        self._last_decision = None

    def pause(self) -> None:
        """Pauses processing if in active recording state."""
        if self._state not in (FSMState.STOPPED, FSMState.PAUSED):
            self._set_state(FSMState.PAUSED, observation=None)

    def resume(self, has_geometry: bool = True, orientation_ready: bool = True) -> None:
        """Resumes processing from paused or uncertain state based on board readiness."""
        if self._state == FSMState.PAUSED:
            if not has_geometry:
                self._set_state(FSMState.LOCATING_BOARD, observation=None)
            elif not orientation_ready:
                self._set_state(FSMState.CALIBRATING, observation=None)
            else:
                self._set_state(FSMState.STABLE, observation=None)
        elif self._state == FSMState.UNCERTAIN:
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
            # Check orientation confirmation. When explicit geometry is attached,
            # require confirmed orientation calibration so we do not enter STABLE
            # with an uncalibrated board and flip sides mid-game.
            if observation.geometry is not None:
                orientation_ready = bool(observation.geometry.orientation_calibrated)
            else:
                orientation_ready = observation.geometry_confidence >= 0.70

            if orientation_ready and not active_disturbance:
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
                if self._settle_start_time is None:
                    self._settle_start_time = observation.timestamp
                elapsed = observation.timestamp - self._settle_start_time
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
                    # Verified consecutive stable sequence: evaluate move reasoning if configured
                    if (
                        self.decision_gate is not None
                        and self.game_state_manager is not None
                        and self.candidate_generator is not None
                        and self.scorer is not None
                    ):
                        board = self.game_state_manager.current_board
                        changed_sqs = self.candidate_generator.detect_changed_squares(board, observation)

                        if not changed_sqs:
                            logger.info("Verification finished: no visual board changes detected; returning to STABLE")
                            self._set_state(FSMState.STABLE, observation)
                        else:
                            candidates = self.candidate_generator.generate_candidates(board, observation)
                            ranked = self.scorer.rank_candidates(board, candidates, observation)
                            decision = self.decision_gate.evaluate(ranked, observation)
                            self._last_decision = decision

                            if decision.status == DecisionStatus.NO_CANDIDATE and changed_sqs:
                                decision.uncertain_squares = decision.uncertain_squares or sorted(changed_sqs)
                                decision.rejection_reason = (
                                    decision.rejection_reason
                                    or f"Unrecognized or illegal change on {', '.join(decision.uncertain_squares)}"
                                )

                            if self.on_move_decision:
                                try:
                                    self.on_move_decision(decision, observation)
                                except Exception as exc:
                                    logger.error("Error in on_move_decision callback: %s", exc, exc_info=True)

                            if decision.status == DecisionStatus.ACCEPTED and decision.accepted_hypothesis:
                                self.game_state_manager.commit(decision.accepted_hypothesis, observation)
                                self._set_state(FSMState.STABLE, observation)
                            elif decision.status in (
                                DecisionStatus.UNCERTAIN,
                                DecisionStatus.REJECTED,
                                DecisionStatus.NO_CANDIDATE,
                            ):
                                self._set_state(FSMState.UNCERTAIN, observation)
                            else:
                                self._set_state(FSMState.STABLE, observation)
                    else:
                        self._set_state(FSMState.STABLE, observation)

                    if self.on_settled_observation:
                        self.on_settled_observation(observation)
                    self._verify_buffer.clear()
                    self._settle_start_time = None

        elif self._state == FSMState.UNCERTAIN:
            strong_hand = hand and len(observation.get_occluded_squares() if hasattr(observation, "get_occluded_squares") else []) >= 2
            high_motion = observation.motion_detected and observation.motion_level >= max(
                0.10, self.config.motion_diff_threshold * 2.5
            )
            mod_motion = observation.motion_level >= (self.config.motion_diff_threshold * 1.45)
            if mod_motion or hand:
                self._uncertain_motion_streak += 1
            else:
                self._uncertain_motion_streak = 0

            # Immediate re-entry on unmistakable motion/hand; require 3 frames for moderate shimmer
            if high_motion or strong_hand or self._uncertain_motion_streak >= 3:
                self._uncertain_motion_streak = 0
                self._verify_buffer.clear()
                self._settle_start_time = None
                self._set_state(FSMState.MOVING, observation)
            else:
                self._verify_buffer.clear()

        observation.fsm_state = self._state
        self._last_timestamp = observation.timestamp
        return self._state
