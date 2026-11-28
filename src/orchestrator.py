"""End-to-End System Orchestrator for Chess Game Recording System.

Coordinates video acquisition, board localization, homography transformations,
optical motion tracking, hand occlusion detection, neural piece classification,
finite state machine (FSM) debouncing, symbolic move reasoning, transactional
event persistence, evidence preservation, and asynchronous Stockfish UCI analysis.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import uuid

import chess
import cv2
import numpy as np

from src.analysis.engine_service import AsyncEngineService
from src.core.config import AppConfig, load_config
from src.core.datatypes import (
    CHESS_SQUARES,
    PIECE_CLASSES,
    AnalysisResult,
    BoardGeometry,
    BoardObservation,
    BoardOrientation,
    DecisionResult,
    DecisionStatus,
    FSMState,
    Frame,
    HandDetection,
    MoveEvent,
    MoveHypothesis,
    PipelineState,
    SessionManifest,
)
from src.ml.piece_classifier import (
    BasePieceClassifier,
    DINOv2PatchChangeDetector,
    MockPieceClassifier,
    ONNXOccupancyClassifier,
    ONNXPieceClassifier,
    WholeBoardChessQClassifier,
    create_piece_classifier,
    probabilities_to_cell_observation,
)
from src.persistence.event_repository import JSONLEventRepository
from src.persistence.evidence_writer import EvidenceWriter
from src.reasoning.candidate_generator import (
    LegalMoveCandidateGenerator,
    board_square_to_piece_class,
)
from src.reasoning.decision_gate import DecisionGate
from src.reasoning.fsm import GameFSM
from src.reasoning.game_state_manager import GameStateManager
from src.reasoning.scorer import MoveScorer
from src.vision.board_localizer import (
    BaseBoardLocalizer,
    BoardGeometryTracker,
    ManualCornerLocalizer,
    create_board_localizer,
)
from src.vision.corner_refinement import order_corners_canonical
from src.vision.hand_detector import BaseHandDetector, MediaPipeHandDetector, create_hand_detector
from src.vision.homography import GeometryTransformer
from src.vision.motion_detector import MotionDetector
from src.vision.occupancy_tracker import BoardOccupancyTracker
from src.vision.orientation import OrientationResolver
from src.vision.roi_extractor import CellRoiExtractor

logger = logging.getLogger(__name__)


class SystemOrchestrator:
    """Thread-safe unified facade orchestrating real-time vision-to-PGN pipeline."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        session_id: Optional[str] = None,
        board_localizer: Optional[BaseBoardLocalizer] = None,
        piece_classifier: Optional[BasePieceClassifier] = None,
        engine_service: Optional[AsyncEngineService] = None,
        hand_detector: Optional[BaseHandDetector] = None,
    ) -> None:
        """Initializes orchestrator subsystems with configuration and dependency injection.

        Args:
            config: Optional AppConfig instance (loads default if None).
            session_id: Unique session ID string (auto-generates timestamped ID if None).
            board_localizer: Optional custom board localizer instance.
            piece_classifier: Optional custom piece classifier instance.
            engine_service: Optional custom asynchronous engine worker service.
            hand_detector: Optional custom hand detector instance.
        """
        self.config = config or load_config()
        self._lock = threading.RLock()

        # Session persistence directory initialization
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or f"session_{ts_str}_{uuid.uuid4().hex[:6]}"
        self.session_dir = Path(self.config.persistence.base_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.white_player = "White Player"
        self.black_player = "Black Player"
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._is_paused = False
        self._active_uncertainty = False

        # ---------------------------------------------------------------------
        # Vision Subsystems
        # ---------------------------------------------------------------------
        self.board_localizer = board_localizer or create_board_localizer(
            config=self.config.vision
        )
        self.orientation_resolver = OrientationResolver(config=self.config.vision.orientation)
        self.geometry_transformer = GeometryTransformer(
            target_size=self.config.vision.target_board_size
        )
        self.roi_extractor = CellRoiExtractor(
            board_size=self.config.vision.target_board_size,
            parallax_alpha=self.config.vision.parallax_height_alpha,
        )
        self.motion_detector = MotionDetector(
            config=self.config.temporal,
            board_size=self.config.vision.target_board_size,
        )
        self.hand_detector = hand_detector or create_hand_detector(
            config=self.config.vision.hand_detector
        )

        # ---------------------------------------------------------------------
        # Machine Learning Subsystems
        # ---------------------------------------------------------------------
        self._custom_piece_classifier = piece_classifier is not None
        self.piece_classifier = piece_classifier or create_piece_classifier(
            config=self.config.piece_classifier
        )
        self.whole_board_classifier = WholeBoardChessQClassifier(
            model_path=self.config.piece_classifier.chessq_model_path,
            device=self.config.piece_classifier.device,
            occupancy_threshold=self.config.piece_classifier.occupancy_threshold,
        )
        self.dino_change_detector = DINOv2PatchChangeDetector(
            model_path=self.config.piece_classifier.dinov2_model_path,
            device=self.config.piece_classifier.device,
        )

        # ---------------------------------------------------------------------
        # Persistence and Analysis Subsystems
        # ---------------------------------------------------------------------
        self.event_repo = JSONLEventRepository(
            session_dir=self.session_dir,
            sync_to_disk=self.config.persistence.sync_to_disk,
        )
        self.evidence_writer = EvidenceWriter(
            evidence_dir=self.session_dir / "evidence",
            policy=self.config.persistence.evidence_retention,
            format=self.config.persistence.evidence_format,
        )
        self.engine_service = engine_service or AsyncEngineService(
            config=self.config.analysis,
            on_analysis_completed=self._on_engine_analysis_completed,
        )

        # ---------------------------------------------------------------------
        # Reasoning and State Management Subsystems
        # ---------------------------------------------------------------------
        self.candidate_generator = LegalMoveCandidateGenerator()
        self.move_scorer = MoveScorer(config=self.config.decision)
        self.decision_gate = DecisionGate(config=self.config.decision)
        self.state_manager = GameStateManager(on_move_committed=self._on_move_committed)

        self.fsm = GameFSM(
            config=self.config.temporal,
            candidate_generator=self.candidate_generator,
            scorer=self.move_scorer,
            decision_gate=self.decision_gate,
            game_state_manager=self.state_manager,
            on_move_decision=self._on_move_decision,
        )

        # Dynamic State Variables
        self._board_geometry: Optional[BoardGeometry] = None
        self.geometry_tracker = BoardGeometryTracker()
        self.occupancy_tracker = BoardOccupancyTracker()
        occ_model_path = Path(self.config.piece_classifier.model_path).parent / "occupancy_classifier.onnx"
        self.occupancy_classifier = ONNXOccupancyClassifier(
            model_path=occ_model_path,
            device=self.config.piece_classifier.device,
        )
        self._orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM
        self._last_confirmed_orientation: Optional[BoardOrientation] = None
        self._orientation_calibrated = False
        self._last_observation: Optional[BoardObservation] = None
        self._last_cell_obs: Optional[Dict[str, Any]] = None
        self._last_stable_raw_obs: Optional[Dict[str, Any]] = None
        self._last_warped_board: Optional[np.ndarray] = None
        self._stable_warped_board: Optional[np.ndarray] = None
        self._stable_warped_history: deque[np.ndarray] = deque(maxlen=8)
        self._pre_commit_warped_board: Optional[np.ndarray] = None
        self._last_committed_move_squares: List[str] = []
        self._pending_two_ply_second_move: Optional[chess.Move] = None
        self._pending_rapid_queue: deque[chess.Move] = deque()
        self._pending_two_ply_raw_obs: Optional[Dict[str, Any]] = None
        self._pending_physical_anomaly: Optional[List[str]] = None
        self._last_hand_detection: HandDetection = HandDetection(is_hand_present=False, occluded_squares=[])
        self._frame_counter: int = 0
        self._last_raw_frame: Optional[Frame] = None
        self._recovery_anomaly_since: Optional[float] = None
        self._last_committed_event: Optional[MoveEvent] = None
        self._last_decision: Optional[DecisionResult] = None
        self._latest_eval: Optional[AnalysisResult] = None
        self._illegal_attempts_count: int = 0
        self._last_logged_illegal_key: Optional[str] = None
        self._stale_event_ids: Set[str] = set()

        # Frame Rate Telemetry
        self._last_frame_time: Optional[float] = None
        self._current_fps: float = 0.0

        logger.info(
            "SystemOrchestrator initialized session=%s base_dir=%s",
            self.session_id,
            self.session_dir,
        )

    # -------------------------------------------------------------------------
    # Callbacks and Subsystem Event Sinks
    # -------------------------------------------------------------------------
    def _on_move_committed(self, event: MoveEvent) -> None:
        """Invoked atomically whenever GameStateManager commits a verified move."""
        logger.info(
            "Orchestrator move committed ply=%d: %s (%s)",
            event.ply,
            event.move_san,
            event.move_uci,
        )
        if self._stable_warped_board is not None:
            self._pre_commit_warped_board = self._stable_warped_board.copy()
        try:
            self._last_committed_move_squares = self.candidate_generator.get_affected_squares(
                chess.Board(event.fen_before),
                chess.Move.from_uci(event.move_uci),
            )
        except Exception:
            self._last_committed_move_squares = []
        self._recovery_anomaly_since = None
        self._pending_physical_anomaly = None
        self._last_logged_illegal_key = None
        self._last_committed_event = event
        self._active_uncertainty = False

        # 1. Write event to cryptographic hash-chained JSONL
        try:
            self.event_repo.append_event(event)
        except Exception as exc:
            logger.error("Failed to append event to repository: %s", exc, exc_info=True)

        # 2. Save photographic evidence and metadata sidecar
        try:
            if self._last_observation is not None:
                self.evidence_writer.save_evidence(
                    event=event,
                    observation=self._last_observation,
                    board_image=self._last_warped_board,
                )
        except Exception as exc:
            logger.error("Failed to save move evidence: %s", exc, exc_info=True)

        # 3. Dispatch move to non-blocking Stockfish analysis queue
        try:
            self.engine_service.submit_move(event)
        except Exception as exc:
            logger.error("Failed to submit move for analysis: %s", exc, exc_info=True)

    def _on_move_decision(self, decision: DecisionResult, observation: BoardObservation) -> None:
        """Invoked when DecisionGate reaches a conclusion during VERIFYING phase."""
        if self._pending_physical_anomaly:
            anomaly_sqs = list(self._pending_physical_anomaly)
            self._pending_physical_anomaly = None
            decision.status = DecisionStatus.UNCERTAIN
            decision.accepted_hypothesis = None
            decision.uncertain_squares = anomaly_sqs
            decision.rejection_reason = (
                f"Illegal move or unrecognized board change on squares {', '.join(anomaly_sqs)}. "
                f"Restore physical pieces to auto-resume or enter valid UCI."
            )
            decision.top_hypotheses = []

        self._last_decision = decision
        if decision.status in (DecisionStatus.UNCERTAIN, DecisionStatus.REJECTED, DecisionStatus.NO_CANDIDATE):
            self._active_uncertainty = True
            top_score = decision.top_hypotheses[0].total_score if decision.top_hypotheses else 0.0
            is_illegal = (
                decision.status in (DecisionStatus.REJECTED, DecisionStatus.NO_CANDIDATE)
                or not decision.top_hypotheses
                or top_score < 0.48
            )
            if is_illegal:
                cur_fen = self.state_manager.current_fen
                sqs_tag = decision.uncertain_squares if decision.uncertain_squares else ["board"]
                illegal_key = f"{self.state_manager.ply}:{','.join(sqs_tag)}:{cur_fen}"
                if illegal_key != self._last_logged_illegal_key:
                    self._last_logged_illegal_key = illegal_key
                    self._illegal_attempts_count += 1
                    try:
                        sq_summary = ",".join(sqs_tag[:4])
                        ill_ev = MoveEvent(
                            event_type="illegal_attempt",
                            ply=max(1, self.state_manager.ply),
                            fen_before=cur_fen,
                            move_uci="0000",
                            move_san=f"ILLEGAL:{sq_summary}",
                            fen_after=cur_fen,
                            confidence=0.0,
                            timestamp=float(observation.timestamp),
                            evidence_frame_id=int(observation.frame_id),
                        )
                        self.event_repo.append_event(ill_ev)
                    except Exception as exc:
                        logger.warning("Failed to append illegal_attempt audit event: %s", exc)
            logger.warning(
                "Orchestrator: DecisionGate reported %s (top candidate: %s)",
                decision.status.value.upper(),
                decision.accepted_hypothesis.move_uci if decision.accepted_hypothesis else "None",
            )
        else:
            self._active_uncertainty = False

    def _on_engine_analysis_completed(self, result: AnalysisResult) -> None:
        """Invoked asynchronously when Stockfish finishes move evaluation."""
        with self._lock:
            if result.event_id in self._stale_event_ids:
                logger.debug("Ignored stale engine evaluation for event %s", result.event_id)
                return
            self._latest_eval = result
        logger.debug(
            "Orchestrator: Engine evaluated ply=%d (%s) quality=%s",
            result.ply,
            result.move_uci,
            result.quality.value if result.quality else "None",
        )

    # -------------------------------------------------------------------------
    # Lifecycle Controls
    # -------------------------------------------------------------------------
    def start_session(
        self,
        white_player: str = "White Player",
        black_player: str = "Black Player",
    ) -> SessionManifest:
        """Initializes a new recording session and starts background services.

        Args:
            white_player: Name of white player.
            black_player: Name of black player.

        Returns:
            Initial SessionManifest descriptor.
        """
        with self._lock:
            self.white_player = white_player
            self.black_player = black_player
            self.engine_service.start()

            manifest = SessionManifest(
                session_id=self.session_id,
                created_at=self._created_at,
                player_white=self.white_player,
                player_black=self.black_player,
                initial_fen=self.state_manager.current_fen,
                current_fen=self.state_manager.current_fen,
                total_plies=self.state_manager.ply,
                journal_path=str(self.event_repo.file_path.relative_to(self.session_dir)),
                pgn_path="game.pgn",
                journal_checksum="",
                is_completed=False,
            )
            return manifest

    def process_frame(self, frame: Frame) -> PipelineState:
        """Processes a single video frame through all perception and reasoning layers.

        Args:
            frame: Frame container containing timestamp, image [H, W, 3], and frame_id.

        Returns:
            Complete snapshot of the system state (PipelineState).
        """
        with self._lock:
            now = time.perf_counter()
            if self._last_frame_time is not None:
                dt = now - self._last_frame_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self._current_fps = 0.85 * self._current_fps + 0.15 * inst_fps if self._current_fps > 0 else inst_fps
            self._last_frame_time = now
            self._last_raw_frame = frame

            # If paused by operator, preserve current state (no auto-calibration or move tracking)
            if self._is_paused:
                return self._build_pipeline_state(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    fsm_state=self.fsm.current_state,
                    board_detected=(self._board_geometry is not None),
                    hand_detected=False,
                    motion_level=0.0,
                )

            self._frame_counter += 1

            # -----------------------------------------------------------------
            # Step 1: Board Localization & Drift Tracking (ADR-019)
            # -----------------------------------------------------------------
            if self._board_geometry is None:
                geom = self.board_localizer.locate(frame)
                if geom is not None and geom.is_valid:
                    self._board_geometry = geom
                    self.geometry_tracker.reset()
                    logger.info("Board geometry successfully locked: corners=%s", geom.corners)
            elif (
                self.fsm.current_state in (FSMState.STABLE, FSMState.UNCERTAIN)
                and not self._last_hand_detection.is_hand_present
                and self.motion_detector.smoothed_motion_score < (self.config.temporal.motion_diff_threshold * 1.2)
                and self._board_geometry.inverse_homography is not None
                and self._board_geometry.detection_method not in ("manual", "aruco", "synthetic")
                and (self._frame_counter % max(1, self.config.temporal.drift_check_interval_frames) == 0)
            ):
                new_geom, drift_status = self.geometry_tracker.check_drift(
                    frame,
                    self._board_geometry,
                    min_area_ratio=0.05,
                )
                if drift_status == "updated" and new_geom is not None and new_geom.homography_matrix:
                    h_old_inv = np.array(self._board_geometry.inverse_homography, dtype=np.float64)
                    h_new = np.array(new_geom.homography_matrix, dtype=np.float64)
                    self._reproject_reference_boards(h_old_inv, h_new)
                    self._board_geometry = new_geom.model_copy(
                        update={
                            "orientation_calibrated": self._orientation_calibrated,
                            "orientation": self._orientation,
                        }
                    )
                    self._last_cell_obs = None
                    self._last_stable_raw_obs = None
                    self.fsm._verify_buffer.clear()
                    self.motion_detector.reset()
                    logger.info("Board geometry drift updated: corners=%s", new_geom.corners)
                elif drift_status == "lost":
                    self._board_geometry = None
                    self._orientation_calibrated = False
                    self._last_cell_obs = None
                    self._last_stable_raw_obs = None
                    self._stable_warped_board = None
                    self._stable_warped_history.clear()
                    self._last_warped_board = None
                    self._pre_commit_warped_board = None
                    self.dino_change_detector.reset()
                    self.geometry_tracker.reset()
                    logger.warning("Board geometry lost during drift tracking; requesting relocalization.")

            board_detected = self._board_geometry is not None

            # -----------------------------------------------------------------
            # Step 2: Hand Detection & Occlusion (motion-gated idle skip)
            # -----------------------------------------------------------------
            detector_impl = getattr(self.hand_detector, "underlying_detector", self.hand_detector)
            can_skip_hand = (
                isinstance(detector_impl, MediaPipeHandDetector)
                and self.fsm.current_state == FSMState.STABLE
                and self.motion_detector.smoothed_motion_score < 0.005
                and not self._last_hand_detection.is_hand_present
                and self._frame_counter > 10
                and (self._frame_counter % max(1, self.config.temporal.hand_idle_skip_stride) != 0)
            )
            if can_skip_hand:
                hand_detection = self._last_hand_detection.model_copy(
                    update={"is_hand_present": False, "occluded_squares": []}
                )
            else:
                hand_detection = self.hand_detector.detect(frame, geometry=self._board_geometry)
                self._last_hand_detection = hand_detection
            is_hand_present = hand_detection.is_hand_present
            occluded_squares = hand_detection.occluded_squares

            # -----------------------------------------------------------------
            # Step 3: Board Warping, Orientation & Dynamic Tracking
            # -----------------------------------------------------------------
            warped: Optional[np.ndarray] = None
            motion_level = 0.0
            motion_detected = False
            cell_obs: Dict[str, Any] = {}

            if board_detected and self._board_geometry and self._board_geometry.homography_matrix:
                h_mat = np.array(self._board_geometry.homography_matrix, dtype=np.float64)
                warped = GeometryTransformer.warp_board(
                    frame.image,
                    h_mat,
                    target_size=self.config.vision.target_board_size,
                )
                self._last_warped_board = warped

                # Optical Motion Detection MUST run before delayed orientation calibration
                motion_metrics = self.motion_detector.process_board_image(
                    warped,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    orientation=self._orientation,
                )
                motion_detected = motion_metrics.is_motion_detected
                motion_level = motion_metrics.global_motion_score

                # Orientation Calibration (wait for calm, hand-free frame before locking orientation)
                if self._orientation_calibrated and not self._board_geometry.orientation_calibrated:
                    self._board_geometry = self._board_geometry.model_copy(
                        update={"orientation_calibrated": True, "orientation": self._orientation}
                    )
                elif not self._orientation_calibrated:
                    prev_ori = self._orientation
                    if self._board_geometry.orientation_calibrated:
                        if self._board_geometry.orientation != BoardOrientation.UNKNOWN:
                            self._orientation = self._board_geometry.orientation
                        self._last_confirmed_orientation = self._orientation
                        self._orientation_calibrated = True
                    elif not is_hand_present and not motion_detected:
                        if self.state_manager.ply > 0 and self._last_confirmed_orientation is not None:
                            detected_ori = self._last_confirmed_orientation
                        else:
                            padded_ori = GeometryTransformer.warp_padded_board(
                                frame.image,
                                h_mat,
                                target_size=self.config.vision.target_board_size,
                                padding=80,
                            )
                            detected_ori = self._resolve_orientation_non_brightness(warped, padded_ori)
                        if detected_ori != BoardOrientation.UNKNOWN:
                            self._orientation = detected_ori
                            self._last_confirmed_orientation = detected_ori
                            self._orientation_calibrated = True
                            self._board_geometry = self._board_geometry.model_copy(
                                update={"orientation_calibrated": True, "orientation": detected_ori}
                            )
                            logger.info("Orientation calibrated: %s", self._orientation.value)
                    if self._orientation != prev_ori:
                        self._last_cell_obs = None
                        self._last_stable_raw_obs = None
                        self.motion_detector.reset()
                        motion_metrics = self.motion_detector.process_board_image(
                            warped,
                            frame_id=frame.frame_id,
                            timestamp=frame.timestamp,
                            orientation=self._orientation,
                        )
                        motion_detected = motion_metrics.is_motion_detected
                        motion_level = motion_metrics.global_motion_score

                # Extract square ROIs and classify cells (ADR-018 + Sparse Changed-Cell Inference)
                curr_fsm = self.fsm.current_state
                should_classify = (
                    self._last_cell_obs is None
                    or (
                        curr_fsm == FSMState.VERIFYING
                        and not is_hand_present
                        and not motion_detected
                        and (len(self.fsm._verify_buffer) + 1) >= self.config.temporal.verifying_stable_frames
                    )
                )
                if should_classify:
                    # Check physical takeback before committing/escalating a new move
                    if (
                        curr_fsm in (FSMState.VERIFYING, FSMState.UNCERTAIN)
                        and self._check_physical_takeback(warped)
                    ):
                        self.rollback_move()
                        if self._pre_commit_warped_board is not None:
                            self._stable_warped_board = self._pre_commit_warped_board.copy()
                            self._stable_warped_history.clear()
                            self._stable_warped_history.append(self._stable_warped_board.copy())
                        self._pre_commit_warped_board = None
                        self._last_committed_move_squares = []
                        self._pending_two_ply_second_move = None
                        self.fsm.force_state(FSMState.STABLE)
                        self.fsm._verify_buffer.clear()
                        self.fsm._settle_start_time = None
                        self._active_uncertainty = False
                        logger.info("Physical takeback detected; rolled back last move and returned to STABLE.")
                        return self._build_pipeline_state(
                            frame_id=frame.frame_id,
                            timestamp=frame.timestamp,
                            fsm_state=FSMState.STABLE,
                            board_detected=board_detected,
                            hand_detected=is_hand_present,
                            motion_level=motion_level,
                        )

                    # Unclipped padded warp preserves heads of tall edge pieces on a/h/1/8 (ADR-021)
                    padded_warped = GeometryTransformer.warp_padded_board(
                        frame.image,
                        h_mat,
                        target_size=self.config.vision.target_board_size,
                        padding=80,
                    )
                    rois = self.roi_extractor.extract_all_rois(padded_warped, orientation=self._orientation)
                    patches = {sq: item[1] for sq, item in rois.items()}

                    deltas_info: Optional[Tuple[Dict[str, float], float, float]] = None
                    used_whole_board = False
                    if self._should_use_whole_board():
                        cell_obs, _ = self.whole_board_classifier.classify_board_image(
                            padded_warped,
                            orientation=self._orientation,
                            add_padding=0,
                        )
                        used_whole_board = True
                    elif (
                        isinstance(self.piece_classifier, ONNXPieceClassifier)
                        and curr_fsm in (FSMState.VERIFYING, FSMState.UNCERTAIN)
                        and self._last_cell_obs is not None
                        and self._stable_warped_board is not None
                        and self._stable_warped_board.shape == warped.shape
                    ):
                        deltas_info = self._compute_base_core_deltas(warped, self._stable_warped_board)
                        deltas, _, change_thresh = deltas_info
                        sparse_targets = {
                            sq for sq, d in deltas.items() if d >= change_thresh * 0.45
                        }
                        # Include image-below neighbor of any changed square (upward parallax spillover)
                        sq_pos, pos_to_sq = self.occupancy_tracker.extract_grid_mapping(
                            warped.shape, self._orientation, self.roi_extractor
                        )
                        for sq in list(sparse_targets):
                            r, c = sq_pos[sq]
                            down_sq = pos_to_sq.get((r + 1, c))
                            if down_sq is not None:
                                sparse_targets.add(down_sq)
                        sparse_targets.update(occluded_squares)
                        sparse_targets.update(
                            sq for sq, prev_obs in self._last_cell_obs.items() if prev_obs.is_occluded
                        )
                        if not sparse_targets:
                            cell_obs = {
                                sq: obs.model_copy(update={"is_occluded": (sq in occluded_squares)})
                                for sq, obs in self._last_cell_obs.items()
                            }
                        elif len(sparse_targets) <= 12:
                            updated_obs = self.piece_classifier.classify_squares(
                                patches, squares=sorted(sparse_targets)
                            )
                            cell_obs = {
                                sq: updated_obs[sq] if sq in updated_obs else self._last_cell_obs[sq].model_copy(
                                    update={"is_occluded": (sq in occluded_squares)}
                                )
                                for sq in CHESS_SQUARES
                            }
                        else:
                            cell_obs = self.piece_classifier.classify_board(patches)
                    else:
                        cell_obs = self.piece_classifier.classify_board(patches)

                    # Preserve raw (un-blended) NN classifier output for subsequent frame caching
                    self._last_cell_obs = {
                        sq: obs.model_copy(deep=True) for sq, obs in cell_obs.items()
                    }
                    if curr_fsm in (FSMState.STABLE, FSMState.CALIBRATING):
                        self._last_stable_raw_obs = self._last_cell_obs
                    if (
                        curr_fsm in (FSMState.VERIFYING, FSMState.UNCERTAIN)
                        and self._stable_warped_board is not None
                        and self._stable_warped_board.shape == warped.shape
                    ):
                        if used_whole_board:
                            cell_obs = self._refine_with_whole_board_hybrid(
                                cell_obs=cell_obs,
                                warped_after=warped,
                                warped_before=self._stable_warped_board,
                            )
                        else:
                            cell_obs = self._refine_zero_shot_with_temporal_delta(
                                cell_obs=cell_obs,
                                rois=rois,
                                warped_after=warped,
                                warped_before=self._stable_warped_board,
                                precomputed_deltas=deltas_info,
                            )
                else:
                    # Copy cached cell observations so occlusion/motion flags update per-frame
                    cell_obs = {
                        sq: obs.model_copy(update={"is_occluded": (sq in occluded_squares)})
                        for sq, obs in self._last_cell_obs.items()
                    }

                if (
                    curr_fsm == FSMState.STABLE
                    and not motion_detected
                    and not is_hand_present
                    and motion_level < (self.config.temporal.motion_diff_threshold * 0.5)
                ):
                    self._stable_warped_history.append(warped.copy())
                    self._stable_warped_board = self._stable_warped_history[0]
                    if self._last_cell_obs is not None:
                        self._last_stable_raw_obs = self._last_cell_obs

                # Attach per-square hand occlusion ratios and motion intensity
                occ_ratios = hand_detection.occlusion_ratios or {}
                sq_motion = motion_metrics.square_motion_scores or {}
                for sq in CHESS_SQUARES:
                    if sq in cell_obs:
                        cell_obs[sq] = cell_obs[sq].model_copy(
                            update={
                                "is_occluded": (sq in occluded_squares) or cell_obs[sq].is_occluded,
                                "occlusion_ratio": float(occ_ratios.get(sq, 1.0 if sq in occluded_squares else 0.0)),
                                "motion_level": float(sq_motion.get(sq, 0.0)),
                            }
                        )

                geom_conf = float(getattr(self._board_geometry, "confidence", 1.0))
                if not self._board_geometry.is_valid:
                    geom_conf = 0.0

                observation = BoardObservation(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    cells=cell_obs,
                    motion_detected=motion_detected,
                    motion_level=motion_level,
                    hand_present=is_hand_present,
                    geometry_confidence=max(0.0, min(1.0, geom_conf)),
                    corners=self._board_geometry.corners,
                    geometry=self._board_geometry,
                    hand_detection=hand_detection,
                )
            else:
                # Board not yet located
                observation = BoardObservation.create_default(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    motion_detected=False,
                    motion_level=0.0,
                    hand_present=is_hand_present,
                    geometry_confidence=0.0,
                    hand_detection=hand_detection,
                )

            self._last_observation = observation

            # -----------------------------------------------------------------
            # Step 4: Advance State Machine
            # -----------------------------------------------------------------
            prev_fsm = self.fsm.current_state
            fsm_state = self.fsm.update(observation)

            # Complete pending rapid multi-ply sequence (2..3 blitz plies settled within one calm window)
            if (
                (self._pending_two_ply_second_move is not None or self._pending_rapid_queue)
                and prev_fsm != FSMState.STABLE
                and fsm_state == FSMState.STABLE
            ):
                queued: List[chess.Move] = []
                if self._pending_two_ply_second_move is not None:
                    queued.append(self._pending_two_ply_second_move)
                while self._pending_rapid_queue:
                    queued.append(self._pending_rapid_queue.popleft())
                raw_cells_2 = self._pending_two_ply_raw_obs
                self._pending_two_ply_second_move = None
                self._pending_two_ply_raw_obs = None
                obs_base = (
                    observation.model_copy(update={"cells": raw_cells_2})
                    if raw_cells_2 is not None
                    else observation
                )
                for idx_q, m_next in enumerate(queued):
                    try:
                        board_cur = self.state_manager.current_board
                        if m_next not in board_cur.legal_moves:
                            break
                        # Compute virtual intermediate observation covering remaining queued moves cleanly
                        remaining_after = queued[idx_q + 1 :]
                        b_step = board_cur.copy()
                        b_step.push(m_next)
                        aff_cur = self.candidate_generator.get_affected_squares(board_cur, m_next)
                        if remaining_after:
                            cells_step = {sq: o.model_copy(deep=True) for sq, o in obs_base.cells.items()}
                            # Keep future-ply squares matching post-m_next intermediate state while verifying m_next
                            b_rem = b_step.copy()
                            for mf in remaining_after:
                                for sqf in self.candidate_generator.get_affected_squares(b_rem, mf):
                                    if sqf not in aff_cur:
                                        mid_cls = board_square_to_piece_class(b_step, chess.parse_square(sqf))
                                        mv_vec = np.full(len(PIECE_CLASSES), 0.01 / max(1, len(PIECE_CLASSES) - 1), dtype=np.float32)
                                        mv_vec[PIECE_CLASSES.index(mid_cls)] = 0.99
                                        cells_step[sqf] = probabilities_to_cell_observation(
                                            sqf, mv_vec, occupancy_threshold=self.config.piece_classifier.occupancy_threshold
                                        )
                                b_rem.push(mf)
                            obs_step = obs_base.model_copy(update={"cells": cells_step})
                        else:
                            obs_step = obs_base

                        hypo_q = self.move_scorer.score_hypothesis(board_cur, m_next, aff_cur, obs_step)
                        dec_q = self.decision_gate.evaluate([hypo_q], obs_step)
                        if dec_q.status == DecisionStatus.ACCEPTED:
                            self.state_manager.commit(hypo_q, obs_step)
                            self._last_decision = dec_q
                            logger.info("Committed rapid follow-up ply: %s", m_next.uci())
                        else:
                            logger.info(
                                "Follow-up ply '%s' gated (%s): %s",
                                m_next.uci(),
                                dec_q.status.value,
                                dec_q.rejection_reason,
                            )
                            break
                    except Exception as exc:
                        logger.warning("Could not commit rapid follow-up ply '%s': %s", m_next.uci(), exc)
                        break
            elif fsm_state not in (FSMState.VERIFYING, FSMState.UNCERTAIN):
                self._pending_two_ply_second_move = None
                self._pending_rapid_queue.clear()
                self._pending_two_ply_raw_obs = None

            # Grace window for knocked-over pieces / single-square adjustment before surfacing modal
            if fsm_state == FSMState.UNCERTAIN:
                last_dec = self.fsm.last_decision or self._last_decision
                is_anomaly = (
                    last_dec is None
                    or len(last_dec.uncertain_squares) > 4
                    or len(last_dec.uncertain_squares) <= 1
                )
                if is_anomaly and self.config.temporal.recovery_grace_sec > 0.0:
                    if self._recovery_anomaly_since is None:
                        self._recovery_anomaly_since = frame.timestamp
                    elapsed_rec = frame.timestamp - self._recovery_anomaly_since
                    self._active_uncertainty = elapsed_rec >= self.config.temporal.recovery_grace_sec
                else:
                    self._active_uncertainty = True
            else:
                self._recovery_anomaly_since = None
                self._active_uncertainty = False

            if (
                prev_fsm == FSMState.STABLE
                and fsm_state != FSMState.STABLE
                and self._stable_warped_history
            ):
                # Freeze the oldest clean STABLE frame (~0.6-0.8s before motion onset)
                self._stable_warped_board = self._stable_warped_history[0].copy()
            elif (
                prev_fsm != FSMState.STABLE
                and fsm_state == FSMState.STABLE
                and warped is not None
                and not motion_detected
                and not is_hand_present
            ):
                self._stable_warped_history.clear()
                self._stable_warped_history.append(warped.copy())
                self._stable_warped_board = self._stable_warped_history[0]

            return self._build_pipeline_state(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                fsm_state=fsm_state,
                board_detected=board_detected,
                hand_detected=is_hand_present,
                motion_level=motion_level,
            )

    def _reproject_reference_boards(self, h_old_inv: np.ndarray, h_new: np.ndarray) -> None:
        """Reprojects stored reference board snapshots when camera/table drifts slightly (ADR-019)."""
        size = self.config.vision.target_board_size
        transform = h_new @ h_old_inv
        reproj_hist: deque[np.ndarray] = deque(maxlen=self._stable_warped_history.maxlen or 8)
        for snap in self._stable_warped_history:
            reproj_hist.append(cv2.warpPerspective(snap, transform, (size, size)))
        self._stable_warped_history = reproj_hist
        if self._stable_warped_board is not None:
            self._stable_warped_board = cv2.warpPerspective(self._stable_warped_board, transform, (size, size))
        if self._pre_commit_warped_board is not None:
            self._pre_commit_warped_board = cv2.warpPerspective(
                self._pre_commit_warped_board, transform, (size, size)
            )

    def _compute_base_core_deltas(
        self,
        warped_after: np.ndarray,
        warped_before: np.ndarray,
    ) -> Tuple[Dict[str, float], float, float]:
        """Computes normalized per-square change metrics focused on each cell's physical base core."""
        return self.occupancy_tracker.compute_pair_deltas(
            warped_after=warped_after,
            warped_before=warped_before,
            orientation=self._orientation,
            roi_extractor=self.roi_extractor,
        )

    def _check_physical_takeback(self, warped_after: np.ndarray) -> bool:
        """Checks whether the last committed move was physically reverted to pre-commit board."""
        if (
            not self.config.temporal.auto_takeback_enabled
            or self._pre_commit_warped_board is None
            or self._stable_warped_board is None
            or self.state_manager.ply == 0
            or not self._last_committed_move_squares
        ):
            return False
        if (
            self._pre_commit_warped_board.shape != warped_after.shape
            or self._stable_warped_board.shape != warped_after.shape
        ):
            return False

        deltas_pre, _, _ = self._compute_base_core_deltas(warped_after, self._pre_commit_warped_board)
        deltas_cur, _, cur_thresh = self._compute_base_core_deltas(warped_after, self._stable_warped_board)

        pre_err = max(deltas_pre[sq] for sq in self._last_committed_move_squares)
        cur_diff = min(deltas_cur[sq] for sq in self._last_committed_move_squares)
        other_changed = [
            sq for sq, d in deltas_pre.items()
            if sq not in self._last_committed_move_squares and d >= max(0.08, cur_thresh)
        ]
        return pre_err < 0.045 and cur_diff >= max(0.06, pre_err * 2.5) and len(other_changed) == 0

    def _refine_zero_shot_with_temporal_delta(
        self,
        cell_obs: Dict[str, Any],
        rois: Dict[str, Any],
        warped_after: np.ndarray,
        warped_before: np.ndarray,
        precomputed_deltas: Optional[Tuple[Dict[str, float], float, float]] = None,
    ) -> Dict[str, Any]:
        """Refines zero-shot classifier probabilities using 3D column occupancy and transition scoring."""
        self._pending_physical_anomaly = None
        board = self.state_manager.current_board
        raw_changed = 0
        for sq in CHESS_SQUARES:
            exp_cls = board_square_to_piece_class(board, chess.parse_square(sq))
            if sq in cell_obs and cell_obs[sq].predicted_piece != exp_cls:
                raw_changed += 1

        if raw_changed <= 6 and not isinstance(self.piece_classifier, ONNXPieceClassifier):
            return cell_obs

        self._pending_two_ply_second_move = None
        nn_occ_before: Optional[Dict[str, float]] = None
        nn_occ_after: Optional[Dict[str, float]] = None
        if self.occupancy_classifier.is_available:
            occ_crops_b = self.roi_extractor.extract_occupancy_crops(warped_before, orientation=self._orientation)
            occ_crops_a = self.roi_extractor.extract_occupancy_crops(warped_after, orientation=self._orientation)
            nn_occ_before = self.occupancy_classifier.predict_squares(occ_crops_b)
            nn_occ_after = self.occupancy_classifier.predict_squares(occ_crops_a)
        elif isinstance(self.piece_classifier, ONNXPieceClassifier):
            nn_occ_after = {
                sq: 1.0 - float(obs.probabilities.get("empty", 1.0))
                for sq, obs in cell_obs.items()
            }
            prev_raw_obs = self._last_stable_raw_obs or self._last_cell_obs
            if prev_raw_obs:
                nn_occ_before = {
                    sq: 1.0 - float(obs.probabilities.get("empty", 1.0))
                    for sq, obs in prev_raw_obs.items()
                }
        evidence = self.occupancy_tracker.evaluate_transition(
            board=board,
            warped_after=warped_after,
            warped_before=warped_before,
            orientation=self._orientation,
            roi_extractor=self.roi_extractor,
            candidate_generator=self.candidate_generator,
            base_deltas=precomputed_deltas,
            nn_occ_before=nn_occ_before,
            nn_occ_after=nn_occ_after,
        )
        deltas = evidence.deltas
        change_thresh = evidence.change_thresh
        secondary_thresh = evidence.secondary_thresh
        changed_sqs = evidence.changed_sqs
        expected_dest_piece = evidence.expected_dest_piece
        self._pending_two_ply_second_move = evidence.pending_two_ply_second_move

        anomaly_set: set[str] = set()
        if not changed_sqs and evidence.anomaly_squares:
            self._pending_physical_anomaly = list(evidence.anomaly_squares)
            anomaly_set = set(evidence.anomaly_squares)

        refined: Dict[str, Any] = {}
        turn_prefix = "w" if board.turn == chess.WHITE else "b"
        opp_prefix = "b" if board.turn == chess.WHITE else "w"
        max_changed_delta = max((deltas[sq] for sq in changed_sqs), default=change_thresh)

        for sq in CHESS_SQUARES:
            obs = cell_obs[sq]
            exp_cls = board_square_to_piece_class(board, chess.parse_square(sq))
            prior_vec = np.full(len(PIECE_CLASSES), 0.01 / max(1, len(PIECE_CLASSES) - 1), dtype=np.float32)
            prior_vec[PIECE_CLASSES.index(exp_cls)] = 0.99

            if sq in anomaly_set:
                alt_vec = np.full(len(PIECE_CLASSES), 0.15 / max(1, len(PIECE_CLASSES) - 1), dtype=np.float32)
                if exp_cls != "empty":
                    alt_vec[0] = 0.85
                else:
                    alt_vec[PIECE_CLASSES.index(f"{opp_prefix}P")] = 0.85
                refined[sq] = probabilities_to_cell_observation(
                    sq, alt_vec, occupancy_threshold=self.config.piece_classifier.occupancy_threshold
                )
            elif sq not in changed_sqs:
                probs_vec = np.array([obs.probabilities.get(c, 0.0) for c in PIECE_CLASSES], dtype=np.float32)
                blended = 0.20 * probs_vec + 0.80 * prior_vec
                refined[sq] = probabilities_to_cell_observation(
                    sq, blended, occupancy_threshold=self.config.piece_classifier.occupancy_threshold
                )
            else:
                probs_vec = np.array([obs.probabilities.get(c, 0.0) for c in PIECE_CLASSES], dtype=np.float32)
                boost = np.zeros(len(PIECE_CLASSES), dtype=np.float32)
                if exp_cls.startswith(turn_prefix):
                    boost[0] = 0.92
                    rem = 0.08 / max(1, len(PIECE_CLASSES) - 1)
                    for k in range(1, len(PIECE_CLASSES)):
                        boost[k] = rem
                else:
                    boost[0] = 0.04
                    active_indices = [i for i, c in enumerate(PIECE_CLASSES) if c.startswith(turn_prefix)]
                    raw_active = np.array([probs_vec[i] for i in active_indices], dtype=np.float32)
                    if float(np.sum(raw_active)) > 1e-4:
                        raw_active = raw_active / float(np.sum(raw_active))
                    else:
                        raw_active = np.full(len(active_indices), 1.0 / len(active_indices), dtype=np.float32)

                    pref_cls = expected_dest_piece.get(sq)
                    if pref_cls in PIECE_CLASSES:
                        pref_idx = PIECE_CLASSES.index(pref_cls)
                        for idx, cls_idx in enumerate(active_indices):
                            boost[cls_idx] = 0.62 if cls_idx == pref_idx else (0.28 / max(1, len(active_indices) - 1))
                    else:
                        for idx, cls_idx in enumerate(active_indices):
                            boost[cls_idx] = 0.90 * raw_active[idx]
                    opp_indices = [i for i, c in enumerate(PIECE_CLASSES) if c.startswith(opp_prefix)]
                    for cls_idx in opp_indices:
                        boost[cls_idx] = 0.06 / max(1, len(opp_indices))

                denom = max(1e-3, max_changed_delta - secondary_thresh)
                rel = min(1.0, max(0.0, (deltas[sq] - secondary_thresh) / denom))
                strength = 0.72 + 0.28 * rel
                weighted_boost = strength * boost + (1.0 - strength) * prior_vec
                blended = 0.25 * probs_vec + 0.75 * weighted_boost
                refined[sq] = probabilities_to_cell_observation(
                    sq, blended, occupancy_threshold=self.config.piece_classifier.occupancy_threshold
                )

        return refined

    def _should_use_whole_board(self) -> bool:
        """True when ChessQ-Lite whole-board classifier is active and no test double was injected."""
        return bool(
            self.config.piece_classifier.use_chessq_whole_board
            and self.whole_board_classifier.is_available
            and not self._custom_piece_classifier
            and not isinstance(self.piece_classifier, MockPieceClassifier)
        )

    def _resolve_orientation_non_brightness(
        self,
        warped: np.ndarray,
        padded_warped: Optional[np.ndarray] = None,
    ) -> BoardOrientation:
        """Resolves board orientation via non-brightness ChessQ viewing vector when >=12 real pieces are visible."""
        if self._should_use_whole_board():
            target_img = padded_warped if padded_warped is not None else warped
            try:
                pad_arg = 0 if min(target_img.shape[:2]) > 660 else 60
                ori_cq, _ = self.whole_board_classifier.detect_orientation(target_img, add_padding=pad_arg)
                if ori_cq != BoardOrientation.UNKNOWN:
                    cells_cq, _ = self.whole_board_classifier.classify_board_image(
                        target_img, orientation=ori_cq, add_padding=pad_arg
                    )
                    occ_cnt = sum(1 for c in cells_cq.values() if c.is_occupied)
                    if occ_cnt >= 12:
                        return ori_cq
            except Exception as exc:
                logger.debug("ChessQ orientation fallback triggered: %s", exc)
        return self.orientation_resolver.resolve(warped)

    def _refine_with_whole_board_hybrid(
        self,
        cell_obs: Dict[str, Any],
        warped_after: np.ndarray,
        warped_before: np.ndarray,
    ) -> Dict[str, Any]:
        """Combines ChessQ-Lite raw [64, 13] posteriors with DINOv2 + SSIM/VecGrad non-brightness deltas."""
        self._pending_physical_anomaly = None
        board = self.state_manager.current_board
        occ_thr = self.config.piece_classifier.occupancy_threshold

        # 1. Structural SSIM + Vector-Gradient base-core delta
        struct_deltas, _, thresh_s = self._compute_base_core_deltas(warped_after, warped_before)

        # 2. DINOv2-Small semantic cosine change map (cached before-embedding ~100ms)
        use_dino = (
            self.config.piece_classifier.use_dinov2_change_detector
            and self.dino_change_detector.is_available
        )
        if use_dino:
            dino_deltas = self.dino_change_detector.compute_square_change_map(
                warped_after=warped_after,
                warped_before=warped_before,
                orientation=self._orientation,
                roi_extractor=self.roi_extractor,
            )
            dino_med = float(np.median(list(dino_deltas.values())))
            dino_thresh = max(0.080, dino_med * 2.50)
        else:
            dino_deltas = {sq: 0.0 for sq in CHESS_SQUARES}
            dino_med = 0.0
            dino_thresh = 1.0

        changed_vis = {
            sq: (struct_deltas[sq] >= thresh_s * 0.90) or (dino_deltas[sq] >= dino_thresh)
            for sq in CHESS_SQUARES
        }
        any_vis_change = any(changed_vis.values())

        prev_stable = self._last_stable_raw_obs or {}
        high_conf_flips = 0
        for sq in CHESS_SQUARES:
            exp_cls = board_square_to_piece_class(board, chess.parse_square(sq))
            raw_obs = cell_obs[sq]
            prev_raw = prev_stable.get(sq)
            if (
                raw_obs.predicted_piece != exp_cls
                and raw_obs.confidence >= 0.78
                and (prev_raw is None or prev_raw.predicted_piece != raw_obs.predicted_piece)
            ):
                high_conf_flips += 1

        # Static/quiet board guard: no semantic/structural delta and no multi-square high-conf flip
        if not any_vis_change and high_conf_flips < 2:
            quiet_obs: Dict[str, Any] = {}
            for sq in CHESS_SQUARES:
                exp_cls = board_square_to_piece_class(board, chess.parse_square(sq))
                vec = np.full(len(PIECE_CLASSES), 0.01 / max(1, len(PIECE_CLASSES) - 1), dtype=np.float32)
                vec[PIECE_CLASSES.index(exp_cls)] = 0.99
                quiet_obs[sq] = probabilities_to_cell_observation(sq, vec, occupancy_threshold=occ_thr)
            return quiet_obs

        refined: Dict[str, Any] = {}
        active_changed: Set[str] = set()
        for sq in CHESS_SQUARES:
            obs = cell_obs[sq]
            exp_cls = board_square_to_piece_class(board, chess.parse_square(sq))
            prev_raw = prev_stable.get(sq)
            is_flipped_high = (
                obs.predicted_piece != exp_cls
                and obs.confidence >= 0.72
                and (prev_raw is None or prev_raw.predicted_piece != obs.predicted_piece)
            )
            if changed_vis[sq] or is_flipped_high:
                refined[sq] = obs.model_copy(deep=True)
                if obs.predicted_piece != exp_cls:
                    active_changed.add(sq)
            else:
                # Stabilize stationary background squares so edge blur doesn't penalize coverage_factor
                probs_vec = np.array([obs.probabilities.get(c, 0.0) for c in PIECE_CLASSES], dtype=np.float32)
                prior_vec = np.full(len(PIECE_CLASSES), 0.01 / max(1, len(PIECE_CLASSES) - 1), dtype=np.float32)
                prior_vec[PIECE_CLASSES.index(exp_cls)] = 0.99
                blended = 0.30 * probs_vec + 0.70 * prior_vec
                refined[sq] = probabilities_to_cell_observation(sq, blended, occupancy_threshold=occ_thr)

        # Rapid 2-ply check: if both players moved within one settling window (3..8 changed squares),
        # check whether a legal 2-ply sequence (m1, m2) matches the whole-board posterior much better than any 1-ply move.
        fused_deltas = {
            sq: max(struct_deltas.get(sq, 0.0), (dino_deltas.get(sq, 0.0) * 0.5) if use_dino else 0.0)
            for sq in CHESS_SQUARES
        }
        changed_for_2ply = active_changed | {sq for sq, ch in changed_vis.items() if ch}
        best_1ply_v = 0.0
        if len(changed_for_2ply) >= 3:
            for m1 in board.legal_moves:
                aff1 = self.candidate_generator.get_affected_squares(board, m1)
                if not active_changed.intersection(aff1):
                    continue
                b_tmp = board.copy()
                b_tmp.push(m1)
                ok_probs = [
                    refined[sq].probabilities.get(
                        board_square_to_piece_class(b_tmp, chess.parse_square(sq)), 0.0
                    )
                    for sq in active_changed
                ]
                if ok_probs:
                    best_1ply_v = max(best_1ply_v, float(sum(ok_probs) / len(ok_probs)))

        best_pair: Optional[Tuple[float, chess.Move, chess.Move, Set[str]]] = None
        if 3 <= len(changed_for_2ply) <= 8:
            two_ply_cands = self.candidate_generator.find_two_ply_sequence(
                board=board,
                changed_squares=changed_for_2ply,
                square_deltas=fused_deltas,
            )
            for m1, m2 in two_ply_cands[:12]:
                b2 = board.copy()
                aff1 = set(self.candidate_generator.get_affected_squares(b2, m1))
                b2.push(m1)
                aff2 = set(self.candidate_generator.get_affected_squares(b2, m2))
                b2.push(m2)
                union = aff1 | aff2
                sc_list = [
                    refined[sq].probabilities.get(
                        board_square_to_piece_class(b2, chess.parse_square(sq)), 0.0
                    )
                    for sq in union
                ]
                pair_vis = float(sum(sc_list) / len(sc_list)) if sc_list else 0.0
                if pair_vis >= 0.72 and pair_vis > best_1ply_v + 0.15:
                    if best_pair is None or pair_vis > best_pair[0]:
                        best_pair = (pair_vis, m1, m2, aff2)

        if best_pair is not None:
            _, m1_j, m2_j, aff2_j = best_pair
            self._pending_two_ply_second_move = m2_j
            self._pending_rapid_queue.clear()
            self._pending_two_ply_raw_obs = {sq: o.model_copy(deep=True) for sq, o in refined.items()}
            b_mid = board.copy()
            b_mid.push(m1_j)
            aff1_j = set(self.candidate_generator.get_affected_squares(board, m1_j))
            for sq in (aff2_j - aff1_j):
                mid_cls = board_square_to_piece_class(b_mid, chess.parse_square(sq))
                mid_vec = np.full(len(PIECE_CLASSES), 0.01 / max(1, len(PIECE_CLASSES) - 1), dtype=np.float32)
                mid_vec[PIECE_CLASSES.index(mid_cls)] = 0.99
                refined[sq] = probabilities_to_cell_observation(sq, mid_vec, occupancy_threshold=occ_thr)
                active_changed.discard(sq)
            logger.info(
                "Detected rapid 2-ply whole-board sequence: %s -> %s",
                m1_j.uci(),
                m2_j.uci(),
            )
        elif len(active_changed) >= 3:
            # Depth-2..5 beam catch-up search for rapid blitz sequences (>=2 plies in one hand window)
            beam: List[Tuple[float, chess.Board, List[chess.Move]]] = [(0.0, board.copy(), [])]
            target_map = {sq: refined[sq] for sq in CHESS_SQUARES}
            for depth in range(5):
                next_beam: List[Tuple[float, chess.Board, List[chess.Move]]] = []
                seen_fens: Set[str] = set()
                for _, b_cur, path in beam:
                    for m_cand in b_cur.legal_moves:
                        f_sq = chess.square_name(m_cand.from_square)
                        t_sq = chess.square_name(m_cand.to_square)
                        if (
                            f_sq not in changed_for_2ply
                            and t_sq not in changed_for_2ply
                            and target_map[t_sq].predicted_piece == board_square_to_piece_class(b_cur, m_cand.to_square)
                        ):
                            continue
                        b_nxt = b_cur.copy()
                        b_nxt.push(m_cand)
                        b_fen = b_nxt.board_fen()
                        if b_fen in seen_fens:
                            continue
                        seen_fens.add(b_fen)
                        match_sum = 0.0
                        mismatches = 0
                        for sq in CHESS_SQUARES:
                            ec = board_square_to_piece_class(b_nxt, chess.parse_square(sq))
                            p_val = target_map[sq].probabilities.get(ec, 0.0)
                            match_sum += p_val
                            if target_map[sq].predicted_piece != ec and (changed_vis[sq] or target_map[sq].confidence >= 0.78):
                                mismatches += 1
                        next_beam.append((match_sum - 1.5 * mismatches, b_nxt, path + [m_cand]))
                next_beam.sort(key=lambda x: x[0], reverse=True)
                beam = next_beam[:8]
                if not beam:
                    break
                top_b = beam[0][1]
                rem_err = [
                    sq for sq in active_changed
                    if board_square_to_piece_class(top_b, chess.parse_square(sq)) != target_map[sq].predicted_piece
                ]
                if len(rem_err) <= 1 and len(beam[0][2]) >= 2:
                    break

            if beam and len(beam[0][2]) >= 2:
                top_path = beam[0][2]
                top_board = beam[0][1]
                union_aff: Set[str] = set()
                b_walk = board.copy()
                for mp in top_path:
                    union_aff.update(self.candidate_generator.get_affected_squares(b_walk, mp))
                    b_walk.push(mp)
                sc_union = [
                    target_map[sq].probabilities.get(
                        board_square_to_piece_class(top_board, chess.parse_square(sq)), 0.0
                    )
                    for sq in union_aff
                ]
                mean_union = float(sum(sc_union) / len(sc_union)) if sc_union else 0.0
                if mean_union >= 0.72 and mean_union > best_1ply_v + 0.15:
                    m1_b = top_path[0]
                    self._pending_two_ply_second_move = top_path[1]
                    self._pending_rapid_queue = deque(top_path[2:])
                    self._pending_two_ply_raw_obs = {sq: o.model_copy(deep=True) for sq, o in refined.items()}
                    b_mid = board.copy()
                    b_mid.push(m1_b)
                    aff1_b = set(self.candidate_generator.get_affected_squares(board, m1_b))
                    for sq in (union_aff - aff1_b):
                        mid_cls = board_square_to_piece_class(b_mid, chess.parse_square(sq))
                        mid_vec = np.full(len(PIECE_CLASSES), 0.01 / max(1, len(PIECE_CLASSES) - 1), dtype=np.float32)
                        mid_vec[PIECE_CLASSES.index(mid_cls)] = 0.99
                        refined[sq] = probabilities_to_cell_observation(sq, mid_vec, occupancy_threshold=occ_thr)
                        active_changed.discard(sq)
                    logger.info(
                        "Detected %d-ply rapid whole-board sequence: %s",
                        len(top_path),
                        " -> ".join(m.uci() for m in top_path),
                    )

        # Legal-move consistency verification & anomaly guard:
        # 1) If whole-board transition changed squares yet NO legal 1..5-ply move explains them
        #    (e.g., knight/king teleported across the board, piece swapped illegally), escalate as illegal anomaly
        #    with empty candidate list instead of inventing unrelated "closest" legal moves.
        if (
            self._pending_two_ply_second_move is None
            and active_changed
        ):
            def _expected_match_prob(obs_cell: Any, exp_c: str) -> float:
                p_exact = float(obs_cell.probabilities.get(exp_c, 0.0))
                if exp_c == "empty":
                    return p_exact
                pref = exp_c[0]
                p_color = sum(float(v) for k, v in obs_cell.probabilities.items() if k.startswith(pref))
                return max(p_exact, 0.85 * p_color)

            best_1ply_ok = False
            for m1 in board.legal_moves:
                aff1 = set(self.candidate_generator.get_affected_squares(board, m1))
                if not active_changed.intersection(aff1):
                    continue
                unexplained = [
                    sq for sq in active_changed
                    if sq not in aff1 and refined[sq].confidence >= 0.78 and changed_vis[sq]
                ]
                if unexplained:
                    continue
                b_tmp = board.copy()
                b_tmp.push(m1)
                aff_scores = [
                    _expected_match_prob(
                        refined[sq],
                        board_square_to_piece_class(b_tmp, chess.parse_square(sq)),
                    )
                    for sq in aff1
                ]
                if aff_scores and min(aff_scores) >= 0.28 and (sum(aff_scores) / len(aff_scores)) >= 0.48:
                    best_1ply_ok = True
                    break

            if not best_1ply_ok:
                anom_sqs = sorted(active_changed | {
                    sq for sq in CHESS_SQUARES
                    if (use_dino and dino_deltas[sq] >= max(0.16, dino_med * 3.80))
                    or struct_deltas[sq] >= thresh_s * 1.40
                })
                self._pending_physical_anomaly = anom_sqs[:6]
                return refined

        # 2) Strong localized physical change without ChessQ flip
        strong_vis = [
            sq for sq in CHESS_SQUARES
            if (use_dino and dino_deltas[sq] >= max(0.16, dino_med * 3.80))
            or struct_deltas[sq] >= thresh_s * 1.40
        ]
        if strong_vis and not active_changed:
            strong_vis.sort(key=lambda s: max(dino_deltas[s], struct_deltas[s]), reverse=True)
            best_overlap = 0
            for m in board.legal_moves:
                aff = set(self.candidate_generator.get_affected_squares(board, m))
                best_overlap = max(best_overlap, len(aff.intersection(strong_vis)))
            if best_overlap == 0:
                self._pending_physical_anomaly = strong_vis[:4]

        return refined

    def _build_pipeline_state(
        self,
        frame_id: int,
        timestamp: float,
        fsm_state: FSMState,
        board_detected: bool,
        hand_detected: bool,
        motion_level: float,
    ) -> PipelineState:
        """Assembles a clean PipelineState snapshot for GUI and CLI consumers."""
        return PipelineState(
            frame_id=frame_id,
            timestamp=timestamp,
            fsm_state=fsm_state,
            board_detected=board_detected,
            geometry_confidence=1.0 if board_detected else 0.0,
            hand_detected=hand_detected,
            motion_level=float(motion_level),
            current_ply=self.state_manager.ply,
            current_fen=self.state_manager.current_fen,
            last_move_san=self._last_committed_event.move_san if self._last_committed_event else None,
            last_move_uci=self._last_committed_event.move_uci if self._last_committed_event else None,
            last_decision=self.fsm.last_decision or self._last_decision,
            latest_eval=self._latest_eval,
            is_paused=self._is_paused,
            active_uncertainty=self._active_uncertainty,
            processing_fps=float(self._current_fps),
        )

    # -------------------------------------------------------------------------
    # Operator Interventions & Controls
    # -------------------------------------------------------------------------
    def _reset_tracking_state(self, preserve_geometry: bool = False) -> None:
        """Resets cached perception and recovery tracking buffers."""
        if not preserve_geometry:
            self._orientation_calibrated = False
            self._last_confirmed_orientation = None
        self._last_cell_obs = None
        self._last_stable_raw_obs = None
        self._last_observation = None
        self._stable_warped_board = None
        self._stable_warped_history.clear()
        self._pre_commit_warped_board = None
        self._last_committed_move_squares = []
        self._pending_two_ply_second_move = None
        self._pending_rapid_queue.clear()
        self._pending_two_ply_raw_obs = None
        self._pending_physical_anomaly = None
        self._recovery_anomaly_since = None
        self._active_uncertainty = False
        self._last_decision = None
        self.occupancy_tracker.reset()
        self.geometry_tracker.reset()
        self.dino_change_detector.reset()
        self.fsm.reset()
        self.motion_detector.reset()
        if self._is_paused:
            self.fsm.pause()
        elif preserve_geometry and self._board_geometry is not None:
            self.fsm.resume(
                has_geometry=True,
                orientation_ready=self._orientation_calibrated,
            )

    def recalibrate(self) -> None:
        """Forces board re-detection and orientation re-calibration."""
        with self._lock:
            self._board_geometry = None
            self._reset_tracking_state(preserve_geometry=False)
            self.board_localizer = create_board_localizer(self.config.vision)
            logger.info("Operator triggered board re-detection and orientation recalibration.")

    def _anchor_current_geometry_from_raw(self, resolve_orientation: bool = False) -> None:
        """Warps latest raw frame under current geometry to seed baseline and optional orientation."""
        if (
            self._last_raw_frame is None
            or self._board_geometry is None
            or not self._board_geometry.homography_matrix
        ):
            return
        warped = GeometryTransformer.warp_board(
            self._last_raw_frame.image,
            np.array(self._board_geometry.homography_matrix, dtype=np.float64),
            target_size=self.config.vision.target_board_size,
        )
        self._last_warped_board = warped
        if resolve_orientation:
            if self.state_manager.ply > 0 and self._last_confirmed_orientation is not None:
                detected_ori = self._last_confirmed_orientation
            else:
                padded = GeometryTransformer.warp_padded_board(
                    self._last_raw_frame.image,
                    np.array(self._board_geometry.homography_matrix, dtype=np.float64),
                    target_size=self.config.vision.target_board_size,
                    padding=80,
                )
                detected_ori = self._resolve_orientation_non_brightness(warped, padded)
            if detected_ori != BoardOrientation.UNKNOWN:
                self._orientation = detected_ori
                self._last_confirmed_orientation = detected_ori
                self._orientation_calibrated = True
                self._board_geometry = self._board_geometry.model_copy(
                    update={"orientation_calibrated": True, "orientation": detected_ori}
                )
        self._stable_warped_board = warped.copy()
        self._stable_warped_history.clear()
        self._stable_warped_history.append(warped.copy())

    def manual_calibrate(self) -> bool:
        """Detects board geometry and calibrates orientation on the latest camera frame (works while paused)."""
        with self._lock:
            if self._last_raw_frame is None:
                logger.warning("Manual calibration requested before any camera frame was received.")
                return False

            self.board_localizer = create_board_localizer(self.config.vision)
            geom = self.board_localizer.locate(self._last_raw_frame)
            if geom is None or not geom.is_valid or not geom.homography_matrix:
                logger.warning("Manual calibration failed to detect board on current frame.")
                return False

            self._board_geometry = geom
            self._reset_tracking_state(preserve_geometry=False)
            self._anchor_current_geometry_from_raw(resolve_orientation=True)
            if self._is_paused:
                self.fsm.pause()
            else:
                self.fsm.resume(has_geometry=True, orientation_ready=self._orientation_calibrated)
            logger.info(
                "Manual calibration completed on current frame: corners=%s ori=%s",
                self._board_geometry.corners,
                self._orientation.value,
            )
            return True

    def set_manual_corners(self, corners: Sequence[Tuple[float, float]]) -> bool:
        """Manually sets the 4 board corners and locks geometry.

        Args:
            corners: Sequence of 4 (x, y) coordinates.

        Returns:
            True if corners form a valid quadrilateral and geometry was locked.
        """
        with self._lock:
            if len(corners) != 4:
                return False
            ordered = order_corners_canonical(corners)
            is_valid, reason, _ = GeometryTransformer.validate_quadrilateral(
                ordered,
                min_area_ratio=0.02,
            )
            if not is_valid:
                logger.warning("Manual corners rejected: %s", reason)
                return False

            geom = GeometryTransformer.build_board_geometry(
                corners=ordered,
                target_size=self.config.vision.target_board_size,
                min_area_ratio=0.02,
                detection_method="manual",
            )
            if geom is not None and geom.is_valid:
                self._board_geometry = geom
                self._reset_tracking_state(preserve_geometry=False)
                self.board_localizer = ManualCornerLocalizer(
                    corners=ordered,
                    target_size=self.config.vision.target_board_size,
                    subpixel_refinement=True,
                )
                self._anchor_current_geometry_from_raw(resolve_orientation=True)
                if self._is_paused:
                    self.fsm.pause()
                logger.info("Operator manually locked board corners: %s", ordered)
                return True
            return False

    def resolve_uncertainty(
        self,
        move_uci: Optional[str] = None,
        auto_accept_top: bool = False,
        candidate_index: Optional[int] = None,
    ) -> bool:
        """Resolves active UNCERTAIN state by operator intervention or policy.

        Args:
            move_uci: Explicit move in UCI notation to commit.
            auto_accept_top: If True, automatically commits highest scoring candidate.
            candidate_index: 0-based index of candidate from last decision ranking.

        Returns:
            True if uncertainty was successfully resolved and move committed.
        """
        with self._lock:
            target_hypo: Optional[MoveHypothesis] = None

            if (auto_accept_top or candidate_index is not None) and self._last_decision:
                rankings = self._last_decision.top_hypotheses
                if not rankings and self._last_decision.accepted_hypothesis:
                    rankings = [self._last_decision.accepted_hypothesis]
                if rankings:
                    idx = candidate_index if candidate_index is not None else 0
                    if 0 <= idx < len(rankings):
                        target_hypo = rankings[idx]

            if target_hypo is not None:
                try:
                    if self._last_observation is not None:
                        self.state_manager.commit(target_hypo, self._last_observation)
                    else:
                        self.state_manager.manual_commit(target_hypo.move_uci, None)
                    if self._is_paused:
                        self.fsm.pause()
                    else:
                        self.fsm.resume(
                            has_geometry=(self._board_geometry is not None),
                            orientation_ready=self._orientation_calibrated,
                        )
                    self._active_uncertainty = False
                    self._pending_physical_anomaly = None
                    self._pending_two_ply_second_move = None
                    self._pending_rapid_queue.clear()
                    self._pending_two_ply_raw_obs = None
                    self._last_cell_obs = None
                    self._last_stable_raw_obs = None
                    self._last_decision = None
                    self.fsm._last_decision = None
                    self._stable_warped_history.clear()
                    if self._last_warped_board is not None:
                        self._stable_warped_board = self._last_warped_board.copy()
                    logger.info("Operator resolved uncertainty with candidate %s", target_hypo.move_uci)
                    return True
                except ValueError as exc:
                    logger.warning("Failed to commit candidate '%s': %s", target_hypo.move_uci, exc)
                    return False

            if move_uci:
                try:
                    self.state_manager.manual_commit(move_uci, self._last_observation)
                    if self._is_paused:
                        self.fsm.pause()
                    else:
                        self.fsm.resume(
                            has_geometry=(self._board_geometry is not None),
                            orientation_ready=self._orientation_calibrated,
                        )
                    self._active_uncertainty = False
                    self._pending_physical_anomaly = None
                    self._pending_two_ply_second_move = None
                    self._pending_rapid_queue.clear()
                    self._pending_two_ply_raw_obs = None
                    self._last_cell_obs = None
                    self._last_stable_raw_obs = None
                    self._last_decision = None
                    self.fsm._last_decision = None
                    self._stable_warped_history.clear()
                    if self._last_warped_board is not None:
                        self._stable_warped_board = self._last_warped_board.copy()
                    logger.info("Operator resolved uncertainty with manual move %s", move_uci)
                    return True
                except Exception as exc:
                    logger.warning("Failed to commit manual move '%s': %s", move_uci, exc)
                    return False

            return False

    def rollback_move(self) -> Optional[MoveEvent]:
        """Rolls back the most recently committed move (operator undo).

        Returns:
            Popped MoveEvent, or None if history was empty.
        """
        with self._lock:
            popped = self.state_manager.rollback()
            if popped is not None:
                self._stale_event_ids.add(popped.event_id)
                try:
                    rb_event = MoveEvent(
                        event_type="rollback",
                        ply=popped.ply,
                        fen_before=popped.fen_after,
                        move_uci=popped.move_uci,
                        move_san=popped.move_san,
                        fen_after=popped.fen_before,
                        confidence=1.0,
                        timestamp=popped.timestamp,
                        evidence_frame_id=popped.evidence_frame_id,
                    )
                    self.event_repo.append_event(rb_event)
                except Exception as exc:
                    logger.warning("Failed to append rollback audit event: %s", exc)
            if self._is_paused:
                self.fsm.pause()
            else:
                self.fsm.resume(
                    has_geometry=(self._board_geometry is not None),
                    orientation_ready=self._orientation_calibrated,
                )
            self._active_uncertainty = False
            self._pending_physical_anomaly = None
            self._last_logged_illegal_key = None
            self._pre_commit_warped_board = None
            self._last_committed_move_squares = []
            self._pending_two_ply_second_move = None
            self._pending_rapid_queue.clear()
            self._pending_two_ply_raw_obs = None
            self._last_decision = None
            self.fsm._last_decision = None
            self._latest_eval = None

            history = self.state_manager.history
            self._last_committed_event = history[-1] if history else None
            self._last_cell_obs = None
            self._last_stable_raw_obs = None
            self._stable_warped_history.clear()
            self._stable_warped_board = (
                self._last_warped_board.copy() if self._last_warped_board is not None else None
            )
            if popped is not None:
                try:
                    self.save_session_snapshot(is_completed=False)
                except Exception as exc:
                    logger.warning("Failed saving snapshot after rollback: %s", exc)
            return popped

    def pause(self) -> None:
        """Pauses video frame processing."""
        with self._lock:
            self._is_paused = True
            self.fsm.pause()

    def resume(self) -> None:
        """Resumes video frame processing."""
        with self._lock:
            self._is_paused = False
            self.fsm.resume(
                has_geometry=(self._board_geometry is not None),
                orientation_ready=self._orientation_calibrated,
            )

    def toggle_pause(self) -> bool:
        """Toggles between paused and active processing states. Returns new state."""
        with self._lock:
            if self._is_paused:
                self.resume()
            else:
                self.pause()
            return self._is_paused

    def export_pgn(self, headers: Optional[Dict[str, str]] = None) -> str:
        """Exports full recorded game history to standard FIDE PGN string.

        Args:
            headers: Optional dictionary of PGN header tags.

        Returns:
            Canonical PGN string representation.
        """
        with self._lock:
            hdr = {
                "Event": "Chess Recording System Session",
                "Site": "Autonomous Recording Station",
                "Date": datetime.now().strftime("%Y.%m.%d"),
                "White": self.white_player,
                "Black": self.black_player,
            }
            if headers:
                hdr.update(headers)
            return self.state_manager.export_pgn(headers=hdr)

    def save_session_snapshot(self, is_completed: bool = False) -> SessionManifest:
        """Persists game.pgn and manifest.json for the current session without stopping workers."""
        with self._lock:
            pgn_content = self.export_pgn()
            pgn_file = self.session_dir / "game.pgn"
            pgn_file.write_text(pgn_content, encoding="utf-8")

            checksum = ""
            if self.event_repo.file_path.exists():
                sha = hashlib.sha256()
                with open(self.event_repo.file_path, "rb") as f:
                    while chunk := f.read(65536):
                        sha.update(chunk)
                checksum = sha.hexdigest()

            initial_fen = (
                self.state_manager.history[0].fen_before
                if self.state_manager.history
                else self.state_manager.current_fen
            )
            manifest = SessionManifest(
                session_id=self.session_id,
                created_at=self._created_at,
                player_white=self.white_player,
                player_black=self.black_player,
                initial_fen=initial_fen,
                current_fen=self.state_manager.current_fen,
                total_plies=self.state_manager.ply,
                journal_path=str(self.event_repo.file_path.relative_to(self.session_dir)),
                pgn_path=str(pgn_file.relative_to(self.session_dir)),
                journal_checksum=checksum,
                is_completed=is_completed,
            )
            (self.session_dir / "manifest.json").write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
            return manifest

    def reset_session(self) -> SessionManifest:
        """Resets current game back to initial chess starting position and clears its move journal."""
        with self._lock:
            for ev in self.state_manager.history:
                self._stale_event_ids.add(ev.event_id)
            self.engine_service.clear_pending()
            self.state_manager.reset()
            self.event_repo.clear()
            self._latest_eval = None
            self._last_committed_event = None
            self._illegal_attempts_count = 0
            self._last_logged_illegal_key = None
            self._last_confirmed_orientation = None
            keep_geom = self._board_geometry is not None
            self._reset_tracking_state(preserve_geometry=keep_geom)
            if keep_geom:
                self._anchor_current_geometry_from_raw(resolve_orientation=False)
            logger.info("Operator reset session %s back to starting position.", self.session_id)
            return self.save_session_snapshot(is_completed=False)

    def new_session(
        self,
        white_player: Optional[str] = None,
        black_player: Optional[str] = None,
        session_id: Optional[str] = None,
        pause: bool = True,
    ) -> SessionManifest:
        """Saves current session snapshot and switches to a brand-new recording session."""
        with self._lock:
            self.save_session_snapshot(is_completed=True)
            for ev in self.state_manager.history:
                self._stale_event_ids.add(ev.event_id)
            self.engine_service.clear_pending()

            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = session_id or f"session_{ts_str}_{uuid.uuid4().hex[:6]}"
            self.session_dir = Path(self.config.persistence.base_dir) / self.session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._created_at = datetime.now(timezone.utc).isoformat()
            if white_player:
                self.white_player = white_player
            if black_player:
                self.black_player = black_player

            self.event_repo = JSONLEventRepository(
                session_dir=self.session_dir,
                sync_to_disk=self.config.persistence.sync_to_disk,
            )
            self.evidence_writer = EvidenceWriter(
                evidence_dir=self.session_dir / "evidence",
                policy=self.config.persistence.evidence_retention,
                format=self.config.persistence.evidence_format,
            )
            self.state_manager.reset()
            self._latest_eval = None
            self._last_committed_event = None
            self._illegal_attempts_count = 0
            self._last_logged_illegal_key = None
            self._last_confirmed_orientation = None
            if pause:
                self._is_paused = True

            keep_geom = self._board_geometry is not None
            self._reset_tracking_state(preserve_geometry=keep_geom)
            if keep_geom:
                self._anchor_current_geometry_from_raw(resolve_orientation=False)

            logger.info("Started new session %s (paused=%s)", self.session_id, self._is_paused)
            return self.save_session_snapshot(is_completed=False)

    def resume_session(self, session_id: str, pause: bool = True) -> SessionManifest:
        """Loads and resumes a previously persisted session from disk."""
        from src.persistence.recovery_manager import RecoveryManager

        with self._lock:
            target_dir = Path(self.config.persistence.base_dir) / session_id
            if not target_dir.is_dir():
                raise FileNotFoundError(f"Session directory not found: {target_dir}")

            if session_id != self.session_id:
                self.save_session_snapshot(is_completed=False)
            for ev in self.state_manager.history:
                self._stale_event_ids.add(ev.event_id)
            self.engine_service.clear_pending()
            self._last_confirmed_orientation = None

            rec = RecoveryManager(target_dir)
            recovered_mgr, events = rec.recover_session()

            manifest_file = target_dir / "manifest.json"
            if manifest_file.is_file():
                try:
                    prev_manifest = SessionManifest.model_validate_json(
                        manifest_file.read_text(encoding="utf-8")
                    )
                    self.white_player = prev_manifest.player_white
                    self.black_player = prev_manifest.player_black
                    self._created_at = prev_manifest.created_at
                except Exception as exc:
                    logger.warning("Could not parse manifest.json for %s: %s", session_id, exc)
            else:
                self._created_at = datetime.now(timezone.utc).isoformat()

            self.session_id = session_id
            self.session_dir = target_dir
            self.event_repo = JSONLEventRepository(
                session_dir=self.session_dir,
                sync_to_disk=self.config.persistence.sync_to_disk,
            )
            self.evidence_writer = EvidenceWriter(
                evidence_dir=self.session_dir / "evidence",
                policy=self.config.persistence.evidence_retention,
                format=self.config.persistence.evidence_format,
            )

            recovered_mgr.on_move_committed = self._on_move_committed
            self.state_manager = recovered_mgr
            self.fsm.game_state_manager = self.state_manager
            self._last_committed_event = events[-1] if events else None
            self._illegal_attempts_count = self.event_repo.count_illegal_attempts()
            self._last_logged_illegal_key = None
            self._latest_eval = None
            if pause:
                self._is_paused = True

            keep_geom = self._board_geometry is not None
            self._reset_tracking_state(preserve_geometry=keep_geom)
            if keep_geom:
                self._anchor_current_geometry_from_raw(resolve_orientation=False)

            logger.info("Resumed session %s at ply=%d (paused=%s)", session_id, self.state_manager.ply, self._is_paused)
            return self.save_session_snapshot(is_completed=False)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all persisted recording sessions sorted newest first."""
        base_dir = Path(self.config.persistence.base_dir)
        if not base_dir.exists():
            return []

        entries: List[Dict[str, Any]] = []
        with self._lock:
            current_id = self.session_id

        for item in base_dir.iterdir():
            if not item.is_dir():
                continue
            sid = item.name
            manifest_p = item / "manifest.json"
            journal_p = item / "events.jsonl"
            if not manifest_p.exists() and not journal_p.exists():
                continue

            created_at = ""
            white_p = "White Player"
            black_p = "Black Player"
            plies = 0
            current_fen = chess.STARTING_FEN
            is_completed = False
            moves_preview = ""

            if manifest_p.is_file():
                try:
                    m = SessionManifest.model_validate_json(manifest_p.read_text(encoding="utf-8"))
                    created_at = m.created_at
                    white_p = m.player_white
                    black_p = m.player_black
                    plies = m.total_plies
                    current_fen = m.current_fen
                    is_completed = m.is_completed
                except Exception:
                    pass

            if journal_p.is_file() and journal_p.stat().st_size > 0:
                try:
                    repo = JSONLEventRepository(item, sync_to_disk=False)
                    evs = repo.read_active_moves()
                    if evs:
                        plies = len(evs)
                        current_fen = evs[-1].fen_after
                        moves_preview = " ".join(ev.move_san for ev in evs[-6:])
                except Exception:
                    pass

            if not created_at:
                created_at = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()

            entries.append(
                {
                    "session_id": sid,
                    "created_at": created_at,
                    "white_player": white_p,
                    "black_player": black_p,
                    "total_plies": plies,
                    "current_fen": current_fen,
                    "is_completed": is_completed,
                    "is_current": sid == current_id,
                    "moves_preview": moves_preview,
                }
            )

        entries.sort(key=lambda e: e["created_at"], reverse=True)
        return entries

    def stop(self) -> SessionManifest:
        """Finalizes session, saves PGN and manifest, calculates checksums, and shuts down."""
        with self._lock:
            self.fsm.stop()

            # 1. Export PGN to disk
            pgn_content = self.export_pgn()
            pgn_file = self.session_dir / "game.pgn"
            pgn_file.write_text(pgn_content, encoding="utf-8")

            # 2. Compute SHA-256 integrity checksum of journal
            checksum = ""
            if self.event_repo.file_path.exists():
                sha = hashlib.sha256()
                with open(self.event_repo.file_path, "rb") as f:
                    while chunk := f.read(65536):
                        sha.update(chunk)
                checksum = sha.hexdigest()

            # 3. Assemble and save SessionManifest
            manifest = SessionManifest(
                session_id=self.session_id,
                created_at=self._created_at,
                player_white=self.white_player,
                player_black=self.black_player,
                initial_fen=self.state_manager.current_fen if not self.state_manager.history else self.state_manager.history[0].fen_before,
                current_fen=self.state_manager.current_fen,
                total_plies=self.state_manager.ply,
                journal_path=str(self.event_repo.file_path.relative_to(self.session_dir)),
                pgn_path=str(pgn_file.relative_to(self.session_dir)),
                journal_checksum=checksum,
                is_completed=True,
            )

            manifest_file = self.session_dir / "manifest.json"
            manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

            # 4. Gracefully terminate engine background worker and hand detector
            self.engine_service.stop(timeout_sec=2.0)
            if hasattr(self.hand_detector, "close"):
                try:
                    self.hand_detector.close()
                except Exception as exc:
                    logger.debug("Hand detector close skipped: %s", exc)
            logger.info("Session %s finalized: total plies=%d", self.session_id, manifest.total_plies)

            return manifest
