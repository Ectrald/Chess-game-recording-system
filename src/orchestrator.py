"""End-to-End System Orchestrator for Chess Game Recording System.

Coordinates video acquisition, board localization, homography transformations,
optical motion tracking, hand occlusion detection, neural piece classification,
finite state machine (FSM) debouncing, symbolic move reasoning, transactional
event persistence, evidence preservation, and asynchronous Stockfish UCI analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import chess
import numpy as np

from src.analysis.engine_service import AsyncEngineService
from src.core.config import AppConfig, load_config
from src.core.datatypes import (
    CHESS_SQUARES,
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
from src.ml.piece_classifier import BasePieceClassifier, create_piece_classifier
from src.persistence.event_repository import JSONLEventRepository
from src.persistence.evidence_writer import EvidenceWriter
from src.reasoning.candidate_generator import LegalMoveCandidateGenerator
from src.reasoning.decision_gate import DecisionGate
from src.reasoning.fsm import GameFSM
from src.reasoning.game_state_manager import GameStateManager
from src.reasoning.scorer import MoveScorer
from src.vision.board_localizer import BaseBoardLocalizer, create_board_localizer
from src.vision.corner_refinement import order_corners_canonical
from src.vision.hand_detector import BaseHandDetector, create_hand_detector
from src.vision.homography import GeometryTransformer
from src.vision.motion_detector import MotionDetector
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
            config=self.config.vision.localizer
        )
        self.orientation_resolver = OrientationResolver(config=self.config.vision.orientation)
        self.geometry_transformer = GeometryTransformer(
            target_size=self.config.vision.target_board_size
        )
        self.roi_extractor = CellRoiExtractor(
            board_size=self.config.vision.target_board_size,
            parallax_alpha=self.config.vision.parallax_height_alpha,
        )
        self.motion_detector = MotionDetector(config=self.config.temporal)
        self.hand_detector = hand_detector or create_hand_detector(
            config=self.config.vision.hand_detector
        )

        # ---------------------------------------------------------------------
        # Machine Learning Subsystems
        # ---------------------------------------------------------------------
        self.piece_classifier = piece_classifier or create_piece_classifier(
            config=self.config.piece_classifier
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
        self._orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM
        self._orientation_calibrated = False
        self._last_observation: Optional[BoardObservation] = None
        self._last_warped_board: Optional[np.ndarray] = None
        self._last_committed_event: Optional[MoveEvent] = None
        self._last_decision: Optional[DecisionResult] = None
        self._latest_eval: Optional[AnalysisResult] = None

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
        self._last_decision = decision
        if decision.status == DecisionStatus.UNCERTAIN:
            self._active_uncertainty = True
            logger.warning(
                "Orchestrator: DecisionGate reported UNCERTAIN (top candidate: %s)",
                decision.accepted_hypothesis.move_uci if decision.accepted_hypothesis else "None",
            )
        else:
            self._active_uncertainty = False

    def _on_engine_analysis_completed(self, result: AnalysisResult) -> None:
        """Invoked asynchronously when Stockfish finishes move evaluation."""
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

            # If paused by operator, preserve current state
            if self._is_paused:
                return self._build_pipeline_state(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    fsm_state=self.fsm.current_state,
                    board_detected=(self._board_geometry is not None),
                    hand_detected=False,
                    motion_level=0.0,
                )

            # -----------------------------------------------------------------
            # Step 1: Board Localization
            # -----------------------------------------------------------------
            if self._board_geometry is None:
                geom = self.board_localizer.locate(frame)
                if geom is not None and geom.is_valid:
                    self._board_geometry = geom
                    logger.info("Board geometry successfully locked: corners=%s", geom.corners)

            board_detected = self._board_geometry is not None

            # -----------------------------------------------------------------
            # Step 2: Hand Detection & Occlusion
            # -----------------------------------------------------------------
            hand_detection = self.hand_detector.detect(frame, geometry=self._board_geometry)
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

                # Orientation Calibration
                if not self._orientation_calibrated:
                    detected_ori = self.orientation_resolver.resolve(warped)
                    if detected_ori != BoardOrientation.UNKNOWN:
                        self._orientation = detected_ori
                        self._orientation_calibrated = True
                        logger.info("Orientation calibrated: %s", self._orientation.value)

                # Optical Motion Detection
                motion_metrics = self.motion_detector.process_board_image(
                    warped,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    orientation=self._orientation,
                )
                motion_detected = motion_metrics.is_motion_detected
                motion_level = motion_metrics.global_motion_score

                # Extract square ROIs and classify cells
                rois = self.roi_extractor.extract_all_rois(warped, orientation=self._orientation)
                patches = {sq: item[1] for sq, item in rois.items()}
                cell_obs = self.piece_classifier.classify_board(patches)

                # Apply hand occlusion flags
                for sq in occluded_squares:
                    if sq in cell_obs:
                        cell_obs[sq].is_occluded = True

                observation = BoardObservation(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    cells=cell_obs,
                    motion_detected=motion_detected,
                    motion_level=motion_level,
                    hand_present=is_hand_present,
                    geometry_confidence=1.0,
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
            fsm_state = self.fsm.update(observation)
            if fsm_state == FSMState.UNCERTAIN:
                self._active_uncertainty = True

            return self._build_pipeline_state(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                fsm_state=fsm_state,
                board_detected=board_detected,
                hand_detected=is_hand_present,
                motion_level=motion_level,
            )

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
    def recalibrate(self) -> None:
        """Forces board re-detection and orientation re-calibration."""
        with self._lock:
            self._board_geometry = None
            self._orientation_calibrated = False
            from src.vision.board_localizer import create_board_localizer
            self.board_localizer = create_board_localizer(self.config.vision)
            logger.info("Operator triggered board re-detection and orientation recalibration.")

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
                self._orientation_calibrated = False
                from src.vision.board_localizer import ManualCornerLocalizer
                self.board_localizer = ManualCornerLocalizer(
                    corners=ordered,
                    target_size=self.config.vision.target_board_size,
                    subpixel_refinement=True,
                )
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
                self.state_manager.commit(target_hypo, self._last_observation)
                self.fsm.resume()
                self._active_uncertainty = False
                logger.info("Operator resolved uncertainty with candidate %s", target_hypo.move_uci)
                return True

            if move_uci:
                try:
                    self.state_manager.manual_commit(move_uci, self._last_observation)
                    self.fsm.resume()
                    self._active_uncertainty = False
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
            self.fsm.resume()
            self._active_uncertainty = False

            history = self.state_manager.history
            self._last_committed_event = history[-1] if history else None
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
            self.fsm.resume()

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

            # 4. Gracefully terminate engine background worker
            self.engine_service.stop(timeout_sec=2.0)
            logger.info("Session %s finalized: total plies=%d", self.session_id, manifest.total_plies)

            return manifest
