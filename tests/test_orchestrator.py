"""Test suite for SystemOrchestrator and end-to-end processing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional
import pytest

import chess
import numpy as np

from src.analysis.engine_service import AsyncEngineService
from src.analysis.stockfish_worker import MockStockfishEngine
from src.core.config import AppConfig, load_config
from src.core.datatypes import (
    BoardGeometry,
    CellObservation,
    FSMState,
    Frame,
    HandDetection,
    MoveEvent,
    MoveHypothesis,
    PipelineState,
    SessionManifest,
)
from src.ml.piece_classifier import BasePieceClassifier, MockPieceClassifier
from src.orchestrator import SystemOrchestrator
from src.vision.board_localizer import BaseBoardLocalizer


class StaticMockLocalizer(BaseBoardLocalizer):
    """Deterministic localizer returning fixed 4-corners quadrilateral."""

    def __init__(self, corners: Optional[List[List[float]]] = None) -> None:
        self.corners = corners or [[100.0, 100.0], [740.0, 100.0], [740.0, 740.0], [100.0, 740.0]]
        self.geom = BoardGeometry(
            corners=self.corners,
            homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            target_size=640,
            is_valid=True,
            orientation_calibrated=True,
        )

    def locate(self, frame: Frame) -> Optional[BoardGeometry]:
        return self.geom


class ControlledPieceClassifier(BasePieceClassifier):
    """Piece classifier whose predictions are dynamically controlled by test."""

    def __init__(self) -> None:
        self._current_board = chess.Board()

    def set_board(self, board: chess.Board) -> None:
        self._current_board = board.copy()

    def classify_cell(self, cell_patch: np.ndarray, square: str) -> CellObservation:
        raise NotImplementedError

    def classify_board(
        self,
        cell_patches: Dict[str, np.ndarray],
        rois: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, CellObservation]:
        results: Dict[str, CellObservation] = {}
        for sq_idx in chess.SQUARES:
            sq_name = chess.square_name(sq_idx)
            piece = self._current_board.piece_at(sq_idx)
            if piece is None:
                piece_label = "empty"
                is_occ = False
                probs = {"empty": 0.99, "wP": 0.001}
            else:
                prefix = "w" if piece.color == chess.WHITE else "b"
                symbol = piece.symbol().upper()
                piece_label = f"{prefix}{symbol}"
                is_occ = True
                probs = {piece_label: 0.98, "empty": 0.01}

            results[sq_name] = CellObservation(
                square=sq_name,
                probabilities=probs,
                predicted_piece=piece_label,
                confidence=0.98,
                is_occupied=is_occ,
            )
        return results


@pytest.fixture
def orchestrator_fixture(tmp_path: Path) -> Iterator[SystemOrchestrator]:
    """Creates a configured orchestrator instance with mock engine and temporary storage."""
    config: AppConfig = load_config()
    config.persistence.base_dir = str(tmp_path / "sessions")
    config.persistence.sync_to_disk = False
    config.analysis.stockfish_enabled = True

    mock_worker = MockStockfishEngine()
    engine_service = AsyncEngineService(config=config.analysis, worker=mock_worker)
    localizer = StaticMockLocalizer()
    classifier = ControlledPieceClassifier()

    orch = SystemOrchestrator(
        config=config,
        session_id="test_orchestrator_session",
        board_localizer=localizer,
        piece_classifier=classifier,
        engine_service=engine_service,
    )
    yield orch
    try:
        orch.stop()
    except Exception:
        pass


class TestSystemOrchestrator:
    """Test suite covering SystemOrchestrator lifecycle and event flows."""

    def test_initialization_and_session_start(self, orchestrator_fixture: SystemOrchestrator) -> None:
        orch = orchestrator_fixture
        assert orch.session_dir.exists()

        manifest = orch.start_session(white_player="Grandmaster A", black_player="Grandmaster B")
        assert isinstance(manifest, SessionManifest)
        assert manifest.player_white == "Grandmaster A"
        assert manifest.player_black == "Grandmaster B"
        assert manifest.total_plies == 0
        assert manifest.is_completed is False
        assert orch.engine_service.is_running is True

    def test_process_frame_fsm_progression(self, orchestrator_fixture: SystemOrchestrator) -> None:
        orch = orchestrator_fixture
        orch.start_session()

        dummy_img = np.full((1080, 1920, 3), 120, dtype=np.uint8)

        # Feed consecutive frames to transition from IDLE -> CALIBRATING -> STABLE
        state = None
        for i in range(5):
            frame = Frame(frame_id=i, timestamp=i * 0.033, image=dummy_img)
            state = orch.process_frame(frame)

        assert state is not None
        assert state.board_detected is True
        assert state.fsm_state == FSMState.STABLE
        assert state.current_ply == 0

    def test_process_frame_unlocated_board(self, tmp_path) -> None:
        """Verifies process_frame executes smoothly when board is not yet located."""
        from src.vision.board_localizer import BaseBoardLocalizer
        from src.core.datatypes import BoardGeometry

        class NoneLocalizer(BaseBoardLocalizer):
            def locate(self, frame):
                return None

        config = load_config()
        config.persistence.base_dir = str(tmp_path / "sessions_unlocated")
        config.analysis.stockfish_enabled = False
        orch = SystemOrchestrator(config=config, board_localizer=NoneLocalizer())
        orch.start_session()

        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = Frame(frame_id=1, timestamp=0.033, image=dummy_img)
        state = orch.process_frame(frame)

        assert state is not None
        assert state.board_detected is False
        assert state.fsm_state == FSMState.LOCATING_BOARD

    def test_pause_and_resume_controls(self, orchestrator_fixture: SystemOrchestrator) -> None:
        orch = orchestrator_fixture
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        orch.pause()
        assert orch._is_paused is True

        frame = Frame(frame_id=1, timestamp=0.033, image=dummy_img)
        state = orch.process_frame(frame)
        assert state.is_paused is True

        orch.resume()
        assert orch._is_paused is False

        is_now_paused = orch.toggle_pause()
        assert is_now_paused is True
        is_now_paused = orch.toggle_pause()
        assert is_now_paused is False

    def test_uncertainty_resolution_manual(self, orchestrator_fixture: SystemOrchestrator) -> None:
        orch = orchestrator_fixture
        orch.start_session()

        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = Frame(frame_id=1, timestamp=0.033, image=dummy_img)
        orch.process_frame(frame)

        # Force UNCERTAIN state
        orch.fsm.force_state(FSMState.UNCERTAIN)
        orch._active_uncertainty = True

        resolved = orch.resolve_uncertainty(move_uci="e2e4")
        assert resolved is True
        assert orch._active_uncertainty is False
        assert orch.state_manager.ply == 1
        assert orch.fsm.current_state == FSMState.STABLE

    def test_rollback_move(self, orchestrator_fixture: SystemOrchestrator) -> None:
        orch = orchestrator_fixture
        orch.start_session()

        # Commit move manually
        orch.resolve_uncertainty(move_uci="e2e4")
        assert orch.state_manager.ply == 1

        popped = orch.rollback_move()
        assert popped is not None
        assert popped.move_uci == "e2e4"
        assert orch.state_manager.ply == 0

    def test_pgn_export(self, orchestrator_fixture: SystemOrchestrator) -> None:
        orch = orchestrator_fixture
        orch.start_session(white_player="Alice", black_player="Bob")
        orch.resolve_uncertainty(move_uci="e2e4")

        pgn_text = orch.export_pgn()
        assert 'White "Alice"' in pgn_text
        assert 'Black "Bob"' in pgn_text
        assert "1. e4" in pgn_text

    def test_stop_and_manifest_finalization(self, orchestrator_fixture: SystemOrchestrator) -> None:
        orch = orchestrator_fixture
        orch.start_session()
        orch.resolve_uncertainty(move_uci="e2e4")

        manifest = orch.stop()
        assert manifest.is_completed is True
        assert manifest.total_plies == 1
        assert len(manifest.journal_checksum) == 64  # Valid SHA-256 hex string

        # Check files exist on disk
        assert (orch.session_dir / "game.pgn").exists()
        assert (orch.session_dir / "manifest.json").exists()
        assert (orch.session_dir / manifest.journal_path).exists()
        assert orch.engine_service.is_running is False

    def test_zero_shot_single_square_noise_produces_no_phantom_changes(
        self, orchestrator_fixture: SystemOrchestrator
    ) -> None:
        """Single-square glare/noise must not cause phantom changed squares or UNCERTAIN state."""
        orch = orchestrator_fixture
        orch.start_session()

        from src.ml.piece_classifier import probabilities_to_cell_observation
        from src.core.datatypes import BoardObservation, BoardOrientation, CHESS_SQUARES

        img = np.full((640, 640, 3), 120, dtype=np.uint8)
        warped_before = img.copy()
        warped_after = img.copy()
        # Inject localized single-square glare onto only one square (e.g. a1 region)
        rois = orch.roi_extractor.extract_all_rois(warped_after, BoardOrientation.WHITE_BOTTOM)
        x1, y1, x2, y2 = rois["a1"][0].board_bbox
        warped_after[y1 + 6 : y2 - 6, x1 + 6 : x2 - 6] = 240

        # Simulate raw classifier predicting empty everywhere (32 mismatches)
        probs_empty = np.zeros(13, dtype=np.float32)
        probs_empty[0] = 0.99
        probs_empty[1:] = 0.01 / 12.0

        raw_obs = {sq: probabilities_to_cell_observation(sq, probs_empty) for sq in CHESS_SQUARES}
        refined = orch._refine_zero_shot_with_temporal_delta(
            cell_obs=raw_obs,
            rois=rois,
            warped_after=warped_after,
            warped_before=warped_before,
        )

        # Every square should reconcile with starting board without phantom changed squares
        obs = BoardObservation.create_default(frame_id=1, timestamp=1.0)
        obs.cells = refined
        assert orch.candidate_generator.detect_changed_squares(orch.state_manager.current_board, obs) == []

    def test_recalibrate_resets_fsm_and_resolve_stale_candidate_safe(
        self, orchestrator_fixture: SystemOrchestrator
    ) -> None:
        orch = orchestrator_fixture
        orch.start_session()
        orch.fsm.force_state(FSMState.UNCERTAIN)
        orch._active_uncertainty = True

        # Inject an illegal stale hypothesis (not legal in starting board)
        from src.core.datatypes import DecisionResult, DecisionStatus, MoveHypothesis
        stale_hypo = MoveHypothesis(
            move_uci="e4e5",
            move_san="e5",
            score_visual=0.5,
            score_temporal=1.0,
            score_geometry=1.0,
            penalty_occlusion=0.0,
            total_score=0.6,
            changed_squares=["e4", "e5"],
        )
        orch._last_decision = DecisionResult(
            status=DecisionStatus.UNCERTAIN,
            top_hypotheses=[stale_hypo],
        )
        assert orch.resolve_uncertainty(candidate_index=0) is False

        orch.recalibrate()
        assert orch.fsm.current_state == FSMState.IDLE
        assert orch._active_uncertainty is False

    def test_two_ply_sequence_and_physical_takeback(
        self, orchestrator_fixture: SystemOrchestrator
    ) -> None:
        """Verifies 2-ply rapid move burst discovery and physical takeback detection."""
        orch = orchestrator_fixture
        orch.start_session()

        # 1. Verify bounded 2-ply path finder (1. e4 e5)
        b = chess.Board()
        pairs = orch.candidate_generator.find_two_ply_sequence(b, {"e2", "e4", "e7", "e5"})
        assert any(m1.uci() == "e2e4" and m2.uci() == "e7e5" for m1, m2 in pairs)

        # 2. Verify physical takeback detection when board returns to pre-commit snapshot
        pre_img = np.full((640, 640, 3), 120, dtype=np.uint8)
        post_img = pre_img.copy()
        rois = orch.roi_extractor.extract_all_rois(pre_img, orch._orientation)
        for sq in ("e2", "e4"):
            bx1, by1, bx2, by2 = orch.roi_extractor.get_base_core_bbox(sq, orch._orientation)
            post_img[by1:by2, bx1:bx2] = 220

        orch.resolve_uncertainty(move_uci="e2e4")
        assert orch.state_manager.ply == 1
        orch._pre_commit_warped_board = pre_img
        orch._stable_warped_board = post_img
        orch._last_committed_move_squares = ["e2", "e4"]

        assert orch._check_physical_takeback(pre_img) is True
        assert orch._check_physical_takeback(post_img) is False

    def test_edge_2ply_completion_with_subtle_dark_square(
        self, orchestrator_fixture: SystemOrchestrator
    ) -> None:
        """Verifies high-contrast 2-ply completion catches dark/edge moves with subtle origin deltas."""
        from src.core.datatypes import CHESS_SQUARES, PIECE_CLASSES
        from src.ml.piece_classifier import probabilities_to_cell_observation

        orch = orchestrator_fixture
        orch.start_session()

        empty_probs = np.zeros(len(PIECE_CLASSES), dtype=np.float32)
        empty_probs[0] = 0.96
        empty_probs[1:] = 0.04 / (len(PIECE_CLASSES) - 1)
        cell_obs = {sq: probabilities_to_cell_observation(sq, empty_probs) for sq in CHESS_SQUARES}

        deltas = {sq: 0.020 for sq in CHESS_SQUARES}
        deltas["d2"] = 0.14
        deltas["d4"] = 0.28
        deltas["g6"] = 0.60
        deltas["g7"] = 0.026  # Subtle dark-on-dark edge square > 1.10 * 0.020

        dummy_board = np.zeros((640, 640, 3), dtype=np.uint8)
        refined = orch._refine_zero_shot_with_temporal_delta(
            cell_obs=cell_obs,
            rois={},
            warped_after=dummy_board,
            warped_before=dummy_board,
            precomputed_deltas=(deltas, 0.020, 0.056),
        )
        assert orch._pending_two_ply_second_move == chess.Move.from_uci("g7g6")
        assert refined["d2"].predicted_piece == "empty"
        assert refined["d4"].is_occupied is True

    def test_uncertain_clears_verify_buffer_no_spin(
        self, orchestrator_fixture: SystemOrchestrator
    ) -> None:
        """Verifies UNCERTAIN evaluation clears _verify_buffer instead of spinning every frame."""
        from src.core.datatypes import CHESS_SQUARES, PIECE_CLASSES, BoardObservation
        from src.ml.piece_classifier import probabilities_to_cell_observation

        orch = orchestrator_fixture
        orch.config.temporal.verifying_stable_frames = 3
        orch.start_session()
        orch.fsm.force_state(FSMState.UNCERTAIN)

        # Build ambiguous observation between e2e4 and e2e3 so DecisionGate emits UNCERTAIN
        empty_probs = np.zeros(len(PIECE_CLASSES), dtype=np.float32)
        empty_probs[0] = 0.96
        empty_probs[1:] = 0.04 / (len(PIECE_CLASSES) - 1)
        deltas = {sq: 0.015 for sq in CHESS_SQUARES}
        deltas["e2"] = 0.25
        deltas["e3"] = 0.24
        deltas["e4"] = 0.24
        dummy_board = np.zeros((640, 640, 3), dtype=np.uint8)
        obs_cells = orch._refine_zero_shot_with_temporal_delta(
            cell_obs={sq: probabilities_to_cell_observation(sq, empty_probs) for sq in CHESS_SQUARES},
            rois={},
            warped_after=dummy_board,
            warped_before=dummy_board,
            precomputed_deltas=(deltas, 0.015, 0.056),
        )
        for fid in range(1, 4):
            obs = BoardObservation(
                frame_id=fid,
                timestamp=float(fid) * 0.2,
                cells=obs_cells,
                motion_detected=False,
                motion_level=0.0,
                hand_present=False,
                geometry_confidence=1.0,
            )
            orch.fsm.update(obs)

        assert orch.fsm.current_state == FSMState.UNCERTAIN
        assert len(orch.fsm._verify_buffer) == 0

    def test_onnx_classifier_small_raw_change_does_not_bypass_3d_lean_refinement(
        self, orchestrator_fixture: SystemOrchestrator
    ) -> None:
        """Even when ONNXPieceClassifier reports raw_changed <= 6 due to lean, 3D tracker must refine."""
        from src.core.datatypes import BoardObservation, CHESS_SQUARES, PIECE_CLASSES
        from src.ml.piece_classifier import ONNXPieceClassifier, probabilities_to_cell_observation
        from src.reasoning.candidate_generator import board_square_to_piece_class

        orch = orchestrator_fixture
        orch.start_session()
        onnx_stub = ONNXPieceClassifier.__new__(ONNXPieceClassifier)
        onnx_stub._session = None
        onnx_stub.confidence_threshold = 0.2
        onnx_stub.occupancy_threshold = 0.5
        orch.piece_classifier = onnx_stub
        orch.occupancy_classifier._session = None

        board = orch.state_manager.current_board
        # Build raw classifier output where e2->e4 pawn leaned onto f4 (raw_changed == 2 <= 6: e2=empty, f4=wP)
        cell_obs = {}
        for sq in CHESS_SQUARES:
            cls = board_square_to_piece_class(board, chess.parse_square(sq))
            if sq == "e2":
                cls = "empty"
            elif sq == "f4":
                cls = "wP"
            probs = np.array(
                [0.92 if c == cls else 0.08 / (len(PIECE_CLASSES) - 1) for c in PIECE_CLASSES],
                dtype=np.float32,
            )
            cell_obs[sq] = probabilities_to_cell_observation(sq, probs)

        deltas = {sq: 0.015 for sq in CHESS_SQUARES}
        deltas["e2"] = 0.14
        deltas["e4"] = 0.12
        deltas["f4"] = 0.02
        dummy_board = np.zeros((640, 640, 3), dtype=np.uint8)
        refined = orch._refine_zero_shot_with_temporal_delta(
            cell_obs=cell_obs,
            rois={},
            warped_after=dummy_board,
            warped_before=dummy_board,
            precomputed_deltas=(deltas, 0.015, 0.056),
        )
        changed = orch.candidate_generator.detect_changed_squares(
            board, BoardObservation.create_default(1, 1.0, cells=refined)
        )
        assert sorted(changed) == ["e2", "e4"]

    def test_pending_physical_anomaly_blocks_accepted_commit(
        self, orchestrator_fixture: SystemOrchestrator
    ) -> None:
        """If 3D tracker flags _pending_physical_anomaly, ACCEPTED decisions must still become UNCERTAIN."""
        from src.core.datatypes import BoardObservation, DecisionResult, DecisionStatus, MoveHypothesis

        orch = orchestrator_fixture
        orch.start_session()
        orch._pending_physical_anomaly = ["f4", "f2"]
        dec = DecisionResult(
            status=DecisionStatus.ACCEPTED,
            accepted_hypothesis=MoveHypothesis(
                move_uci="f2f4",
                move_san="f4",
                changed_squares=["f2", "f4"],
                score_visual=0.80,
                total_score=0.76,
            ),
            top_hypotheses=[],
        )
        orch._on_move_decision(dec, BoardObservation.create_default(1, 1.0))
        assert dec.status == DecisionStatus.UNCERTAIN
        assert dec.accepted_hypothesis is None
        assert dec.top_hypotheses == []
        assert dec.uncertain_squares == ["f4", "f2"]
        assert orch._active_uncertainty is True

