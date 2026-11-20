"""Test suite for SystemOrchestrator and end-to-end processing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
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
def orchestrator_fixture(tmp_path: Path) -> SystemOrchestrator:
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
    return orch


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

