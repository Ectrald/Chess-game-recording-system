"""Regression tests for Phase 1 & Phase 2 fixes: audit events, FSM UNCERTAIN stability, and whole-board contracts."""

from __future__ import annotations

from pathlib import Path
import chess
import numpy as np
import pytest

from src.core.config import AppConfig, TemporalConfig
from src.core.datatypes import (
    CHESS_SQUARES,
    PIECE_CLASSES,
    BoardGeometry,
    BoardObservation,
    BoardOrientation,
    CellObservation,
    FSMState,
    Frame,
    MoveEvent,
)
from src.ml.piece_classifier import (
    DINOv2PatchChangeDetector,
    MockPieceClassifier,
    WholeBoardChessQClassifier,
)
from src.orchestrator import SystemOrchestrator
from src.persistence.event_repository import JSONLEventRepository
from src.persistence.recovery_manager import RecoveryManager
from src.reasoning.fsm import GameFSM
from src.vision.board_localizer import ManualCornerLocalizer
from src.vision.hand_detector import DummyHandDetector


def _make_obs(ts: float, motion_level: float = 0.0, motion_detected: bool = False) -> BoardObservation:
    obs = BoardObservation.create_default(
        timestamp=ts,
        frame_id=int(ts * 10),
        motion_detected=motion_detected,
        motion_level=motion_level,
        hand_present=False,
        geometry_confidence=1.0,
    )
    return obs


def test_rollback_and_illegal_attempt_audit_chain(tmp_path: Path):
    """Verify rollback & illegal_attempt events keep cryptographic SHA-256 chain intact and fold cleanly."""
    repo = JSONLEventRepository(session_dir=tmp_path)
    b0 = chess.Board()
    m1 = chess.Move.from_uci("e2e4")
    fen_b1 = b0.fen()
    b0.push(m1)
    fen_a1 = b0.fen()

    ev1 = repo.append_event(
        MoveEvent(
            ply=1,
            fen_before=fen_b1,
            move_uci="e2e4",
            move_san="e4",
            fen_after=fen_a1,
            confidence=0.95,
            timestamp=1.0,
            evidence_frame_id=10,
        )
    )
    # Record illegal attempt
    repo.append_event(
        MoveEvent(
            event_type="illegal_attempt",
            ply=1,
            fen_before=fen_a1,
            move_uci="0000",
            move_san="ILLEGAL:e4,f4",
            fen_after=fen_a1,
            confidence=0.0,
            timestamp=2.0,
            evidence_frame_id=20,
        )
    )
    # Rollback e2e4
    repo.append_event(
        MoveEvent(
            event_type="rollback",
            ply=1,
            fen_before=ev1.fen_after,
            move_uci=ev1.move_uci,
            move_san=ev1.move_san,
            fen_after=ev1.fen_before,
            confidence=1.0,
            timestamp=3.0,
            evidence_frame_id=30,
        )
    )
    # Replay d2d4 as ply 1
    b_alt = chess.Board()
    m_alt = chess.Move.from_uci("d2d4")
    b_alt.push(m_alt)
    repo.append_event(
        MoveEvent(
            ply=1,
            fen_before=chess.STARTING_FEN,
            move_uci="d2d4",
            move_san="d4",
            fen_after=b_alt.fen(),
            confidence=0.94,
            timestamp=4.0,
            evidence_frame_id=40,
        )
    )

    is_valid, reason = repo.verify_integrity()
    assert is_valid is True, f"Integrity failed: {reason}"
    assert repo.count_illegal_attempts() == 1
    active = repo.read_active_moves()
    assert [e.move_uci for e in active] == ["d2d4"]

    rec = RecoveryManager(session_dir=tmp_path)
    state_mgr, recovered = rec.recover_session()
    assert state_mgr.ply == 1
    assert state_mgr.current_fen == b_alt.fen()
    assert [e.move_uci for e in recovered] == ["d2d4"]


def test_fsm_uncertain_ignores_transient_motion_flicker():
    """UNCERTAIN modal state must ignore single-frame lighting shimmer before re-entering MOVING."""
    cfg = TemporalConfig(motion_diff_threshold=0.04, settling_time_sec=0.5, verifying_stable_frames=2)
    fsm = GameFSM(config=cfg)
    fsm.force_state(FSMState.UNCERTAIN)

    # Single-frame transient spike must stay UNCERTAIN
    s1 = fsm.update(_make_obs(1.0, motion_level=0.08, motion_detected=True))
    assert s1 == FSMState.UNCERTAIN
    s2 = fsm.update(_make_obs(1.1, motion_level=0.0, motion_detected=False))
    assert s2 == FSMState.UNCERTAIN

    # Sustained 3-frame disturbance transitions to MOVING
    fsm.update(_make_obs(2.0, motion_level=0.08, motion_detected=True))
    fsm.update(_make_obs(2.1, motion_level=0.08, motion_detected=True))
    s3 = fsm.update(_make_obs(2.2, motion_level=0.08, motion_detected=True))
    assert s3 == FSMState.MOVING


def test_whole_board_and_dino_missing_weights_graceful(tmp_path: Path):
    """Missing ONNX weights initialize cleanly with is_available=False without raising."""
    cq = WholeBoardChessQClassifier(model_path=tmp_path / "missing_cq.onnx")
    dino = DINOv2PatchChangeDetector(model_path=tmp_path / "missing_dino.onnx")
    assert cq.is_available is False
    assert dino.is_available is False
    assert cq.detect_orientation(np.zeros((640, 640, 3), dtype=np.uint8))[0] == BoardOrientation.UNKNOWN


def test_orchestrator_propagates_geometry_conf_and_motion_level(tmp_path: Path):
    """Observation cells must receive real motion_level and geometry_confidence from orchestrator."""
    cfg = AppConfig()
    cfg.persistence.base_dir = str(tmp_path / "sessions")
    cfg.analysis.stockfish_enabled = False
    cfg.piece_classifier.use_chessq_whole_board = False
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    corners = [(80.0, 40.0), (560.0, 40.0), (560.0, 440.0), (80.0, 440.0)]
    localizer = ManualCornerLocalizer(corners=corners, subpixel_refinement=False)
    orch = SystemOrchestrator(
        config=cfg,
        session_id="test_prop",
        board_localizer=localizer,
        hand_detector=DummyHandDetector(),
        piece_classifier=MockPieceClassifier(),
    )
    orch.start_session()
    orch._orientation = BoardOrientation.WHITE_BOTTOM
    orch._orientation_calibrated = True
    state = orch.process_frame(Frame(frame_id=1, timestamp=0.1, image=img, width=640, height=480))
    assert state.board_detected is True
    assert orch._last_observation is not None
    assert 0.0 < orch._last_observation.geometry_confidence <= 1.0
    assert all(0.0 <= c.occlusion_ratio <= 1.0 and c.motion_level >= 0.0 for c in orch._last_observation.cells.values())
    orch.stop()
