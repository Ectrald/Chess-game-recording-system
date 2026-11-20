"""End-to-End integration tests for Persistence and Engine Analysis subsystems."""

from pathlib import Path
import time
import numpy as np
import pytest
import chess
import chess.pgn

from src.core.config import AnalysisConfig
from src.core.datatypes import BoardObservation, CellObservation, FSMState, MoveHypothesis
from src.persistence.event_repository import JSONLEventRepository
from src.persistence.evidence_writer import EvidenceWriter
from src.persistence.recovery_manager import RecoveryManager
from src.analysis.engine_service import AsyncEngineService
from src.analysis.stockfish_worker import MockStockfishEngine
from src.reasoning.game_state_manager import GameStateManager


def make_dummy_observation(frame_id: int) -> BoardObservation:
    cells = {
        sq: CellObservation(
            square=sq,
            probabilities={"empty": 1.0},
            predicted_piece="empty",
            is_occupied=False,
            confidence=1.0,
        )
        for sq in [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]
    }
    return BoardObservation(
        frame_id=frame_id,
        timestamp=float(frame_id) / 10.0,
        cells=cells,
        fsm_state=FSMState.STABLE,
        motion_detected=False,
        motion_level=0.0,
        hand_present=False,
        geometry_confidence=0.98,
    )


def test_persistence_analysis_e2e_pipeline(tmp_path: Path):
    session_dir = tmp_path / "test_session_001"
    evidence_dir = session_dir / "evidence"

    # Initialize components
    event_repo = JSONLEventRepository(session_dir=session_dir)
    evidence_writer = EvidenceWriter(evidence_dir=evidence_dir)

    analysis_completed = []
    analysis_config = AnalysisConfig(stockfish_enabled=True, max_queue_size=10, timeout_sec=1.0)
    engine_service = AsyncEngineService(
        config=analysis_config,
        worker=MockStockfishEngine(),
        on_analysis_completed=lambda res: analysis_completed.append(res),
    )
    engine_service.start()

    # Wire GameStateManager callback
    def on_move_committed(ev):
        # 1. Append to journal
        event_repo.append_event(ev)
        # 2. Write evidence
        obs = make_dummy_observation(ev.evidence_frame_id)
        dummy_img = np.zeros((120, 120, 3), dtype=np.uint8)
        evidence_writer.save_evidence(ev, obs, board_image=dummy_img)
        # 3. Submit to async analysis
        engine_service.submit_move(ev)

    state_mgr = GameStateManager(on_move_committed=on_move_committed)

    # Play short game: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
    moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]

    for idx, m_uci in enumerate(moves, start=1):
        hypo = MoveHypothesis(
            move_uci=m_uci,
            move_san=None,
            changed_squares=[],
            score_visual=0.90,
            score_temporal=1.0,
            score_geometry=1.0,
            penalty_occlusion=0.0,
            total_score=0.90,
            is_legal=True,
        )
        obs = make_dummy_observation(frame_id=idx * 15)
        state_mgr.commit(hypo, obs)

    # Allow background analysis worker to complete all moves
    for _ in range(40):
        if len(analysis_completed) == len(moves):
            break
        time.sleep(0.05)

    assert len(analysis_completed) == len(moves)
    engine_service.stop()

    # 1. Verify Event Repository
    assert event_repo.count == 7
    is_valid, reason = event_repo.verify_integrity()
    assert is_valid is True, f"Integrity failed: {reason}"

    # 2. Verify Evidence Writer created all 7 images and 7 metadata files
    images = list(evidence_dir.glob("*.jpg"))
    metas = list(evidence_dir.glob("*.json"))
    assert len(images) == 7
    assert len(metas) == 7

    # 3. Simulate process termination and RecoveryManager cold-start
    recovery = RecoveryManager(session_dir=session_dir)
    recovered_mgr, recovered_events = recovery.recover_session()

    assert recovered_mgr.ply == 7
    assert recovered_mgr.current_fen == state_mgr.current_fen
    assert recovered_mgr.current_board.is_checkmate() is True
    assert len(recovered_events) == 7
    assert recovered_events[-1].move_san == "Qxf7#"

