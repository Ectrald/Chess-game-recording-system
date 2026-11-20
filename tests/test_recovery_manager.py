"""Tests for RecoveryManager and crash recovery procedures."""

from pathlib import Path
import pytest
import chess

from src.core.datatypes import MoveEvent
from src.persistence.event_repository import JSONLEventRepository
from src.persistence.recovery_manager import RecoveryManager


def test_clean_session_recovery(tmp_path: Path):
    repo = JSONLEventRepository(session_dir=tmp_path)
    # Commit moves 1. e4 e5 2. Nf3
    b = chess.Board()
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5"), chess.Move.from_uci("g1f3")]

    for ply, m in enumerate(moves, start=1):
        san = b.san(m)
        fen_b = b.fen()
        b.push(m)
        fen_a = b.fen()
        repo.append_event(
            MoveEvent(
                ply=ply,
                fen_before=fen_b,
                move_uci=m.uci(),
                move_san=san,
                fen_after=fen_a,
                confidence=0.90,
                timestamp=float(ply),
                evidence_frame_id=ply * 10,
                idempotency_key=f"{ply}_{m.uci()}_{ply * 10}",
            )
        )

    recovery = RecoveryManager(session_dir=tmp_path)
    state_mgr, events = recovery.recover_session()

    assert state_mgr.ply == 3
    assert len(events) == 3
    assert state_mgr.current_fen == b.fen()
    assert state_mgr.history[-1].move_san == "Nf3"


def test_crash_recovery_corrupted_trailing_bytes(tmp_path: Path):
    repo = JSONLEventRepository(session_dir=tmp_path)
    ev1 = repo.append_event(
        MoveEvent(
            ply=1,
            fen_before=chess.STARTING_FEN,
            move_uci="e2e4",
            move_san="e4",
            fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            confidence=0.95,
            timestamp=1.0,
            evidence_frame_id=10,
        )
    )

    # Simulate crash: power loss in the middle of writing ply 2
    journal_file = tmp_path / "events.jsonl"
    with journal_file.open("ab") as f:
        # Incomplete half-written JSON fragment
        f.write(b'{"ply": 2, "fen_before": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "move_')

    recovery = RecoveryManager(session_dir=tmp_path)

    # Sanitize should truncate the incomplete fragment
    truncated_bytes = recovery.sanitize_corrupted_tail()
    assert truncated_bytes > 0

    # Session should now recover cleanly up to ply 1
    state_mgr, events = recovery.recover_session()
    assert state_mgr.ply == 1
    assert len(events) == 1
    assert state_mgr.history[0].move_uci == "e2e4"


def test_recovery_empty_session(tmp_path: Path):
    recovery = RecoveryManager(session_dir=tmp_path)
    state_mgr, events = recovery.recover_session()
    assert state_mgr.ply == 0
    assert len(events) == 0
    assert state_mgr.current_fen == chess.STARTING_FEN

