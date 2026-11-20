"""Tests for JSONLEventRepository and cryptographic SHA-256 audit hashing."""

import json
from pathlib import Path
import pytest
import chess

from src.core.datatypes import MoveEvent
from src.persistence.event_repository import JSONLEventRepository, compute_event_hash


def make_event(ply: int, move_uci: str, move_san: str, fen_before: str, fen_after: str) -> MoveEvent:
    return MoveEvent(
        ply=ply,
        fen_before=fen_before,
        move_uci=move_uci,
        move_san=move_san,
        fen_after=fen_after,
        confidence=0.92,
        timestamp=float(ply),
        evidence_frame_id=ply * 10,
    )


def test_repository_append_and_read(tmp_path: Path):
    repo = JSONLEventRepository(session_dir=tmp_path)
    assert repo.count == 0

    ev1 = make_event(
        ply=1,
        move_uci="e2e4",
        move_san="e4",
        fen_before=chess.STARTING_FEN,
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    )
    committed1 = repo.append_event(ev1)

    assert committed1.prev_event_hash is None
    assert committed1.event_hash is not None
    assert repo.count == 1
    assert repo.last_event_hash == committed1.event_hash

    # Move 2: 1... e5
    ev2 = make_event(
        ply=2,
        move_uci="e7e5",
        move_san="e5",
        fen_before=committed1.fen_after,
        fen_after="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    )
    committed2 = repo.append_event(ev2)

    assert committed2.prev_event_hash == committed1.event_hash
    assert committed2.event_hash is not None
    assert committed2.event_hash != committed1.event_hash
    assert repo.count == 2

    # Read back all events
    events = repo.read_all_events()
    assert len(events) == 2
    assert events[0].move_san == "e4"
    assert events[1].move_san == "e5"


def test_repository_integrity_verification_success(tmp_path: Path):
    repo = JSONLEventRepository(session_dir=tmp_path)
    board = chess.Board()

    for ply in range(1, 5):
        move = list(board.legal_moves)[0]
        san = board.san(move)
        fen_before = board.fen()
        board.push(move)
        fen_after = board.fen()
        repo.append_event(make_event(ply, move.uci(), san, fen_before, fen_after))

    is_valid, reason = repo.verify_integrity()
    assert is_valid is True
    assert reason is None


def test_repository_tamper_detection(tmp_path: Path):
    repo = JSONLEventRepository(session_dir=tmp_path)
    ev1 = make_event(
        1, "e2e4", "e4", chess.STARTING_FEN, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    repo.append_event(ev1)

    # Tamper with the file contents directly (change SAN 'e4' to 'd4')
    journal_file = tmp_path / "events.jsonl"
    with journal_file.open("r", encoding="utf-8") as f:
        content = f.read()

    tampered_content = content.replace('"move_san":"e4"', '"move_san":"d4"')
    with journal_file.open("w", encoding="utf-8") as f:
        f.write(tampered_content)

    # Re-verify integrity on tampered file
    repo_tampered = JSONLEventRepository(session_dir=tmp_path)
    is_valid, reason = repo_tampered.verify_integrity()
    assert is_valid is False
    assert "event_hash verification failed" in reason


def test_repository_hash_chain_break_detection(tmp_path: Path):
    repo = JSONLEventRepository(session_dir=tmp_path)
    ev1 = repo.append_event(
        make_event(1, "e2e4", "e4", chess.STARTING_FEN, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    )
    ev2 = repo.append_event(
        make_event(2, "e7e5", "e5", ev1.fen_after, "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    )

    # Corrupt line 2's prev_event_hash
    journal_file = tmp_path / "events.jsonl"
    lines = journal_file.read_text().splitlines()
    d2 = json.loads(lines[1])
    d2["prev_event_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
    lines[1] = json.dumps(d2)
    journal_file.write_text("\n".join(lines) + "\n")

    is_valid, reason = repo.verify_integrity()
    assert is_valid is False
    assert "prev_event_hash mismatch" in reason

