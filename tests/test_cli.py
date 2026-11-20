"""Test suite for CLI interface, subcommands, and console entrypoints."""

from __future__ import annotations

from pathlib import Path
import pytest

import chess

from src.core.datatypes import MoveEvent
from src.persistence.event_repository import JSONLEventRepository
from src.ui.cli import build_parser, cmd_audit, cmd_benchmark, cmd_record, cmd_replay, main


@pytest.fixture
def sample_session_journal(tmp_path: Path) -> Path:
    """Creates a sample valid events.jsonl file with 2 moves."""
    repo = JSONLEventRepository(session_dir=tmp_path, sync_to_disk=False)
    ev1 = MoveEvent(
        ply=1,
        fen_before=chess.STARTING_FEN,
        move_uci="e2e4",
        move_san="e4",
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        confidence=0.98,
        timestamp=1.0,
        evidence_frame_id=30,
    )
    repo.append_event(ev1)

    ev2 = MoveEvent(
        ply=2,
        fen_before=ev1.fen_after,
        move_uci="e7e5",
        move_san="e5",
        fen_after="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        confidence=0.97,
        timestamp=2.0,
        evidence_frame_id=60,
    )
    repo.append_event(ev2)
    return repo.file_path


class TestCLIParser:
    """Tests CLI argument parsing across all subcommands."""

    def test_parser_record_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["record", "--headless"])
        assert args.subcommand == "record"
        assert args.source == "0"
        assert args.headless is True

    def test_parser_audit_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["audit", "data/sessions/game1/events.jsonl"])
        assert args.subcommand == "audit"
        assert args.journal_path == "data/sessions/game1/events.jsonl"

    def test_parser_replay_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["replay", "data/sessions/game1/events.jsonl", "--delay", "0.01"])
        assert args.subcommand == "replay"
        assert args.journal_path == "data/sessions/game1/events.jsonl"
        assert args.delay == 0.01

    def test_parser_benchmark_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--pgn-ref", "data/ground_truth/test_short_game.pgn", "--max-frames", "50"])
        assert args.subcommand == "benchmark"
        assert args.pgn_ref == "data/ground_truth/test_short_game.pgn"
        assert args.max_frames == 50


class TestCLIExecution:
    """Tests execution of individual CLI commands."""

    def test_cmd_audit_valid_journal(self, sample_session_journal: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(["audit", str(sample_session_journal)])
        exit_code = cmd_audit(args)
        assert exit_code == 0

    def test_cmd_audit_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "non_existent.jsonl"
        parser = build_parser()
        args = parser.parse_args(["audit", str(missing)])
        exit_code = cmd_audit(args)
        assert exit_code == 1

    def test_cmd_audit_tampered_journal(self, sample_session_journal: Path) -> None:
        # Tamper with the file contents
        content = sample_session_journal.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Mutate move in first event without updating hash
        lines[0] = lines[0].replace('"e2e4"', '"d2d4"')
        sample_session_journal.write_text("\n".join(lines) + "\n", encoding="utf-8")

        parser = build_parser()
        args = parser.parse_args(["audit", str(sample_session_journal)])
        exit_code = cmd_audit(args)
        assert exit_code == 1

    def test_cmd_replay_valid_journal(self, sample_session_journal: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(["replay", str(sample_session_journal), "--delay", "0.0"])
        exit_code = cmd_replay(args)
        assert exit_code == 0

    def test_cmd_replay_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "non_existent.jsonl"
        parser = build_parser()
        args = parser.parse_args(["replay", str(missing)])
        exit_code = cmd_replay(args)
        assert exit_code == 1

    def test_cmd_record_headless_short(self, tmp_path: Path) -> None:
        pgn_ref = "data/ground_truth/test_short_game.pgn"
        out_dir = str(tmp_path / "sessions")
        exit_code = main([
            "record",
            "--source", pgn_ref,
            "--out-dir", out_dir,
            "--headless",
            "--max-frames", "15",
        ])
        assert exit_code == 0
        # Ensure session was created
        sessions_created = list(Path(out_dir).glob("session_*"))
        assert len(sessions_created) >= 1
        assert (sessions_created[0] / "manifest.json").exists()
        assert (sessions_created[0] / "game.pgn").exists()

