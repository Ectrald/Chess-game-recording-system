"""Command-Line Interface (CLI) for Chess Game Recording System.

Provides console commands:
- `record`: Real-time camera or video recording with interactive HUD dashboard.
- `audit`: Cryptographic SHA-256 hash-chain verification of session event journals.
- `replay`: Step-by-step authoritative deterministic replaying of recorded games.
- `benchmark`: Automated performance and accuracy benchmark against reference PGNs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import List, Optional

import chess
import chess.pgn

from src.core.config import AppConfig, CameraConfig, load_config
from src.orchestrator import SystemOrchestrator
from src.persistence.event_repository import JSONLEventRepository
from src.persistence.recovery_manager import RecoveryManager
from src.ui.dashboard import InteractiveDashboard
from src.vision.video_adapter import create_video_adapter

logger = logging.getLogger("chess_recording_system")


def setup_logging(verbose: bool = False) -> None:
    """Configures console logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_record(args: argparse.Namespace) -> int:
    """Executes the `record` subcommand."""
    config: AppConfig = load_config(args.config)

    # Determine camera source type
    src = str(args.source)
    if src.endswith(".pgn"):
        src_type = "synthetic"
    elif src.isdigit():
        src_type = "device"
    elif src.startswith("rtsp://") or src.startswith("http://"):
        src_type = "rtsp"
    else:
        src_type = "file"

    config.camera.source_path = src
    config.camera.source_type = src_type
    if args.out_dir:
        config.persistence.base_dir = args.out_dir

    is_headless = args.headless or (not args.display and config.ui.headless)

    logger.info("Initializing recording session: source=%s (type=%s)", src, src_type)
    video_adapter = create_video_adapter(config.camera)
    orchestrator = SystemOrchestrator(config=config, session_id=args.session_id)
    dashboard = InteractiveDashboard(orchestrator=orchestrator, headless=is_headless)

    try:
        manifest = dashboard.run_loop(video_adapter=video_adapter, max_frames=args.max_frames)
    finally:
        video_adapter.release()

    print("\n" + "=" * 60)
    print("SESSION RECORDING COMPLETED")
    print(f"Session ID:       {manifest.session_id}")
    print(f"Total Half-Moves: {manifest.total_plies}")
    print(f"Final FEN:        {manifest.current_fen}")
    print(f"PGN File:         {orchestrator.session_dir / manifest.pgn_path}")
    print(f"Journal Checksum: {manifest.journal_checksum[:16]}...")
    print("=" * 60 + "\n")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Executes the `audit` subcommand to verify cryptographic journal integrity."""
    journal_path = Path(args.journal_path)
    if not journal_path.is_file():
        print(f"Error: Journal file not found: {journal_path}", file=sys.stderr)
        return 1

    print(f"Auditing cryptographic SHA-256 integrity for: {journal_path}")
    repo = JSONLEventRepository(session_dir=journal_path.parent, filename=journal_path.name)
    is_valid, error_msg = repo.verify_integrity()
    count = repo.count

    print("-" * 60)
    if is_valid:
        print(f"STATUS: VALID [PASS]")
        print(f"Verified {count} consecutive tamper-evident move events.")
        print("Hash chain integrity confirmed 100%.")
        print("-" * 60)
        return 0
    else:
        print(f"STATUS: CORRUPTED / TAMPERED [FAIL]")
        print(f"Total events inspected: {count}")
        print(f"Integrity Error: {error_msg}")
        print("-" * 60)
        return 1


def cmd_replay(args: argparse.Namespace) -> int:
    """Executes the `replay` subcommand to step through recorded sessions."""
    journal_path = Path(args.journal_path)
    if not journal_path.is_file():
        print(f"Error: Journal file not found: {journal_path}", file=sys.stderr)
        return 1

    repo = JSONLEventRepository(session_dir=journal_path.parent, filename=journal_path.name)
    events = repo.read_all_events()
    if not events:
        print(f"Journal {journal_path} is empty.")
        return 0

    print(f"Replaying {len(events)} events from: {journal_path}\n")
    board = chess.Board(events[0].fen_before)

    for ev in events:
        move = chess.Move.from_uci(ev.move_uci)
        board.push(move)
        color_prefix = f"{(ev.ply + 1) // 2}. " if ev.ply % 2 == 1 else f"{(ev.ply + 1) // 2}... "
        print(f"Ply {ev.ply:03d}: {color_prefix}{ev.move_san:<7} (UCI: {ev.move_uci}) -> FEN: {ev.fen_after}")
        if args.delay > 0:
            time.sleep(args.delay)

    print("\nFinal Position:")
    print(board)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Executes the `benchmark` subcommand measuring throughput and accuracy."""
    ref_pgn = Path(args.pgn_ref)
    if not ref_pgn.is_file():
        print(f"Error: Reference PGN file not found: {ref_pgn}", file=sys.stderr)
        return 1

    print(f"Starting System Benchmark against: {ref_pgn}")
    config: AppConfig = load_config(args.config)
    config.camera.source_type = "synthetic"
    config.camera.source_path = str(ref_pgn)
    config.camera.fps = args.fps
    config.ui.headless = True

    # Parse reference final FEN
    with open(ref_pgn, "r", encoding="utf-8") as f:
        ref_game = chess.pgn.read_game(f)
    ref_board = ref_game.board() if ref_game else chess.Board()
    if ref_game:
        for m in ref_game.mainline_moves():
            ref_board.push(m)
    expected_fen = ref_board.fen()

    video_adapter = create_video_adapter(config.camera)
    orchestrator = SystemOrchestrator(config=config, session_id="benchmark_run")
    dashboard = InteractiveDashboard(orchestrator=orchestrator, headless=True)

    t0 = time.perf_counter()
    manifest = dashboard.run_loop(video_adapter=video_adapter, max_frames=args.max_frames)
    elapsed = time.perf_counter() - t0
    video_adapter.release()

    fen_matched = (manifest.current_fen == expected_fen)
    avg_fps = video_adapter.get_frame_count() / elapsed if elapsed > 0 else 0.0

    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print(f"Total Time:      {elapsed:.3f} s")
    print(f"Throughput:      {avg_fps:.1f} FPS")
    print(f"Total Plies:     {manifest.total_plies}")
    print(f"Expected FEN:    {expected_fen}")
    print(f"Actual FEN:      {manifest.current_fen}")
    print(f"Accuracy Status: {'MATCH (100% Correct)' if fen_matched else 'MISMATCH'}")
    print("=" * 60 + "\n")

    return 0 if fen_matched else 1


def build_parser() -> argparse.ArgumentParser:
    """Constructs command-line argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="chess-recorder",
        description="Autonomous Chess Game Recording System - CLI and Vision Pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug output")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: record
    p_rec = subparsers.add_parser("record", help="Record game from video or live camera")
    p_rec.add_argument("--source", "-s", default="0", help="Camera device index, video file, or PGN")
    p_rec.add_argument("--config", "-c", default=None, help="Path to YAML configuration file")
    p_rec.add_argument("--out-dir", "-o", default=None, help="Sessions root directory")
    p_rec.add_argument("--session-id", default=None, help="Custom session identifier string")
    p_rec.add_argument("--headless", action="store_true", help="Run in headless mode without GUI")
    p_rec.add_argument("--display", action="store_true", help="Force display window")
    p_rec.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    p_rec.add_argument("--white", default="White Player", help="White player name")
    p_rec.add_argument("--black", default="Black Player", help="Black player name")

    # Subcommand: audit
    p_audit = subparsers.add_parser("audit", help="Verify cryptographic SHA-256 integrity of events.jsonl")
    p_audit.add_argument("journal_path", help="Path to events.jsonl file")

    # Subcommand: replay
    p_replay = subparsers.add_parser("replay", help="Deterministically replay recorded session")
    p_replay.add_argument("journal_path", help="Path to events.jsonl file")
    p_replay.add_argument("--delay", type=float, default=0.05, help="Delay in seconds between plies")

    # Subcommand: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run pipeline benchmark against reference PGN")
    p_bench.add_argument("--pgn-ref", default="data/ground_truth/test_short_game.pgn", help="Ground-truth PGN path")
    p_bench.add_argument("--config", "-c", default=None, help="Configuration file path")
    p_bench.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    p_bench.add_argument("--fps", type=float, default=30.0, help="Synthetic stream FPS")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint function."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.subcommand == "record":
        return cmd_record(args)
    elif args.subcommand == "audit":
        return cmd_audit(args)
    elif args.subcommand == "replay":
        return cmd_replay(args)
    elif args.subcommand == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
