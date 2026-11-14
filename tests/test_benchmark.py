"""Unit tests for benchmark harness and evaluation in src/core/benchmark.py."""

import chess
import pytest

from src.core.benchmark import (
    BenchmarkEvaluator,
    BenchmarkGroundTruth,
    compute_fen_distance,
)
from src.core.datatypes import MoveEvent


class TestFENCalculation:
    def test_identical_fens_have_zero_distance(self):
        fen = chess.STARTING_FEN
        assert compute_fen_distance(fen, fen) == 0

    def test_single_pawn_move_has_distance_two(self):
        # 1. e4: e2 emptied, e4 occupied -> 2 squares changed
        fen_start = chess.STARTING_FEN
        board = chess.Board()
        board.push_san("e4")
        fen_after = board.fen()
        assert compute_fen_distance(fen_start, fen_after) == 2


class TestBenchmarkGroundTruth:
    def test_load_short_game(self, short_game_pgn_path):
        gt = BenchmarkGroundTruth(short_game_pgn_path)
        assert gt.total_plies == 7
        assert gt.headers.get("Event") == "Scholar's Mate Test Game"
        assert gt.ucis[0] == "e2e4"
        assert gt.sans[0] == "e4"
        assert gt.sans[-1] == "Qxf7#"

    def test_load_special_moves(self, special_moves_pgn_path):
        gt = BenchmarkGroundTruth(special_moves_pgn_path)
        assert gt.total_plies == 116
        # Check castling, en passant, promotion exist in moves
        assert "O-O" in gt.sans
        assert "O-O-O" in gt.sans
        assert "exd6" in gt.sans
        assert "g1=Q" in gt.sans


class TestBenchmarkEvaluator:
    def test_perfect_accuracy_evaluation(self, short_game_pgn_path):
        gt = BenchmarkGroundTruth(short_game_pgn_path)
        evaluator = BenchmarkEvaluator(gt)

        # Generate synthetic predicted moves matching ground truth exactly
        predicted_moves = []
        board = chess.Board()
        for i, move in enumerate(gt.moves):
            fen_before = board.fen()
            san = board.san(move)
            board.push(move)
            fen_after = board.fen()

            predicted_moves.append(
                MoveEvent(
                    ply=i + 1,
                    fen_before=fen_before,
                    move_uci=move.uci(),
                    move_san=san,
                    fen_after=fen_after,
                    confidence=0.95,
                    timestamp=float(i) * 2.0 + 0.3,  # 300ms latency
                    evidence_frame_id=(i + 1) * 60,
                )
            )

        gt_timestamps = [float(i) * 2.0 for i in range(len(predicted_moves))]
        result = evaluator.evaluate(predicted_moves, ground_truth_timestamps=gt_timestamps)

        assert result.total_plies == 7
        assert result.matched_plies == 7
        assert result.move_accuracy == 100.0
        assert result.first_error_ply is None
        assert result.mean_fen_distance == 0.0
        assert 0.25 <= result.mean_latency_sec <= 0.35

        report = result.to_markdown_report()
        assert "PASSED" in report
        assert "100.00%" in report

    def test_imperfect_accuracy_first_error(self, short_game_pgn_path):
        gt = BenchmarkGroundTruth(short_game_pgn_path)
        evaluator = BenchmarkEvaluator(gt)

        # Move 1 is correct (e4), Move 2 is wrong (c5 instead of e5)
        m1 = MoveEvent(
            ply=1,
            fen_before=chess.STARTING_FEN,
            move_uci="e2e4",
            move_san="e4",
            fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            confidence=0.9,
            timestamp=1.0,
            evidence_frame_id=30,
        )
        m2_wrong = MoveEvent(
            ply=2,
            fen_before=m1.fen_after,
            move_uci="c7c5",  # Wrong move! Ground truth expects e7e5
            move_san="c5",
            fen_after="rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            confidence=0.8,
            timestamp=2.0,
            evidence_frame_id=60,
        )

        result = evaluator.evaluate([m1, m2_wrong])
        assert result.matched_plies == 1
        assert result.first_error_ply == 2
        assert result.move_accuracy < 100.0
        assert result.mean_fen_distance > 0.0
