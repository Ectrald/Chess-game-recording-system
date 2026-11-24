"""Tests for StockfishUCIWorker, MockStockfishEngine, and timeout controller."""

import pytest
import chess

from src.core.config import AnalysisConfig
from src.analysis.stockfish_worker import (
    MockStockfishEngine,
    StockfishUCIWorker,
    create_engine_worker,
)


def test_mock_engine_starting_position():
    mock = MockStockfishEngine()
    board = chess.Board()

    res = mock.evaluate(board, depth=10, timeout_sec=1.0)
    assert res.eval_cp is not None
    # Material is equal; mobility gives symmetric or close to 0 score
    assert abs(res.eval_cp) < 200
    assert res.best_move_uci is not None
    assert res.depth == 10
    assert res.time_sec >= 0.0


def test_mock_engine_material_advantage():
    mock = MockStockfishEngine()
    # White has extra queen
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    res = mock.evaluate(board, depth=10)
    assert res.eval_cp is not None
    # Queen is ~900 cp advantage
    assert res.eval_cp > 500


def test_mock_engine_checkmate_position():
    mock = MockStockfishEngine()
    # Fool's mate checkmate position: Black delivered Qxf2#
    board = chess.Board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    res = mock.evaluate(board)
    assert res.mate_in is not None
    assert res.mate_in == -1


def test_stockfish_worker_fallback_creation():
    config = AnalysisConfig(stockfish_enabled=True, stockfish_path="/non/existent/path/to/stockfish")
    worker = create_engine_worker(config)
    assert worker is not None

    board = chess.Board()
    eval_res = worker.evaluate(board, depth=5)
    assert eval_res is not None
    assert eval_res.best_move_uci is not None

    worker.restart()
    worker.close()


def test_stockfish_worker_discover_executable_with_real_file(tmp_path):
    """Verifies _discover_executable runs os.access check when pointing to a real file."""
    dummy_exe = tmp_path / "dummy_stockfish"
    dummy_exe.write_text("#!/bin/sh\nexit 0\n")
    dummy_exe.chmod(0o755)

    config = AnalysisConfig(stockfish_enabled=True, stockfish_path=str(dummy_exe))
    worker = StockfishUCIWorker(config)
    assert worker._discover_executable() == str(dummy_exe)


