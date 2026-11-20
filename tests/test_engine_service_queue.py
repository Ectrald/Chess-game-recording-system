"""Tests for AsyncEngineService: bounded queue, non-blocking execution, and fault isolation."""

import time
import pytest
import chess

from src.core.config import AnalysisConfig
from src.core.datatypes import MoveEvent
from src.analysis.engine_service import AsyncEngineService
from src.analysis.stockfish_worker import MockStockfishEngine


def make_test_event(ply: int) -> MoveEvent:
    return MoveEvent(
        ply=ply,
        fen_before=chess.STARTING_FEN,
        move_uci="e2e4",
        move_san="e4",
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        confidence=0.9,
        timestamp=float(ply),
        evidence_frame_id=ply * 10,
    )


def test_async_engine_service_execution():
    completed_results = []
    config = AnalysisConfig(stockfish_enabled=True, max_queue_size=10, timeout_sec=1.0)
    service = AsyncEngineService(
        config=config,
        worker=MockStockfishEngine(),
        on_analysis_completed=lambda res: completed_results.append(res),
    )

    service.start()
    assert service.is_running is True

    # Submit 3 moves
    for i in range(1, 4):
        ok = service.submit_move(make_test_event(i))
        assert ok is True

    # Wait briefly for worker loop to drain queue
    for _ in range(20):
        if len(completed_results) == 3:
            break
        time.sleep(0.05)

    assert len(completed_results) == 3
    assert completed_results[0].ply == 1
    assert completed_results[0].quality is not None

    service.stop()
    assert service.is_running is False


def test_async_engine_service_bounded_queue_overflow():
    failed_reasons = []
    # Small queue of size 2
    config = AnalysisConfig(stockfish_enabled=True, max_queue_size=2)
    service = AsyncEngineService(
        config=config,
        worker=MockStockfishEngine(),
        on_analysis_failed=lambda eid, reason: failed_reasons.append((eid, reason)),
    )

    # Do NOT start the service so queue fills up
    assert service.submit_move(make_test_event(1)) is True
    assert service.submit_move(make_test_event(2)) is True

    # Third submission should overflow and return False without blocking
    t0 = time.time()
    overflow_submitted = service.submit_move(make_test_event(3))
    elapsed = time.time() - t0

    assert overflow_submitted is False
    assert elapsed < 0.1  # Completely non-blocking!
    assert len(failed_reasons) == 1
    assert "Queue overflow" in failed_reasons[0][1]

    service.stop()


def test_async_engine_service_disabled():
    config = AnalysisConfig(stockfish_enabled=False)
    service = AsyncEngineService(config=config)
    service.start()

    res = service.submit_move(make_test_event(1))
    assert res is False
    assert service.queue_size == 0

    service.stop()

