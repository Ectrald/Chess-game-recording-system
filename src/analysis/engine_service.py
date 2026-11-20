"""Asynchronous chess analysis service with bounded queue and fault isolation.

Coordinates background Stockfish worker threads:
- Non-blocking submission: video capture thread never stalls
- Bounded queue prevents memory leaks during high move frequency
- Failure and timeout resilience: engine crashes do not rollback moves
- Dispatches typed lifecycle events to UI listeners
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional
import chess

from src.core.config import AnalysisConfig
from src.core.datatypes import (
    AnalysisResult,
    EngineTaskStatus,
    MoveEvent,
)
from src.analysis.move_evaluator import MoveEvaluator
from src.analysis.stockfish_worker import BaseEngineWorker, create_engine_worker

logger = logging.getLogger(__name__)


class AsyncEngineService:
    """Asynchronous background service for chess position evaluation."""

    def __init__(
        self,
        config: AnalysisConfig,
        worker: Optional[BaseEngineWorker] = None,
        on_analysis_completed: Optional[Callable[[AnalysisResult], None]] = None,
        on_analysis_failed: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Initializes AsyncEngineService.

        Args:
            config: AnalysisConfig with depths, timeouts, and queue size.
            worker: Optional custom engine worker (creates default worker if None).
            on_analysis_completed: Callback fired upon successful move evaluation.
            on_analysis_failed: Callback fired upon analysis error or skip (event_id, reason).
        """
        self.config = config
        self.worker = worker or create_engine_worker(config)
        self.evaluator = MoveEvaluator(blunder_threshold_cp=config.blunder_threshold_cp)

        self.on_analysis_completed = on_analysis_completed
        self.on_analysis_failed = on_analysis_failed

        self._queue: queue.Queue[MoveEvent] = queue.Queue(maxsize=config.max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    @property
    def queue_size(self) -> int:
        """Current number of pending tasks in queue."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """True if worker thread is actively running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Starts the background evaluation worker thread."""
        with self._lock:
            if self.is_running:
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="StockfishAnalysisWorker",
                daemon=True,
            )
            self._thread.start()
            logger.info("AsyncEngineService worker thread started")

    def stop(self, timeout_sec: float = 2.0) -> None:
        """Gracefully stops worker thread and cleans up engine resources."""
        with self._lock:
            self._stop_event.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=timeout_sec)
                self._thread = None

            self.worker.close()
            logger.info("AsyncEngineService worker stopped and closed")

    def submit_move(self, event: MoveEvent) -> bool:
        """Submits a MoveEvent for background evaluation without blocking the caller.

        Args:
            event: Confirmed MoveEvent.

        Returns:
            True if task was queued, False if disabled or dropped due to queue overflow.
        """
        if not self.config.stockfish_enabled:
            logger.debug("Engine analysis disabled; skipping move ply=%d", event.ply)
            return False

        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            logger.warning(
                "Analysis queue full (%d tasks); dropping analysis for move ply=%d (%s)",
                self.config.max_queue_size,
                event.ply,
                event.move_uci,
            )
            if self.on_analysis_failed:
                try:
                    self.on_analysis_failed(event.event_id, "Queue overflow (backpressure dropped)")
                except Exception as e:
                    logger.error("Error in on_analysis_failed callback: %s", e)
            return False

    def _worker_loop(self) -> None:
        """Internal consumer loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_event(event)
            except Exception as exc:
                logger.error("Failed to process analysis task for event %s: %s", event.event_id, exc, exc_info=True)
                if self.on_analysis_failed:
                    try:
                        self.on_analysis_failed(event.event_id, str(exc))
                    except Exception as e:
                        logger.error("Error in on_analysis_failed callback: %s", e)
            finally:
                self._queue.task_done()

    def _process_event(self, event: MoveEvent) -> None:
        """Evaluates before/after positions for an event and emits completion callback."""
        board_before = chess.Board(event.fen_before)
        board_after = chess.Board(event.fen_after)

        eval_before = self.worker.evaluate(
            board_before,
            depth=self.config.stockfish_depth,
            timeout_sec=self.config.timeout_sec,
        )

        eval_after = self.worker.evaluate(
            board_after,
            depth=self.config.stockfish_depth,
            timeout_sec=self.config.timeout_sec,
        )

        result = self.evaluator.evaluate_move(event, eval_before, eval_after)

        # Mutate event with calculated metrics
        event.eval_centipawns = result.eval_loss_cp
        event.is_blunder = result.is_blunder

        if self.on_analysis_completed:
            try:
                self.on_analysis_completed(result)
            except Exception as exc:
                logger.error("Error in on_analysis_completed callback: %s", exc, exc_info=True)

