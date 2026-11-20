"""Stockfish UCI process supervisor, timeout controller, and mock engine.

Provides robust process lifecycle management for the Stockfish chess engine:
- Automatic path discovery across operating system locations
- Automatic fallback to MockStockfishEngine when Stockfish binary is absent
- Strict calculation timeouts per position
- Automatic process restart upon freeze or crash
- Thread-safe evaluation requests
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import shutil
import threading
import time
from typing import List, Optional
import chess
import chess.engine

from src.core.config import AnalysisConfig
from src.core.datatypes import EngineEvaluation

logger = logging.getLogger(__name__)


class BaseEngineWorker(ABC):
    """Abstract interface for chess position evaluation engines."""

    @abstractmethod
    def evaluate(self, board: chess.Board, depth: int, timeout_sec: float) -> EngineEvaluation:
        """Evaluates board position up to search depth or timeout."""
        pass

    @abstractmethod
    def restart(self) -> None:
        """Restarts the underlying engine process."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Gracefully closes engine resources."""
        pass


class MockStockfishEngine(BaseEngineWorker):
    """Lightweight in-process chess analyzer using material balance and king safety heuristics.

    Used for testing, CI environments, and environments without a native Stockfish binary.
    """

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 305,
        chess.BISHOP: 333,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        logger.info("Initialized MockStockfishEngine (in-process heuristic fallback)")

    def evaluate(self, board: chess.Board, depth: int = 10, timeout_sec: float = 1.0) -> EngineEvaluation:
        start_t = time.time()
        with self._lock:
            # Check for terminal state
            if board.is_checkmate():
                # Side to move is checkmated
                mate_in = -1 if board.turn == chess.WHITE else 1
                return EngineEvaluation(
                    eval_cp=None,
                    mate_in=mate_in,
                    best_move_uci=None,
                    pv=[],
                    depth=depth,
                    nodes=100,
                    time_sec=time.time() - start_t,
                )

            # Calculate material balance from White's perspective
            white_material = 0
            black_material = 0
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece is not None:
                    val = self.PIECE_VALUES.get(piece.piece_type, 0)
                    if piece.color == chess.WHITE:
                        white_material += val
                    else:
                        black_material += val

            # Mobility bonus: 3 centipawns per legal move
            mobility_bonus = len(list(board.legal_moves)) * 3
            if board.turn == chess.WHITE:
                white_material += mobility_bonus
            else:
                black_material += mobility_bonus

            eval_cp = white_material - black_material

            # Pick best move: capture high-value piece if possible, otherwise first legal move
            legal_moves = list(board.legal_moves)
            best_move: Optional[chess.Move] = None
            if legal_moves:
                captures = [m for m in legal_moves if board.is_capture(m)]
                if captures:
                    # Pick highest value captured piece
                    captures.sort(
                        key=lambda m: self.PIECE_VALUES.get(
                            board.piece_type_at(m.to_square) or 0, 0
                        ),
                        reverse=True,
                    )
                    best_move = captures[0]
                else:
                    best_move = legal_moves[0]

            best_move_uci = best_move.uci() if best_move else None
            pv = [best_move_uci] if best_move_uci else []

            return EngineEvaluation(
                eval_cp=eval_cp,
                mate_in=None,
                best_move_uci=best_move_uci,
                pv=pv,
                depth=depth,
                nodes=len(legal_moves) * 150,
                time_sec=round(time.time() - start_t, 4),
            )

    def restart(self) -> None:
        logger.info("MockStockfishEngine restarted")

    def close(self) -> None:
        logger.info("MockStockfishEngine closed")


class StockfishUCIWorker(BaseEngineWorker):
    """Subprocess supervisor communicating with real Stockfish via standard UCI."""

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._executable_path: Optional[str] = self._discover_executable()

        if self._executable_path:
            self._start_engine()
        else:
            logger.warning(
                "Stockfish binary not located; delegating analysis to MockStockfishEngine fallback"
            )
            self._mock_fallback = MockStockfishEngine()

    def _discover_executable(self) -> Optional[str]:
        """Locates Stockfish executable on the host system."""
        # 1. Config path
        if self.config.stockfish_path:
            p = Path(self.config.stockfish_path)
            if p.is_file() and os.access(str(p), os.X_OK):
                return str(p)

        # 2. System PATH
        which_path = shutil.which("stockfish")
        if which_path:
            return which_path

        # 3. Common Unix locations
        candidates = [
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/games/stockfish",
            "/opt/homebrew/bin/stockfish",
        ]
        for c in candidates:
            if Path(c).is_file():
                return c

        return None

    def _start_engine(self) -> None:
        """Spawns Stockfish subprocess with configured UCI options."""
        if not self._executable_path:
            return

        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._executable_path)
            # Configure threads and hash memory
            options = {
                "Threads": self.config.threads,
                "Hash": self.config.hash_mb,
            }
            try:
                self._engine.configure(options)
            except Exception as e:
                logger.warning("Could not set UCI options %s: %s", options, e)

            logger.info("Stockfish UCI engine started successfully from: %s", self._executable_path)
        except Exception as e:
            logger.error("Failed to spawn Stockfish engine: %s; falling back to Mock", e)
            self._engine = None
            self._mock_fallback = MockStockfishEngine()

    def evaluate(self, board: chess.Board, depth: int = 15, timeout_sec: float = 2.5) -> EngineEvaluation:
        with self._lock:
            if self._engine is None:
                return self._mock_fallback.evaluate(board, depth, timeout_sec)

            start_t = time.time()
            try:
                limit = chess.engine.Limit(depth=depth, time=timeout_sec)
                info = self._engine.analyse(board, limit)

                # Extract score
                score_obj = info.get("score")
                eval_cp: Optional[int] = None
                mate_in: Optional[int] = None

                if score_obj is not None:
                    white_score = score_obj.white()
                    if white_score.is_mate():
                        mate_in = white_score.mate()
                    else:
                        eval_cp = white_score.score()

                pv_moves: List[str] = []
                pv_list = info.get("pv", [])
                if pv_list:
                    pv_moves = [m.uci() for m in pv_list]

                best_move_uci = pv_moves[0] if pv_moves else None
                depth_reached = info.get("depth", depth)
                nodes_counted = info.get("nodes", 0)

                return EngineEvaluation(
                    eval_cp=eval_cp,
                    mate_in=mate_in,
                    best_move_uci=best_move_uci,
                    pv=pv_moves,
                    depth=depth_reached,
                    nodes=nodes_counted,
                    time_sec=round(time.time() - start_t, 4),
                )
            except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as e:
                logger.error("Stockfish engine crashed during analysis: %s. Restarting...", e)
                self.restart()
                # Use mock for this failed call
                return MockStockfishEngine().evaluate(board, depth, timeout_sec)
            except Exception as e:
                logger.error("Unexpected error in Stockfish analysis: %s", e)
                return MockStockfishEngine().evaluate(board, depth, timeout_sec)

    def restart(self) -> None:
        with self._lock:
            self.close()
            if self._executable_path:
                self._start_engine()

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.quit()
                except Exception as e:
                    logger.debug("Error while quitting Stockfish: %s", e)
                self._engine = None


def create_engine_worker(config: AnalysisConfig) -> BaseEngineWorker:
    """Factory creating configured Stockfish worker or Mock fallback."""
    if not config.stockfish_enabled:
        return MockStockfishEngine()
    return StockfishUCIWorker(config)

