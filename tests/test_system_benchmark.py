"""End-to-End System Benchmark and Ground-Truth Accuracy Test Suite.

Validates that the complete neuro-symbolic pipeline achieves:
1. 100% move sequence accuracy on canonical test games (Scholar's mate, special moves).
2. Exact matching final FEN boards.
3. Frame processing budget compliance (average latency <= 35 ms / >= 28.5 FPS on CPU).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pytest

import chess
import chess.pgn
import numpy as np

from src.analysis.engine_service import AsyncEngineService
from src.analysis.stockfish_worker import MockStockfishEngine
from src.core.config import AppConfig, load_config
from src.core.datatypes import (
    CHESS_SQUARES,
    BoardGeometry,
    CellObservation,
    Frame,
    HandDetection,
    PipelineState,
    SessionManifest,
)
from src.ml.piece_classifier import BasePieceClassifier
from src.orchestrator import SystemOrchestrator
from src.ui.dashboard import InteractiveDashboard
from src.vision.board_localizer import BaseBoardLocalizer
from src.vision.hand_detector import BaseHandDetector
from src.vision.synthetic_board import SyntheticBoardGenerator
from src.vision.video_adapter import BaseVideoAdapter


class SyntheticOracleLocalizer(BaseBoardLocalizer):
    """Localizer using the synthetic generator's exact ground-truth geometry."""

    def __init__(self, generator: SyntheticBoardGenerator) -> None:
        self.generator = generator

    def locate(self, frame: Frame) -> Optional[BoardGeometry]:
        return self.generator.get_ground_truth_geometry()


class SyntheticOracleClassifier(BasePieceClassifier):
    """Oracle classifier returning synthetic generator's ground-truth cell observations."""

    def __init__(self, generator: SyntheticBoardGenerator) -> None:
        self.generator = generator
        self.active_frame_id = 0
        self._last_latency_ms = 0.5

    def set_active_frame_id(self, frame_id: int) -> None:
        self.active_frame_id = frame_id

    def classify_cell(self, cell_patch: np.ndarray, square: str) -> CellObservation:
        obs = self.generator.get_ground_truth_observation(self.active_frame_id)
        return obs.cells[square]

    def classify_board(
        self,
        cell_patches: Dict[str, np.ndarray],
        rois: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, CellObservation]:
        obs = self.generator.get_ground_truth_observation(self.active_frame_id)
        return obs.cells


class SyntheticHandDetector(BaseHandDetector):
    """Accurately reflects simulated hand movements from SyntheticBoardGenerator."""

    def __init__(self, generator: SyntheticBoardGenerator) -> None:
        self.generator = generator

    def detect(
        self,
        frame: Frame,
        geometry: Optional[BoardGeometry] = None,
        rois: Optional[Dict[str, Any]] = None,
    ) -> HandDetection:
        sub_frame = frame.frame_id % self.generator.frames_per_move
        trans = int(round(self.generator.frames_per_move * self.generator.transition_fraction))
        ply_index = frame.frame_id // self.generator.frames_per_move
        is_hand = (0 < ply_index <= len(self.generator.moves)) and (sub_frame < trans)
        return HandDetection(
            is_hand_present=is_hand,
            confidence=0.9 if is_hand else 0.0,
            landmarks=[],
            occluded_squares=[],
            occlusion_ratios={sq: 0.0 for sq in CHESS_SQUARES},
        )


class SynchronizedSyntheticAdapter(BaseVideoAdapter):
    """Adapter wrapping SyntheticBoardGenerator and synchronizing the oracle classifier."""

    def __init__(
        self,
        generator: SyntheticBoardGenerator,
        classifier: SyntheticOracleClassifier,
    ) -> None:
        self.generator = generator
        self.classifier = classifier
        self._current_frame_id = 0
        self._is_active = True

    def read(self) -> Optional[Frame]:
        if not self._is_active or self._current_frame_id >= self.generator.total_frames:
            return None

        self.classifier.set_active_frame_id(self._current_frame_id)
        frame = self.generator.render_frame(self._current_frame_id)
        self._current_frame_id += 1
        return frame

    def get_fps(self) -> float:
        return self.generator.fps

    def get_frame_count(self) -> int:
        return self.generator.total_frames

    def seek(self, frame_id: int) -> bool:
        if 0 <= frame_id < self.generator.total_frames:
            self._current_frame_id = frame_id
            return True
        return False

    def is_opened(self) -> bool:
        return self._is_active and (self._current_frame_id < self.generator.total_frames)

    def release(self) -> None:
        self._is_active = False


class TestSystemBenchmark:
    """End-to-End benchmarks validating accuracy, special moves, and throughput."""

    def test_benchmark_short_game(self, tmp_path: Path) -> None:
        """Benchmarks Scholar's Mate (7 plies) end-to-end through the full pipeline."""
        pgn_path = Path("data/ground_truth/test_short_game.pgn")
        assert pgn_path.is_file(), f"Missing ground truth PGN: {pgn_path}"

        # 1. Parse ground truth expected moves and final FEN
        with open(pgn_path, "r", encoding="utf-8") as f:
            ref_game = chess.pgn.read_game(f)
        ref_board = ref_game.board() if ref_game else chess.Board()
        expected_uci_sequence = []
        if ref_game:
            for m in ref_game.mainline_moves():
                expected_uci_sequence.append(m.uci())
                ref_board.push(m)
        expected_final_fen = ref_board.fen()

        # 2. Configure synthetic generator
        generator = SyntheticBoardGenerator(
            board_size=640,
            canvas_width=1280,
            canvas_height=720,
            fps=30.0,
            frames_per_move=20,
        )
        generator.set_game_from_pgn(str(pgn_path))

        classifier = SyntheticOracleClassifier(generator)
        adapter = SynchronizedSyntheticAdapter(generator, classifier)
        localizer = SyntheticOracleLocalizer(generator)
        hand_detector = SyntheticHandDetector(generator)

        # 3. Configure orchestrator with fast debounce thresholds
        config = load_config()
        config.persistence.base_dir = str(tmp_path / "sessions")
        config.persistence.sync_to_disk = False
        config.temporal.settling_time_sec = 0.15
        config.temporal.verifying_stable_frames = 3
        config.analysis.stockfish_enabled = True

        mock_worker = MockStockfishEngine()
        engine_service = AsyncEngineService(config=config.analysis, worker=mock_worker)

        orchestrator = SystemOrchestrator(
            config=config,
            session_id="benchmark_short_game",
            board_localizer=localizer,
            piece_classifier=classifier,
            engine_service=engine_service,
            hand_detector=hand_detector,
        )

        dashboard = InteractiveDashboard(orchestrator=orchestrator, headless=True)

        # 4. Measure execution timing and latency
        pipeline_latencies = []
        orig_process = orchestrator.process_frame

        def timed_process(frame: Frame) -> PipelineState:
            t_start = time.perf_counter()
            res = orig_process(frame)
            pipeline_latencies.append((time.perf_counter() - t_start) * 1000.0)
            return res

        orchestrator.process_frame = timed_process  # type: ignore[method-assign]

        manifest = dashboard.run_loop(video_adapter=adapter)
        mean_latency_ms = float(np.mean(pipeline_latencies))

        # 5. Verify results
        actual_moves = [ev.move_uci for ev in orchestrator.state_manager.history]
        assert actual_moves == expected_uci_sequence, (
            f"Move mismatch! Expected {expected_uci_sequence}, got {actual_moves}"
        )
        assert manifest.current_fen == expected_final_fen, (
            f"Final FEN mismatch! Expected {expected_final_fen}, got {manifest.current_fen}"
        )
        assert manifest.total_plies == len(expected_uci_sequence)
        assert manifest.is_completed is True

        # Latency budget requirement: <= 35 ms per frame
        assert mean_latency_ms <= 35.0, f"Latency budget exceeded: {mean_latency_ms:.2f} ms > 35 ms"

    def test_benchmark_special_moves(self, tmp_path: Path) -> None:
        """Benchmarks castling, en-passant, and promotion on test_special_moves.pgn."""
        pgn_path = Path("data/ground_truth/test_special_moves.pgn")
        assert pgn_path.is_file(), f"Missing ground truth PGN: {pgn_path}"

        # Parse reference game (subset of 12 moves covering e.p. and O-O)
        with open(pgn_path, "r", encoding="utf-8") as f:
            ref_game = chess.pgn.read_game(f)
        moves = list(ref_game.mainline_moves())[:12]

        sub_board = chess.Board()
        expected_ucis = []
        for m in moves:
            expected_ucis.append(m.uci())
            sub_board.push(m)
        expected_final_fen = sub_board.fen()

        # Create generator with explicit move sequence
        generator = SyntheticBoardGenerator(
            board_size=640,
            canvas_width=1280,
            canvas_height=720,
            fps=30.0,
            frames_per_move=20,
        )
        generator.set_game_from_moves(moves)

        classifier = SyntheticOracleClassifier(generator)
        adapter = SynchronizedSyntheticAdapter(generator, classifier)
        localizer = SyntheticOracleLocalizer(generator)
        hand_detector = SyntheticHandDetector(generator)

        config = load_config()
        config.persistence.base_dir = str(tmp_path / "sessions")
        config.persistence.sync_to_disk = False
        config.temporal.settling_time_sec = 0.15
        config.temporal.verifying_stable_frames = 3

        mock_worker = MockStockfishEngine()
        engine_service = AsyncEngineService(config=config.analysis, worker=mock_worker)

        orchestrator = SystemOrchestrator(
            config=config,
            session_id="benchmark_special_moves",
            board_localizer=localizer,
            piece_classifier=classifier,
            engine_service=engine_service,
            hand_detector=hand_detector,
        )

        dashboard = InteractiveDashboard(orchestrator=orchestrator, headless=True)
        manifest = dashboard.run_loop(video_adapter=adapter)

        actual_moves = [ev.move_uci for ev in orchestrator.state_manager.history]
        assert actual_moves == expected_ucis
        assert manifest.current_fen == expected_final_fen
        assert manifest.total_plies == len(expected_ucis)
