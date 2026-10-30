"""Benchmark harness and metric evaluation module for chess game recording.

Compares detected MoveEvent streams against ground-truth PGN records,
calculating Move Accuracy (%), FEN distances, first error ply, and latencies.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import chess
import chess.pgn
from pydantic import BaseModel, Field

from src.core.datatypes import MoveEvent


def compute_fen_distance(fen_a: str, fen_b: str) -> int:
    """Computes square-by-square difference between two FEN positions.
    
    Returns the count of squares (0..64) where pieces or empty states differ.
    """
    board_a = chess.Board(fen_a)
    board_b = chess.Board(fen_b)
    diff_count = 0
    for sq in chess.SQUARES:
        if board_a.piece_at(sq) != board_b.piece_at(sq):
            diff_count += 1
    return diff_count


class BenchmarkResult(BaseModel):
    """Detailed evaluation metrics comparing predicted moves against ground truth."""
    total_plies: int = Field(ge=0, description="Total half-moves in ground truth game")
    predicted_plies: int = Field(ge=0, description="Total half-moves produced by recorder")
    matched_plies: int = Field(ge=0, description="Number of correctly identified moves")
    move_accuracy: float = Field(ge=0.0, le=100.0, description="Accuracy percentage of moves")
    first_error_ply: Optional[int] = Field(
        default=None,
        description="First ply where predicted move differed from ground truth",
    )
    fen_distances: List[int] = Field(
        default_factory=list,
        description="FEN distance (number of mismatched squares) at each matched ply",
    )
    mean_fen_distance: float = Field(
        default=0.0,
        ge=0.0,
        description="Average FEN difference across all evaluated plies",
    )
    mean_latency_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Average latency in seconds between move event and ground truth",
    )
    game_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="PGN header metadata (Event, Site, White, Black, Result)",
    )

    def to_markdown_report(self) -> str:
        """Formats the evaluation result into a standardized Markdown report."""
        status = "PASSED" if self.move_accuracy >= 95.0 else "FAILED"
        lines = [
            f"# Benchmark Evaluation Report: {status}",
            "",
            f"- **Event:** {self.game_headers.get('Event', 'N/A')}",
            f"- **Players:** {self.game_headers.get('White', 'N/A')} vs {self.game_headers.get('Black', 'N/A')}",
            f"- **Result:** {self.game_headers.get('Result', '*')}",
            "",
            "## Summary Metrics",
            f"| Metric | Value | Target |",
            f"| :--- | :--- | :--- |",
            f"| **Move Accuracy** | `{self.move_accuracy:.2f}%` | $\\ge 95.0\\%$ |",
            f"| **Matched Plies** | `{self.matched_plies} / {self.total_plies}` | `{self.total_plies}` |",
            f"| **First Error Ply** | `{self.first_error_ply if self.first_error_ply is not None else 'None'}` | `None` |",
            f"| **Mean FEN Distance** | `{self.mean_fen_distance:.3f}` | `0.000` |",
            f"| **Mean Latency** | `{self.mean_latency_sec:.3f} s` | $\\le 1.5$ s |",
            "",
        ]
        return "\n".join(lines)


class BenchmarkGroundTruth:
    """Represents reference moves and FEN positions from an official PGN."""

    def __init__(self, pgn_content_or_path: Union[str, Path]) -> None:
        self.game = self._load_game(pgn_content_or_path)
        self.headers: Dict[str, str] = dict(self.game.headers)
        self.moves: List[chess.Move] = list(self.game.mainline_moves())
        self.fens_after: List[str] = []
        self.sans: List[str] = []
        self.ucis: List[str] = []

        board = self.game.board()
        for m in self.moves:
            self.sans.append(board.san(m))
            self.ucis.append(m.uci())
            board.push(m)
            self.fens_after.append(board.fen())

    @staticmethod
    def _load_game(source: Union[str, Path]) -> chess.pgn.Game:
        if isinstance(source, Path) or (isinstance(source, str) and not ("\n" in source or "[" in source)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"PGN file not found: {path.resolve()}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = source

        game = chess.pgn.read_game(io.StringIO(content))
        if game is None:
            raise ValueError("Failed to parse valid chess game from PGN content")
        return game

    @property
    def total_plies(self) -> int:
        return len(self.moves)


class BenchmarkEvaluator:
    """Evaluates predicted move sequences against ground truth."""

    def __init__(self, ground_truth: Union[BenchmarkGroundTruth, str, Path]) -> None:
        if isinstance(ground_truth, BenchmarkGroundTruth):
            self.gt = ground_truth
        else:
            self.gt = BenchmarkGroundTruth(ground_truth)

    def evaluate(
        self,
        predicted_moves: List[MoveEvent],
        ground_truth_timestamps: Optional[List[float]] = None,
    ) -> BenchmarkResult:
        """Evaluates predicted move events.
        
        Args:
            predicted_moves: Chronological list of MoveEvent instances.
            ground_truth_timestamps: Optional list of timestamps when each ply occurred.
            
        Returns:
            BenchmarkResult with accuracy %, first error, FEN distances, latency.
        """
        matched_plies = 0
        first_error_ply: Optional[int] = None
        fen_distances: List[int] = []
        latencies: List[float] = []

        eval_length = min(len(predicted_moves), self.gt.total_plies)

        for i in range(eval_length):
            pred = predicted_moves[i]
            expected_uci = self.gt.ucis[i]
            expected_fen = self.gt.fens_after[i]

            fen_dist = compute_fen_distance(pred.fen_after, expected_fen)
            fen_distances.append(fen_dist)

            if pred.move_uci == expected_uci and fen_dist == 0:
                matched_plies += 1
            else:
                if first_error_ply is None:
                    first_error_ply = i + 1  # 1-indexed ply

            if ground_truth_timestamps and i < len(ground_truth_timestamps):
                latency = max(0.0, pred.timestamp - ground_truth_timestamps[i])
                latencies.append(latency)

        if first_error_ply is None and len(predicted_moves) < self.gt.total_plies:
            first_error_ply = len(predicted_moves) + 1

        accuracy = (matched_plies / float(self.gt.total_plies) * 100.0) if self.gt.total_plies > 0 else 0.0
        mean_fen = (sum(fen_distances) / float(len(fen_distances))) if fen_distances else 0.0
        mean_lat = (sum(latencies) / float(len(latencies))) if latencies else 0.0

        return BenchmarkResult(
            total_plies=self.gt.total_plies,
            predicted_plies=len(predicted_moves),
            matched_plies=matched_plies,
            move_accuracy=accuracy,
            first_error_ply=first_error_ply,
            fen_distances=fen_distances,
            mean_fen_distance=mean_fen,
            mean_latency_sec=mean_lat,
            game_headers=self.gt.headers,
        )
