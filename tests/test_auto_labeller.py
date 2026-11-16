"""Unit and integration tests for the AutoLabeller dataset harvesting pipeline.

Covers:
- STAGE_2.md Section 6: test_auto_labeller.py specifications
- piece_to_class_name mapping for all 13 chess classes
- Single-position harvesting and crop resolution validation
- End-to-end dataset harvesting with SyntheticVideoAdapter and PGN
- Manifest validation and class distribution sanity checks
"""

import json
from pathlib import Path
import chess
import cv2
import numpy as np
import pytest

from src.core.config import LabellerConfig, TemporalConfig, VisionConfig
from src.core.datatypes import CHESS_SQUARES, PIECE_CLASSES
from src.ml.auto_labeller import AutoLabeller, piece_to_class_name
from src.vision.hand_detector import MockHandDetector
from src.vision.motion_detector import MotionDetector
from src.vision.synthetic_board import SyntheticBoardGenerator
from src.vision.video_adapter import SyntheticVideoAdapter


def test_piece_to_class_name():
    """Verifies mapping of python-chess Piece objects to standard 13-class string identifiers."""
    assert piece_to_class_name(None) == "empty"
    assert piece_to_class_name(chess.Piece(chess.PAWN, chess.WHITE)) == "wP"
    assert piece_to_class_name(chess.Piece(chess.KNIGHT, chess.WHITE)) == "wN"
    assert piece_to_class_name(chess.Piece(chess.BISHOP, chess.WHITE)) == "wB"
    assert piece_to_class_name(chess.Piece(chess.ROOK, chess.WHITE)) == "wR"
    assert piece_to_class_name(chess.Piece(chess.QUEEN, chess.WHITE)) == "wQ"
    assert piece_to_class_name(chess.Piece(chess.KING, chess.WHITE)) == "wK"

    assert piece_to_class_name(chess.Piece(chess.PAWN, chess.BLACK)) == "bP"
    assert piece_to_class_name(chess.Piece(chess.KNIGHT, chess.BLACK)) == "bN"
    assert piece_to_class_name(chess.Piece(chess.BISHOP, chess.BLACK)) == "bB"
    assert piece_to_class_name(chess.Piece(chess.ROOK, chess.BLACK)) == "bR"
    assert piece_to_class_name(chess.Piece(chess.QUEEN, chess.BLACK)) == "bQ"
    assert piece_to_class_name(chess.Piece(chess.KING, chess.BLACK)) == "bK"

    for piece_type in chess.PIECE_TYPES:
        for color in (chess.WHITE, chess.BLACK):
            cls_name = piece_to_class_name(chess.Piece(piece_type, color))
            assert cls_name in PIECE_CLASSES


def test_harvest_position_crops(tmp_path: Path):
    """Verifies that harvest_position extracts 64 correctly dimensioned crops and saves them."""
    gen = SyntheticBoardGenerator(board_size=640)
    flat_board = gen.render_board_flat(chess.Board())

    adapter = SyntheticVideoAdapter(generator=gen)
    labeller = AutoLabeller(
        video_adapter=adapter,
        pgn_content_or_file="1. e4 e5 *",
        config=LabellerConfig(output_dir=str(tmp_path), target_crop_size=(112, 112)),
    )

    board = chess.Board()
    samples = labeller.harvest_position(
        board_image=flat_board,
        board=board,
        ply_index=0,
        timestamp_sec=0.0,
        output_dir=tmp_path,
    )

    assert len(samples) == 64
    for sample in samples:
        img_path = tmp_path / sample.image_rel_path
        assert img_path.exists(), f"Image {img_path} not found"
        img = cv2.imread(str(img_path))
        assert img is not None
        assert img.shape == (112, 112, 3)

    # Initial position check
    sample_a1 = next(s for s in samples if s.square == "a1")
    assert sample_a1.piece_class == "wR"
    sample_e4 = next(s for s in samples if s.square == "e4")
    assert sample_e4.piece_class == "empty"
    sample_e8 = next(s for s in samples if s.square == "e8")
    assert sample_e8.piece_class == "bK"


def test_autolabeller_end_to_end_synthetic(tmp_path: Path):
    """Verifies end-to-end AutoLabeller pipeline on a synthetic 2-move game."""
    # 2 moves: 1. e4 e5 -> 3 positions: ply 0 (start), ply 1 (e4), ply 2 (e5)
    gen = SyntheticBoardGenerator(frames_per_move=15, fps=25)
    gen.set_game_from_moves(["e2e4", "e7e5"])
    adapter = SyntheticVideoAdapter(generator=gen)

    geom = gen.get_ground_truth_geometry()

    temporal_cfg = TemporalConfig(
        settling_time_sec=0.2,
        verifying_stable_frames=2,
        motion_diff_threshold=0.03,
    )
    labeller_cfg = LabellerConfig(
        output_dir=str(tmp_path),
        target_crop_size=(112, 112),
        max_samples_per_move=64,
        save_format="png",
    )

    labeller = AutoLabeller(
        video_adapter=adapter,
        pgn_content_or_file="1. e4 e5 *",
        config=labeller_cfg,
        temporal_config=temporal_cfg,
        geometry=geom,
        hand_detector=MockHandDetector(),
    )

    report = labeller.run()

    # Total samples should be 3 positions * 64 squares = 192 samples
    assert report.total_samples == 3 * 64
    assert report.plies_harvested == 3
    assert report.class_distribution["empty"] > 90
    assert report.class_distribution["wP"] > 0
    assert report.class_distribution["bP"] > 0
    assert report.class_distribution["wK"] == 3
    assert report.class_distribution["bK"] == 3

    # Verify manifest file exists and matches
    manifest_path = tmp_path / "dataset_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_samples"] == 192
    assert manifest["plies_harvested"] == 3
    assert len(manifest["samples"]) == 192
    assert manifest["target_crop_size"] == [112, 112]

    # Verify every recorded sample exists on disk
    for s_meta in manifest["samples"]:
        file_path = tmp_path / s_meta["image_rel_path"]
        assert file_path.exists()


def test_autolabeller_short_game_full(tmp_path: Path, short_game_pgn_path: Path):
    """Verifies complete harvesting of Scholar's Mate test game (7 moves, 512 samples)."""
    gen = SyntheticBoardGenerator(frames_per_move=15, fps=25)
    gen.set_game_from_pgn(str(short_game_pgn_path))
    adapter = SyntheticVideoAdapter(generator=gen)

    geom = gen.get_ground_truth_geometry()

    temporal_cfg = TemporalConfig(
        settling_time_sec=0.15,
        verifying_stable_frames=2,
        motion_diff_threshold=0.03,
    )
    labeller_cfg = LabellerConfig(
        output_dir=str(tmp_path),
        target_crop_size=(112, 112),
        max_samples_per_move=64,
        save_format="png",
    )

    labeller = AutoLabeller(
        video_adapter=adapter,
        pgn_content_or_file=short_game_pgn_path,
        config=labeller_cfg,
        temporal_config=temporal_cfg,
        geometry=geom,
        hand_detector=MockHandDetector(),
    )

    report = labeller.run()

    # 7 plies + initial position = 8 positions * 64 squares = 512 samples
    assert report.total_samples == 8 * 64
    assert report.plies_harvested == 8
    assert report.class_distribution["empty"] == 257
    assert report.class_distribution["wP"] == 64
    assert report.class_distribution["bP"] == 63
    assert report.class_distribution["wK"] == 8
    assert report.class_distribution["bK"] == 8
    assert report.class_distribution["wQ"] == 8

    manifest_path = Path(report.manifest_path)
    assert manifest_path.exists()

