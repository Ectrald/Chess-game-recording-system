"""Unit tests for core data contracts in src/core/datatypes.py."""

import json
import numpy as np
import pytest
from pydantic import ValidationError

from src.core.datatypes import (
    BoardGeometry,
    BoardObservation,
    CellObservation,
    Frame,
    MoveEvent,
    MoveHypothesis,
    CHESS_SQUARES,
)


class TestFrameContract:
    def test_valid_frame(self, sample_bgr_image):
        frame = Frame(
            frame_id=1,
            timestamp=0.033,
            image=sample_bgr_image,
            width=640,
            height=640,
        )
        assert frame.frame_id == 1
        assert frame.channels == 3
        assert frame.shape == (640, 640, 3)

    def test_auto_dimensions_from_image(self, sample_bgr_image):
        frame = Frame(frame_id=0, timestamp=0.0, image=sample_bgr_image)
        assert frame.width == 640
        assert frame.height == 640

    def test_invalid_image_type(self):
        with pytest.raises(ValidationError):
            Frame(frame_id=0, timestamp=0.0, image="not_an_image")

    def test_invalid_negative_frame_id(self, sample_bgr_image):
        with pytest.raises(ValidationError):
            Frame(frame_id=-1, timestamp=0.0, image=sample_bgr_image)


class TestBoardGeometryContract:
    def test_valid_geometry(self, sample_geometry):
        assert sample_geometry.confidence == 0.95
        assert sample_geometry.is_valid is True
        np_corners = sample_geometry.to_numpy_corners()
        assert np_corners.shape == (4, 2)
        np_homo = sample_geometry.to_numpy_homography()
        assert np_homo.shape == (3, 3)

    def test_invalid_corners_count(self):
        with pytest.raises(ValidationError):
            BoardGeometry(
                corners=[(0.0, 0.0), (1.0, 1.0)],  # only 2 points
                confidence=0.8,
                is_valid=True,
            )

    def test_invalid_homography_shape(self, sample_corners):
        with pytest.raises(ValidationError):
            BoardGeometry(
                corners=sample_corners,
                homography_matrix=[[1.0, 0.0], [0.0, 1.0]],  # 2x2 instead of 3x3
                confidence=0.8,
                is_valid=True,
            )


class TestCellObservationContract:
    def test_valid_cell_with_auto_prediction(self):
        cell = CellObservation(
            square="e4",
            probabilities={"empty": 0.05, "wP": 0.95},
        )
        assert cell.predicted_piece == "wP"
        assert cell.confidence == 0.95
        assert cell.is_occupied is True

    def test_invalid_square_name(self):
        with pytest.raises(ValidationError):
            CellObservation(
                square="e9",  # Rank 9 does not exist
                probabilities={"empty": 1.0},
            )
        with pytest.raises(ValidationError):
            CellObservation(
                square="i4",  # File i does not exist
                probabilities={"empty": 1.0},
            )

    def test_invalid_probability_range(self):
        with pytest.raises(ValidationError):
            CellObservation(
                square="d4",
                probabilities={"empty": 1.5},  # > 1.0
            )


class TestBoardObservationContract:
    def test_exactly_64_cells_required(self, sample_board_observation):
        assert len(sample_board_observation.cells) == 64
        assert sample_board_observation.cells["e4"].square == "e4"

    def test_rejects_incomplete_cells(self):
        # 63 cells (missing h8)
        incomplete_cells = {
            sq: CellObservation(
                square=sq,
                probabilities={"empty": 1.0},
                is_occupied=False,
            )
            for sq in CHESS_SQUARES if sq != "h8"
        }
        with pytest.raises(ValidationError) as exc_info:
            BoardObservation(
                timestamp=0.0,
                frame_id=0,
                cells=incomplete_cells,
                motion_detected=False,
                motion_level=0.0,
                hand_present=False,
                geometry_confidence=1.0,
            )
        assert "must contain exactly 64 cells" in str(exc_info.value)

    def test_occupied_squares_and_diff(self, sample_board_observation):
        obs1 = sample_board_observation
        assert len(obs1.get_occupied_squares()) == 0

        # Create second observation with piece on e4
        cells2 = dict(obs1.cells)
        cells2["e4"] = CellObservation(
            square="e4",
            probabilities={"empty": 0.02, "wP": 0.98},
            predicted_piece="wP",
            confidence=0.98,
            is_occupied=True,
        )
        obs2 = obs1.model_copy(update={"cells": cells2})

        assert obs2.get_occupied_squares() == {"e4"}
        assert obs2.get_piece_map() == {"e4": "wP"}
        assert obs1.diff_squares(obs2) == ["e4"]


class TestMoveContracts:
    def test_move_hypothesis(self):
        hypo = MoveHypothesis(
            move_uci="e2e4",
            move_san="e4",
            changed_squares=["e2", "e4"],
            score_visual=0.88,
            score_temporal=0.95,
            score_geometry=0.99,
            penalty_occlusion=0.0,
            total_score=0.91,
        )
        assert hypo.move_uci == "e2e4"
        assert hypo.total_score == 0.91

    def test_move_event_serialization_and_validation(self):
        event = MoveEvent(
            ply=1,
            fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move_uci="e2e4",
            move_san="e4",
            fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            confidence=0.85,
            timestamp=2.5,
            evidence_frame_id=75,
            eval_centipawns=25,
            is_blunder=False,
        )
        # JSON roundtrip
        json_str = event.model_dump_json()
        restored = MoveEvent.model_validate_json(json_str)
        assert restored.move_uci == "e2e4"
        assert restored.ply == 1

    def test_move_event_invalid_fen(self):
        with pytest.raises(ValidationError):
            MoveEvent(
                ply=1,
                fen_before="invalid fen string with wrong fields",
                move_uci="e2e4",
                move_san="e4",
                fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                confidence=0.8,
                timestamp=1.0,
                evidence_frame_id=30,
            )
