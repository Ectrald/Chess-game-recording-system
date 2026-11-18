"""Unit tests for core data contracts in src/core/datatypes.py."""

import json
import numpy as np
import pytest
from pydantic import ValidationError

from src.core.datatypes import (
    AutoLabellerSample,
    BoardGeometry,
    BoardObservation,
    BoardOrientation,
    CellObservation,
    CellROI,
    Frame,
    FSMState,
    HandDetection,
    HandLandmark,
    MotionMetrics,
    MoveEvent,
    MoveHypothesis,
    Point2D,
    Quadrilateral,
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

    def test_self_intersecting_geometry_raises(self):
        # Butterfly / self-intersecting polygon
        butterfly_corners = [
            (0.0, 0.0),
            (100.0, 100.0),
            (100.0, 0.0),
            (0.0, 100.0),
        ]
        with pytest.raises(ValidationError):
            BoardGeometry(
                corners=butterfly_corners,
                confidence=0.8,
                is_valid=True,
            )

    def test_invalid_allowed_when_is_valid_false(self):
        butterfly_corners = [
            (0.0, 0.0),
            (100.0, 100.0),
            (100.0, 0.0),
            (0.0, 100.0),
        ]
        # Should NOT raise when is_valid=False (e.g. tracking rejection)
        geom = BoardGeometry(
            corners=butterfly_corners,
            confidence=0.0,
            is_valid=False,
        )
        assert geom.is_valid is False

    def test_enhanced_geometry_fields(self, sample_geometry):
        assert sample_geometry.corners_image == sample_geometry.corners
        assert sample_geometry.polygon_area_px2 > 0.0
        assert sample_geometry.aspect_ratio > 0.0
        assert sample_geometry.orientation == BoardOrientation.UNKNOWN
        quad = sample_geometry.to_quadrilateral()
        assert isinstance(quad, Quadrilateral)
        assert len(quad.as_list()) == 4
        assert quad.to_numpy().shape == (4, 2)


class TestPointAndQuadrilateralContracts:
    def test_point2d(self):
        pt = Point2D(x=12.5, y=34.5)
        assert pt.x == 12.5
        assert pt.y == 34.5
        assert pt.as_tuple() == (12.5, 34.5)

    def test_quadrilateral(self):
        quad = Quadrilateral(
            top_left=Point2D(0.0, 0.0),
            top_right=Point2D(100.0, 0.0),
            bottom_right=Point2D(100.0, 100.0),
            bottom_left=Point2D(0.0, 100.0),
        )
        as_list = quad.as_list()
        assert len(as_list) == 4
        assert as_list[0] == (0.0, 0.0)
        assert as_list[2] == (100.0, 100.0)
        np_arr = quad.to_numpy()
        assert np_arr.shape == (4, 2)


class TestCellROIContract:
    def test_valid_cell_roi(self):
        roi = CellROI(
            square="e4",
            index=28,
            board_bbox=(320, 320, 400, 400),
            extended_bbox=(320, 300, 400, 400),
            center_image=(960.0, 540.0),
            center_board=(360.0, 360.0),
        )
        assert roi.square == "e4"
        assert roi.index == 28
        assert roi.board_bbox == (320, 320, 400, 400)

    def test_invalid_cell_roi_square(self):
        with pytest.raises(ValidationError):
            CellROI(
                square="z9",
                index=0,
                board_bbox=(0, 0, 80, 80),
                extended_bbox=(0, 0, 80, 80),
                center_image=(0.0, 0.0),
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


class TestStage2Datatypes:
    def test_fsm_states(self):
        assert FSMState.STABLE.value == "stable"
        assert FSMState.MOVING.value == "moving"
        assert FSMState.SETTLING.value == "settling"
        assert FSMState.VERIFYING.value == "verifying"
        assert FSMState.UNCERTAIN.value == "uncertain"
        assert FSMState.BOARD_LOST.value == "board_lost"

    def test_hand_detection_contract(self):
        det = HandDetection(
            is_hand_present=True,
            confidence=0.92,
            landmarks=[(0.5, 0.5, 0.0), (0.6, 0.6, 0.0)],
            bounding_box=(100, 200, 300, 400),
            occluded_squares=["e4", "e5"],
            occlusion_ratios={"e4": 0.45, "e5": 0.20},
        )
        assert det.is_hand_present is True
        assert det.confidence == 0.92
        assert len(det.occluded_squares) == 2
        assert det.occlusion_ratios["e4"] == 0.45

    def test_motion_metrics_contract(self):
        mask = np.zeros((640, 640), dtype=np.uint8)
        mask[100:150, 100:150] = 255
        metrics = MotionMetrics(
            timestamp=1.5,
            frame_id=45,
            global_motion_score=0.06,
            square_motion_scores={"e4": 0.35},
            is_motion_detected=True,
            motion_mask=mask,
        )
        assert metrics.timestamp == 1.5
        assert metrics.frame_id == 45
        assert metrics.global_motion_score == 0.06
        assert metrics.is_motion_detected is True
        assert metrics.motion_mask.shape == (640, 640)

    def test_auto_labeller_sample_contract(self):
        sample = AutoLabellerSample(
            sample_id="0001_e4_1.50",
            ply_index=1,
            square="e4",
            piece_class="wP",
            fen_after_ply="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            timestamp_sec=1.5,
            image_rel_path="train/wP/0001_e4_1.50.png",
        )
        assert sample.sample_id == "0001_e4_1.50"
        assert sample.square == "e4"
        assert sample.piece_class == "wP"

    def test_board_observation_with_stage2_fields(self, sample_board_observation):
        sample_board_observation.fsm_state = FSMState.STABLE
        sample_board_observation.motion_metrics = MotionMetrics(
            global_motion_score=0.01,
            is_motion_detected=False,
        )
        sample_board_observation.hand_detection = HandDetection(
            is_hand_present=False,
            confidence=0.0,
        )
        assert sample_board_observation.fsm_state == FSMState.STABLE
        assert sample_board_observation.motion_metrics.global_motion_score == 0.01
        assert sample_board_observation.hand_detection.is_hand_present is False


class TestStage3Datatypes:
    def test_model_manifest_contract(self):
        from src.core.datatypes import ModelManifest, PIECE_CLASSES

        manifest = ModelManifest(
            model_name="piece_classifier_v1",
            architecture="efficientnet_b0",
            format="onnx",
            val_macro_f1=0.965,
            test_macro_f1=0.952,
            latency_cpu_ms=14.8,
            trained_at="2026-11-18T12:00:00Z",
        )
        assert manifest.model_name == "piece_classifier_v1"
        assert manifest.architecture == "efficientnet_b0"
        assert manifest.format == "onnx"
        assert manifest.input_shape == [64, 3, 112, 112]
        assert manifest.classes == PIECE_CLASSES
        assert manifest.val_macro_f1 == 0.965
        assert manifest.test_macro_f1 == 0.952
        assert manifest.latency_cpu_ms == 14.8

    def test_evaluation_report_contract(self):
        from src.core.datatypes import EvaluationReport

        report = EvaluationReport(
            overall_accuracy=0.98,
            occupancy_accuracy=0.995,
            color_accuracy=0.99,
            macro_precision=0.97,
            macro_recall=0.96,
            macro_f1=0.965,
            per_class_f1={"empty": 0.999, "wP": 0.98},
            confusion_matrix=[[10, 0], [0, 10]],
            expected_calibration_error=0.025,
        )
        assert report.overall_accuracy == 0.98
        assert report.occupancy_accuracy == 0.995
        assert report.macro_f1 == 0.965
        assert report.expected_calibration_error == 0.025


