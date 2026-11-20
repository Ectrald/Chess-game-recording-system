"""Test suite for VisualHUDComposer and GUI rendering components."""

from __future__ import annotations

import chess
import cv2
import numpy as np
import pytest

from src.core.datatypes import (
    AnalysisResult,
    DecisionResult,
    DecisionStatus,
    EngineEvaluation,
    EngineTaskStatus,
    FSMState,
    MoveHypothesis,
    MoveQuality,
    PipelineState,
)
from src.ui.visualizer import FSM_COLOR_MAP, QUALITY_COLOR_MAP, VisualHUDComposer


@pytest.fixture
def dummy_frame_bgr() -> np.ndarray:
    """Creates a synthetic camera frame [1080, 1920, 3]."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Add dummy chessboard-like background
    img[200:880, 640:1320] = 180
    return img


@pytest.fixture
def default_pipeline_state() -> PipelineState:
    """Creates standard pipeline state fixture."""
    return PipelineState(
        frame_id=120,
        timestamp=4.0,
        fsm_state=FSMState.STABLE,
        board_detected=True,
        geometry_confidence=0.98,
        hand_detected=False,
        motion_level=0.012,
        current_ply=1,
        current_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        last_move_san="e4",
        last_move_uci="e2e4",
        is_paused=False,
        active_uncertainty=False,
        processing_fps=29.8,
    )


class TestVisualHUDComposer:
    """Test suite covering HUD layout, 2D board render, eval gauges, and modals."""

    def test_composer_initialization(self) -> None:
        composer = VisualHUDComposer(width=1280, height=720, font_scale=0.6)
        assert composer.width == 1280
        assert composer.height == 720
        assert composer.sidebar_width == int(1280 * 0.35)
        assert composer.video_width == 1280 - composer.sidebar_width

    def test_render_virtual_board_default(self) -> None:
        composer = VisualHUDComposer()
        board = chess.Board()
        last_move = chess.Move.from_uci("e2e4")

        board_img = composer.render_virtual_board(board, size=240, last_move=last_move)
        assert isinstance(board_img, np.ndarray)
        assert board_img.shape == (240, 240, 3)
        assert board_img.dtype == np.uint8
        # Ensure image is not entirely black
        assert np.mean(board_img) > 10.0

    def test_render_virtual_board_custom_size(self) -> None:
        composer = VisualHUDComposer()
        board = chess.Board()
        board_img = composer.render_virtual_board(board, size=320)
        assert board_img.shape == (320, 320, 3)

    def test_compose_standard_layout(
        self,
        dummy_frame_bgr: np.ndarray,
        default_pipeline_state: PipelineState,
    ) -> None:
        composer = VisualHUDComposer(width=1280, height=720)
        corners = np.array([[640, 200], [1320, 200], [1320, 880], [640, 880]], dtype=np.float32)

        composed = composer.compose(
            frame_bgr=dummy_frame_bgr,
            state=default_pipeline_state,
            corners=corners,
        )

        assert isinstance(composed, np.ndarray)
        assert composed.shape == (720, 1280, 3)
        assert composed.dtype == np.uint8

    def test_compose_with_hand_mask(
        self,
        dummy_frame_bgr: np.ndarray,
        default_pipeline_state: PipelineState,
    ) -> None:
        composer = VisualHUDComposer(width=1280, height=720, show_hand_landmarks=True)
        default_pipeline_state.hand_detected = True

        h, w = dummy_frame_bgr.shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        hand_mask[400:600, 800:1000] = 255

        composed = composer.compose(
            frame_bgr=dummy_frame_bgr,
            state=default_pipeline_state,
            hand_mask=hand_mask,
        )
        assert composed.shape == (720, 1280, 3)

    @pytest.mark.parametrize("fsm_state", [
        FSMState.IDLE,
        FSMState.LOCATING_BOARD,
        FSMState.CALIBRATING,
        FSMState.STABLE,
        FSMState.MOVING,
        FSMState.SETTLING,
        FSMState.VERIFYING,
        FSMState.UNCERTAIN,
        FSMState.BOARD_LOST,
        FSMState.PAUSED,
    ])
    def test_compose_all_fsm_states(
        self,
        fsm_state: FSMState,
        dummy_frame_bgr: np.ndarray,
        default_pipeline_state: PipelineState,
    ) -> None:
        composer = VisualHUDComposer(width=1280, height=720)
        default_pipeline_state.fsm_state = fsm_state

        composed = composer.compose(
            frame_bgr=dummy_frame_bgr,
            state=default_pipeline_state,
        )
        assert composed.shape == (720, 1280, 3)

    def test_compose_with_stockfish_eval(
        self,
        dummy_frame_bgr: np.ndarray,
        default_pipeline_state: PipelineState,
    ) -> None:
        composer = VisualHUDComposer(width=1280, height=720, show_eval_bar=True)

        eval_result = AnalysisResult(
            event_id="test_ev",
            ply=1,
            move_uci="e2e4",
            fen_after=default_pipeline_state.current_fen,
            status=EngineTaskStatus.COMPLETED,
            eval_before=EngineEvaluation(eval_cp=20, depth=15),
            eval_after=EngineEvaluation(eval_cp=35, depth=15, best_move_uci="e7e5"),
            centipawn_loss=0,
            quality=MoveQuality.BEST,
        )
        default_pipeline_state.latest_eval = eval_result

        composed = composer.compose(
            frame_bgr=dummy_frame_bgr,
            state=default_pipeline_state,
        )
        assert composed.shape == (720, 1280, 3)

    def test_compose_with_mate_eval(
        self,
        dummy_frame_bgr: np.ndarray,
        default_pipeline_state: PipelineState,
    ) -> None:
        composer = VisualHUDComposer(width=1280, height=720, show_eval_bar=True)

        eval_result = AnalysisResult(
            event_id="test_ev",
            ply=7,
            move_uci="d1h5",
            fen_after=default_pipeline_state.current_fen,
            status=EngineTaskStatus.COMPLETED,
            eval_after=EngineEvaluation(mate_in=2, depth=15, best_move_uci="f7f8q"),
            quality=MoveQuality.EXCELLENT,
        )
        default_pipeline_state.latest_eval = eval_result

        composed = composer.compose(
            frame_bgr=dummy_frame_bgr,
            state=default_pipeline_state,
        )
        assert composed.shape == (720, 1280, 3)

    def test_compose_uncertainty_modal(
        self,
        dummy_frame_bgr: np.ndarray,
        default_pipeline_state: PipelineState,
    ) -> None:
        composer = VisualHUDComposer(width=1280, height=720)
        default_pipeline_state.active_uncertainty = True
        default_pipeline_state.fsm_state = FSMState.UNCERTAIN

        hypo1 = MoveHypothesis(
            move_uci="e2e4",
            changed_squares=["e2", "e4"],
            total_score=0.52,
            score_visual=0.50,
        )
        hypo2 = MoveHypothesis(
            move_uci="e2e3",
            changed_squares=["e2", "e3"],
            total_score=0.48,
            score_visual=0.45,
        )
        decision = DecisionResult(
            status=DecisionStatus.UNCERTAIN,
            accepted_hypothesis=hypo1,
            top_hypotheses=[hypo1, hypo2],
            rejection_reason="Score margin Delta < min_margin",
        )
        default_pipeline_state.last_decision = decision

        composed = composer.compose(
            frame_bgr=dummy_frame_bgr,
            state=default_pipeline_state,
        )
        assert composed.shape == (720, 1280, 3)

    def test_compose_paused_banner(
        self,
        dummy_frame_bgr: np.ndarray,
        default_pipeline_state: PipelineState,
    ) -> None:
        composer = VisualHUDComposer(width=1280, height=720)
        default_pipeline_state.is_paused = True

        composed = composer.compose(
            frame_bgr=dummy_frame_bgr,
            state=default_pipeline_state,
        )
        assert composed.shape == (720, 1280, 3)
