"""Unit tests for vertical parallax disambiguation and gamma-0.5 dark-square sensitivity."""

from pathlib import Path
from typing import Dict

import chess
import cv2
import numpy as np
import pytest

from src.core.config import AppConfig
from src.core.datatypes import CHESS_SQUARES, PIECE_CLASSES, BoardObservation, BoardOrientation
from src.ml.piece_classifier import probabilities_to_cell_observation
from src.orchestrator import SystemOrchestrator


def _make_orchestrator(tmp_path: Path) -> SystemOrchestrator:
    cfg = AppConfig()
    cfg.persistence.base_dir = str(tmp_path)
    cfg.piece_classifier.backend = "mock"
    cfg.vision.hand_detector.enabled = False
    cfg.vision.localizer.mode = "manual"
    cfg.analysis.stockfish_enabled = False
    orch = SystemOrchestrator(config=cfg)
    orch._orientation = BoardOrientation.WHITE_RIGHT
    return orch


def _draw_textured_pawn(
    img: np.ndarray,
    roi_extractor,
    square: str,
    orientation: BoardOrientation,
    base_val: int = 215,
    dark: bool = False,
) -> None:
    x0, y0, x1, y1 = roi_extractor.get_square_coordinates(square, orientation)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cw, ch = x1 - x0, y1 - y0
    # Draw high-contrast upper head + mid collar so top_std is high on occupied squares
    head_color = (18, 18, 18) if dark else (base_val, base_val, base_val)
    rim_color = (95, 95, 95) if dark else (65, 65, 65)
    cv2.circle(img, (cx, int(y0 + 0.30 * ch)), max(6, cw // 5), head_color, -1)
    cv2.circle(img, (cx, int(y0 + 0.30 * ch)), max(6, cw // 5), rim_color, 2)
    cv2.ellipse(
        img,
        (cx, int(y0 + 0.65 * ch)),
        (max(8, cw // 4), max(8, ch // 5)),
        0,
        0,
        360,
        head_color,
        -1,
    )
    cv2.ellipse(
        img,
        (cx, int(y0 + 0.65 * ch)),
        (max(8, cw // 4), max(8, ch // 5)),
        0,
        0,
        360,
        rim_color,
        2,
    )


def _make_empty_board(roi_extractor, orientation: BoardOrientation) -> np.ndarray:
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    for sq in CHESS_SQUARES:
        x0, y0, x1, y1 = roi_extractor.get_square_coordinates(sq, orientation)
        file_idx = chess.square_file(chess.parse_square(sq))
        rank_idx = chess.square_rank(chess.parse_square(sq))
        is_light = (file_idx + rank_idx) % 2 == 1
        val = 190 if is_light else 42
        img[y0:y1, x0:x1] = (val, val, val)
    return img


def _make_noisy_zero_shot_obs(mirage_h4: bool = True) -> Dict[str, object]:
    """Builds zero-shot-like predictions (>6 mismatches vs initial board) that favor h2->h4 mirage."""
    obs: Dict[str, object] = {}
    for sq in CHESS_SQUARES:
        vec = np.full(len(PIECE_CLASSES), 0.02, dtype=np.float32)
        if mirage_h4 and sq == "h4":
            vec[PIECE_CLASSES.index("wP")] = 0.78
        elif sq == "h2":
            vec[0] = 0.75
        else:
            vec[PIECE_CLASSES.index("bP")] = 0.65  # Ensures raw_changed > 6 like zero-shot classifier
        vec /= float(np.sum(vec))
        obs[sq] = probabilities_to_cell_observation(sq, vec, occupancy_threshold=0.50)
    return obs


def _rank_top_move(orch: SystemOrchestrator, cell_obs: Dict[str, object]) -> str:
    board_obs = BoardObservation(
        timestamp=1.0,
        frame_id=10,
        cells=cell_obs,
        motion_detected=False,
        motion_level=0.0,
        hand_present=False,
        geometry_confidence=1.0,
    )
    board = orch.state_manager.current_board
    candidates = orch.candidate_generator.generate_candidates(board, board_obs)
    ranked = orch.move_scorer.rank_candidates(board, candidates, board_obs)
    return ranked[0].move_uci


class TestOrchestratorParallaxAndGamma:
    def test_parallax_disambiguation_promotes_g4_over_h4_mirage(self, tmp_path: Path):
        orch = _make_orchestrator(tmp_path)
        before = _make_empty_board(orch.roi_extractor, BoardOrientation.WHITE_RIGHT)
        for sq in ("h2", "g2", "f2", "e2", "d2", "c2", "b2", "a2"):
            _draw_textured_pawn(before, orch.roi_extractor, sq, BoardOrientation.WHITE_RIGHT)

        after = before.copy()
        # Real move is g2 -> g4: g2 becomes clean/empty, g4 gets pawn base, h2 STILL holds its standing pawn
        gx0, gy0, gx1, gy1 = orch.roi_extractor.get_square_coordinates("g2", BoardOrientation.WHITE_RIGHT)
        after[gy0:gy1, gx0:gx1] = (42, 42, 42)  # g2 is dark square, vacated
        # Modest base change on g4 (white pawn on light g4)
        _draw_textured_pawn(after, orch.roi_extractor, "g4", BoardOrientation.WHITE_RIGHT, base_val=235)

        # Simulate tall pawn on g4 projecting strongly into bottom/core of h4 (dark square),
        # plus slight lighting ripple in h2 base while h2 upper head stays intact!
        hx0, hy0, hx1, hy1 = orch.roi_extractor.get_square_coordinates("h4", BoardOrientation.WHITE_RIGHT)
        cv2.circle(after, ((hx0 + hx1) // 2, int(hy0 + 0.65 * (hy1 - hy0))), 18, (230, 230, 230), -1)
        h2x0, h2y0, h2x1, h2y1 = orch.roi_extractor.get_square_coordinates("h2", BoardOrientation.WHITE_RIGHT)
        cv2.rectangle(
            after,
            (int(h2x0 + 0.25 * (h2x1 - h2x0)), int(h2y0 + 0.58 * (h2y1 - h2y0))),
            (int(h2x0 + 0.75 * (h2x1 - h2x0)), int(h2y0 + 0.78 * (h2y1 - h2y0))),
            (110, 110, 110),
            -1,
        )

        raw_obs = _make_noisy_zero_shot_obs(mirage_h4=True)
        refined = orch._refine_zero_shot_with_temporal_delta(
            cell_obs=raw_obs,
            rois={},
            warped_after=after,
            warped_before=before,
        )
        assert refined["h2"].predicted_piece == "wP"
        assert refined["g2"].predicted_piece == "empty"
        assert _rank_top_move(orch, refined) == "g2g4"

    def test_parallax_does_not_suppress_real_h4_move(self, tmp_path: Path):
        orch = _make_orchestrator(tmp_path)
        before = _make_empty_board(orch.roi_extractor, BoardOrientation.WHITE_RIGHT)
        for sq in ("h2", "g2", "f2", "e2", "d2", "c2", "b2", "a2"):
            _draw_textured_pawn(before, orch.roi_extractor, sq, BoardOrientation.WHITE_RIGHT)

        after = before.copy()
        # Real move is h2 -> h4: h2 actually empties, g2 keeps its pawn, h4 gets white pawn on dark square
        hx0, hy0, hx1, hy1 = orch.roi_extractor.get_square_coordinates("h2", BoardOrientation.WHITE_RIGHT)
        after[hy0:hy1, hx0:hx1] = (190, 190, 190)  # h2 is light square, vacated
        _draw_textured_pawn(after, orch.roi_extractor, "h4", BoardOrientation.WHITE_RIGHT, base_val=230)

        raw_obs = _make_noisy_zero_shot_obs(mirage_h4=True)
        refined = orch._refine_zero_shot_with_temporal_delta(
            cell_obs=raw_obs,
            rois={},
            warped_after=after,
            warped_before=before,
        )
        assert _rank_top_move(orch, refined) == "h2h4"

    def test_gamma_luminance_detects_dark_on_dark_a7_to_a5(self, tmp_path: Path):
        orch = _make_orchestrator(tmp_path)
        before = np.full((640, 640, 3), 26, dtype=np.uint8)
        # Keep most board squares clean dark/light checkerboard so median noise is low
        before = _make_empty_board(orch.roi_extractor, BoardOrientation.WHITE_RIGHT)
        a7x0, a7y0, a7x1, a7y1 = orch.roi_extractor.get_square_coordinates("a7", BoardOrientation.WHITE_RIGHT)
        a5x0, a5y0, a5x1, a5y1 = orch.roi_extractor.get_square_coordinates("a5", BoardOrientation.WHITE_RIGHT)

        # Dark squares with subtle ~20 vs ~27 contrast (low in linear space, lifted by gamma 0.5)
        before[a7y0:a7y1, a7x0:a7x1] = (22, 22, 22)
        cv2.circle(before, ((a7x0 + a7x1) // 2, (a7y0 + a7y1) // 2), 14, (10, 10, 10), -1)
        before[a5y0:a5y1, a5x0:a5x1] = (26, 26, 26)

        after = before.copy()
        after[a7y0:a7y1, a7x0:a7x1] = (26, 26, 26)
        after[a5y0:a5y1, a5x0:a5x1] = (22, 22, 22)
        cv2.circle(after, ((a5x0 + a5x1) // 2, (a5y0 + a5y1) // 2), 14, (10, 10, 10), -1)

        deltas, _, change_thresh = orch._compute_base_core_deltas(after, before)
        assert deltas["a7"] >= change_thresh
        assert deltas["a5"] >= change_thresh

    def test_parallax_destination_lean_recovers_nf3(self, tmp_path: Path):
        """Bright white knight on light f3 leaning into dark g3 must recover g1f3 via upward-lean fallback."""
        orch = _make_orchestrator(tmp_path)
        before = _make_empty_board(orch.roi_extractor, BoardOrientation.WHITE_RIGHT)
        _draw_textured_pawn(before, orch.roi_extractor, "g1", BoardOrientation.WHITE_RIGHT, base_val=230)

        after = before.copy()
        # g1 becomes empty (dark square -> big change)
        g1x0, g1y0, g1x1, g1y1 = orch.roi_extractor.get_square_coordinates("g1", BoardOrientation.WHITE_RIGHT)
        after[g1y0:g1y1, g1x0:g1x1] = (42, 42, 42)
        # f3 gets very subtle white-on-white base footprint (~0.025, < secondary_thresh 0.035), while knight head leans strongly into dark g3
        f3x0, f3y0, f3x1, f3y1 = orch.roi_extractor.get_square_coordinates("f3", BoardOrientation.WHITE_RIGHT)
        cv2.circle(after, ((f3x0 + f3x1) // 2, (f3y0 + f3y1) // 2), 14, (145, 145, 145), 2)
        _draw_textured_pawn(after, orch.roi_extractor, "g3", BoardOrientation.WHITE_RIGHT, base_val=235)

        raw_obs = _make_noisy_zero_shot_obs(mirage_h4=False)
        refined = orch._refine_zero_shot_with_temporal_delta(
            cell_obs=raw_obs,
            rois={},
            warped_after=after,
            warped_before=before,
        )
        assert _rank_top_move(orch, refined) == "g1f3"

    def test_illegal_move_triggers_arbiter_alert_and_recovers_on_restore(self, tmp_path: Path):
        """An illegal physical move (2 changed squares, no legal match) must enter UNCERTAIN, not STABLE."""
        from src.core.datatypes import FSMState

        orch = _make_orchestrator(tmp_path)
        before = _make_empty_board(orch.roi_extractor, BoardOrientation.WHITE_RIGHT)
        _draw_textured_pawn(before, orch.roi_extractor, "d4", BoardOrientation.WHITE_RIGHT, base_val=230)
        _draw_textured_pawn(before, orch.roi_extractor, "d5", BoardOrientation.WHITE_RIGHT, dark=True)

        after = before.copy()
        # Illegal move: White moves d4 pawn onto d6/c1->f4? Simulate moving c1 bishop illegally onto occupied e2
        _draw_textured_pawn(after, orch.roi_extractor, "c1", BoardOrientation.WHITE_RIGHT, base_val=230)
        e2x0, e2y0, e2x1, e2y1 = orch.roi_extractor.get_square_coordinates("e2", BoardOrientation.WHITE_RIGHT)
        cv2.circle(after, ((e2x0 + e2x1) // 2, (e2y0 + e2y1) // 2), 14, (235, 235, 235), -1)

        raw_obs = _make_noisy_zero_shot_obs(mirage_h4=False)
        refined = orch._refine_zero_shot_with_temporal_delta(
            cell_obs=raw_obs,
            rois={},
            warped_after=after,
            warped_before=before,
        )
        board = orch.state_manager.current_board
        obs = BoardObservation(
            timestamp=1.0,
            frame_id=10,
            cells=refined,
            motion_detected=False,
            motion_level=0.0,
            hand_present=False,
            geometry_confidence=1.0,
        )
        cands = orch.candidate_generator.generate_candidates(board, obs)
        ranked = orch.move_scorer.rank_candidates(board, cands, obs)
        dec = orch.decision_gate.evaluate(ranked, obs)
        orch._on_move_decision(dec, obs)

        assert dec.status.value == "uncertain"
        assert set(dec.uncertain_squares) == {"c1", "e2"}
        assert dec.top_hypotheses == []
        assert "Illegal move or unrecognized board change" in (dec.rejection_reason or "")
        assert orch._active_uncertainty is True

        # Restoring physical board to `before` clears anomaly cleanly
        restored_refined = orch._refine_zero_shot_with_temporal_delta(
            cell_obs=raw_obs,
            rois={},
            warped_after=before,
            warped_before=before,
        )
        restored_obs = BoardObservation(
            timestamp=2.0,
            frame_id=20,
            cells=restored_refined,
            motion_detected=False,
            motion_level=0.0,
            hand_present=False,
            geometry_confidence=1.0,
        )
        assert orch.candidate_generator.detect_changed_squares(board, restored_obs) == []
