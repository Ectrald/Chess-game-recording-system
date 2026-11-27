"""Unit tests for 3D occupancy, same-color camouflage, and parallax disambiguation."""

from __future__ import annotations

import chess
import cv2
import numpy as np

from src.core.datatypes import BoardOrientation
from src.reasoning.candidate_generator import LegalMoveCandidateGenerator
from src.vision.occupancy_tracker import BoardOccupancyTracker
from src.vision.roi_extractor import CellRoiExtractor


def _make_board_img(light_val: int = 230, dark_val: int = 75) -> np.ndarray:
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            # WHITE_RIGHT top-left (0,0) = h8 (dark), (r+c)%2 == 0 -> dark
            val = dark_val if (r + c) % 2 == 0 else light_val
            img[r * 80 : (r + 1) * 80, c * 80 : (c + 1) * 80] = (val, val, val)
    return img


def _draw_circle_piece(img: np.ndarray, sq: str, orientation: BoardOrientation, color_bgr: tuple, edge_bgr: tuple) -> None:
    roi_ex = CellRoiExtractor(640)
    x0, y0, x1, y1 = roi_ex.get_square_coordinates(sq, orientation)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.circle(img, (cx, cy), 22, color_bgr, -1)
    cv2.circle(img, (cx, cy), 22, edge_bgr, 2)


def _draw_upward_head(img: np.ndarray, sq_above: str, orientation: BoardOrientation, color_bgr: tuple) -> None:
    roi_ex = CellRoiExtractor(640)
    x0, y0, x1, y1 = roi_ex.get_square_coordinates(sq_above, orientation)
    cx = (x0 + x1) // 2
    # Tall piece from square below leaning into bottom band (and overlapping core) of sq_above
    cv2.ellipse(img, (cx, y0 + 56), (14, 18), 0, 0, 360, color_bgr, -1)


def _populate_starting_board(img: np.ndarray, board: chess.Board, orientation: BoardOrientation) -> None:
    for sq_idx in chess.SQUARES:
        piece = board.piece_at(sq_idx)
        if piece is None:
            continue
        sq = chess.square_name(sq_idx)
        if piece.color == chess.WHITE:
            _draw_circle_piece(img, sq, orientation, (240, 240, 240), (170, 170, 170))
        else:
            _draw_circle_piece(img, sq, orientation, (35, 35, 35), (15, 15, 15))


def test_e7e5_beats_f7f5_when_e5_head_leans_into_f5():
    """Black pawn e7->e5 leaning upward into empty light f5 must not trigger f7->f5."""
    board = chess.Board()
    board.push_uci("f2f4")
    ori = BoardOrientation.WHITE_RIGHT
    roi_ex = CellRoiExtractor(640)
    cand = LegalMoveCandidateGenerator()
    tracker = BoardOccupancyTracker()

    before = _make_board_img()
    _populate_starting_board(before, board, ori)

    after_board = board.copy()
    after_board.push_uci("e7e5")
    after = _make_board_img()
    _populate_starting_board(after, after_board, ori)
    # Dark pawn on dark e5 projects strongly upward into light f5 (above e5 in WHITE_RIGHT)
    _draw_upward_head(after, "f5", ori, (25, 25, 25))

    ev = tracker.evaluate_transition(board, after, before, ori, roi_ex, cand)
    assert ev.changed_sqs == {"e7", "e5"}
    assert ev.expected_dest_piece.get("e5") == "bP"


def test_same_color_nf3_recovered_via_upward_projection_even_with_f1_lean():
    """White Knight g1->f3 (white piece on white f3 leaning into dark g3) beats g2->g3."""
    board = chess.Board()
    ori = BoardOrientation.WHITE_RIGHT
    roi_ex = CellRoiExtractor(640)
    cand = LegalMoveCandidateGenerator()
    tracker = BoardOccupancyTracker()

    before = _make_board_img()
    _populate_starting_board(before, board, ori)
    # White bishop on f1 leans upward across dark g1 in both before & after
    _draw_upward_head(before, "g1", ori, (220, 220, 220))

    after_board = board.copy()
    after_board.push_uci("g1f3")
    after = _make_board_img()
    _populate_starting_board(after, after_board, ori)
    _draw_upward_head(after, "g1", ori, (220, 220, 220))
    # White Knight on white f3 leans heavily into dark g3 above f3
    _draw_upward_head(after, "g3", ori, (240, 240, 240))

    ev = tracker.evaluate_transition(board, after, before, ori, roi_ex, cand)
    assert ev.changed_sqs == {"g1", "f3"}
    assert ev.expected_dest_piece.get("f3") == "wN"


def test_untouched_shadowed_pawn_does_not_block_real_move():
    """An illumination/shadow shift on an untouched black pawn (h7) must not block 2. Bf4."""
    board = chess.Board()
    board.push_uci("d2d4")
    board.push_uci("g7g6")
    ori = BoardOrientation.WHITE_RIGHT
    roi_ex = CellRoiExtractor(640)
    cand = LegalMoveCandidateGenerator()
    tracker = BoardOccupancyTracker()

    before = _make_board_img()
    _populate_starting_board(before, board, ori)

    after_board = board.copy()
    after_board.push_uci("c1f4")
    after = _make_board_img()
    _populate_starting_board(after, after_board, ori)
    # Darken h7 locally by shadow without removing h7 pawn
    x0, y0, x1, y1 = roi_ex.get_square_coordinates("h7", ori)
    after[y0:y1, x0:x1] = np.clip(after[y0:y1, x0:x1].astype(np.int32) - 45, 0, 255).astype(np.uint8)

    ev = tracker.evaluate_transition(board, after, before, ori, roi_ex, cand)
    assert ev.changed_sqs == {"c1", "f4"}
    assert ev.pending_two_ply_second_move is None
    assert ev.anomaly_squares == []


def test_extract_occupancy_crops_are_unexpanded_100x100():
    """Verify chesscog occupancy crops are strictly 100x100 and map all 64 squares."""
    roi_ex = CellRoiExtractor(640)
    img = _make_board_img()
    crops = roi_ex.extract_occupancy_crops(img, BoardOrientation.WHITE_RIGHT, output_size=(100, 100))
    assert len(crops) == 64
    assert crops["e4"].shape == (100, 100, 3)


def test_calibrated_nn_occupancy_fusion_respects_chesscog_when_accurate():
    """When neural occupancy predicts occupied squares accurately, fusion strengthens e7->e5."""
    board = chess.Board()
    board.push_uci("f2f4")
    ori = BoardOrientation.WHITE_RIGHT
    roi_ex = CellRoiExtractor(640)
    cand = LegalMoveCandidateGenerator()
    tracker = BoardOccupancyTracker()

    before = _make_board_img()
    _populate_starting_board(before, board, ori)
    after_board = board.copy()
    after_board.push_uci("e7e5")
    after = _make_board_img()
    _populate_starting_board(after, after_board, ori)
    _draw_upward_head(after, "f5", ori, (25, 25, 25))

    nn_b = {sq: (0.95 if board.piece_at(chess.parse_square(sq)) else 0.05) for sq in roi_ex.extract_occupancy_crops(before, ori)}
    nn_a = {sq: (0.95 if after_board.piece_at(chess.parse_square(sq)) else 0.05) for sq in roi_ex.extract_occupancy_crops(after, ori)}
    ev = tracker.evaluate_transition(
        board, after, before, ori, roi_ex, cand, nn_occ_before=nn_b, nn_occ_after=nn_a
    )
    assert ev.changed_sqs == {"e7", "e5"}
