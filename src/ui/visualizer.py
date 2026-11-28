"""Visual HUD Composer for Chess Game Recording System.

Composes real-time operator overlays: board boundary and square grids,
last move arrows, player hand occlusion masks, finite state machine (FSM)
status badges, authoritative 2D virtual chessboard projection, SAN move history,
Stockfish evaluation gauges, and operator alert dialogs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import chess
import cv2
import numpy as np

from src.core.datatypes import (
    CHESS_SQUARES,
    PIECE_CLASSES,
    AnalysisResult,
    DecisionResult,
    FSMState,
    MoveQuality,
    PipelineState,
)

logger = logging.getLogger(__name__)

# FSM State Color Palette (BGR)
FSM_COLOR_MAP: Dict[FSMState, Tuple[int, int, int]] = {
    FSMState.IDLE: (128, 128, 128),            # Gray
    FSMState.LOCATING_BOARD: (0, 165, 255),    # Orange
    FSMState.CALIBRATING: (255, 191, 0),       # Deep Sky Blue
    FSMState.STABLE: (50, 205, 50),            # Lime Green
    FSMState.MOVING: (0, 215, 255),            # Yellow
    FSMState.SETTLING: (255, 140, 0),          # Amber / Dark Orange
    FSMState.VERIFYING: (211, 0, 148),         # Purple / Violet
    FSMState.UNCERTAIN: (0, 0, 220),           # Crimson Red
    FSMState.BOARD_LOST: (80, 80, 80),         # Dark Gray
    FSMState.PAUSED: (0, 255, 255),            # Cyan
    FSMState.STOPPED: (50, 50, 50),            # Very Dark Gray
}

# Quality Badge Colors (BGR)
QUALITY_COLOR_MAP: Dict[MoveQuality, Tuple[int, int, int]] = {
    MoveQuality.BEST: (50, 205, 50),           # Green
    MoveQuality.EXCELLENT: (100, 220, 100),    # Light Green
    MoveQuality.GOOD: (180, 220, 180),         # Pale Green
    MoveQuality.INACCURACY: (0, 215, 255),     # Yellow
    MoveQuality.MISTAKE: (0, 140, 255),        # Orange
    MoveQuality.BLUNDER: (0, 0, 220),          # Red
}

# ASCII Piece Label Table (Symbol -> 2-char token)
PIECE_LABELS: Dict[str, str] = {
    "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
    "p": "p", "n": "n", "b": "b", "r": "r", "q": "q", "k": "k",
}


class VisualHUDComposer:
    """Composes composite graphical HUD layout for operator dashboard."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        font_scale: float = 0.6,
        show_virtual_board: bool = True,
        show_move_list: bool = True,
        show_eval_bar: bool = True,
        show_hand_landmarks: bool = True,
    ) -> None:
        """Initializes HUD composer with display dimensions and feature flags.

        Args:
            width: Target canvas width in pixels (e.g. 1280 or 1920).
            height: Target canvas height in pixels (e.g. 720 or 1080).
            font_scale: Font scale multiplier for overlay text.
            show_virtual_board: Whether to render 2D virtual board sidebar.
            show_move_list: Whether to display recent move history.
            show_eval_bar: Whether to display Stockfish evaluation indicator.
            show_hand_landmarks: Whether to render hand occlusion overlay.
        """
        self.width = width
        self.height = height
        self.font_scale = font_scale
        self.show_virtual_board = show_virtual_board
        self.show_move_list = show_move_list
        self.show_eval_bar = show_eval_bar
        self.show_hand_landmarks = show_hand_landmarks

        # Layout geometry: 65% left video pane, 35% right analytics sidebar
        self.sidebar_width = int(self.width * 0.35)
        self.video_width = self.width - self.sidebar_width

    def render_virtual_board(
        self,
        board: chess.Board,
        size: int = 240,
        last_move: Optional[chess.Move] = None,
    ) -> np.ndarray:
        """Renders an authoritative 8x8 virtual chessboard projection.

        Args:
            board: Current python-chess Board instance.
            size: Width and height of the rendered square board in pixels.
            last_move: Optional Move to highlight source and target squares.

        Returns:
            Rendered BGR image array [size, size, 3].
        """
        sq_size = size // 8
        canvas_size = sq_size * 8
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        # Chessboard square palette (BGR)
        light_color = (220, 225, 230)
        dark_color = (120, 145, 170)
        highlight_src = (100, 220, 150)   # Greenish for origin
        highlight_dst = (230, 200, 50)    # Cyan/Gold for destination

        from_sq = last_move.from_square if last_move else None
        to_sq = last_move.to_square if last_move else None

        for row in range(8):
            rank = 7 - row  # Rank 8 down to 1
            for col in range(8):
                file = col  # File a to h
                square = chess.square(file, rank)
                is_light = (file + rank) % 2 == 1

                x1 = col * sq_size
                y1 = row * sq_size
                x2 = x1 + sq_size
                y2 = y1 + sq_size

                # Determine base cell color
                if square == from_sq:
                    cell_color = highlight_src
                elif square == to_sq:
                    cell_color = highlight_dst
                else:
                    cell_color = light_color if is_light else dark_color

                cv2.rectangle(canvas, (x1, y1), (x2, y2), cell_color, -1)

                # Render chess piece
                piece = board.piece_at(square)
                if piece is not None:
                    cx = x1 + sq_size // 2
                    cy = y1 + sq_size // 2
                    radius = int(sq_size * 0.38)

                    # Piece token background
                    token_bg = (245, 245, 245) if piece.color == chess.WHITE else (35, 35, 35)
                    token_border = (50, 50, 50) if piece.color == chess.WHITE else (200, 200, 200)
                    cv2.circle(canvas, (cx, cy), radius, token_bg, -1)
                    cv2.circle(canvas, (cx, cy), radius, token_border, 1, cv2.LINE_AA)

                    # Piece letter
                    symbol = piece.symbol()
                    char = symbol.upper()
                    text_color = (10, 10, 10) if piece.color == chess.WHITE else (240, 240, 240)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = sq_size / 60.0
                    thickness = 2 if piece.color == chess.WHITE else 1

                    (tw, th), _ = cv2.getTextSize(char, font, scale, thickness)
                    tx = cx - tw // 2
                    ty = cy + th // 2
                    cv2.putText(canvas, char, (tx, ty), font, scale, text_color, thickness, cv2.LINE_AA)

        # Draw outer border
        cv2.rectangle(canvas, (0, 0), (canvas_size - 1, canvas_size - 1), (60, 60, 60), 1)

        if canvas_size != size:
            canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

        return canvas

    @staticmethod
    def _ensure_bgr_uint8(frame_bgr: Optional[np.ndarray]) -> np.ndarray:
        """Normalizes arbitrary input frame arrays into non-empty 3-channel BGR uint8."""
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return np.zeros((360, 640, 3), dtype=np.uint8)
        img = frame_bgr
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img.copy()

    def compose(
        self,
        frame_bgr: np.ndarray,
        state: PipelineState,
        corners: Optional[np.ndarray] = None,
        hand_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composes full composite display canvas containing video feed and analytics sidebar.

        Args:
            frame_bgr: Raw input camera image [H, W, 3].
            state: Full snapshot of current pipeline state.
            corners: Optional 4-corner coordinates of chessboard in input image.
            hand_mask: Optional binary mask of player hand occlusion.

        Returns:
            Composite BGR canvas of size [self.height, self.width, 3].
        """
        safe_frame = self._ensure_bgr_uint8(frame_bgr)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # ---------------------------------------------------------------------
        # 1. Left Video Feed Pane
        # ---------------------------------------------------------------------
        video_pane = self._render_video_feed(safe_frame, state, corners, hand_mask)
        canvas[:, :self.video_width] = video_pane

        # ---------------------------------------------------------------------
        # 2. Right Analytics Sidebar Pane
        # ---------------------------------------------------------------------
        sidebar_pane = self._render_sidebar(state)
        canvas[:, self.video_width:] = sidebar_pane

        # Divider line between video and sidebar
        cv2.line(canvas, (self.video_width, 0), (self.video_width, self.height), (70, 70, 70), 2)

        # ---------------------------------------------------------------------
        # 3. Global Modals and Operator Alerts (Overlays)
        # ---------------------------------------------------------------------
        if state.active_uncertainty:
            self._draw_uncertainty_modal(canvas, state)
        elif state.is_paused:
            self._draw_paused_banner(canvas)

        return canvas

    def compose_video_overlay(
        self,
        frame_bgr: np.ndarray,
        state: PipelineState,
        corners: Optional[np.ndarray] = None,
        hand_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Renders video feed with board boundary, hand overlay and status HUD directly on frame."""
        vis = self._ensure_bgr_uint8(frame_bgr)
        h, w = vis.shape[:2]

        # Draw board boundary polygon & corners
        if corners is not None and len(corners) == 4 and state.board_detected:
            poly = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            for cx, cy in corners:
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 200, 0), -1)

        # Hand occlusion overlay
        if hand_mask is not None and self.show_hand_landmarks and state.hand_detected:
            hm_h, hm_w = hand_mask.shape[:2]
            if hm_h == h and hm_w == w:
                red_tint = np.full_like(vis, (0, 0, 180))
                mask_bool = hand_mask > 0
                if np.any(mask_bool):
                    vis[mask_bool] = cv2.addWeighted(vis[mask_bool], 0.6, red_tint[mask_bool], 0.4, 0)

        # FSM State Badge (Top-left)
        state_color = FSM_COLOR_MAP.get(state.fsm_state, (100, 100, 100))
        badge_w, badge_h = 240, 48
        bx, by = 16, 16
        cv2.rectangle(vis, (bx, by), (bx + badge_w, by + badge_h), (25, 25, 25), -1)
        cv2.rectangle(vis, (bx, by), (bx + 8, by + badge_h), state_color, -1)
        cv2.rectangle(vis, (bx, by), (bx + badge_w, by + badge_h), (60, 60, 60), 1)

        state_text = state.fsm_state.value.upper()
        cv2.putText(
            vis,
            f"FSM: {state_text}",
            (bx + 16, by + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        # Telemetry metrics (Bottom-left)
        tele_y = h - 16
        tele_str = f"FPS: {state.processing_fps:.1f} | Frame: {state.frame_id} | Ply: {state.current_ply}"
        cv2.putText(
            vis,
            tele_str,
            (16, tele_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Hand & Motion status
        hand_text = "HAND DETECTED" if state.hand_detected else "HAND: CLEAR"
        hand_color = (0, 0, 220) if state.hand_detected else (50, 180, 50)
        cv2.putText(
            vis,
            f"[{hand_text}]  Motion: {state.motion_level:.3f}",
            (16, tele_y - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            hand_color,
            1,
            cv2.LINE_AA,
        )

        # Paused banner if paused
        if state.is_paused:
            p_text = "PAUSED"
            cv2.putText(vis, p_text, (max(16, w // 2 - 80), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        return vis

    def _render_video_feed(
        self,
        frame_bgr: np.ndarray,
        state: PipelineState,
        corners: Optional[np.ndarray],
        hand_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Renders video feed with board boundary, hand overlay and status HUD."""
        target_w = self.video_width
        target_h = self.height

        # Resize video preserving aspect ratio with letterboxing
        h_src, w_src = frame_bgr.shape[:2]
        if h_src <= 0 or w_src <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        scale = min(target_w / w_src, target_h / h_src)
        scaled_w = max(1, int(w_src * scale))
        scaled_h = max(1, int(h_src * scale))

        resized = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        pane = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        off_x = (target_w - scaled_w) // 2
        off_y = (target_h - scaled_h) // 2
        pane[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = resized

        # Scale corners and draw board boundary polygon
        if corners is not None and len(corners) == 4 and state.board_detected:
            scaled_corners = []
            for pt in corners:
                x = int(pt[0] * scale + off_x)
                y = int(pt[1] * scale + off_y)
                scaled_corners.append([x, y])
            poly = np.array(scaled_corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(pane, [poly], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # Draw corner vertex markers
            for i, (cx, cy) in enumerate(scaled_corners):
                cv2.circle(pane, (cx, cy), 4, (0, 200, 0), -1)

        # Hand occlusion mask overlay
        if hand_mask is not None and self.show_hand_landmarks and state.hand_detected:
            h_m, w_m = hand_mask.shape[:2]
            if h_m == h_src and w_m == w_src:
                resized_mask = cv2.resize(hand_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                sub_pane = pane[off_y:off_y + scaled_h, off_x:off_x + scaled_w]
                red_tint = np.full_like(sub_pane, (0, 0, 180))
                mask_bool = resized_mask > 0
                if np.any(mask_bool):
                    sub_pane[mask_bool] = cv2.addWeighted(sub_pane[mask_bool], 0.6, red_tint[mask_bool], 0.4, 0)
                    pane[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = sub_pane

        # FSM State Badge (Top-left)
        state_color = FSM_COLOR_MAP.get(state.fsm_state, (100, 100, 100))
        badge_w, badge_h = 240, 48
        bx, by = 16, 16
        cv2.rectangle(pane, (bx, by), (bx + badge_w, by + badge_h), (25, 25, 25), -1)
        cv2.rectangle(pane, (bx, by), (bx + 8, by + badge_h), state_color, -1)
        cv2.rectangle(pane, (bx, by), (bx + badge_w, by + badge_h), (60, 60, 60), 1)

        state_text = state.fsm_state.value.upper()
        cv2.putText(
            pane,
            f"FSM: {state_text}",
            (bx + 16, by + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        # Performance & telemetry metrics (Bottom-left)
        tele_y = target_h - 16
        tele_str = f"FPS: {state.processing_fps:.1f} | Frame: {state.frame_id} | Ply: {state.current_ply}"
        cv2.putText(
            pane,
            tele_str,
            (16, tele_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Hand & Motion status pills
        hand_text = "HAND DETECTED" if state.hand_detected else "HAND: CLEAR"
        hand_color = (0, 0, 220) if state.hand_detected else (50, 180, 50)
        cv2.putText(
            pane,
            f"[{hand_text}]  Motion: {state.motion_level:.3f}",
            (16, tele_y - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            hand_color,
            1,
            cv2.LINE_AA,
        )

        return pane

    def _render_sidebar(self, state: PipelineState) -> np.ndarray:
        """Renders right sidebar: 2D virtual board, move list, and Stockfish eval gauge."""
        w = self.sidebar_width
        h = self.height
        sidebar = np.full((h, w, 3), 28, dtype=np.uint8)

        # Header Title
        cv2.putText(
            sidebar,
            "CHESS RECORDING HUD",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.line(sidebar, (20, 44), (w - 20, 44), (60, 60, 60), 1)

        # ---------------------------------------------------------------------
        # 1. 2D Virtual Board Render
        # ---------------------------------------------------------------------
        curr_y = 60
        if self.show_virtual_board:
            try:
                board = chess.Board(state.current_fen)
            except Exception:
                board = chess.Board()

            last_move = None
            if state.last_move_uci:
                try:
                    last_move = chess.Move.from_uci(state.last_move_uci)
                except Exception:
                    last_move = None

            board_size = min(max(16, w - 40), max(16, h - curr_y - 40), 240)
            if w > 40 and curr_y + board_size <= h:
                board_img = self.render_virtual_board(board, size=board_size, last_move=last_move)
                bx = (w - board_size) // 2
                sidebar[curr_y:curr_y + board_size, bx:bx + board_size] = board_img
                curr_y += board_size + 20

        # ---------------------------------------------------------------------
        # 2. Stockfish Evaluation Gauge
        # ---------------------------------------------------------------------
        if self.show_eval_bar:
            curr_y = self._render_eval_gauge(sidebar, state.latest_eval, curr_y)

        # ---------------------------------------------------------------------
        # 3. Recorded Moves List (SAN)
        # ---------------------------------------------------------------------
        if self.show_move_list:
            self._render_move_list(sidebar, state, curr_y)

        return sidebar

    def _render_eval_gauge(
        self,
        sidebar: np.ndarray,
        latest_eval: Optional[AnalysisResult],
        start_y: int,
    ) -> int:
        """Draws Stockfish evaluation bar and best-move indicator."""
        w = self.sidebar_width
        y = start_y

        cv2.putText(
            sidebar,
            "STOCKFISH 16 UCI",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        y += 12

        bar_x = 20
        bar_w = w - 40
        bar_h = 16
        cv2.rectangle(sidebar, (bar_x, y), (bar_x + bar_w, y + bar_h), (45, 45, 45), -1)

        score_text = "0.00"
        best_move_text = "None"
        eval_ratio = 0.5  # 0.5 is dead equal

        if latest_eval is not None and latest_eval.eval_after is not None:
            eva = latest_eval.eval_after
            if eva.mate_in is not None:
                score_text = f"M{eva.mate_in:+d}"
                eval_ratio = 1.0 if eva.mate_in > 0 else 0.0
            elif eva.eval_cp is not None:
                cp = eva.eval_cp
                score_text = f"{cp / 100.0:+.2f}"
                # Sigmoid-like clamp [-600, +600] cp
                clamped_cp = max(-600, min(600, cp))
                eval_ratio = (clamped_cp + 600) / 1200.0

            if eva.best_move_uci:
                best_move_text = eva.best_move_uci

        # Draw Advantage Fill
        fill_w = int(bar_w * eval_ratio)
        cv2.rectangle(sidebar, (bar_x, y), (bar_x + fill_w, y + bar_h), (220, 220, 220), -1)
        # Center marker line
        mid_x = bar_x + bar_w // 2
        cv2.line(sidebar, (mid_x, y - 2), (mid_x, y + bar_h + 2), (0, 140, 255), 1)
        cv2.rectangle(sidebar, (bar_x, y), (bar_x + bar_w, y + bar_h), (80, 80, 80), 1)
        y += bar_h + 18

        # Eval metrics text
        cv2.putText(
            sidebar,
            f"Eval: {score_text}  |  Best: {best_move_text}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        y += 24
        cv2.line(sidebar, (20, y), (w - 20, y), (50, 50, 50), 1)
        return y + 16

    def _render_move_list(
        self,
        sidebar: np.ndarray,
        state: PipelineState,
        start_y: int,
    ) -> None:
        """Draws recent committed move history with SAN and quality badges."""
        w = self.sidebar_width
        y = start_y

        cv2.putText(
            sidebar,
            "RECORDED MOVES",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        y += 20

        # Recent move item
        if state.last_move_san:
            ply_num = state.current_ply
            move_num = (ply_num + 1) // 2
            color_prefix = f"{move_num}. " if ply_num % 2 == 1 else f"{move_num}... "
            move_str = f"{color_prefix}{state.last_move_san} ({state.last_move_uci})"

            cv2.putText(
                sidebar,
                move_str,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

            # Draw move quality pill if available
            if state.latest_eval and state.latest_eval.quality:
                q = state.latest_eval.quality
                q_color = QUALITY_COLOR_MAP.get(q, (180, 180, 180))
                q_text = f"[{q.value.upper()}]"
                cv2.putText(
                    sidebar,
                    q_text,
                    (w - 110, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    q_color,
                    1,
                    cv2.LINE_AA,
                )
            y += 24
        else:
            cv2.putText(
                sidebar,
                "No moves committed yet",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (100, 100, 100),
                1,
                cv2.LINE_AA,
            )
            y += 24

        # Keyboard hotkey quick reference at the bottom of the sidebar
        footer_y = self.height - 70
        cv2.line(sidebar, (20, footer_y - 10), (w - 20, footer_y - 10), (50, 50, 50), 1)
        cv2.putText(sidebar, "[Space] Pause/Resume", (20, footer_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
        cv2.putText(sidebar, "[Z] Rollback  [U] Accept Top", (20, footer_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
        cv2.putText(sidebar, "[P] Export PGN  [Q] Exit", (20, footer_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    def _draw_uncertainty_modal(self, canvas: np.ndarray, state: PipelineState) -> None:
        """Draws operator intervention alert modal when FSM enters UNCERTAIN state."""
        h, w = canvas.shape[:2]
        modal_w = min(560, max(120, w - 20))
        modal_h = min(220, max(80, h - 20))
        mx = max(0, (w - modal_w) // 2)
        my = max(0, (h - modal_h) // 2)

        # Dim background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (mx, my), (mx + modal_w, my + modal_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        # Modal border
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (0, 0, 220), 2)
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + 36), (0, 0, 180), -1)

        cv2.putText(
            canvas,
            "ACTION REQUIRED: UNCERTAIN MOVE RESOLUTION",
            (mx + 16, my + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y = my + 64
        cv2.putText(
            canvas,
            "Visual decision gate confidence below acceptance margin.",
            (mx + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        y += 24

        # List candidate moves if available
        candidates = []
        if state.last_decision:
            if state.last_decision.top_hypotheses:
                candidates = state.last_decision.top_hypotheses
            elif state.last_decision.accepted_hypothesis:
                candidates = [state.last_decision.accepted_hypothesis]

        if candidates:
            cv2.putText(
                canvas,
                "Candidate moves:",
                (mx + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 215, 255),
                1,
                cv2.LINE_AA,
            )
            y += 20
            for i, cand in enumerate(candidates[:3]):
                cand_str = f"[{i + 1}] {cand.move_uci} (score: {cand.total_score:.3f}, visual: {cand.score_visual:.2f})"
                cv2.putText(
                    canvas,
                    cand_str,
                    (mx + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                y += 20

        # Operator controls hint
        hint_y = my + modal_h - 18
        cv2.putText(
            canvas,
            "Press [1..3] to accept candidate | [U] Auto-accept Top | [Z] Reject & Rollback",
            (mx + 20, hint_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (100, 255, 100),
            1,
            cv2.LINE_AA,
        )

    def _draw_paused_banner(self, canvas: np.ndarray) -> None:
        """Draws flashing PAUSED overlay across the top center."""
        h, w = canvas.shape[:2]
        bw, bh = 380, 44
        bx = (w - bw) // 2
        by = 20

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 0, 0), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
        cv2.putText(
            canvas,
            "PAUSED - Press [Space] to Resume",
            (bx + 18, by + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
