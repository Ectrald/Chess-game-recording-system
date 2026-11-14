"""Synthetic chess board and video stream generator for testing and benchmarks.

Generates realistic simulated video frames of chess games by rendering
positions from python-chess, applying perspective homography, simulated
hand/shadow occlusions during move execution, and synthetic noise.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import chess
import chess.pgn
import cv2
import numpy as np

from src.core.datatypes import (
    BoardGeometry,
    BoardObservation,
    CellObservation,
    Frame,
    CHESS_SQUARES,
)


class SyntheticBoardGenerator:
    """Renders synthetic video frames of chess games with realistic transformations.
    
    Provides ground truth states (FEN, move events, 4 corners, homography)
    to validate vision detectors and symbolic solvers without real camera video.
    """

    def __init__(
        self,
        board_size: int = 640,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        fps: float = 30.0,
        frames_per_move: int = 60,
        transition_fraction: float = 0.35,
        corners: Optional[List[Tuple[float, float]]] = None,
        simulate_occlusion: bool = True,
        noise_std: float = 0.0,
    ) -> None:
        """Initializes synthetic board generator.
        
        Args:
            board_size: Normalized flat board size in pixels (square, e.g. 640).
            canvas_width: Output canvas width in pixels.
            canvas_height: Output canvas height in pixels.
            fps: Frame rate of the synthetic video.
            frames_per_move: Number of video frames allocated to each half-move (ply).
            transition_fraction: Fraction of move duration during which hand/piece moves.
            corners: 4 quadrilateral corners on canvas (TL, TR, BR, BL).
                     If None, a realistic perspective view is automatically computed.
            simulate_occlusion: Whether to draw simulated hand shadow during transitions.
            noise_std: Standard deviation of Gaussian sensor noise to add.
        """
        self.board_size = board_size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.fps = fps
        self.frames_per_move = max(10, frames_per_move)
        self.transition_fraction = max(0.1, min(0.8, transition_fraction))
        self.simulate_occlusion = simulate_occlusion
        self.noise_std = noise_std

        # Default realistic perspective corners (simulating 65° camera elevation)
        if corners is None:
            margin_x = canvas_width * 0.22
            margin_y_top = canvas_height * 0.15
            margin_y_bot = canvas_height * 0.88
            tilt_dx = canvas_width * 0.08
            self.corners: List[Tuple[float, float]] = [
                (margin_x + tilt_dx, margin_y_top),           # TL
                (canvas_width - margin_x - tilt_dx, margin_y_top),  # TR
                (canvas_width - margin_x + tilt_dx, margin_y_bot),  # BR
                (margin_x - tilt_dx, margin_y_bot),           # BL
            ]
        else:
            if len(corners) != 4:
                raise ValueError(f"corners must have exactly 4 points, got {len(corners)}")
            self.corners = list(corners)

        # Precompute homography matrix: flat 640x640 -> canvas quadrilateral
        src_pts = np.array(
            [[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]],
            dtype=np.float32,
        )
        dst_pts = np.array(self.corners, dtype=np.float32)
        self._homography_forward = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self._homography_inverse = cv2.getPerspectiveTransform(dst_pts, src_pts)

        # Game state tracking
        self.moves: List[chess.Move] = []
        self.fens: List[str] = [chess.STARTING_FEN]
        self._board_history: List[chess.Board] = [chess.Board()]
        self._total_frames: int = self.frames_per_move  # At least initial position

    def set_game_from_moves(self, moves: Sequence[Union[str, chess.Move]]) -> None:
        """Sets the sequence of moves to simulate.
        
        Args:
            moves: List of moves in UCI strings ('e2e4') or chess.Move objects.
        """
        board = chess.Board()
        self.moves = []
        self.fens = [board.fen()]
        self._board_history = [board.copy()]

        for m in moves:
            if isinstance(m, str):
                move_obj = chess.Move.from_uci(m)
            else:
                move_obj = m
            if not board.is_legal(move_obj):
                raise ValueError(f"Illegal move {move_obj.uci()} in position {board.fen()}")
            board.push(move_obj)
            self.moves.append(move_obj)
            self.fens.append(board.fen())
            self._board_history.append(board.copy())

        # Total frames: initial pause + each move's frames + trailing pause
        self._total_frames = max(1, len(self.moves) + 1) * self.frames_per_move

    def set_game_from_pgn(self, pgn_content_or_file: str) -> None:
        """Sets the game sequence by parsing a PGN text or file."""
        import io
        if "\n" in pgn_content_or_file or "[" in pgn_content_or_file:
            pgn_io = io.StringIO(pgn_content_or_file)
        else:
            with open(pgn_content_or_file, "r", encoding="utf-8") as f:
                pgn_io = io.StringIO(f.read())
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            raise ValueError("No valid chess game found in PGN")
        moves = list(game.mainline_moves())
        self.set_game_from_moves(moves)

    @property
    def total_frames(self) -> int:
        """Total frame count of the synthetic sequence."""
        return self._total_frames

    def get_ground_truth_geometry(self) -> BoardGeometry:
        """Returns the exact ground-truth quadrilateral geometry and homography."""
        return BoardGeometry(
            corners=self.corners,
            homography_matrix=self._homography_forward.tolist(),
            confidence=1.0,
            is_valid=True,
            orientation_calibrated=True,
        )

    def get_ground_truth_board(self, frame_id: int) -> chess.Board:
        """Returns the ground truth chess.Board state corresponding to frame_id."""
        ply_index = min(len(self._board_history) - 1, frame_id // self.frames_per_move)
        return self._board_history[ply_index].copy()

    def get_ground_truth_observation(self, frame_id: int) -> BoardObservation:
        """Generates exact ground-truth BoardObservation for testing downstream components."""
        board = self.get_ground_truth_board(frame_id)
        cells: Dict[str, CellObservation] = {}

        for square_name in CHESS_SQUARES:
            sq_idx = chess.parse_square(square_name)
            piece = board.piece_at(sq_idx)
            probs = {cls_name: 0.0 for cls_name in [
                "empty", "wP", "wN", "wB", "wR", "wQ", "wK",
                "bP", "bN", "bB", "bR", "bQ", "bK"
            ]}
            if piece is None:
                piece_label = "empty"
                probs["empty"] = 0.99
                probs["wP"] = 0.001
                occupied = False
            else:
                color_prefix = "w" if piece.color == chess.WHITE else "b"
                symbol = piece.symbol().upper()
                piece_label = f"{color_prefix}{symbol}"
                probs[piece_label] = 0.98
                probs["empty"] = 0.01
                occupied = True

            cells[square_name] = CellObservation(
                square=square_name,
                probabilities=probs,
                predicted_piece=piece_label,
                confidence=probs[piece_label],
                is_occupied=occupied,
                is_occluded=False,
            )

        timestamp = frame_id / self.fps
        return BoardObservation(
            timestamp=timestamp,
            frame_id=frame_id,
            cells=cells,
            motion_detected=False,
            motion_level=0.0,
            hand_present=False,
            geometry_confidence=1.0,
        )

    def render_board_flat(
        self,
        board: chess.Board,
        highlight_squares: Optional[Sequence[int]] = None,
        lifted_piece: Optional[Tuple[str, float, float]] = None,
    ) -> np.ndarray:
        """Renders canonical 8x8 chess board to a normalized square image.
        
        Args:
            board: Current chess position.
            highlight_squares: Optional list of chess square indices to highlight.
            lifted_piece: Optional tuple (piece_symbol, center_x_px, center_y_px) for transition.
            
        Returns:
            RGB/BGR numpy array of shape (board_size, board_size, 3), dtype uint8.
        """
        img = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)
        sq_size = self.board_size / 8.0

        # Colors in BGR
        light_color = (181, 217, 240)  # #F0D9B5 in BGR
        dark_color = (99, 136, 181)    # #B58863 in BGR
        highlight_color = (100, 200, 240)

        highlight_set = set(highlight_squares or [])

        # 1. Draw 64 squares
        for rank in range(8):
            for file in range(8):
                sq_idx = chess.square(file, rank)
                x1 = int(round(file * sq_size))
                y1 = int(round((7 - rank) * sq_size))
                x2 = int(round((file + 1) * sq_size))
                y2 = int(round((7 - rank + 1) * sq_size))

                if sq_idx in highlight_set:
                    color = highlight_color
                else:
                    color = light_color if (file + rank) % 2 != 0 else dark_color

                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        # 2. Draw pieces
        for sq_idx, piece in board.piece_map().items():
            file = chess.square_file(sq_idx)
            rank = chess.square_rank(sq_idx)
            cx = int(round((file + 0.5) * sq_size))
            cy = int(round((7 - rank + 0.5) * sq_size))
            self._draw_piece_glyph(img, piece, cx, cy, int(sq_size * 0.4))

        # 3. Draw lifted/moving piece if present
        if lifted_piece is not None:
            symbol, px, py = lifted_piece
            piece_obj = chess.Piece.from_symbol(symbol)
            # Draw shadow beneath lifted piece
            cv2.circle(img, (int(px) + 5, int(py) + 8), int(sq_size * 0.38), (40, 40, 40), -1)
            self._draw_piece_glyph(img, piece_obj, int(px), int(py), int(sq_size * 0.42))

        return img

    def _draw_piece_glyph(
        self,
        img: np.ndarray,
        piece: chess.Piece,
        cx: int,
        cy: int,
        radius: int,
    ) -> None:
        """Draws a clean, high-contrast piece marker with label."""
        is_white = piece.color == chess.WHITE
        bg_color = (250, 250, 250) if is_white else (30, 30, 30)
        fg_color = (15, 15, 15) if is_white else (245, 245, 245)
        border_color = (80, 80, 80)

        # Draw piece circular base
        cv2.circle(img, (cx, cy), radius, bg_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, border_color, 2, lineType=cv2.LINE_AA)

        # Text label (e.g. 'P', 'N', 'B', 'R', 'Q', 'K')
        label = piece.symbol().upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = radius / 22.0
        thickness = max(1, int(round(font_scale * 2.2)))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(img, label, (tx, ty), font, font_scale, fg_color, thickness, cv2.LINE_AA)

    def render_frame(self, frame_id: int) -> Frame:
        """Renders full perspective frame at given frame_id.
        
        Simulates:
        - Settled positions when no hand is moving.
        - Dynamic piece translation and hand shadow during transition window.
        - Perspective warping onto canvas background.
        - Additive noise / lighting perturbations.
        """
        # Determine current game ply and sub-frame offset within this move
        ply_index = frame_id // self.frames_per_move
        sub_frame = frame_id % self.frames_per_move
        transition_frames = int(round(self.frames_per_move * self.transition_fraction))

        # Background canvas (simulating wooden table surface)
        canvas = np.full((self.canvas_height, self.canvas_width, 3), (210, 218, 224), dtype=np.uint8)

        # Draw soft table grid/texture
        cv2.rectangle(canvas, (0, 0), (self.canvas_width, self.canvas_height), (190, 200, 205), 4)

        if ply_index == 0 or ply_index > len(self.moves):
            # Static position before game starts or after all moves played
            board_idx = min(len(self._board_history) - 1, ply_index)
            board = self._board_history[board_idx]
            flat_board = self.render_board_flat(board)
        else:
            # During move execution (ply_index - 1 is move index)
            move = self.moves[ply_index - 1]
            board_before = self._board_history[ply_index - 1]
            board_after = self._board_history[ply_index]

            if sub_frame < transition_frames:
                # Active transition: piece moving from move.from_square to move.to_square
                progress = sub_frame / float(max(1, transition_frames))
                sq_size = self.board_size / 8.0

                from_file = chess.square_file(move.from_square)
                from_rank = chess.square_rank(move.from_square)
                to_file = chess.square_file(move.to_square)
                to_rank = chess.square_rank(move.to_square)

                x0 = (from_file + 0.5) * sq_size
                y0 = (7 - from_rank + 0.5) * sq_size
                x1 = (to_file + 0.5) * sq_size
                y1 = (7 - to_rank + 0.5) * sq_size

                curr_x = x0 + (x1 - x0) * progress
                curr_y = y0 + (y1 - y0) * progress - math.sin(progress * math.pi) * (sq_size * 0.4)

                moving_piece = board_before.piece_at(move.from_square)
                moving_symbol = moving_piece.symbol() if moving_piece else "P"

                # Render board with origin square highlighted and moving piece lifted
                temp_board = board_before.copy()
                temp_board.remove_piece_at(move.from_square)
                flat_board = self.render_board_flat(
                    temp_board,
                    highlight_squares=[move.from_square, move.to_square],
                    lifted_piece=(moving_symbol, curr_x, curr_y),
                )
            else:
                # Settled position after move completion
                flat_board = self.render_board_flat(
                    board_after,
                    highlight_squares=[move.from_square, move.to_square] if (sub_frame < transition_frames + 5) else None,
                )

        # 4. Warp flat board to perspective canvas
        warped_board = cv2.warpPerspective(
            flat_board,
            self._homography_forward,
            (self.canvas_width, self.canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        # Create quadrilateral mask
        pts = np.array(self.corners, dtype=np.int32)
        mask = np.zeros((self.canvas_height, self.canvas_width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)

        # Composite warped board onto table canvas
        canvas[mask > 0] = warped_board[mask > 0]

        # 5. Simulate hand occlusion during transition if requested
        if self.simulate_occlusion and 0 < ply_index <= len(self.moves) and sub_frame < transition_frames:
            self._draw_simulated_hand(canvas, frame_id, sub_frame, transition_frames)

        # 6. Apply sensor noise if configured
        if self.noise_std > 0.0:
            noise = np.random.normal(0, self.noise_std, canvas.shape).astype(np.float32)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        timestamp = float(frame_id) / self.fps
        return Frame(
            frame_id=frame_id,
            timestamp=timestamp,
            image=canvas,
            width=self.canvas_width,
            height=self.canvas_height,
        )

    def _draw_simulated_hand(
        self,
        canvas: np.ndarray,
        frame_id: int,
        sub_frame: int,
        transition_frames: int,
    ) -> None:
        """Draws realistic simulated player hand shadow encroaching the board."""
        progress = sub_frame / float(max(1, transition_frames))
        # Hand enters from bottom edge and reaches toward board center
        start_y = self.canvas_height
        target_y = int(self.canvas_height * 0.55)
        curr_y = int(start_y - (start_y - target_y) * math.sin(progress * math.pi))
        curr_x = int(self.canvas_width * 0.52 + math.sin(progress * math.pi * 2) * 50)

        overlay = canvas.copy()
        # Hand ellipse with skin-like tone and alpha blending
        cv2.ellipse(
            overlay,
            (curr_x, curr_y),
            (140, 260),
            angle=15,
            startAngle=0,
            endAngle=360,
            color=(170, 185, 220),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        # Shadow cast below hand
        cv2.ellipse(
            overlay,
            (curr_x - 30, curr_y + 40),
            (120, 220),
            angle=15,
            startAngle=0,
            endAngle=360,
            color=(40, 40, 40),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        # Alpha blend 50%
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
