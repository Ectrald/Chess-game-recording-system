"""Automated dataset harvesting module (AutoLabeller).

Extracts and labels square image crops from confirmed STABLE positions
synchronized with ground-truth PGN game records according to:
- STAGE_2.md, Section 4.4
- 04_VISION_PIPELINE.md, Section 4.4
- Explanatory Note, Section 2.4.4
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import cv2
import chess
import chess.pgn
import numpy as np
from pydantic import BaseModel, Field

from src.core.config import LabellerConfig, TemporalConfig, VisionConfig
from src.core.datatypes import (
    AutoLabellerSample,
    BoardGeometry,
    BoardObservation,
    BoardOrientation,
    CellROI,
    CHESS_SQUARES,
    FSMState,
    PIECE_CLASSES,
)
from src.reasoning.fsm import GameFSM
from src.vision.board_localizer import BaseBoardLocalizer, create_board_localizer
from src.vision.hand_detector import BaseHandDetector, create_hand_detector
from src.vision.homography import GeometryTransformer
from src.vision.motion_detector import MotionDetector
from src.vision.roi_extractor import CellRoiExtractor
from src.vision.video_adapter import BaseVideoAdapter

logger = logging.getLogger(__name__)


def piece_to_class_name(piece: Optional[chess.Piece]) -> str:
    """Converts python-chess Piece instance to standard 13-class name (empty, wP..bK)."""
    if piece is None:
        return "empty"
    color_prefix = "w" if piece.color == chess.WHITE else "b"
    return f"{color_prefix}{piece.symbol().upper()}"


class AutoLabellerReport(BaseModel):
    """Summary metrics of an executed AutoLabeller harvesting session."""
    total_samples: int = Field(description="Total number of harvested cell crop images")
    class_distribution: Dict[str, int] = Field(description="Sample count per piece class")
    plies_harvested: int = Field(description="Number of game positions harvested")
    manifest_path: str = Field(description="Absolute path to generated dataset_manifest.json")
    execution_time_sec: float = Field(description="Total wall-clock runtime in seconds")


class AutoLabeller:
    """Pipeline for harvesting classified square training patches from video streams."""

    def __init__(
        self,
        video_adapter: BaseVideoAdapter,
        pgn_content_or_file: Union[str, Path],
        config: Optional[LabellerConfig] = None,
        temporal_config: Optional[TemporalConfig] = None,
        vision_config: Optional[VisionConfig] = None,
        geometry: Optional[BoardGeometry] = None,
        localizer: Optional[BaseBoardLocalizer] = None,
        motion_detector: Optional[MotionDetector] = None,
        hand_detector: Optional[BaseHandDetector] = None,
        fsm: Optional[GameFSM] = None,
        roi_extractor: Optional[CellRoiExtractor] = None,
        orientation: BoardOrientation = BoardOrientation.WHITE_BOTTOM,
    ) -> None:
        self.video_adapter = video_adapter
        self.pgn_input = pgn_content_or_file
        self.config = config or LabellerConfig()
        self.temporal_config = temporal_config or TemporalConfig()
        self.vision_config = vision_config or VisionConfig()
        self.geometry = geometry
        self.orientation = orientation

        self.target_board_size = self.vision_config.target_board_size
        self.localizer = localizer or create_board_localizer(config=self.vision_config)
        self.roi_extractor = roi_extractor or CellRoiExtractor(
            board_size=self.target_board_size,
            output_cell_size=self.config.target_crop_size,
            config=self.vision_config,
        )
        self.motion_detector = motion_detector or MotionDetector(config=self.temporal_config)
        self.hand_detector = hand_detector or create_hand_detector()
        self.fsm = fsm or GameFSM(config=self.temporal_config)

        # Precompute square ROIs in 640x640 space
        self.rois: Dict[str, CellROI] = {
            sq: self.roi_extractor.build_cell_roi(sq, orientation=self.orientation)
            for sq in CHESS_SQUARES
        }

        # Parse PGN moves
        self._moves, self._starting_board = self._parse_pgn(self.pgn_input)

    def _parse_pgn(self, pgn_input: Union[str, Path]) -> Tuple[List[chess.Move], chess.Board]:
        """Loads and validates mainline moves from PGN string or file."""
        if isinstance(pgn_input, Path) or (isinstance(pgn_input, str) and "\n" not in pgn_input and Path(pgn_input).exists()):
            with open(pgn_input, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = str(pgn_input)

        pgn_io = io.StringIO(content)
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            raise ValueError("No valid chess game found in PGN input.")

        moves = list(game.mainline_moves())
        board = game.board()
        return moves, board

    def harvest_position(
        self,
        board_image: np.ndarray,
        board: chess.Board,
        ply_index: int,
        timestamp_sec: float,
        output_dir: Path,
    ) -> List[AutoLabellerSample]:
        """Extracts and saves cell crops for the given settled board position."""
        train_dir = output_dir / "train"
        samples: List[AutoLabellerSample] = []
        fen = board.fen()

        # Extract all 64 square crops at once
        rois_and_patches = self.roi_extractor.extract_all_rois(
            board_image=board_image,
            orientation=self.orientation,
            output_size=self.config.target_crop_size,
        )

        squares_to_harvest = CHESS_SQUARES[: self.config.max_samples_per_move]

        for sq in squares_to_harvest:
            cell_roi, patch = rois_and_patches[sq]
            sq_idx = chess.parse_square(sq)
            piece = board.piece_at(sq_idx)
            piece_class = piece_to_class_name(piece)

            cls_dir = train_dir / piece_class
            cls_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{ply_index:04d}_{sq}_{timestamp_sec:.2f}.{self.config.save_format}"
            rel_path = f"train/{piece_class}/{filename}"
            full_path = output_dir / rel_path

            # Save image
            cv2.imwrite(str(full_path), patch)

            sample = AutoLabellerSample(
                sample_id=f"{ply_index:04d}_{sq}_{timestamp_sec:.2f}",
                ply_index=ply_index,
                square=sq,
                piece_class=piece_class,
                fen_after_ply=fen,
                timestamp_sec=float(timestamp_sec),
                image_rel_path=rel_path,
            )
            samples.append(sample)

        return samples

    def run(self) -> AutoLabellerReport:
        """Executes full harvesting pipeline across the video adapter stream."""
        start_time = time.time()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        samples: List[AutoLabellerSample] = []
        class_counts: Dict[str, int] = {c: 0 for c in PIECE_CLASSES}

        board = self._starting_board.copy()
        current_ply = 0
        harvested_plies: Set[int] = set()

        homography_matrix: Optional[np.ndarray] = None
        if self.geometry is not None and self.geometry.is_valid:
            homography_matrix, _ = GeometryTransformer.compute_homography(
                self.geometry.corners, target_size=self.target_board_size
            )

        for frame in self.video_adapter:
            # 1. Acquire / update board geometry if not preset
            if homography_matrix is None:
                geom = self.localizer.localize(frame)
                if geom.is_valid:
                    self.geometry = geom
                    homography_matrix, _ = GeometryTransformer.compute_homography(
                        geom.corners, target_size=self.target_board_size
                    )
                else:
                    # Waiting for board localization
                    obs_lost = BoardObservation.create_default(
                        timestamp=frame.timestamp,
                        frame_id=frame.frame_id,
                        geometry=None,
                        geometry_confidence=0.0,
                    )
                    self.fsm.update(obs_lost)
                    continue

            # 2. Warp board to normalized square
            warped = GeometryTransformer.warp_board(
                frame.image, homography_matrix, target_size=self.target_board_size
            )

            # 3. Detect differential motion & hand occlusion
            motion_metrics = self.motion_detector.process_board_image(
                board_image=warped,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                orientation=self.orientation,
                rois=self.rois,
            )
            hand_det = self.hand_detector.detect(
                frame=frame,
                geometry=self.geometry,
                rois=self.rois,
            )

            # 4. Construct observation and update FSM
            obs = BoardObservation.create_default(
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                geometry=self.geometry,
                geometry_confidence=1.0,
                motion_detected=motion_metrics.is_motion_detected,
                motion_level=motion_metrics.global_motion_score,
                hand_present=hand_det.is_hand_present,
                motion_metrics=motion_metrics,
                hand_detection=hand_det,
            )

            prev_state = self.fsm.current_state
            curr_state = self.fsm.update(obs)

            # 5. Check harvesting triggers
            # A. Harvest Initial Position (ply 0) upon first reaching STABLE
            if curr_state == FSMState.STABLE and 0 not in harvested_plies:
                if not motion_metrics.is_motion_detected and not hand_det.is_hand_present:
                    new_samples = self.harvest_position(
                        board_image=warped,
                        board=board,
                        ply_index=0,
                        timestamp_sec=frame.timestamp,
                        output_dir=output_dir,
                    )
                    samples.extend(new_samples)
                    for s in new_samples:
                        class_counts[s.piece_class] += 1
                    harvested_plies.add(0)
                    current_ply = 1
                    logger.info("Harvested initial board position (ply 0, %d samples).", len(new_samples))

            # B. Harvest Settled Position after move transition (VERIFYING -> STABLE)
            elif prev_state == FSMState.VERIFYING and curr_state == FSMState.STABLE:
                if current_ply <= len(self._moves) and current_ply not in harvested_plies:
                    # Apply corresponding move from PGN
                    move = self._moves[current_ply - 1]
                    board.push(move)

                    new_samples = self.harvest_position(
                        board_image=warped,
                        board=board,
                        ply_index=current_ply,
                        timestamp_sec=frame.timestamp,
                        output_dir=output_dir,
                    )
                    samples.extend(new_samples)
                    for s in new_samples:
                        class_counts[s.piece_class] += 1
                    harvested_plies.add(current_ply)
                    logger.info(
                        "Harvested settled position after move %s (ply %d, %d samples).",
                        move.uci(),
                        current_ply,
                        len(new_samples),
                    )
                    current_ply += 1

        # 6. Generate and save dataset manifest
        manifest_path = output_dir / "dataset_manifest.json"
        manifest_data = {
            "dataset_version": "1.0.0",
            "total_samples": len(samples),
            "plies_harvested": len(harvested_plies),
            "class_distribution": class_counts,
            "target_crop_size": list(self.config.target_crop_size),
            "save_format": self.config.save_format,
            "samples": [s.model_dump() for s in samples],
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        exec_time = time.time() - start_time
        logger.info(
            "AutoLabeller finished: %d samples harvested across %d plies in %.2fs.",
            len(samples),
            len(harvested_plies),
            exec_time,
        )

        return AutoLabellerReport(
            total_samples=len(samples),
            class_distribution=class_counts,
            plies_harvested=len(harvested_plies),
            manifest_path=str(manifest_path.resolve()),
            execution_time_sec=float(exec_time),
        )
