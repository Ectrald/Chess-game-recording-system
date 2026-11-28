"""Interactive Operator Dashboard for Chess Game Recording System.

Provides real-time visualization through OpenCV HighGUI with hotkey controls
for pause/resume, move rollback, orientation recalibration, and operator
intervention during UNCERTAIN move arbitration. Automatically detects and
falls back to headless execution on servers without X11/Wayland display.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import cv2

from src.core.datatypes import PipelineState, SessionManifest
from src.orchestrator import SystemOrchestrator
from src.ui.visualizer import VisualHUDComposer
from src.vision.video_adapter import BaseVideoAdapter

logger = logging.getLogger(__name__)


class InteractiveDashboard:
    """Interactive desktop dashboard and loop manager with OpenCV HighGUI."""

    def __init__(
        self,
        orchestrator: SystemOrchestrator,
        headless: bool = False,
        composer: Optional[VisualHUDComposer] = None,
        window_name: str = "Chess Recording System HUD",
    ) -> None:
        """Initializes operator dashboard.

        Args:
            orchestrator: SystemOrchestrator instance coordinating pipeline.
            headless: Whether to force headless mode without GUI display.
            composer: Optional VisualHUDComposer instance.
            window_name: Title string for OpenCV display window.
        """
        self.orchestrator = orchestrator
        self.window_name = window_name
        self.composer = composer or VisualHUDComposer(
            width=self.orchestrator.config.ui.display_width,
            height=self.orchestrator.config.ui.display_height,
            font_scale=self.orchestrator.config.ui.font_scale,
            show_virtual_board=self.orchestrator.config.ui.show_virtual_board,
            show_move_list=self.orchestrator.config.ui.show_move_list,
            show_eval_bar=self.orchestrator.config.ui.show_eval_bar,
            show_hand_landmarks=self.orchestrator.config.ui.show_hand_landmarks,
        )

        # Detect display availability
        self.headless = headless or self.orchestrator.config.ui.headless
        if not self.headless:
            display = os.environ.get("DISPLAY")
            wayland = os.environ.get("WAYLAND_DISPLAY")
            if not display and not wayland:
                logger.warning("No DISPLAY or WAYLAND_DISPLAY detected; switching to headless mode.")
                self.headless = True
            else:
                try:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name, self.composer.width, self.composer.height)
                except cv2.error as err:
                    logger.warning("Failed to initialize OpenCV HighGUI window (%s); falling back to headless.", err)
                    self.headless = True

    def run_loop(
        self,
        video_adapter: BaseVideoAdapter,
        max_frames: Optional[int] = None,
        on_frame_processed: Optional[Callable[[PipelineState], None]] = None,
    ) -> SessionManifest:
        """Executes main processing loop over incoming video adapter frames.

        Args:
            video_adapter: Video source adapter providing sequential video frames.
            max_frames: Optional frame limit for benchmark and batch runs.
            on_frame_processed: Optional per-frame hook callback.

        Returns:
            Final SessionManifest after stopping orchestrator and finalizing session.
        """
        self.orchestrator.start_session()
        nominal_fps = video_adapter.get_fps()
        delay_ms = max(1, int(1000.0 / max(1.0, nominal_fps))) if nominal_fps > 0 else 33

        processed_count = 0
        logger.info(
            "Entering dashboard run_loop: headless=%s nominal_fps=%.1f",
            self.headless,
            nominal_fps,
        )

        try:
            while video_adapter.is_opened():
                frame = video_adapter.read()
                if frame is None:
                    logger.info("Video stream reached end of file or closed.")
                    break

                # Advance orchestrator pipeline
                state = self.orchestrator.process_frame(frame)
                processed_count += 1

                if on_frame_processed:
                    on_frame_processed(state)

                # GUI rendering and hotkey processing
                if not self.headless:
                    geom = getattr(self.orchestrator, "_board_geometry", None)
                    corners = geom.corners if geom is not None else None
                    vis_frame = self.composer.compose(frame.image, state, corners=corners)
                    cv2.imshow(self.window_name, vis_frame)

                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key != 255:
                        # Space: Toggle Pause
                        if key == ord(" "):
                            is_paused = self.orchestrator.toggle_pause()
                            logger.info("Operator triggered Pause=%s", is_paused)

                        # 'R': Orientation & Geometry Recalibration
                        elif key in (ord("r"), ord("R")):
                            self.orchestrator.recalibrate()
                            logger.info("Operator requested board & orientation recalibration.")

                        # 'Z': Rollback last move
                        elif key in (ord("z"), ord("Z")):
                            reverted = self.orchestrator.rollback_move()
                            logger.info("Operator triggered rollback: %s", reverted.move_san if reverted else "None")

                        # 'U': Auto-accept top candidate on UNCERTAIN
                        elif key in (ord("u"), ord("U")):
                            self.orchestrator.resolve_uncertainty(auto_accept_top=True)

                        # '1'..'5': Accept specific ranked candidate
                        elif ord("1") <= key <= ord("5"):
                            cand_idx = key - ord("1")
                            self.orchestrator.resolve_uncertainty(candidate_index=cand_idx)

                        # 'P': Export PGN snapshot
                        elif key in (ord("p"), ord("P")):
                            pgn_str = self.orchestrator.export_pgn()
                            logger.info("PGN snapshot generated (%d chars)", len(pgn_str))

                        # 'Q' or Esc: Graceful exit
                        elif key in (ord("q"), ord("Q"), 27):
                            logger.info("Operator initiated shutdown via hotkey.")
                            break

                if max_frames is not None and processed_count >= max_frames:
                    logger.info("Reached maximum requested frames limit (%d)", max_frames)
                    break

        finally:
            if not self.headless:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass

        manifest = self.orchestrator.stop()
        return manifest

