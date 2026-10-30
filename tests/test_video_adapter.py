"""Unit tests for video adapter and synthetic board generator in src/vision/."""

import numpy as np
import pytest

from src.core.config import CameraConfig
from src.core.exceptions import VideoSourceError
from src.vision.synthetic_board import SyntheticBoardGenerator
from src.vision.video_adapter import (
    OpenCVVideoAdapter,
    SyntheticVideoAdapter,
    create_video_adapter,
)


class TestSyntheticBoardGenerator:
    def test_default_initialization(self):
        gen = SyntheticBoardGenerator(board_size=640, canvas_width=1280, canvas_height=720)
        assert gen.total_frames >= 10
        assert len(gen.corners) == 4

    def test_set_game_and_render_frame(self):
        gen = SyntheticBoardGenerator(frames_per_move=20)
        # 1. e4 e5
        gen.set_game_from_moves(["e2e4", "e7e5"])
        assert len(gen.moves) == 2
        assert gen.total_frames == 3 * 20  # 60 frames

        frame = gen.render_frame(0)
        assert frame.frame_id == 0
        assert frame.timestamp == 0.0
        assert frame.width == 1920
        assert frame.height == 1080
        assert isinstance(frame.image, np.ndarray)

    def test_ground_truth_observation(self):
        gen = SyntheticBoardGenerator()
        obs = gen.get_ground_truth_observation(frame_id=0)
        assert len(obs.cells) == 64
        # a1 should have white rook in initial position
        assert obs.cells["a1"].predicted_piece == "wR"
        assert obs.cells["e1"].predicted_piece == "wK"
        assert obs.cells["e8"].predicted_piece == "bK"
        assert obs.cells["e4"].predicted_piece == "empty"

    def test_ground_truth_geometry(self):
        gen = SyntheticBoardGenerator()
        geom = gen.get_ground_truth_geometry()
        assert geom.is_valid is True
        assert geom.confidence == 1.0
        assert len(geom.corners) == 4
        assert geom.homography_matrix is not None


class TestSyntheticVideoAdapter:
    def test_read_sequence_and_eof(self, short_game_pgn_path):
        gen = SyntheticBoardGenerator(frames_per_move=10)
        gen.set_game_from_pgn(str(short_game_pgn_path))
        adapter = SyntheticVideoAdapter(generator=gen)

        assert adapter.is_opened() is True
        assert adapter.get_frame_count() == gen.total_frames
        assert adapter.get_fps() == gen.fps

        frames_read = 0
        while True:
            frame = adapter.read()
            if frame is None:
                break
            frames_read += 1

        assert frames_read == gen.total_frames
        # Next read should continue returning None (EOF)
        assert adapter.read() is None

    def test_seek(self):
        gen = SyntheticBoardGenerator(frames_per_move=10)
        gen.set_game_from_moves(["e2e4"])
        adapter = SyntheticVideoAdapter(generator=gen)

        ok = adapter.seek(5)
        assert ok is True
        frame = adapter.read()
        assert frame.frame_id == 5

        # Seeking beyond total frames should fail
        bad_seek = adapter.seek(999999)
        assert bad_seek is False

    def test_context_manager(self):
        gen = SyntheticBoardGenerator(frames_per_move=10)
        with SyntheticVideoAdapter(generator=gen) as adapter:
            assert adapter.is_opened() is True
            frame = adapter.read()
            assert frame is not None
        assert adapter.is_opened() is False


class TestVideoAdapterFactory:
    def test_create_synthetic_adapter(self, short_game_pgn_path):
        cfg = CameraConfig(
            source_type="synthetic",
            source_path=str(short_game_pgn_path),
            fps=30.0,
            width=1280,
            height=720,
        )
        adapter = create_video_adapter(cfg)
        assert isinstance(adapter, SyntheticVideoAdapter)
        assert adapter.get_frame_count() > 0

    def test_invalid_source_type(self):
        cfg = CameraConfig()
        cfg.source_type = "invalid_protocol"  # bypass validator via direct set if possible
        with pytest.raises(VideoSourceError):
            create_video_adapter(cfg)

    def test_opencv_adapter_invalid_file_raises_error(self):
        cfg = CameraConfig(
            source_type="file",
            source_path="/nonexistent/video/file.mp4",
        )
        with pytest.raises(VideoSourceError):
            create_video_adapter(cfg)
