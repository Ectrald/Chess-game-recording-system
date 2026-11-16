"""Unit tests for diagnostic visualization in src/vision/diagnostics.py."""

import numpy as np
import pytest

from src.core.datatypes import BoardOrientation, Frame
from src.vision.diagnostics import BoardDiagnosticVisualizer, draw_board_diagnostics
from src.vision.homography import GeometryTransformer
from src.vision.roi_extractor import CellRoiExtractor


class TestDiagnosticsOverlay:
    def test_draw_diagnostics_with_valid_geometry(self, sample_corners, sample_frame):
        geom = GeometryTransformer.build_board_geometry(
            sample_corners,
            target_size=640,
            image_shape=(sample_frame.height, sample_frame.width),
            orientation=BoardOrientation.WHITE_BOTTOM,
            detection_method="pure_hough",
        )
        extractor = CellRoiExtractor(board_size=640)
        h_inv = geom.to_numpy_inverse_homography()
        rois = {
            sq: extractor.build_cell_roi(sq, orientation=geom.orientation, inverse_homography=h_inv)
            for sq in ["a1", "e4", "h8"]
        }
        thumb = np.zeros((640, 640, 3), dtype=np.uint8)

        vis = draw_board_diagnostics(
            frame=sample_frame,
            geometry=geom,
            rois=rois,
            normalized_board=thumb,
            draw_labels=True,
            draw_grid=True,
        )

        assert isinstance(vis, np.ndarray)
        assert vis.shape == sample_frame.image.shape
        assert vis.dtype == np.uint8
        # Ensure diagnostic overlay added visual changes
        assert not np.array_equal(vis, sample_frame.image)

    def test_draw_diagnostics_empty_geometry(self, sample_frame):
        vis = draw_board_diagnostics(
            frame=sample_frame,
            geometry=None,
            rois=None,
            normalized_board=None,
        )
        assert isinstance(vis, np.ndarray)
        assert vis.shape == sample_frame.image.shape

