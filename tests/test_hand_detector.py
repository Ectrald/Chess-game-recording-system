"""Unit tests for hand detection and occlusion masking.

Covers:
- STAGE_2.md Section 6: test_hand_detector.py specifications
- MockHandDetector behavior and dynamic state alteration
- ColorSilhouetteHandDetector skin-tone segmentation
- Occlusion masking and per-square ratio calculations
- Fallback handling when MediaPipe is unavailable
"""

import numpy as np
import pytest
import cv2

from src.core.config import HandDetectorConfig
from src.core.datatypes import (
    BoardGeometry,
    BoardOrientation,
    CHESS_SQUARES,
    Frame,
    HandDetection,
)
from src.vision.hand_detector import (
    BaseHandDetector,
    ColorSilhouetteHandDetector,
    DummyHandDetector,
    HandDetector,
    MediaPipeHandDetector,
    MockHandDetector,
    compute_occlusion_from_board_mask,
    create_hand_detector,
)
from src.vision.roi_extractor import CellRoiExtractor


@pytest.fixture
def dummy_geometry() -> BoardGeometry:
    """Fixture providing simple canonical 640x640 board geometry."""
    corners = [(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)]
    dst = np.array([[0, 0], [640, 0], [640, 640], [0, 640]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), dst)
    return BoardGeometry(
        corners=corners,
        homography_matrix=H.tolist(),
        inverse_homography=np.linalg.inv(H).tolist(),
        target_size=640,
        is_valid=True,
    )


@pytest.fixture
def sample_frame() -> Frame:
    """Fixture providing a blank 600x600 BGR frame."""
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    return Frame(image=img, frame_id=1, timestamp=0.033)


def test_dummy_hand_detector(sample_frame: Frame):
    """Verifies that disabled detector produces empty detection without errors."""
    detector = DummyHandDetector()
    result = detector.detect(sample_frame)
    assert not result.is_hand_present
    assert result.confidence == 0.0
    assert len(result.occluded_squares) == 0
    assert all(result.occlusion_ratios[sq] == 0.0 for sq in CHESS_SQUARES)


def test_mock_hand_detector(sample_frame: Frame):
    """Verifies MockHandDetector state toggling and square occlusion simulation."""
    detector = MockHandDetector()
    res1 = detector.detect(sample_frame)
    assert not res1.is_hand_present

    detector.set_hand_state(is_present=True, squares=["e2", "e4"])
    res2 = detector.detect(sample_frame)
    assert res2.is_hand_present
    assert "e2" in res2.occluded_squares
    assert "e4" in res2.occluded_squares
    assert res2.occlusion_ratios["e2"] == 0.85
    assert res2.occlusion_ratios["a1"] == 0.0


def test_compute_occlusion_from_board_mask():
    """Verifies per-square occlusion calculation using CellROI dictionary."""
    roi_ext = CellRoiExtractor(board_size=640)
    rois = {sq: roi_ext.build_cell_roi(sq, BoardOrientation.WHITE_BOTTOM) for sq in CHESS_SQUARES}

    mask = np.zeros((640, 640), dtype=np.uint8)
    # Draw hand covering field e2 (x: 320..400, y: 480..560)
    # Fill exactly 50% of square e2: 80 x 40 px
    mask[480:520, 320:400] = 255

    occluded_sqs, ratios = compute_occlusion_from_board_mask(
        board_mask=mask,
        rois=rois,
        occlusion_threshold=0.15,
        board_size=640,
    )

    assert "e2" in occluded_sqs
    assert pytest.approx(ratios["e2"], rel=1e-2) == 0.50
    assert ratios["e4"] == 0.0
    assert ratios["a1"] == 0.0
    assert len(occluded_sqs) == 1


def test_color_silhouette_hand_detector_with_skin_patch(dummy_geometry: BoardGeometry):
    """Verifies ColorSilhouetteHandDetector detects skin-colored regions and occluded squares."""
    # Create image with a skin-tone colored ellipse in the board region
    # In BGR: skin tone e.g. B=110, G=140, R=195 (Cr ~ 152, Cb ~ 100 in YCrCb)
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    # Draw skin-colored patch at board center (300, 300) in image space
    cv2.circle(img, (300, 300), radius=50, color=(110, 140, 195), thickness=-1)

    frame = Frame(image=img, frame_id=10, timestamp=0.33)
    detector = ColorSilhouetteHandDetector(min_hand_area_px=100)
    result = detector.detect(frame, geometry=dummy_geometry)

    assert result.is_hand_present
    assert result.bounding_box is not None
    assert len(result.occluded_squares) > 0
    # In canonical 640x640 board, center (320, 320) corresponds to rank 4/5, file d/e
    assert any(sq in result.occluded_squares for sq in ["d4", "d5", "e4", "e5"])


def test_color_silhouette_hand_detector_empty_and_gray(dummy_geometry: BoardGeometry):
    """Verifies ColorSilhouetteHandDetector returns negative on non-skin and grayscale images."""
    detector = ColorSilhouetteHandDetector()

    # Black BGR image
    black_img = np.zeros((600, 600, 3), dtype=np.uint8)
    res_black = detector.detect(Frame(image=black_img, frame_id=1, timestamp=0.0), geometry=dummy_geometry)
    assert not res_black.is_hand_present

    # Grayscale image
    gray_img = np.full((600, 600), 128, dtype=np.uint8)
    res_gray = detector.detect(Frame(image=gray_img, frame_id=2, timestamp=0.0), geometry=dummy_geometry)
    assert not res_gray.is_hand_present


def test_mediapipe_fallback_when_unavailable(sample_frame: Frame, dummy_geometry: BoardGeometry):
    """Verifies MediaPipeHandDetector handles absence of mediapipe package gracefully."""
    detector = MediaPipeHandDetector()
    # On environments where mediapipe is not installed, is_available should be False
    if not detector.is_available:
        res = detector.detect(sample_frame, geometry=dummy_geometry)
        assert not res.is_hand_present
        assert res.confidence == 0.0


def test_hand_detector_facade_and_factory(sample_frame: Frame):
    """Verifies HandDetector facade initialization, backend selection, and disable toggle."""
    # Disabled detector
    cfg_disabled = HandDetectorConfig(enabled=False)
    det_disabled = create_hand_detector(config=cfg_disabled)
    assert isinstance(det_disabled, HandDetector)
    assert det_disabled.active_backend == "disabled"
    assert not det_disabled.detect(sample_frame).is_hand_present

    # Mock backend
    det_mock = create_hand_detector(backend="mock")
    assert isinstance(det_mock, HandDetector)
    assert det_mock.active_backend == "mock"

    # Color backend
    det_color = create_hand_detector(backend="color")
    assert isinstance(det_color, HandDetector)
    assert det_color.active_backend == "color"

    # Auto backend
    det_auto = create_hand_detector(backend="auto")
    assert isinstance(det_auto, HandDetector)
    assert det_auto.active_backend in ["mediapipe", "color"]
