"""Vision perception package: frame acquisition, board localization and cell classification."""

from src.vision.board_localizer import (
    BaseBoardLocalizer,
    CoarseToFineLocalizer,
    ManualCornerLocalizer,
    PureHoughLocalizer,
    ArucoLocalizer,
    create_board_localizer,
)
from src.vision.corner_refinement import (
    CornerRefiner,
    order_corners_canonical,
    refine_subpixel_corners,
    detect_corners_hough_lines,
    detect_corners_contour_hull,
)
from src.vision.diagnostics import BoardDiagnosticVisualizer, draw_board_diagnostics
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
from src.vision.homography import GeometryTransformer
from src.vision.motion_detector import MotionDetector
from src.vision.orientation import OrientationResolver
from src.vision.roi_extractor import CellRoiExtractor
from src.vision.synthetic_board import SyntheticBoardGenerator
from src.vision.video_adapter import (
    BaseVideoAdapter,
    OpenCVVideoAdapter,
    SyntheticVideoAdapter,
    create_video_adapter,
)

__all__ = [
    "BaseBoardLocalizer",
    "CoarseToFineLocalizer",
    "ManualCornerLocalizer",
    "PureHoughLocalizer",
    "ArucoLocalizer",
    "create_board_localizer",
    "CornerRefiner",
    "order_corners_canonical",
    "refine_subpixel_corners",
    "detect_corners_hough_lines",
    "detect_corners_contour_hull",
    "BoardDiagnosticVisualizer",
    "draw_board_diagnostics",
    "GeometryTransformer",
    "OrientationResolver",
    "CellRoiExtractor",
    "SyntheticBoardGenerator",
    "BaseVideoAdapter",
    "OpenCVVideoAdapter",
    "SyntheticVideoAdapter",
    "create_video_adapter",
    "MotionDetector",
    "BaseHandDetector",
    "HandDetector",
    "MediaPipeHandDetector",
    "ColorSilhouetteHandDetector",
    "MockHandDetector",
    "DummyHandDetector",
    "create_hand_detector",
    "compute_occlusion_from_board_mask",
]

