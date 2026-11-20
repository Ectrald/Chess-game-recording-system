"""User Interface package for Chess Game Recording System.

Provides visual HUD composition, interactive operator dashboard,
and command-line interface (CLI).
"""

from src.ui.cli import build_parser, main
from src.ui.dashboard import InteractiveDashboard
from src.ui.visualizer import VisualHUDComposer

__all__ = [
    "VisualHUDComposer",
    "InteractiveDashboard",
    "main",
    "build_parser",
]
