"""
sight sprite: machine vision with voice alerts 
"""
__version__ = "0.1.0"

from .capture import get_snapshot
from .capture import get_snapshots
from .capture import capture_video
from .capture import show_test_image
from .capture import show_test_video


__all__ = ["get_snapshot",
           "get_snapshots",
           "capture_video",
           "show_test_image",
           "show_test_video"]

