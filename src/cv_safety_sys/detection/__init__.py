"""Detection utilities for the CV Safety System."""

from .yolov7_tracker import (
    SimpleTracker,
    VideoRelicTracker,
    download_yolov7_tiny,
    load_model,
    DEFAULT_YOLO_MODEL_PATH,
)

__all__ = [
    "SimpleTracker",
    "VideoRelicTracker",
    "download_yolov7_tiny",
    "load_model",
    "DEFAULT_YOLO_MODEL_PATH",
]
