#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen
# File purpose: One-command launcher for the integrated safety monitoring desktop client.
"""Launch the integrated relic safety client and validate required model resources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from PySide6.QtWidgets import QApplication

from cv_safety_sys.detection.yolov7_tracker import (
    DEFAULT_YOLO_MODEL_PATH,
    download_yolov7_tiny,
)
from cv_safety_sys.pose.model_downloader import (
    DEFAULT_MODEL_PATH as DEFAULT_POSE_MODEL_PATH,
    download_model as download_pose_model,
)
from cv_safety_sys.ui.qt_monitor import SafetyMonitorWindow, prepare_monitor


YOLO_REPO_URL = "https://github.com/WongKinYiu/yolov7.git"


def ensure_pose_model(path: Path) -> Path:
    """Ensure that the pose model exists."""

    if path.exists():
        return path

    downloaded = download_pose_model(path)
    if downloaded is None:
        raise RuntimeError(
            "Failed to download the pose model. Check your network or place it in models/."
        )
    return Path(downloaded)


def ensure_yolo_model(path: Path) -> Path:
    """Ensure that the YOLOv7-tiny model exists."""

    downloaded = download_yolov7_tiny(path)
    if downloaded is None:
        raise RuntimeError(
            "Failed to prepare the YOLO model. Check your network or place weights in models/."
        )
    return downloaded


def check_yolov7_repo() -> None:
    """Verify that the local yolov7 source directory exists."""

    yolo_dir = REPO_ROOT / "yolov7"
    if not yolo_dir.exists():
        raise RuntimeError(
            "Missing local yolov7 source directory. Run:\n"
            f"  git clone --depth 1 {YOLO_REPO_URL} yolov7"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated relic safety client entrypoint")
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam or a video path)')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--pose-model', type=str, default=str(DEFAULT_POSE_MODEL_PATH), help='Pose model path')
    parser.add_argument('--yolo-model', type=str, default=str(DEFAULT_YOLO_MODEL_PATH), help='YOLO model path')
    parser.add_argument('--alert-sound', type=str, default=None, help='Optional alert sound file path')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    check_yolov7_repo()

    pose_model_path = ensure_pose_model(Path(args.pose_model))
    yolo_model_path = ensure_yolo_model(Path(args.yolo_model))

    monitor = prepare_monitor(args.conf, pose_model_path, yolo_model_path)

    video_source: int | str = int(args.source) if args.source.isdigit() else args.source
    alert_sound = Path(args.alert_sound) if args.alert_sound else None

    app = QApplication(sys.argv)
    window = SafetyMonitorWindow(
        monitor,
        video_source,
        alert_sound if alert_sound and alert_sound.exists() else None,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
