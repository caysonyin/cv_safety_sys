#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键启动文物安全协同防护客户端，并自动校验模型资源。"""

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


REPO_ROOT = Path(__file__).resolve().parent
YOLO_REPO_URL = "https://github.com/WongKinYiu/yolov7.git"


def ensure_pose_model(path: Path) -> Path:
    """确保姿态模型存在。"""

    if path.exists():
        return path

    downloaded = download_pose_model(path)
    if downloaded is None:
        raise RuntimeError(
            "无法下载姿态模型，请检查网络连接或手动放置模型到 models/ 目录"
        )
    return Path(downloaded)


def ensure_yolo_model(path: Path) -> Path:
    """确保 YOLOv7-tiny 模型存在。"""

    downloaded = download_yolov7_tiny(path)
    if downloaded is None:
        raise RuntimeError(
            "无法准备YOLO模型，请确认网络连接或手动将权重放到 models/ 目录"
        )
    return downloaded


def check_yolov7_repo() -> None:
    """简单检测 YOLOv7 源码目录是否存在。"""

    yolo_dir = REPO_ROOT / "yolov7"
    if not yolo_dir.exists():
        raise RuntimeError(
            "未找到 yolov7 源码目录，请先运行\n"
            f"  git clone --depth 1 {YOLO_REPO_URL} yolov7"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文物安全协同防护客户端入口")
    parser.add_argument('--source', type=str, default='0', help='视频源(0=摄像头或视频路径)')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO 置信度阈值')
    parser.add_argument('--pose-model', type=str, default=str(DEFAULT_POSE_MODEL_PATH), help='姿态模型路径')
    parser.add_argument('--yolo-model', type=str, default=str(DEFAULT_YOLO_MODEL_PATH), help='YOLO 模型路径')
    parser.add_argument('--alert-sound', type=str, default=None, help='报警提示音文件路径（可选）')
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
