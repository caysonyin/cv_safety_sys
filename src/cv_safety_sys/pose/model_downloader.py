#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 MediaPipe Pose Landmarker 模型到项目 ``models`` 目录。"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_full.task"


def download_model(destination: Optional[Path] = None) -> Optional[str]:
    """确保姿态识别模型存在并返回其路径。"""

    destination = destination or DEFAULT_MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"模型文件已存在: {destination}")
        return str(destination)

    try:
        print("正在下载模型文件，请稍候...")
        urllib.request.urlretrieve(MODEL_URL, destination)
        print(f"模型下载完成: {destination}")
        return str(destination)
    except Exception as e:
        print(f"下载失败: {e}")
        return None

if __name__ == "__main__":
    model_path = download_model()
    if model_path:
        print("模型准备就绪，可以运行姿态检测程序了！")
    else:
        print("模型下载失败，请检查网络连接")
