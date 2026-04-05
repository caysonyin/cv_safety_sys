#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen
# File purpose: Download and cache the MediaPipe pose model into the local models directory.
"""Download the MediaPipe Pose Landmarker model into the project's ``models`` directory."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_full.task"


def download_model(destination: Optional[Path] = None) -> Optional[str]:
    """Ensure the pose model exists and return the local path."""

    destination = destination or DEFAULT_MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"Model already exists: {destination}")
        return str(destination)

    try:
        print("Downloading pose model, please wait...")
        urllib.request.urlretrieve(MODEL_URL, destination)
        print(f"Model download completed: {destination}")
        return str(destination)
    except Exception as e:
        print(f"Download failed: {e}")
        return None

if __name__ == "__main__":
    model_path = download_model()
    if model_path:
        print("Model is ready. You can now run pose detection.")
    else:
        print("Model download failed. Please check your network connection.")
