"""Model management utilities for the safety monitoring system."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import requests
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

from WebcamPoseDetection.download_model import download_model as download_pose_landmarker

LOGGER = logging.getLogger(__name__)

MODELS_DIR = Path("models")
POSE_MODEL_NAME = "pose_landmarker_full.task"
DETECTION_WEIGHTS_NAME = "fasterrcnn_resnet50_fpn_coco.pth"
DETECTION_WEIGHTS_URL = (
    "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
)


def ensure_models() -> Tuple[Path, Path]:
    """Ensure that both pose and detection models are downloaded."""

    MODELS_DIR.mkdir(exist_ok=True)

    pose_result = download_pose_landmarker()
    if not pose_result:
        raise RuntimeError("Failed to download MediaPipe pose model")
    pose_path = Path(pose_result)
    if not pose_path.exists():
        raise RuntimeError("MediaPipe pose model missing after download")

    detection_path = MODELS_DIR / DETECTION_WEIGHTS_NAME
    if not detection_path.exists():
        LOGGER.info("Downloading detection weights from %s", DETECTION_WEIGHTS_URL)
        _download_file(DETECTION_WEIGHTS_URL, detection_path)
    return pose_path, detection_path


def load_detection_model(weights_path: Path):
    """Load a Faster R-CNN detection model on CPU."""

    device = torch.device("cpu")
    model = fasterrcnn_resnet50_fpn(weights=None, progress=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]


def _download_file(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    tmp_path = destination.with_suffix(".tmp")
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with tmp_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            file.write(chunk)
            downloaded += len(chunk)
    os.replace(tmp_path, destination)
    LOGGER.info("Downloaded %.2f MB to %s", downloaded / (1024 * 1024), destination)
