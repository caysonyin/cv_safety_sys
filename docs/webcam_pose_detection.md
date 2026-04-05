<!-- Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen -->
<!-- File purpose: Pose module usage, model handling, and runtime guidance. -->

# Pose Detection Module Guide

The repository uses **MediaPipe Tasks Pose Landmarker** for pose estimation.

Model preparation is handled by `src/cv_safety_sys/pose/model_downloader.py`, and inference is integrated into `IntegratedSafetyMonitor`.

## Module Locations

- Model downloader: `src/cv_safety_sys/pose/model_downloader.py`
- Pose inference wrapper: `PoseLandmarkHelper` in `src/cv_safety_sys/monitoring/integrated_monitor.py`

## Model File

- Default model path: `models/pose_landmarker_full.task`
- Missing model files are downloaded automatically on first run.

Manual pre-download command:

```bash
PYTHONPATH=src python -m cv_safety_sys.pose.model_downloader
```

## Role in the Safety System

1. Run per-frame human pose inference (33 keypoints).
2. Build pose bounding boxes from keypoints.
3. Match pose entries to YOLO person detections using IoU.
4. Provide key geometric signals for fence intrusion checks and dangerous-object/person association.

## Practical Notes

- If runtime is slow, lower input resolution or use a faster machine.
- If model download fails, place the `.task` file under `models/` manually.
- To use a custom model file, pass `--pose-model <path>`.
