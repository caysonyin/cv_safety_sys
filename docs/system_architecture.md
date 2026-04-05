<!-- Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen -->
<!-- File purpose: High-level architecture, module boundaries, and data flow. -->

# System Architecture

This project performs real-time safety monitoring by combining **protected relics, human pose landmarks, and dangerous objects**.

The main runtime pipeline is:

**Video Input → Detection/Tracking → Pose Estimation → Safety Fusion Logic → UI/Alerts**

## Modules and Responsibilities

| Module | Key File | Responsibility |
| --- | --- | --- |
| Startup and resource checks | `run.py` | Validates local `yolov7/` repository, prepares model paths, and launches the Qt client. |
| Detection and tracking | `src/cv_safety_sys/detection/yolov7_tracker.py` | Runs YOLOv7-tiny detection, class filtering, `SimpleTracker`, and relic selection interaction. |
| Pose model management | `src/cv_safety_sys/pose/model_downloader.py` | Downloads and caches the MediaPipe Pose Landmarker model. |
| Safety fusion logic | `src/cv_safety_sys/monitoring/integrated_monitor.py` | Fuses person/relic/dangerous-object detections with pose points and produces fences, alerts, and stats. |
| Visualization and interaction | `src/cv_safety_sys/ui/qt_monitor.py` | Implements PySide6 UI, video panel, alert list, status widgets, and mouse/keyboard interactions. |

## Data Flow

1. **Video capture**: OpenCV reads frames from a camera or a video file.
2. **Detection and class filtering**: YOLOv7 outputs bounding boxes and classes, including `cup`, `person`, and configured dangerous classes.
3. **Tracking**: `SimpleTracker` maintains stable `track_id`s across frames.
4. **Pose estimation**: MediaPipe Pose produces 33 keypoints and associates pose entries to person detections via IoU.
5. **Safety logic**:
   - Build protection fences around selected relics.
   - Detect whether human keypoints enter the fenced region.
   - Associate dangerous objects to nearby persons and escalate alert severity.
6. **Output rendering**: Push structured status/alerts to OpenCV/Qt views and update alert history and counters.

## Runtime Entry Points

- `python run.py --source 0`: recommended integrated desktop client.
- `PYTHONPATH=src python -m cv_safety_sys.monitoring.integrated_monitor --source 0`: OpenCV monitoring view.
- `PYTHONPATH=src python -m cv_safety_sys.ui.qt_monitor --source 0`: direct Qt module entry.

## Models and Dependencies

- Default YOLO weights path: `models/yolov7-tiny.pt`
- Default pose model path: `models/pose_landmarker_full.task`
- YOLO inference code folder: repository root `yolov7/` (must be cloned manually)
