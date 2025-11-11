# Cup Tracking & Safety Monitor

This module combines YOLOv7-tiny detection, centroid-based tracking, and MediaPipe pose landmarks to guard protected cups and flag tennis rackets as hazardous objects. The pipeline is tuned for the final demo configuration and omits earlier experimentation notes.

## Folder Overview

```
src/cv_safety_sys/
├── detection/yolov7_tracker.py      # YOLOv7-tiny cup detector with interactive tracking
├── monitoring/integrated_monitor.py # Cup fence monitoring + pose + tennis racket alerts
└── ui/qt_monitor.py                 # PySide6 desktop client and monitor bootstrap helpers
```

## Final Feature Set

- **Cup-centric detection**: `video_relic_tracking.py` limits the selectable inventory to the `cup` class. Selected instances receive persistent IDs, confidence reporting, and an expanded safety fence for intrusion monitoring.
- **Tennis racket hazard detection**: `integrated_safety_monitor.py` filters YOLOv7 results to `{cup, person, tennis racket}`. Any detected tennis racket is highlighted and associated with nearby tracked people to generate carry alerts.
- **Pose-aware risk analysis**: MediaPipe pose landmarks are projected onto the frame to determine whether tracked joints breach a cup’s safety fence.
- **Unified user interface**: Both scripts expose consistent mouse/keyboard controls and overlay conventions so the interaction model remains the same across the subsystem.

## Operating the Scripts

```bash
# Launch the full desktop client (defaults to the first webcam)
python run.py --source 0

# Optional: run the detector headlessly for testing
python -m cv_safety_sys.detection.yolov7_tracker --source 0
```

Key flags (applies to both entry points):
- `--source`: camera index or path to a video file.
- `--conf`: confidence threshold override (default `0.25` in the GUI, `0.1` for the CLI tracker).
- `--pose-model`: override the cached MediaPipe pose model path (defaults to `models/pose_landmarker_full.task`).
- `--yolo-model`: override the cached YOLOv7-tiny weights (defaults to `models/yolov7-tiny.pt`).

The monitor bootstrap automatically downloads missing assets and reuses the shared `SimpleTracker` implementation.

## Implementation Notes

- The YOLOv7-tiny weights are retrieved from the official release and cached locally; place the file manually if the host lacks network access.
- Hazard filtering relies on the shared `DANGEROUS_CLASSES = {'tennis racket'}` constant. Update that value across the subsystem if the hazard policy changes.
- All detections are mapped back to the original frame size before tracking to ensure that pose landmarks and safety fences align perfectly with the rendered frame.
