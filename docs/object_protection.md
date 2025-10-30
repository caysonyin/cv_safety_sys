# Cup Tracking & Safety Monitor

This module combines YOLOv7-tiny detection, centroid-based tracking, and MediaPipe pose landmarks to guard protected cups and flag tennis rackets as hazardous objects. The pipeline is tuned for the final demo configuration and omits earlier experimentation notes.

## Folder Overview

```
object_protection/
├── video_relic_tracking.py       # YOLOv7-tiny cup detector with interactive tracking
├── integrated_safety_monitor.py  # Cup fence monitoring + pose + tennis racket alerts
├── general.py                    # Trimmed utility helpers imported from YOLOv7
├── yolov7-tiny.pt                # Inference weights (downloaded on first run)
└── yolov7/                       # Vendor directory required by the detector
```

## Final Feature Set

- **Cup-centric detection**: `video_relic_tracking.py` limits the selectable inventory to the `cup` class. Selected instances receive persistent IDs, confidence reporting, and an expanded safety fence for intrusion monitoring.
- **Tennis racket hazard detection**: `integrated_safety_monitor.py` filters YOLOv7 results to `{cup, person, tennis racket}`. Any detected tennis racket is highlighted and associated with nearby tracked people to generate carry alerts.
- **Pose-aware risk analysis**: MediaPipe pose landmarks are projected onto the frame to determine whether tracked joints breach a cup’s safety fence.
- **Unified user interface**: Both scripts expose consistent mouse/keyboard controls and overlay conventions so the interaction model remains the same across the subsystem.

## Operating the Scripts

```bash
# Interactive cup tracking (default webcam)
python object_protection/video_relic_tracking.py --source 0

# Integrated monitor combining cups, people, and tennis rackets
python object_protection/integrated_safety_monitor.py --source 0
```

Optional flags:
- `--source`: camera index or path to a video file.
- `--conf`: confidence threshold override (default `0.1`).

The integrated monitor automatically loads the MediaPipe pose model downloaded via `WebcamPoseDetection/download_model.py` and reuses the shared `SimpleTracker`.

## Implementation Notes

- The YOLOv7-tiny weights are retrieved from the official release and cached locally; place the file manually if the host lacks network access.
- Hazard filtering relies on the shared `DANGEROUS_CLASSES = {'tennis racket'}` constant. Update that value across the subsystem if the hazard policy changes.
- All detections are mapped back to the original frame size before tracking to ensure that pose landmarks and safety fences align perfectly with the rendered frame.
