# CV Safety System

A consolidated computer vision toolkit for safeguarding exhibit areas. The repository provides two runtime-ready subsystems that can operate independently or in tandem:

- **WebcamPoseDetection** — MediaPipe-based, 33-keypoint human pose estimation with minimal, developer-friendly, and performance-optimized pipelines.
- **object_protection** — YOLOv7-tiny powered detection, interactive object tracking, and an integrated safety monitor that cross-references pose landmarks with detected items.

The current configuration treats **cups as the protected exhibit proxy** and flags a **tennis racket as the hazardous object**. All UI labels, filtering logic, and alerts align with this setup.

## Key Capabilities

- Real-time YOLOv7-tiny inference with centroid tracking and interactive selection of protected cups.
- Automatic safety fence calculation around selected cups with intrusion detection based on MediaPipe pose landmarks.
- Hazard detection workflow that highlights tennis rackets and binds them to the nearest tracked person for alerting.
- Lightweight dependency footprint tested on Python 3.9 / Ubuntu 22.04 (CPU-only baseline).

## Quick Start

```bash
# Install shared dependencies
pip install -r requirements.txt

# Download MediaPipe pose model
python WebcamPoseDetection/download_model.py

# Run the cup tracker (default webcam)
python object_protection/video_relic_tracking.py --source 0

# Run the integrated safety monitor (cups + tennis rackets)
python object_protection/integrated_safety_monitor.py --source 0
```

Each script accepts `--source` to select a camera index or video file. The first YOLO-based run downloads `yolov7-tiny.pt` automatically unless the file is already present in `object_protection/`.

## Repository Layout

```
cv_safety_sys/
├── WebcamPoseDetection/       # Pose estimation subsystem
├── object_protection/         # Cup tracking and safety monitor
├── docs/                      # Detailed subsystem documentation
└── envs/                      # Optional environment descriptors
```

See the `docs/` directory for subsystem deep dives and implementation notes that correspond to the final configuration described above.
