# CV Safety System

A consolidated computer vision toolkit for safeguarding exhibit areas. The repository provides two runtime-ready subsystems that can operate independently or in tandem:

- **Pose demos (`examples/pose/`)** — MediaPipe-based, 33-keypoint human pose estimation with minimal, developer-friendly, and performance-optimized pipelines.
- **Safety monitor core (`src/cv_safety_sys/`)** — YOLOv7-tiny powered detection, interactive object tracking, and an integrated safety monitor that cross-references pose landmarks with detected items.

The current configuration treats **cups as the protected exhibit proxy** and flags a **tennis racket as the hazardous object**. All UI labels, filtering logic, and alerts align with this setup.

## Key Capabilities

- Real-time YOLOv7-tiny inference with centroid tracking and interactive selection of protected cups.
- Automatic safety fence calculation around selected cups with intrusion detection based on MediaPipe pose landmarks.
- Hazard detection workflow that highlights tennis rackets and binds them to the nearest tracked person for alerting.
- Lightweight dependency footprint tested on Python 3.10 / Ubuntu 22.04 (CPU-only baseline).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Clone the YOLOv7 operator code used by the detector (only required once)
git clone --depth 1 https://github.com/WongKinYiu/yolov7.git

# Launch the full desktop client (auto-downloads required models to ./models)
python run.py --source 0

# Run the integrated safety monitor (cups + tennis rackets)
python object_protection/integrated_safety_monitor.py --source 0

# Launch the PySide6 desktop client with visual alerts & optional custom sound
python object_protection/qt_monitor_app.py --source 0 [--alert-sound path/to/sound.wav]
```

`run.py` automatically downloads the MediaPipe pose model and the YOLOv7-tiny
weights into the shared `models/` directory if they are missing. To reuse
custom weights, place them under `models/` and pass `--yolo-model`.

## Chinese Text Rendering

The monitoring overlays inside the video feed now use a Unicode-capable text renderer.
The helper automatically looks for common Chinese fonts (STHeiti, Microsoft YaHei,
WenQuanYi, Noto Sans CJK, etc.). If your system stores fonts elsewhere, point the
renderer to a specific file by exporting `CV_SAFETY_FONT=/absolute/path/to/font.ttf`
before launching `run.py` or `object_protection/qt_monitor_app.py`.

## Repository Layout

```
cv_safety_sys/
├── models/                    # Cached pose & detector weights (auto-created)
├── run.py                     # One-click launcher for the PySide6 desktop client
├── src/cv_safety_sys/         # Python package with reusable modules
│   ├── detection/             # YOLOv7 tracking utilities
│   ├── monitoring/            # Integrated safety monitor core
│   ├── pose/                  # MediaPipe pose helpers & downloader
│   └── ui/                    # PySide6 desktop application
├── examples/pose/             # Stand-alone pose estimation demos
└── docs/                      # Detailed subsystem documentation
```

See the `docs/` directory for subsystem deep dives and implementation notes that correspond to the final configuration described above.
