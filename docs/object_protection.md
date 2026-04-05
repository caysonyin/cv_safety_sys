<!-- Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen -->
<!-- File purpose: Relic protection workflow and alert fusion behavior. -->

# Relic Protection and Safety Fusion

This document explains the relic-protection workflow that combines relic detection, person pose landmarks, and dangerous-object association. The central orchestrator is `IntegratedSafetyMonitor`.


```text
src/cv_safety_sys/
├── detection/yolov7_tracker.py      # Detection, tracking, relic selection
├── monitoring/integrated_monitor.py # Fusion logic and alert decisions
└── ui/qt_monitor.py                 # Qt interface and interactions
```

## End-to-End Workflow

1. **Detection stage**: YOLOv7-tiny detects relics, persons, and dangerous objects.
2. **Tracking stage**: detections are assigned stable `track_id`s for temporal consistency.
3. **Relic fence stage**: selected relic boxes are expanded into protection fences.
4. **Pose association stage**: 33 pose keypoints are matched to person detections and checked against fences.
5. **Danger association stage**: dangerous objects are linked to nearby persons and may trigger higher severity alerts.
6. **Alert output stage**: structured alert objects are emitted for UI list rendering and frame overlays.

## Default Categories

- Protected relic class: `cup`
- Dangerous classes: `knife`, `scissors`, `baseball bat`

> To extend dangerous categories, update `DANGEROUS_CLASSES` in `integrated_monitor.py`.

## Common Commands


```bash
# Recommended integrated runtime
python run.py --source 0

# Debug fused monitoring logic (OpenCV window)
PYTHONPATH=src python -m cv_safety_sys.monitoring.integrated_monitor --source 0 --conf 0.25

# Detection/tracking only
PYTHONPATH=src python -m cv_safety_sys.detection.yolov7_tracker --source 0 --conf 0.1
```

## Qt Interaction Notes

- Clicking targets in the video area enters relic-selection workflow (see in-app status/toast messages).
- The alert panel lists current events; selecting an item highlights the corresponding target context.
- Use `--alert-sound` to provide a custom local sound file for alerts.
