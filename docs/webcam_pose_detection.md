# 姿态检测模块说明

当前仓库中的姿态能力由 `MediaPipe Tasks Pose Landmarker` 提供，主要由 `src/cv_safety_sys/pose/model_downloader.py` 负责模型准备，并在 `IntegratedSafetyMonitor` 中调用。

## 模块位置

- 模型下载器：`src/cv_safety_sys/pose/model_downloader.py`
- 姿态推理封装：`src/cv_safety_sys/monitoring/integrated_monitor.py` 中的 `PoseLandmarkHelper`

## 模型文件

- 默认路径：`models/pose_landmarker_full.task`
- 首次运行 `run.py` 或监控模块时，会自动下载缺失模型。

如需手动预下载：

```bash
PYTHONPATH=src python -m cv_safety_sys.pose.model_downloader
```

## 在系统中的作用

1. 对每帧图像进行人体姿态推理（33 关键点）。
2. 生成每个人体的关键点包围框（pose bbox）。
3. 将姿态结果与 YOLO 的 person 检测框进行 IoU 匹配。
4. 为“入侵围栏判定”和“危险物-人员关联”提供关键人体几何信息。

## 运行建议

- 若画面卡顿，可降低输入分辨率或使用更高性能设备。
- 若模型下载失败，可手动将 `.task` 文件放入 `models/` 再重试。
- 若需要替换模型，可通过 `--pose-model` 指向自定义路径。

