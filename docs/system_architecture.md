# 系统总体架构

该项目面向“受保护展品 + 人体姿态 + 危险物品”联动防护场景，所有子模块围绕统一的摄像头输入与模型缓存目录协同工作。整体架构由 **输入采集层 → 感知推理层 → 安全策略层 → 展示与告警层** 四个部分组成。

## 模块划分

| 模块 | 关键文件 | 核心职责 |
| --- | --- | --- |
| 摄像头采集层 | `examples/pose/*.py`、`run.py` | 拉取本地或 RTSP 视频流，统一输出 `BGR` 帧。 |
| 姿态感知层 | `cv_safety_sys/pose/*` | 封装 MediaPipe 33 关节点推理、模型下载与坐标变换。 |
| 目标检测与跟踪层 | `cv_safety_sys/detection/yolov7_tracker.py` | 载入 YOLOv7-tiny、执行类别筛选与质心跟踪。 |
| 安全策略层 | `cv_safety_sys/monitoring/integrated_monitor.py` | 结合姿态与目标结果，计算安全围栏、危险关联和告警等级。 |
| 展示与交互层 | `cv_safety_sys/ui/qt_monitor.py`、`object_protection/qt_monitor_app.py` | PySide6 桌面 UI、鼠标选取、防护区可视化与历史记录。 |

所有模型文件默认存放在仓库根目录的 `models/` 下，通过 `run.py` 或各子模块的 `download_*` 方法自动拉取，避免重复配置。

## 数据流与关键接口

1. **帧采集**：`VideoSource`（PySide6 UI）或 OpenCV Webcam 拉取原始帧，并带上时间戳、分辨率。
2. **目标检测 → 追踪**：
   - `YOLOv7TinyDetector` 返回 `BBox(xyxy) + class_id + score`。
   - `SimpleTracker` 维护 `track_id → bbox`，并记录“受保护展品”的自定义属性。
3. **姿态推理 → 坐标映射**：
   - `PoseLandmarker` 输出 33 个归一化坐标。
   - `cv_safety_sys.pose.postprocess.scale_landmarks` 将归一化点映射回与检测框同一坐标系。
4. **安全策略聚合**：
   - `CupFence` 根据被标记的展品生成扩展矩形。
   - `HazardBinder` 计算危险物体与最近人体骨架之间的欧氏距离，决定告警级别。
5. **UI 呈现**：`QtMonitorWidget` 将跟踪数据、姿态关节点、围栏与告警文本绘制到显示层，并暴露鼠标事件给 `SimpleTracker`。

以上流程通过 `IntegratedSafetyMonitor` 这个 orchestrator 串联，任何脚本只要实例化它就能获得相同的业务逻辑。

## 线程与性能策略

- **推理并行**：在高性能脚本中，会使用 `ThreadPoolExecutor` 将姿态与 YOLO 推理拆分，最后在主线程聚合结果。
- **缓冲队列**：摄像头采集与推理解耦，超时后自动丢弃最旧帧，保持整体延迟稳定。
- **降采样与 ROI**：默认对输入帧执行短边 640 像素的缩放，再映射回原尺寸，兼顾速度与精度。

## 与文档的对应关系

- `docs/webcam_pose_detection.md`：详细说明姿态模块的脚本、关键参数与排障。
- `docs/object_protection.md`：聚焦展品保护、危险物识别与 Qt 客户端的协同流程。

阅读本章可以快速定位需要修改的模块，并理解跨模块接口如何协同。
