# 展品保护与安全联动

该子系统在单路视频流上串联 **YOLOv7-tiny 目标检测 → 质心跟踪 → MediaPipe 姿态推理 → 安全策略**，用于实时守护受保护展品（默认 `cup`）并识别危险物品（默认 `tennis racket`）。下文说明各核心脚本的职责、调用方式与关键技术细节。

## 目录速览

```
src/cv_safety_sys/
├── detection/yolov7_tracker.py      # YOLOv7 检测 + 质心跟踪 + 鼠标交互
├── monitoring/integrated_monitor.py # 展品围栏、危险绑定与告警决策
└── ui/qt_monitor.py                 # PySide6 前端与监控编排器
object_protection/
├── qt_monitor_app.py                # 桌面端入口（加载 UI + Monitor）
└── integrated_safety_monitor.py     # CLI 入口，复用同一 Monitor
```

## 运行方式

```bash
# 桌面端（含 PySide6 UI、姿态与告警逻辑）
python run.py --source 0

# 仅运行检测/追踪逻辑用于调试
python -m cv_safety_sys.detection.yolov7_tracker --source 0
```

常用参数：

- `--source`：摄像头索引或视频文件路径，Qt 版本与 CLI 版本一致。
- `--conf`：YOLO 置信度阈值，GUI 默认 `0.25`，CLI 默认 `0.1` 以方便调试。
- `--pose-model` / `--yolo-model`：覆盖默认模型路径（位于 `models/`）。
- `--alert-sound`：Qt 客户端可选的本地告警音。

首次运行会自动下载缺失的模型，若机器无法联网，可手动把模型放入 `models/`。

## 技术流程

1. **检测阶段**：`YOLOv7TinyDetector` 在 640×640 输入上推理，筛选 `{cup, person, tennis racket}`，其余类别直接丢弃。
2. **跟踪阶段**：`SimpleTracker` 采用质心距离与 IoU 混合策略，为 `cup` 分配稳定的 `track_id`，并通过鼠标事件将“受保护展品”状态写入该 ID。
3. **围栏生成**：对被标记的展品框进行 `expand_ratio=1.15` 的矩形扩展，得到安全围栏，并实时显示剩余安全距离。
4. **姿态关联**：`PoseBridge` 将 MediaPipe 33 关键点投影到原始帧，并与安全围栏或危险物体计算最短距离，用于判断“侵入”与“危险携带”。
5. **告警策略**：
   - **侵入告警**：任意人体关键点落入围栏，触发黄色或红色高亮，强度与侵入深度相关。
   - **危险携带**：检测到 `tennis racket` 时，`HazardBinder` 搜索最近的人体并展示绑定提示。
   - 所有事件会在 Qt UI 顶层滚动显示，并可触发声音提示。

## 关键实现细节

- **统一时间轴**：检测、姿态与跟踪共享同一帧编号，UI 只渲染最新结果，避免多线程乱序。
- **自适应缩放**：YOLO 推理在缩放后的方形帧上执行，推理结束后再映射回原分辨率，保证与姿态坐标对齐。
- **交互协议**：鼠标左键选中检测框，`Enter` 确认保护列表，`Backspace` 清空；这些事件由 UI 透传给 `SimpleTracker`，因此 CLI 与 GUI 行为一致。
- **危险类别配置**：`cv_safety_sys.monitoring.constants.DANGEROUS_CLASSES` 集中管理危险物品；如需扩展，只需改动该常量并刷新告警文案。
- **性能监控**：Qt 客户端在状态栏显示检测与姿态耗时，帮助确认 CPU/GPU 负载。长时间阻塞会触发“丢帧”提示。

## 与其他模块的耦合

- 姿态模块通过 `cv_safety_sys.pose` 暴露的 `download_model()`、`create_pose_detector()` 接口使用，不直接依赖 MediaPipe API，便于替换实现。
- PySide6 UI 只关心 `IntegratedSafetyMonitor` 暴露的结构化结果（展品列表、告警列表、关节点坐标），因此可以在保持 UI 稳定的前提下替换底层模型。

阅读本文件可搭建或扩展展品保护、危险感知逻辑；若需了解姿态模块细节，请参考 `docs/webcam_pose_detection.md`。
