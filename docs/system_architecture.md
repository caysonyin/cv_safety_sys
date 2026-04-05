# 系统总体架构

本项目围绕“受保护文物 + 人体姿态 + 危险物品”三类信息做实时联动监控。运行主链路可以概括为：

**视频输入 → 目标检测/跟踪 → 姿态识别 → 安全策略融合 → UI/告警输出**。

## 模块与职责

| 模块 | 关键文件 | 职责 |
| --- | --- | --- |
| 启动与资源校验 | `run.py` | 校验 `yolov7/` 仓库、准备模型路径并启动 Qt 客户端。 |
| 检测与跟踪 | `src/cv_safety_sys/detection/yolov7_tracker.py` | YOLOv7-tiny 检测、目标筛选、`SimpleTracker` 跟踪与文物选择交互。 |
| 姿态模型管理 | `src/cv_safety_sys/pose/model_downloader.py` | 下载并缓存 MediaPipe Pose Landmarker 模型。 |
| 安全策略融合 | `src/cv_safety_sys/monitoring/integrated_monitor.py` | 融合人/文物/危险物与姿态点，输出围栏、报警和统计信息。 |
| 可视化与交互 | `src/cv_safety_sys/ui/qt_monitor.py` | PySide6 客户端、视频显示、报警列表、状态面板、鼠标/键盘交互。 |

## 数据流

1. **视频采集**：OpenCV 从摄像头或文件读取帧。
2. **检测与分类**：YOLOv7 输出边界框与类别，筛出 `cup/person` 及危险类别。
3. **目标跟踪**：`SimpleTracker` 为目标分配稳定 `track_id`，用于跨帧关联。
4. **姿态推理**：MediaPipe Pose 输出人体关键点，并与 person 检测框做 IoU 关联。
5. **策略计算**：
   - 对选中文物生成防护围栏。
   - 判断人体关键点是否侵入围栏。
   - 对危险物与最近人员进行关联并触发更高等级告警。
6. **结果输出**：将结构化状态同步到 OpenCV/Qt 显示层，并更新报警列表与统计数据。

## 运行入口

- `python run.py --source 0`：推荐，一键启动桌面端。
- `PYTHONPATH=src python -m cv_safety_sys.monitoring.integrated_monitor --source 0`：OpenCV 窗口版本。
- `PYTHONPATH=src python -m cv_safety_sys.ui.qt_monitor --source 0`：直接运行 Qt 模块。

## 模型与依赖

- YOLO 权重默认路径：`models/yolov7-tiny.pt`
- 姿态模型默认路径：`models/pose_landmarker_full.task`
- YOLO 推理代码目录：仓库根目录下 `yolov7/`（需手动 `git clone`）

