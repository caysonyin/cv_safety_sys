# 文物保护与安全联动

本文聚焦项目中的“文物防护 + 人员姿态 + 危险物品”联动逻辑，核心由 `IntegratedSafetyMonitor` 统一编排。

## 关键模块

```text
src/cv_safety_sys/
├── detection/yolov7_tracker.py      # 检测、跟踪、文物选择
├── monitoring/integrated_monitor.py # 联动策略与报警决策
└── ui/qt_monitor.py                 # Qt 界面与交互
```

## 联动流程

1. **检测阶段**：YOLOv7-tiny 检测文物、人员和危险物体。
2. **跟踪阶段**：为检测结果分配 `track_id`，维持跨帧一致性。
3. **文物保护区**：对被标记为保护对象的文物框扩展出“电子围栏”。
4. **姿态关联**：将人体 33 个关键点与人员框关联，判断是否侵入围栏。
5. **危险关联**：将危险物体绑定到最近人员，并提升报警等级。
6. **告警输出**：生成结构化报警消息，供 UI 列表与视频叠层展示。

## 默认类别

- 受保护文物：`cup`
- 危险物品：`knife`、`scissors`、`baseball bat`

> 若需要扩展危险类别，可在 `integrated_monitor.py` 的 `DANGEROUS_CLASSES` 中增改。

## 常用命令

```bash
# 一键运行（推荐）
python run.py --source 0

# 调试联动策略（OpenCV窗口）
PYTHONPATH=src python -m cv_safety_sys.monitoring.integrated_monitor --source 0 --conf 0.25

# 仅调试检测与跟踪
PYTHONPATH=src python -m cv_safety_sys.detection.yolov7_tracker --source 0 --conf 0.1
```

## 交互说明（Qt）

- 在视频区点击目标可触发文物选择流程（具体提示以界面状态栏/Toast 为准）。
- 报警面板会展示当前事件摘要，点击列表项可定位对应目标。
- 可通过 `--alert-sound` 指定本地音频文件作为报警音。

