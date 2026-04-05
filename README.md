# CV Safety System

面向展区安全场景的计算机视觉监控系统：在同一路视频中联动 **文物检测/跟踪、人体姿态识别、危险物品识别与报警可视化**。

> 当前仓库代码以 `cup` 作为默认受保护文物类别，并将 `knife`、`scissors`、`baseball bat` 视为危险物品类别（可在代码中扩展）。

## 功能概览

- YOLOv7-tiny 检测与简易多目标跟踪（含文物选择与围栏逻辑）
- MediaPipe Pose 33 关键点识别
- 文物防护区入侵检测 + 危险物品/人员关联报警
- PySide6 实时桌面监控界面（报警列表、状态面板、视频叠层）
- 首次运行自动下载姿态模型与 YOLO 权重

## 环境要求

- Python 3.10+
- Linux / macOS / Windows（建议使用具备摄像头访问权限的环境）
- 可访问互联网（首次自动下载模型时需要）

## 安装

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 拉取 YOLOv7 官方仓库（当前实现会从仓库目录动态加载推理代码）
git clone --depth 1 https://github.com/WongKinYiu/yolov7.git
```

## 快速启动

```bash
# 一键启动 PySide6 客户端（推荐）
python run.py --source 0
```

可选参数：

- `--source`：摄像头索引（如 `0`）或视频文件路径
- `--conf`：YOLO 置信度阈值（默认 `0.25`）
- `--pose-model`：姿态模型路径（默认 `models/pose_landmarker_full.task`）
- `--yolo-model`：YOLO 权重路径（默认 `models/yolov7-tiny.pt`）
- `--alert-sound`：自定义报警音文件

## 其他运行方式

```bash
# 仅运行协同监控逻辑（OpenCV 窗口）
PYTHONPATH=src python -m cv_safety_sys.monitoring.integrated_monitor --source 0

# 直接启动 Qt 客户端模块
PYTHONPATH=src python -m cv_safety_sys.ui.qt_monitor --source 0

# 仅运行文物检测/跟踪调试
PYTHONPATH=src python -m cv_safety_sys.detection.yolov7_tracker --source 0
```

## 项目结构

```text
cv_safety_sys/
├── run.py
├── requirements.txt
├── src/cv_safety_sys/
│   ├── detection/           # YOLOv7 检测与跟踪
│   ├── monitoring/          # 文物+姿态+危险物品联动策略
│   ├── pose/                # MediaPipe 姿态模型下载
│   ├── ui/                  # PySide6 客户端
│   └── utils/               # 文本渲染等工具
└── docs/                    # 架构与模块文档
```

## 文档

- 系统架构：`docs/system_architecture.md`
- 文物保护联动：`docs/object_protection.md`
- 姿态模块说明：`docs/webcam_pose_detection.md`

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
You may copy, modify, and redistribute this project under the terms of GPL-3.0.
See the [LICENSE](./LICENSE) file for details.

## 开源说明

本项目采用 GPL-3.0 开源许可证发布。
这意味着你可以在 GPL-3.0 条件下使用、修改和再分发本项目。
若你分发修改后的版本，通常也需要按 GPL-3.0 要求继续提供相应源码。
详细条款请见仓库根目录的 [LICENSE](./LICENSE)。

## Copyright

Copyright (C) 2026 Cayson
