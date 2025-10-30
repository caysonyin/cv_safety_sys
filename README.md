# CV Safety System / 计算机视觉安全系统

该仓库聚合了两个围绕文物安全场景构建的计算机视觉子系统：

- **WebcamPoseDetection** – 基于 MediaPipe 的实时 33 关键点人体姿态识别。
- **object_protection** – 基于 YOLOv7-tiny 的文物检测、目标跟踪与安全联动监控。

文档与环境配置文件位于 `docs/` 与 `envs/` 目录，下文给出整体指引与快速上手方式。

## 快速开始

### 1. 准备 Python 环境

仓库所有脚本均在 **Python 3.9** 下开发，并已在 **Ubuntu 22.04 + CPU** 环境完成验证。建议通过 `conda` 或 `venv` 创建隔离环境，再安装统一的依赖集合：

```bash
# 示例：使用 conda 创建环境
conda create -n cv-safety python=3.9
conda activate cv-safety

# 安装统一依赖（同时支持姿态识别与文物保护脚本）
pip install -r requirements.txt
```

`envs/example_human_det_environments.yml` 仍然提供了一个可导入的 Conda 环境描述，其内容与 `requirements.txt` 保持一致，可用于一次性安装所有运行所需依赖。

### 2. 下载模型文件

- **姿态识别模型**：运行 `WebcamPoseDetection/download_model.py` 自动下载 `pose_landmarker_full.task` 到仓库根目录的 `models/` 文件夹。
- **YOLOv7 权重**：`object_protection/video_relic_tracking.py` 在首次运行时会尝试下载 `yolov7-tiny.pt`。如果无法联网，请手动放置文件并更新脚本中的路径。

### 3. 运行示例脚本

```bash
# 启动最简姿态识别（默认摄像头）
python WebcamPoseDetection/webcam_pose_minimal.py

# 启动带 FPS 统计的姿态识别
python WebcamPoseDetection/webcam_pose_simple.py

# 启动文物检测与跟踪（摄像头）
python object_protection/video_relic_tracking.py --source 0

# 启动文物安全联动监控（摄像头）
python object_protection/integrated_safety_monitor.py --source 0
```

详细参数说明请查阅各模块文档。

> **提示**：所有实时脚本均支持 `--source`（摄像头索引或视频文件）参数，可在同一环境下为两个子系统指定不同输入源，实现可视化的同时运行。

## 目录概览

```
cv_safety_sys/
├── WebcamPoseDetection/       # 姿态识别子系统
│   ├── download_model.py
│   ├── pose33_realtime_optimized.py
│   ├── webcam_pose_minimal.py
│   └── webcam_pose_simple.py
├── object_protection/         # 文物检测与安全联动
│   ├── integrated_safety_monitor.py
│   ├── video_relic_tracking.py
│   ├── general.py
│   └── yolov7-tiny.pt (可选，首次运行会下载)
├── docs/                      # 详细说明文档
└── envs/                      # 依赖与环境配置文件
```

## 文档与使用说明

- `docs/webcam_pose_detection.md` – 姿态识别功能简介、安装与运行说明。
- `docs/webcam_pose_detection_structure.md` – 姿态识别目录结构与代码导读。
- `docs/object_protection.md` – 文物检测、跟踪及安全联动功能说明。
- `docs/exam_doc.md` – 系统整体概览与演示 checklist。

欢迎在对应模块的文档中查看更多操作细节与故障排查建议。

## 环境与依赖

- `requirements.txt` – 统一的依赖列表，覆盖姿态识别与文物保护两个子系统。
- `envs/example_human_det_environments.yml` – Conda 环境描述，使用 `requirements.txt` 中的同一套依赖。

若需新增依赖，请同步更新上述文件并在文档中注明用途。

## 贡献指南

1. 为每个子模块使用独立分支进行开发，避免直接推送到 `main`。
2. 新增或修改脚本时，请在 `docs/` 中补充对应使用说明。
3. 不要向仓库提交体积较大的模型权重或中间结果，可将其加入 `.gitignore`。
4. 在提交前运行相关脚本或测试，确保核心功能可用。

祝你探索顺利，也欢迎补充更多安全场景的子系统。
