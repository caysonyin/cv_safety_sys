# CV Safety System

面向展区安全的计算机视觉一体化方案，提供 **姿态感知示例（`examples/pose/`）** 与 **安全监控主干（`src/cv_safety_sys/`）** 两个可独立运行的子系统。当前默认场景将 `cup` 视为受保护文物，`tennis racket` 视为危险物品，所有 UI 标签与告警策略均为该配置调优。

## 技术栈

- **语言与运行环境**：Python 3.10，Ubuntu 22.04 + CPU 基线，可扩展到 GPU。
- **计算机视觉**：
  - `YOLOv7-tiny`（PyTorch）：检测 `cup/person/tennis racket` 并输出置信度。
  - `MediaPipe Tasks Pose`：33 关键点姿态推理，含模型自动下载器。
  - `OpenCV`：摄像头采集、图像预处理与可视化。
- **跟踪与策略**：
  - 自研 `SimpleTracker`（质心 + IoU 混合策略）和 `CupFence`/`HazardBinder` 安全逻辑。
  - `NumPy`/`SciPy`（可选）用于距离计算与向量化操作。
- **桌面客户端**：`PySide6` 绘制实时叠层、表单和告警列表；Qt 事件与检测器共享统一接口。
- **工程工具**：`requirements.txt` 管理依赖，`run.py` 统一调度、下载模型并暴露命令行参数。

## 功能亮点

- YOLOv7-tiny + 质心跟踪，支持鼠标交互式选择受保护展品。
- 根据选定展品自动生成安全围栏，并结合 MediaPipe 姿态判断人体侵入情况。
- 危险物绑定逻辑可将 `tennis racket` 关联到最近人员并触发告警。
- 轻量依赖，单机 CPU 可实时运行；可在 CLI 或 PySide6 UI 中任意组合。

## 快速上手

```bash
# 安装依赖
pip install -r requirements.txt

# （首次）拉取 YOLOv7 推理所需的官方代码
git clone --depth 1 https://github.com/WongKinYiu/yolov7.git

# 启动完整桌面端，自动下载缺失模型到 ./models
python run.py --source 0

# 以 CLI 方式运行安全联动（展品 + 危险物）
python object_protection/integrated_safety_monitor.py --source 0

# 启动 Qt 客户端并指定自定义告警音
python object_protection/qt_monitor_app.py --source 0 --alert-sound path/to/sound.wav
```

`run.py` 会在首次启动时自动下载 MediaPipe 姿态模型与 YOLOv7-tiny 权重到 `models/`。若需要自定义权重，可直接放入该目录并通过 `--yolo-model` 或 `--pose-model` 覆盖路径。

## 中文字体渲染

视频叠层使用支持 Unicode 的文字渲染器，会自动搜索常用中文字体（STHeiti、Microsoft YaHei、WenQuanYi、Noto Sans CJK 等）。如字体安装在其他路径，可在运行前设置：

```bash
export CV_SAFETY_FONT=/absolute/path/to/font.ttf
python run.py --source 0
```

同样适用于 `object_protection/qt_monitor_app.py`。

## 目录结构

```
cv_safety_sys/
├── models/                    # 姿态/检测模型缓存（首次运行自动创建）
├── run.py                     # PySide6 客户端统一入口
├── src/cv_safety_sys/         # 可复用的核心 Python 包
│   ├── detection/             # YOLOv7 检测与跟踪模块
│   ├── monitoring/            # 安全集成逻辑
│   ├── pose/                  # MediaPipe 封装与模型下载器
│   └── ui/                    # PySide6 应用与渲染
├── examples/pose/             # 姿态推理示例脚本
└── docs/                      # 技术文档（架构 / 姿态 / 展品保护）
```

详细技术说明请查阅 `docs/system_architecture.md`、`docs/webcam_pose_detection.md` 与 `docs/object_protection.md`。
