# 实时摄像头 33 关节点姿态检测

基于 MediaPipe Tasks 的实时人体姿态识别，提供从极简示例到多线程优化的三种实现。适用于需要快速验证或扩展到更复杂协同场景（如文物保护联动）的项目。

## 文件概览

| 文件 | 说明 |
| --- | --- |
| `webcam_pose_minimal.py` | 最小化脚本，仅包含摄像头读取与姿态绘制，便于快速验证环境。 |
| `webcam_pose_simple.py` | 面向开发的类封装版本，增加 FPS 统计与异常处理。 |
| `pose33_realtime_optimized.py` | 多线程优化版本，包含异步处理队列、性能监控等高级特性。 |
| `download_model.py` | 下载 MediaPipe `pose_landmarker_full.task` 模型的工具脚本。 |
| `test_setup.py` | 基础环境自检工具（摄像头可用性、依赖版本）。 |
| `run_webcam.bat` | Windows 批处理脚本，可交互选择不同运行模式。 |

更多目录结构说明可参考 `docs/webcam_pose_detection_structure.md`。

## 环境准备

1. 创建 Python 3.9 环境并安装依赖：
   ```bash
   pip install -r envs/webcam_pose_detection_requirements.txt
   ```
2. 下载模型：
   ```bash
   python WebcamPoseDetection/download_model.py
   ```
   默认会将模型保存至仓库根目录下的 `models/pose_landmarker_full.task`。

> **提示**：`test_setup.py` 可以验证摄像头可用性并在缺少依赖时给出提示。

## 运行方式

| 场景 | 命令 | 特点 |
| --- | --- | --- |
| 快速体验 | `python WebcamPoseDetection/webcam_pose_minimal.py` | 逻辑最少，终端输出仅提示键盘退出。 |
| 开发调试 | `python WebcamPoseDetection/webcam_pose_simple.py` | 类封装、实时 FPS、异常提示，更适合扩展。 |
| 高性能需求 | `python WebcamPoseDetection/pose33_realtime_optimized.py --webcam` | 队列 + 后台线程优化，适用对延迟敏感的场景。 |

程序启动后将打开默认摄像头，按 `q` 键即可退出。

## 关键特性

- 支持同时检测多人的 33 个关键点。
- 通过 MediaPipe VIDEO 模式降低延迟并提升稳定性。
- 可选队列/多线程优化以适配复杂场景。
- 兼容 Windows、macOS、Linux 等常见桌面系统。

## 故障排查

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| 摄像头无法打开 | 设备被占用或权限不足 | 关闭其他应用，或在 Linux 下确认 `/dev/video*` 权限。 |
| 提示模型缺失 | 未运行下载脚本 | 先执行 `download_model.py`，或将模型手动放置到 `models/` 目录。 |
| 画面卡顿 | CPU 性能不足或分辨率过高 | 调低摄像头分辨率，或改用优化版本并启用 GPU。 |
| Mediapipe 安装失败 | 操作系统或 Python 版本不兼容 | 确保使用 64 位 Python 3.9，必要时切换到官方 wheel 支持的平台。 |

## 延伸阅读

- `docs/webcam_pose_detection_structure.md`：逐文件结构说明与代码导读。
- `object_protection/integrated_safety_monitor.py`：示范如何复用姿态检测结果实现多模块联动。
