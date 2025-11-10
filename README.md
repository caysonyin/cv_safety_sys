# CV Safety System

一个面向展陈安全场景的计算机视觉平台，现已集成统一的 Web 控制台，可在浏览器中查看视频流、实时统计信息与告警提示。本仓库仍保留原有的独立子模块以便研究，但日常部署推荐使用新的 Web 方案。

## 新增特性概览

- **一键启动**：执行根目录下的 `run_system.py` 即可启动服务，自动检查并下载所需的 MediaPipe 姿态模型与 Faster R-CNN 检测权重。
- **Web 实时监控台**：在浏览器中同时查看视频流、统计面板、实时告警与历史记录，支持声音及可视化双重告警。
- **融合分析**：结合人体姿态、重点文物与危险物检测结果，自动识别靠近文物或携带危险物的人员行为。
- **模块化结构**：`safety_monitor` 目录提供清晰的模型管理、分析管线与 Web 服务划分，方便二次开发或替换模型。

## 快速上手

```bash
# 安装依赖
pip install -r requirements.txt

# 启动融合安放系统（默认调用本地摄像头）
python run_system.py --source 0

# 如需改用视频文件
python run_system.py --source path/to/video.mp4
```

运行后访问 `http://127.0.0.1:8000/`（或根据 `--host/--port` 参数自定义地址）即可使用 Web 控制台。系统首次启动会在 `models/` 目录下载并缓存权重文件。

## 目录结构

```
cv_safety_sys/
├── safety_monitor/            # 新的融合安放系统实现（模型管理、管线、Web UI）
├── WebcamPoseDetection/       # 原始的 MediaPipe 姿态检测脚本
├── object_protection/         # 历史 YOLOv7 方案（仍可参考）
├── docs/                      # 现有功能的说明文档
└── run_system.py              # 一键启动入口
```

更多实现细节可参考 `safety_monitor/` 目录下的源码与注释。旧版脚本仍然可运行，用于对比或迁移历史流程。
