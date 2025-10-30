# 文物检测、跟踪与安全联动系统

该子系统以 YOLOv7-tiny 为基础，实现了从单纯的文物检测、目标跟踪到联合姿态识别的安全预警流程。

## 目录概览

```
object_protection/
├── video_relic_tracking.py       # YOLOv7 文物检测 + 交互式跟踪
├── integrated_safety_monitor.py  # 文物检测 × 人体姿态 × 危险物联动
├── general.py                    # YOLO/YOLOR 通用工具函数（精简版）
├── yolov7-tiny.pt                # 预训练权重（首次运行可自动下载）
└── yolov7/                       # 官方 YOLOv7 仓库（需单独克隆或下载）
```

## 环境准备

1. 创建 Python 3.9 虚拟环境。
2. 安装统一依赖（同时覆盖检测、姿态与联动功能）：
   ```bash
   pip install -r requirements.txt
   ```
3. 根据网络情况准备权重：
   - 首次运行 `video_relic_tracking.py` 或 `integrated_safety_monitor.py` 时会尝试下载 `yolov7-tiny.pt` 至当前目录。
   - 若下载失败，可从官方渠道获取后放置在 `object_protection/` 目录，并确保脚本中的路径正确。

## 核心脚本

### `video_relic_tracking.py`
- 通过 YOLOv7-tiny 进行实时检测，支持摄像头和视频文件输入。
- 鼠标点击即可选中/取消文物目标，按 `Enter` 确认，`ESC` 退出，`S` 保存当前帧。
- 内置基于质心距离的 `SimpleTracker`，可保持目标 ID。
- 提供电子围栏判定与文物重要性评分示例，便于扩展到安全策略。

常用命令：
```bash
# 默认摄像头
python object_protection/video_relic_tracking.py --source 0

# 指定视频文件 + 自定义置信度阈值
python object_protection/video_relic_tracking.py --source museum.mp4 --conf 0.25
```

### `integrated_safety_monitor.py`
- 在文物检测基础上引入人体姿态识别与危险物品检测。
- 通过 `PoseLandmarkHelper` 复用 MediaPipe 模型，判断人员是否靠近文物或持有危险物体。
- 支持告警历史记录、危险区域标记以及多目标跟踪。
- 适合作为文物安全演示或进一步产品化的参考实现。

运行前需确保已下载姿态模型（参见 `WebcamPoseDetection/download_model.py`）。

## 常见问题

| 问题 | 处理建议 |
| --- | --- |
| GPU 推理报错 | 确认 CUDA 与 PyTorch 版本匹配，必要时改用 CPU。 |
| `yolov7` 模块找不到 | 在仓库根目录克隆 `https://github.com/WongKinYiu/yolov7`，或将其加入 `PYTHONPATH`。 |
| 帧率过低 | 调整 `--img` 参数降低输入分辨率，或减少电子围栏数量。 |
| Mediapipe 未安装 | 若运行联动监控，请使用完整依赖文件重新安装。 |

## 延伸阅读

- `docs/webcam_pose_detection.md`：姿态识别模块说明。
- `README.md`：仓库顶层概览与快速上手。
