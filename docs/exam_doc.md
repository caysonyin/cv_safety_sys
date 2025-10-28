# 系统演示速查表

该文档用于快速确认环境是否准备就绪，以及在演示或验收时需要验证的关键功能。

## 环境检查

1. **Python 版本**：确保运行 `python --version` 输出为 3.9.x。
2. **依赖安装**：
   - 姿态识别：`pip list | grep mediapipe`、`pip list | grep opencv-python`。
   - 文物安全联动：额外确认 `torch`、`torchvision` 与 `mediapipe` 版本匹配。
3. **模型文件**：
   - `models/pose_landmarker_full.task` 存在。
   - `object_protection/yolov7-tiny.pt` 已下载或可联网获取。

## 功能验证

### 1. 姿态识别
- 运行 `python WebcamPoseDetection/webcam_pose_simple.py`。
- 确认窗口中显示人体骨架，并且左上角 FPS 正常更新。

### 2. 文物检测与跟踪
- 运行 `python object_protection/video_relic_tracking.py --source 0`。
- 鼠标点击某个检测框将其标记为受保护文物；按 `Enter` 固定选择。
- 拖动物体穿过电子围栏，观察告警提示是否触发。

### 3. 安全联动监控
- 确认姿态模型已下载后，运行：
  ```bash
  python object_protection/integrated_safety_monitor.py --source 0
  ```
- 验证以下交互：
  - 人员靠近文物时的距离提示；
  - 检测到危险物品（如刀具）时的告警弹窗；
  - 告警历史是否在界面上滚动显示。

## 故障排除记录

| 时间 | 模块 | 问题 | 处理方案 |
| --- | --- | --- | --- |
| | | | |

可在每次调试或演示后补充以上表格，方便团队共享经验。
