# WebcamPoseDetection 目录结构

```
WebcamPoseDetection/
├── download_model.py             # 下载 MediaPipe 姿态模型
├── pose33_realtime_optimized.py  # 多线程 + 队列优化的实时检测器
├── webcam_pose_minimal.py        # 极简示例脚本
├── webcam_pose_simple.py         # 面向扩展的类封装版本
├── run_webcam.bat                # Windows 启动脚本（菜单式）
├── test_setup.py                 # 运行环境自检脚本
└── __init__.py (若存在则用于包化)
```

## 代码导读

### `webcam_pose_minimal.py`
- 直接在 `main()` 中完成摄像头采集、MediaPipe 推理与关键点绘制。
- 适合作为快速检查依赖是否安装成功的入口。

### `webcam_pose_simple.py`
- 定义 `SimpleWebcamPoseDetector` 类，封装模型初始化与帧处理。
- 支持 FPS 统计与基本异常信息输出。
- 是二次开发或与其他模块集成时的推荐基线。

### `pose33_realtime_optimized.py`
- 提供 `RealtimePoseDetector` 类，使用线程安全队列拆分采集与推理流程。
- 内置帧尺寸自适应、性能统计与丢帧策略，适用于延迟敏感的场景。
- 通过命令行参数选择输入源和运行模式，便于部署。

### `download_model.py`
- 下载官方 MediaPipe `pose_landmarker_full` 模型至仓库根目录的 `models/` 目录。
- 在其他脚本调用前执行一次即可。

### `test_setup.py`
- 检查必要依赖是否可用并测试摄像头打开情况。
- 适合在部署前进行快速健康检查。

## 与其他子系统的关系

- `object_protection/integrated_safety_monitor.py` 通过导入 `download_model.py` 获取模型路径，并在安全联动场景中复用姿态检测能力。
- 环境依赖与其他子系统统一在仓库根目录的 `requirements.txt` 中维护。

了解以上结构，有助于快速定位需要修改或复用的脚本位置。
