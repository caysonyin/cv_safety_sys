# 实时摄像头33关节姿态检测

## 功能描述
基于MediaPipe的实时摄像头33关节人体姿态检测系统，支持实时显示和性能监控。

## 文件说明

### 核心文件
- `webcam_pose_minimal.py` - 最简化版本，只包含核心功能
- `webcam_pose_simple.py` - 简化版本，带类封装和性能监控
- `pose33_realtime_optimized.py` - 完整优化版本，支持高级功能

### 依赖文件
- `requirements.txt` - Python依赖包列表
- `models/pose_landmarker_full.task` - MediaPipe姿态检测模型（需要下载）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 最简单的方式（推荐）
```bash
python webcam_pose_minimal.py
```

### 2. 带性能监控
```bash
python webcam_pose_simple.py
```

### 3. 完整功能版本
```bash
python pose33_realtime_optimized.py --webcam
```

## 功能特点

- ✅ 实时摄像头输入
- ✅ 33关节人体姿态检测
- ✅ 实时显示检测结果
- ✅ 性能监控（FPS显示）
- ✅ 低延时处理
- ✅ 简单易用

## 操作说明

1. 运行程序后会自动打开摄像头
2. 在摄像头前做动作，系统会实时检测并显示33个关节
3. 按 'q' 键退出程序

## 系统要求

- Python 3.7+
- 摄像头设备
- 支持的操作系统：Windows, macOS, Linux

## 模型下载

首次运行时会自动下载MediaPipe模型文件到 `models/` 目录。

## 性能优化

- 使用VIDEO模式获得最佳性能
- 智能图像缩放减少处理时间
- 优化的绘制算法提高显示速度
- 多线程处理支持（完整版本）

## 故障排除

1. **摄像头无法打开**：检查摄像头是否被其他程序占用
2. **模型下载失败**：检查网络连接，手动下载模型文件
3. **性能问题**：降低摄像头分辨率或使用简化版本

## 技术架构

- **OpenCV**: 摄像头捕获和图像处理
- **MediaPipe**: 姿态检测模型
- **NumPy**: 数值计算
- **多线程**: 异步处理（完整版本）

## 更新日志

- v1.0: 基础实时姿态检测功能
- v1.1: 添加性能监控
- v1.2: 优化处理速度和稳定性
