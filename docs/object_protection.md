# 文物检测与跟踪系统

基于YOLOv7-tiny的实时文物检测、选择和跟踪系统。

## 📁 项目结构

```
object_protection/
├── video_relic_tracking.py       # 主程序：视频文物跟踪
├── yolov7/                       # YOLOv7框架（需单独下载）
└── ...

envs/
├── object_protection_requirements.txt          # 完整依赖包
└── object_protection_requirements_minimal.txt  # 最小依赖包
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 完整版本（推荐）
pip install -r envs/object_protection_requirements.txt

# 或最小版本
pip install -r envs/object_protection_requirements_minimal.txt
```

### 2. 运行程序

```bash
# 使用摄像头（默认）
python video_relic_tracking.py --source 0

# 使用视频文件
python video_relic_tracking.py --source "video.mp4"

# 调整检测阈值
python video_relic_tracking.py --source 0 --conf 0.1
```

## 🎯 功能特性

- ✅ 实时文物检测
- ✅ 交互式文物选择
- ✅ 目标跟踪（保持ID）
- ✅ 电子栅栏保护
- ✅ 支持摄像头和视频文件

## 🎮 操作说明

1. **点击红色框** - 选择文物
2. **点击绿色框** - 取消选择
3. **按Enter键** - 确认选择
4. **按ESC键** - 退出程序
5. **按S键** - 保存当前帧

## 📋 系统要求

- Python 3.7+
- CUDA支持（可选，用于GPU加速）
- 摄像头或视频文件

## 🔧 依赖包说明

### 核心依赖
- `torch` - PyTorch深度学习框架
- `opencv-python` - 计算机视觉处理
- `numpy` - 数值计算
- `requests` - 网络请求

### 可选依赖
- `matplotlib` - 可视化
- `tqdm` - 进度条
- `psutil` - 系统监控
