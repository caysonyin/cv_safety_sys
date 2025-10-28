# 项目结构说明

## 文件夹结构
```
WebcamPoseDetection/
├── webcam_pose_minimal.py       # 最简化版本（推荐使用）
├── webcam_pose_simple.py        # 简化版本（带性能监控）
├── pose33_realtime_optimized.py # 完整优化版本
├── download_model.py            # 模型下载脚本
├── test_setup.py               # 环境测试脚本
├── run_webcam.bat              # Windows启动脚本
├── .gitignore                  # Git忽略文件
└── ...

envs/
└── webcam_pose_detection_requirements.txt  # Python依赖包

docs/
├── webcam_pose_detection.md               # 项目说明文档
└── webcam_pose_detection_structure.md     # 本文件
```

## 文件功能说明

### 核心程序文件
1. **webcam_pose_minimal.py** (推荐)
   - 最简化的实现
   - 只包含核心功能
   - 代码量最少，易于理解
   - 适合学习和快速使用

2. **webcam_pose_simple.py**
   - 带类封装的版本
   - 包含性能监控
   - 代码结构更清晰
   - 适合进一步开发

3. **pose33_realtime_optimized.py**
   - 完整功能版本
   - 支持多线程处理
   - 包含高级优化
   - 适合生产环境

### 辅助文件
4. **download_model.py**
   - 自动下载MediaPipe模型
   - 首次运行前执行
   - 处理网络下载问题

5. **test_setup.py**
   - 环境依赖检查
   - 自动安装缺失包
   - 摄像头可用性测试

6. **run_webcam.bat**
   - Windows批处理启动脚本
   - 提供选择菜单
   - 简化使用流程

### 配置文件与文档
7. **envs/webcam_pose_detection_requirements.txt**
   - Python依赖包列表
   - 版本要求说明
   - 便于环境复现

8. **docs/webcam_pose_detection.md**
   - 详细使用说明
   - 功能特点介绍
   - 故障排除指南

9. **.gitignore**
   - Git版本控制忽略文件
   - 排除临时文件和模型文件
   - 保持仓库整洁

## 使用流程

### 首次使用
1. 运行 `test_setup.py` 检查环境
2. 运行 `download_model.py` 下载模型
3. 双击 `run_webcam.bat` 启动程序

### 日常使用
- 直接运行 `webcam_pose_minimal.py`
- 或使用 `run_webcam.bat` 选择版本

### 开发使用
- 参考 `webcam_pose_simple.py` 进行扩展
- 使用 `pose33_realtime_optimized.py` 作为高级版本

## 技术特点

### 简化版本特点
- 单文件实现
- 最少依赖
- 易于理解和修改
- 快速启动

### 优化版本特点
- 多线程处理
- 性能监控
- 错误处理
- 资源管理

## 部署说明

### 本地部署
- 直接运行Python文件
- 使用批处理脚本启动

### 服务器部署
- 安装依赖包
- 配置摄像头权限
- 使用完整版本

### Git部署
- 克隆仓库
- 安装依赖
- 下载模型
- 运行程序
