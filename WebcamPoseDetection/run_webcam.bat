@echo off
echo 启动实时摄像头姿态检测...
echo.
echo 选择运行模式:
echo 1. 最简化版本 (推荐)
echo 2. 带性能监控版本
echo 3. 完整优化版本
echo.
set /p choice=请输入选择 (1-3): 

if "%choice%"=="1" (
    echo 启动最简化版本...
    python webcam_pose_minimal.py
) else if "%choice%"=="2" (
    echo 启动带性能监控版本...
    python webcam_pose_simple.py
) else if "%choice%"=="3" (
    echo 启动完整优化版本...
    python pose33_realtime_optimized.py --webcam
) else (
    echo 无效选择，启动默认版本...
    python webcam_pose_minimal.py
)

pause
