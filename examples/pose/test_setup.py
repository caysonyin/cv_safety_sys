#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试环境设置和依赖检查
"""

import sys
import subprocess
import importlib

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python版本过低，需要Python 3.7+")
        return False
    else:
        print("✅ Python版本符合要求")
        return True

def check_dependencies():
    """检查依赖包"""
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            importlib.import_module(module)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """安装缺失的依赖包"""
    if not packages:
        return True
    
    print(f"\n需要安装以下包: {', '.join(packages)}")
    print("正在安装...")
    
    try:
        for package in packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False

def test_camera():
    """测试摄像头"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ 摄像头可用")
            cap.release()
            return True
        else:
            print("❌ 摄像头不可用")
            return False
    except Exception as e:
        print(f"❌ 摄像头测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 实时摄像头姿态检测环境检查 ===\n")
    
    # 检查Python版本
    python_ok = check_python_version()
    print()
    
    # 检查依赖包
    missing = check_dependencies()
    print()
    
    # 安装缺失的包
    if missing:
        install_ok = install_dependencies(missing)
        if not install_ok:
            print("❌ 依赖安装失败，请手动安装")
            return False
        print()
    
    # 测试摄像头
    camera_ok = test_camera()
    print()
    
    # 总结
    if python_ok and not missing and camera_ok:
        print("🎉 环境检查通过！可以运行姿态检测程序了")
        print("\n运行方式:")
        print("1. 运行: python webcam_pose_minimal.py")
        print("2. 或运行: python webcam_pose_simple.py")
        return True
    else:
        print("⚠️ 环境检查未完全通过，请解决上述问题")
        return False

if __name__ == "__main__":
    success = main()
    input("\n按回车键退出...")
