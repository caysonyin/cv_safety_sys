#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载MediaPipe姿态检测模型
"""

import os
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

def download_model():
    """下载模型文件"""
    print("开始下载MediaPipe姿态检测模型...")
    
    # 创建models目录
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "pose_landmarker_full.task"
    
    if model_path.exists():
        print(f"模型文件已存在: {model_path}")
        return str(model_path)
    
    try:
        print("正在下载模型文件，请稍候...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"模型下载完成: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"下载失败: {e}")
        return None

if __name__ == "__main__":
    model_path = download_model()
    if model_path:
        print("模型准备就绪，可以运行姿态检测程序了！")
    else:
        print("模型下载失败，请检查网络连接")
