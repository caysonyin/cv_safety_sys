#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简化的实时摄像头 33 关节姿态检测。"""

import argparse
import os
import time
from typing import Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import Image as MPImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class SimpleWebcamPoseDetector:
    """简化的实时摄像头姿态检测器"""
    
    def __init__(self, model_path: str = "models/pose_landmarker_full.task"):
        self.model_path = model_path
        self.timestamp_ms = 0
        
        # 初始化MediaPipe
        self._initialize_mediapipe()
        
    def _initialize_mediapipe(self):
        """初始化MediaPipe模型"""
        if not os.path.exists(self.model_path):
            print(f"模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            num_poses=5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        print("MediaPipe模型初始化完成")
    
    def _draw_pose(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """绘制姿态"""
        if not landmarks:
            return frame
        
        annotated = frame.copy()
        h, w = frame.shape[:2]
        connections = mp.solutions.pose.POSE_CONNECTIONS
        
        for lm_list in landmarks:
            # 转换坐标
            points = []
            for lm in lm_list:
                x = int(lm.x * w)
                y = int(lm.y * h)
                if 0 <= x < w and 0 <= y < h:
                    points.append((x, y))
                else:
                    points.append((-1, -1))
            
            # 绘制连接线
            for a, b in connections:
                if 0 <= a < len(points) and 0 <= b < len(points):
                    pa, pb = points[a], points[b]
                    if pa[0] >= 0 and pb[0] >= 0:
                        cv2.line(annotated, pa, pb, (0, 255, 0), 2)
            
            # 绘制关节点
            for x, y in points:
                if x >= 0:
                    cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)
        
        return annotated
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        # 转换颜色空间
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        
        # 检测姿态
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        self.timestamp_ms += 1
        
        # 绘制结果
        annotated = self._draw_pose(frame, result.pose_landmarks)
        
        return annotated


def run_webcam_pose(video_source: Union[int, str] = 0, window_name: str = "实时姿态检测"):
    """运行实时摄像头姿态检测"""
    print("启动实时摄像头姿态检测...")
    print("按 'q' 键退出")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"无法打开视频源: {video_source}")
        return

    # 设置摄像头参数
    if isinstance(video_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 创建检测器
    try:
        detector = SimpleWebcamPoseDetector()
    except FileNotFoundError:
        print("请确保模型文件存在: models/pose_landmarker_full.task")
        cap.release()
        return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 处理帧
            annotated_frame = detector.process_frame(frame)
            
            # 添加FPS信息
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow(window_name, annotated_frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
    
    # 显示统计信息
    total_time = time.time() - start_time
    print(f"\n处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {total_time:.2f}秒")
    if total_time > 0:
        print(f"平均FPS: {frame_count / total_time:.2f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时姿态检测示例")
    parser.add_argument(
        "--source",
        default="0",
        help="视频源 (摄像头索引或视频文件路径)",
    )
    parser.add_argument(
        "--window-name",
        default="实时姿态检测",
        help="OpenCV 显示窗口名称",
    )
    return parser.parse_args()


def _coerce_source(value: str) -> Union[int, str]:
    value = value.strip()
    return int(value) if value.lstrip("-").isdigit() else value


if __name__ == "__main__":
    args = _parse_args()
    run_webcam_pose(
        video_source=_coerce_source(str(args.source)),
        window_name=args.window_name,
    )
