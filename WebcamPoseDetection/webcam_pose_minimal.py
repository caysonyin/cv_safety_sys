#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最简化的实时摄像头33关节姿态检测
只包含最核心的功能
"""

import cv2
import mediapipe as mp
from mediapipe import Image as MPImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def main():
    """主函数 - 最简化的实时摄像头姿态检测"""
    print("启动实时摄像头姿态检测...")
    print("按 'q' 键退出")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 初始化MediaPipe
    model_path = "models/pose_landmarker_full.task"
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    
    timestamp_ms = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换颜色空间
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
            
            # 检测姿态
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 1
            
            # 绘制姿态
            if result.pose_landmarks:
                annotated = frame.copy()
                h, w = frame.shape[:2]
                connections = mp.solutions.pose.POSE_CONNECTIONS
                
                for lm_list in result.pose_landmarks:
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
                
                cv2.imshow('实时姿态检测', annotated)
            else:
                cv2.imshow('实时姿态检测', frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
    
    print("处理完成!")


if __name__ == "__main__":
    main()
