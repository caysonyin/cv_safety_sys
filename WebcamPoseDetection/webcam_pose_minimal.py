#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最简化的实时摄像头 33 关节姿态检测。"""

import argparse
from typing import Union

import cv2
import mediapipe as mp
from mediapipe import Image as MPImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def main(video_source: Union[int, str] = 0, window_name: str = "实时姿态检测"):
    """主函数 - 最简化的实时摄像头姿态检测"""
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
                
                cv2.imshow(window_name, annotated)
            else:
                cv2.imshow(window_name, frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
    
    print("处理完成!")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最简姿态检测示例")
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
    main(
        video_source=_coerce_source(str(args.source)),
        window_name=args.window_name,
    )
