import os
import argparse
import time
import threading
import queue
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import Image as MPImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class RealtimePoseDetector:
    """实时姿态检测器，专为低延时优化"""
    
    def __init__(self, model_path: str, max_queue_size: int = 5):
        self.model_path = model_path
        self.max_queue_size = max_queue_size
        
        # 性能监控
        self.frame_times = deque(maxlen=30)  # 保留最近30帧的时间
        self.processing_times = deque(maxlen=30)
        
        # 线程安全队列
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        
        # 控制标志
        self.running = False
        self.processing_thread = None
        
        # 初始化MediaPipe
        self._initialize_mediapipe()
        
        # 时间戳管理
        self.timestamp_ms = 0
        
    def _initialize_mediapipe(self):
        """初始化MediaPipe模型"""
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,  # 使用VIDEO模式以获得最佳性能
            min_pose_detection_confidence=0.3,  # 降低阈值以提高速度
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            num_poses=5,  # 限制检测人数以提高速度
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """优化的帧预处理"""
        h, w = frame.shape[:2]
        
        # 智能缩放：如果图像太大，进行缩放以提高处理速度
        max_dimension = 640  # 最大尺寸
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 确保尺寸是16的倍数，便于统一推理流程
        h, w = frame.shape[:2]
        new_h = ((h + 15) // 16) * 16
        new_w = ((w + 15) // 16) * 16
        
        if new_h != h or new_w != w:
            frame = cv2.resize(frame, (new_w, new_w), interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def _draw_pose_fast(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """快速姿态绘制"""
        if not landmarks:
            return frame
        
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # 预计算所有连接
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
            
            # 绘制连接线（使用更细的线条以提高速度）
            for a, b in connections:
                if 0 <= a < len(points) and 0 <= b < len(points):
                    pa, pb = points[a], points[b]
                    if pa[0] >= 0 and pb[0] >= 0:
                        cv2.line(annotated, pa, pb, (0, 255, 0), 1)
            
            # 绘制关节点（使用更小的圆点）
            for x, y in points:
                if x >= 0:
                    cv2.circle(annotated, (x, y), 2, (0, 255, 255), -1)
        
        return annotated
    
    def _processing_worker(self):
        """后台处理线程"""
        while self.running:
            try:
                # 获取帧（带超时）
                frame_data = self.frame_queue.get(timeout=0.1)
                frame, frame_id = frame_data
                
                start_time = time.time()
                
                # 预处理
                processed_frame = self._preprocess_frame(frame)
                
                # 转换颜色空间
                rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
                
                # 检测姿态
                result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
                self.timestamp_ms += 1
                
                # 绘制结果
                annotated = self._draw_pose_fast(processed_frame, result.pose_landmarks)
                
                # 记录处理时间
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # 将结果放入队列
                self.result_queue.put((frame_id, annotated, processing_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理线程错误: {e}")
                continue
    
    def start_processing(self):
        """开始处理"""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """停止处理"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Optional[np.ndarray]:
        """处理单帧（非阻塞）"""
        if not self.running:
            self.start_processing()
        
        # 尝试添加帧到队列
        try:
            self.frame_queue.put_nowait((frame, frame_id))
        except queue.Full:
            # 队列满时，丢弃最旧的帧
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame, frame_id))
            except queue.Empty:
                pass
        
        # 尝试获取结果
        try:
            result_id, annotated_frame, processing_time = self.result_queue.get_nowait()
            return annotated_frame
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.processing_times:
            return {"fps": 0, "avg_processing_time": 0}
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        return {
            "fps": fps,
            "avg_processing_time": avg_processing_time,
            "queue_size": self.frame_queue.qsize(),
            "result_queue_size": self.result_queue.qsize()
        }


def process_video_realtime(
    input_path: str,
    output_path: str | None = None,
    model_path: str = "models/pose_landmarker_full.task",
    show_preview: bool = True,
    max_queue_size: int = 3
) -> str:
    """实时视频处理"""
    
    # 确保模型存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return ""
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {input_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_realtime_pose33.mp4"
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 创建检测器
    detector = RealtimePoseDetector(model_path, max_queue_size)
    
    print("开始实时处理...")
    print("按 'q' 键退出预览")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 处理帧
            annotated_frame = detector.process_frame(frame, frame_count)
            
            if annotated_frame is not None:
                # 写入输出视频
                out.write(annotated_frame)
                
                # 显示预览
                if show_preview:
                    # 添加性能信息
                    stats = detector.get_performance_stats()
                    cv2.putText(annotated_frame, f"FPS: {stats['fps']:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Queue: {stats['queue_size']}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('实时姿态检测', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # 如果没有处理结果，写入原帧
                out.write(frame)
    
    finally:
        # 清理资源
        detector.stop_processing()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # 显示最终统计
    total_time = time.time() - start_time
    final_stats = detector.get_performance_stats()
    
    print(f"\n处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均FPS: {frame_count / total_time:.2f}")
    print(f"处理FPS: {final_stats['fps']:.2f}")
    print(f"平均处理时间: {final_stats['avg_processing_time']:.3f}秒")
    print(f"输出文件: {output_path}")
    
    return output_path


def process_webcam_realtime(
    model_path: str = "models/pose_landmarker_full.task",
    max_queue_size: int = 2
) -> None:
    """实时摄像头处理"""
    
    # 确保模型存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 创建检测器
    detector = RealtimePoseDetector(model_path, max_queue_size)
    
    print("开始实时摄像头处理...")
    print("按 'q' 键退出")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 处理帧
            annotated_frame = detector.process_frame(frame, frame_count)
            
            if annotated_frame is not None:
                # 添加性能信息
                stats = detector.get_performance_stats()
                cv2.putText(annotated_frame, f"FPS: {stats['fps']:.1f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Queue: {stats['queue_size']}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('实时姿态检测', annotated_frame)
            else:
                cv2.imshow('实时姿态检测', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 清理资源
        detector.stop_processing()
        cap.release()
        cv2.destroyAllWindows()
    
    # 显示最终统计
    total_time = time.time() - start_time
    final_stats = detector.get_performance_stats()
    
    print(f"\n处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均FPS: {frame_count / total_time:.2f}")
    print(f"处理FPS: {final_stats['fps']:.2f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实时33关节姿态检测")
    parser.add_argument("--video", help="输入视频路径")
    parser.add_argument("--webcam", action="store_true", help="使用摄像头")
    parser.add_argument("--output", "-o", help="输出视频路径")
    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="模型路径")
    parser.add_argument("--queue-size", type=int, default=3, help="队列大小")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览")
    
    args = parser.parse_args()
    
    if args.webcam:
        # 摄像头模式
        process_webcam_realtime(args.model, args.queue_size)
    elif args.video:
        # 视频文件模式
        process_video_realtime(
            args.video, 
            args.output, 
            args.model, 
            not args.no_preview,
            args.queue_size
        )
    else:
        print("请指定 --video 或使用 --webcam")


if __name__ == "__main__":
    main()
