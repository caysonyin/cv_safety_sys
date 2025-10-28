#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频文物跟踪系统
实时检测、选择和跟踪文物，保持目标ID
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

# 添加yolov7目录到Python路径
sys.path.append('yolov7')

try:  # 优先使用本地yolov7工具函数
    from utils.general import non_max_suppression, scale_coords
except ModuleNotFoundError:  # pragma: no cover - 兼容子目录结构
    from yolov7.utils.general import non_max_suppression, scale_coords  # type: ignore


CLASS_NAMES: Tuple[str, ...] = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

EXCLUDED_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'hair drier', 'toothbrush'
}

HIGH_ANTIQUITY_CLASSES = {'bottle', 'wine glass', 'cup', 'bowl', 'vase', 'book', 'clock', 'scissors'}
MEDIUM_ANTIQUITY_CLASSES = {'teddy bear', 'potted plant'}

class SimpleTracker:
    """简单的目标跟踪器，基于质心距离的贪心匹配。"""

    def __init__(self, max_disappeared: int = 10, max_distance: float = 100.0):
        self.next_object_id = 0
        self.objects: Dict[int, Dict[str, object]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        """注册新目标"""
        self.objects[self.next_object_id] = {
            'centroid': tuple(centroid),
            'bbox': list(bbox),
            'last_seen': time.time()
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        """注销目标"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
    
    def update(self, detections: Sequence[Dict[str, object]]):
        """更新跟踪器"""
        if not detections:
            # 没有检测到目标，增加所有目标的消失计数
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_bboxes = [list(det['bbox']) for det in detections]
        input_centroids = np.array([
            ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            for bbox in input_bboxes
        ], dtype=np.float32)

        if not self.objects:
            for centroid, bbox in zip(input_centroids, input_bboxes):
                self.register(centroid, bbox)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array([
            self.objects[obj_id]['centroid'] for obj_id in object_ids
        ], dtype=np.float32)

        distance_matrix = np.linalg.norm(
            object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2,
        )

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for row in np.argsort(distance_matrix.min(axis=1)):
            col = distance_matrix[row].argmin()
            if row in used_rows or col in used_cols:
                continue
            if distance_matrix[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id]['centroid'] = tuple(input_centroids[col])
            self.objects[object_id]['bbox'] = input_bboxes[col]
            self.objects[object_id]['last_seen'] = time.time()
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distance_matrix.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(distance_matrix.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col], input_bboxes[col])

        return self.objects

class VideoRelicTracker:
    def __init__(self, model, device, confidence_threshold: float = 0.1):
        self.model = model
        self.device = device
        self.tracker = SimpleTracker(max_disappeared=10)
        self.selected_relics = set()  # 选中的文物ID
        self.relic_detections = []  # 当前帧的文物检测结果
        self.tracked_objects = {}  # 跟踪的目标
        self.window_name = "文物跟踪系统 - 点击选择文物，按Enter确认，按ESC退出"
        self.confidence_threshold = confidence_threshold

        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def _prepare_image(self, frame: np.ndarray) -> torch.Tensor:
        """预处理输入图像以便进行模型推理"""
        resized = cv2.resize(frame, (640, 640))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_image.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda':
            return tensor.half()
        return tensor.float()

    @staticmethod
    def _calculate_antiquity_score(
        *,
        area: float,
        confidence: float,
        class_name: str,
        bronze_ratio: float,
        edge_density: float,
    ) -> float:
        score = 0.0

        if area > 100000:
            score += 0.8
        elif area > 50000:
            score += 0.6
        elif area > 10000:
            score += 0.4
        else:
            score += 0.3

        if confidence > 0.8:
            score += 0.3
        elif confidence > 0.6:
            score += 0.2
        elif confidence > 0.4:
            score += 0.1

        if class_name in HIGH_ANTIQUITY_CLASSES:
            score += 0.4
        elif class_name in MEDIUM_ANTIQUITY_CLASSES:
            score += 0.2
        else:
            score += 0.1

        if bronze_ratio > 0.01 or edge_density > 0.05:
            score += 0.3

        return min(score, 1.0)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查点击是否在某个检测框内
            clicked_relic = self.get_clicked_relic(x, y)
            if clicked_relic is not None:
                if clicked_relic in self.selected_relics:
                    # 如果已选中，则取消选择
                    self.selected_relics.remove(clicked_relic)
                    print(f"取消选择文物 {clicked_relic}")
                else:
                    # 如果未选中，则选择
                    self.selected_relics.add(clicked_relic)
                    print(f"选择文物 {clicked_relic}")
    
    def get_clicked_relic(self, x, y):
        """获取点击的文物ID"""
        for detection in self.relic_detections:
            x1, y1, x2, y2 = detection['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return detection.get('track_id', None)
        return None
    
    def detect_relics(self, frame):
        """检测文物"""
        h, w = frame.shape[:2]
        
        # 分析图片特征
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bronze_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
        bronze_ratio = float(np.count_nonzero(bronze_mask)) / (h * w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (h * w)

        # 预处理 + 推理
        image_tensor = self._prepare_image(frame)
        with torch.no_grad():
            predictions = self.model(image_tensor)

        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        predictions = non_max_suppression(predictions, conf_thres=self.confidence_threshold)

        relic_detections: List[Dict[str, object]] = []
        detections_tensor = predictions[0]
        if detections_tensor is None or len(detections_tensor) == 0:
            return relic_detections

        detections_tensor[:, :4] = scale_coords(
            image_tensor.shape[2:],
            detections_tensor[:, :4],
            frame.shape
        ).round()

        for *xyxy, conf, cls in detections_tensor:
            class_id = int(cls)
            confidence = float(conf)
            x1, y1, x2, y2 = map(int, xyxy)

            if class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_id]
            else:
                class_name = f'class_{class_id}'

            if class_name in EXCLUDED_CLASSES:
                continue

            bbox_area = max(0, (x2 - x1) * (y2 - y1))
            score = self._calculate_antiquity_score(
                area=bbox_area,
                confidence=confidence,
                class_name=class_name,
                bronze_ratio=bronze_ratio,
                edge_density=edge_density,
            )

            if score < 0.3:
                continue

            relic_detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name,
                'area': bbox_area,
                'antiquity_score': score,
            })

        return relic_detections
    
    def update_tracking(self, detections):
        """更新跟踪"""
        # 更新跟踪器
        self.tracked_objects = self.tracker.update(detections)
        
        if not self.tracked_objects:
            for detection in detections:
                detection['track_id'] = None
            return

        tracked_ids = list(self.tracked_objects.keys())
        tracked_centroids = np.array([
            self.tracked_objects[track_id]['centroid'] for track_id in tracked_ids
        ], dtype=np.float32)

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            distances = np.linalg.norm(tracked_centroids - centroid, axis=1)

            best_index = distances.argmin()
            if distances[best_index] < 50:
                detection['track_id'] = tracked_ids[best_index]
            else:
                detection['track_id'] = None
    
    def draw_detections(self, frame):
        """绘制检测结果"""
        result_frame = frame.copy()
        
        for detection in self.relic_detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection.get('track_id', None)
            is_selected = track_id in self.selected_relics if track_id is not None else False
            
            # 确保坐标在图片范围内
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # 选择颜色和样式
            if is_selected:
                color = (0, 255, 0)  # 绿色
                thickness = 4
            else:
                color = (0, 0, 255)  # 红色
                thickness = 2
            
            # 绘制边界框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制跟踪ID
            if track_id is not None:
                cv2.putText(
                    result_frame, 
                    f"ID:{track_id}", 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    color, 
                    2
                )
            
            # 如果被选中，绘制红色电子栅栏
            if is_selected:
                fence_info = self.calculate_safety_fence([x1, y1, x2, y2], frame.shape)
                fx1, fy1, fx2, fy2 = fence_info['fence_bbox']
                cv2.rectangle(result_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
        
        return result_frame
    
    def calculate_safety_fence(self, relic_bbox, frame_shape, safety_margin=0.3):
        """计算文物的安全栅栏范围"""
        x1, y1, x2, y2 = relic_bbox
        h, w = frame_shape[:2]
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        relic_width = x2 - x1
        relic_height = y2 - y1
        
        safety_width = int(relic_width * (1 + safety_margin * 2))
        safety_height = int(relic_height * (1 + safety_margin * 2))
        
        fence_x1 = max(0, center_x - safety_width // 2)
        fence_y1 = max(0, center_y - safety_height // 2)
        fence_x2 = min(w, center_x + safety_width // 2)
        fence_y2 = min(h, center_y + safety_height // 2)
        
        return {
            'fence_bbox': [fence_x1, fence_y1, fence_x2, fence_y2],
            'relic_center': [center_x, center_y],
            'safety_margin': safety_margin,
            'fence_area': (fence_x2 - fence_x1) * (fence_y2 - fence_y1)
        }
    
    def process_video(self, video_source=0):
        """处理视频"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return
        
        print("=== 视频文物跟踪系统 ===")
        print("操作说明:")
        print("1. 点击红色框选择文物")
        print("2. 点击绿色框取消选择")
        print("3. 按Enter键确认选择")
        print("4. 按ESC键退出")
        print("5. 按S键保存当前帧")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 检测文物
            self.relic_detections = self.detect_relics(frame)
            
            # 更新跟踪
            self.update_tracking(self.relic_detections)
            
            # 绘制检测结果
            result_frame = self.draw_detections(frame)
            
            # 显示状态信息
            status_text = f"已选择: {len(self.selected_relics)} 个文物"
            cv2.putText(
                result_frame, 
                status_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (255, 255, 255), 
                2
            )
            
            # 显示帧数
            cv2.putText(
                result_frame, 
                f"Frame: {frame_count}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
            # 显示图片
            cv2.imshow(self.window_name, result_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break
            elif key == 13:  # Enter键
                print(f"确认选择 {len(self.selected_relics)} 个文物")
            elif key == ord('s') or key == ord('S'):  # S键保存
                filename = f"tracking_frame_{frame_count}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"保存帧到: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()

def download_yolov7_tiny(destination: Path = Path("yolov7-tiny.pt")) -> Optional[Path]:
    """下载YOLOv7-tiny预训练模型"""
    model_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"

    if destination.exists():
        print(f"模型文件已存在: {destination}")
        return destination

    print("正在下载YOLOv7-tiny模型...")
    try:
        import requests

        response = requests.get(model_url, stream=True, timeout=30)
        response.raise_for_status()

        with destination.open('wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                file.write(chunk)

        print(f"模型下载完成: {destination}")
        return destination
    except Exception as e:
        print(f"下载模型失败: {e}")
        return None

def load_model(model_path: Path):
    """加载YOLOv7模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model = model.to(device).float().eval()
        print("模型加载成功")
        return model, device
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='视频文物跟踪系统')
    parser.add_argument('--source', type=str, default='0', help='视频源 (0=摄像头, 或视频文件路径)')
    parser.add_argument('--conf', type=float, default=0.1, help='置信度阈值')
    
    args = parser.parse_args()
    
    print("=== 视频文物跟踪系统 ===")
    print("实时检测、选择和跟踪文物")
    
    # 下载模型
    model_path = download_yolov7_tiny()
    if model_path is None:
        return

    # 加载模型
    model, device = load_model(model_path)
    if model is None:
        return

    # 创建跟踪器
    tracker = VideoRelicTracker(model, device, confidence_threshold=args.conf)
    
    # 处理视频
    try:
        video_source = int(args.source) if args.source.isdigit() else args.source
        tracker.process_video(video_source)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"处理视频时出错: {e}")

if __name__ == "__main__":
    main()
