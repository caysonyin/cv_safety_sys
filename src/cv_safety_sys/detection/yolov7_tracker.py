#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于YOLOv7-tiny的文物检测与跟踪工具集。"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import inspect
import subprocess
import cv2
import numpy as np
import torch

from cv_safety_sys.utils import put_text


REPO_ROOT = Path(__file__).resolve().parents[3]
YOLO_DIR = REPO_ROOT / "yolov7"
YOLO_REPO_URL = "https://github.com/WongKinYiu/yolov7.git"


def _ensure_yolov7_repo() -> None:
    """确保 yolov7 源码存在（若缺失尝试自动克隆）。"""

    if YOLO_DIR.exists():
        return

    print("未检测到 yolov7 源码目录，正在自动克隆...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", YOLO_REPO_URL, str(YOLO_DIR)],
            check=True,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("yolov7 仓库克隆完成。")
    except Exception as exc:  # pragma: no cover - 依赖外部环境
        raise ModuleNotFoundError(
            "无法自动克隆 yolov7 仓库，请手动运行：\n"
            f"  git clone --depth 1 {YOLO_REPO_URL} {YOLO_DIR}"
        ) from exc


_ensure_yolov7_repo()
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

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

DEFAULT_YOLO_MODEL_PATH = REPO_ROOT / "models" / "yolov7-tiny.pt"

class SimpleTracker:
    """简单的目标跟踪器，基于质心距离的贪心匹配。"""

    def __init__(
        self,
        max_disappeared: int = 10,
        max_distance: float = 100.0,
        *,
        iou_high_threshold: float = 0.45,
        iou_low_threshold: float = 0.2,
    ):
        self.next_object_id = 0
        self.objects: Dict[int, Dict[str, object]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_high_threshold = iou_high_threshold
        self.iou_low_threshold = iou_low_threshold
        self.last_assignments: Dict[int, int] = {}

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
    
    @staticmethod
    def _bbox_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
        ax1, ay1, ax2, ay2 = map(float, box_a)
        bx1, by1, bx2, by2 = map(float, box_b)
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0.0:
            return 0.0
        return float(inter_area / denom)

    def get_last_assignments(self) -> Dict[int, int]:
        """返回上一帧匹配结果（检测索引 -> track_id）。"""
        return dict(self.last_assignments)

    def update(self, detections: Sequence[Dict[str, object]]):
        """更新跟踪器"""
        self.last_assignments = {}
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
            assignments: Dict[int, int] = {}
            for idx, (centroid, bbox) in enumerate(zip(input_centroids, input_bboxes)):
                new_id = self.register(centroid, bbox)
                assignments[idx] = new_id
            self.last_assignments = assignments
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array([
            self.objects[obj_id]['centroid'] for obj_id in object_ids
        ], dtype=np.float32)
        object_bboxes = [self.objects[obj_id]['bbox'] for obj_id in object_ids]

        distance_matrix = np.linalg.norm(
            object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :],
            axis=2,
        )

        iou_matrix = np.zeros((len(object_ids), len(input_bboxes)), dtype=np.float32)
        for row, bbox_a in enumerate(object_bboxes):
            for col, bbox_b in enumerate(input_bboxes):
                iou_matrix[row, col] = self._bbox_iou(bbox_a, bbox_b)

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        assignments: Dict[int, int] = {}

        def _assign(row: int, col: int) -> None:
            obj_id = object_ids[row]
            self.objects[obj_id]['centroid'] = tuple(input_centroids[col])
            self.objects[obj_id]['bbox'] = input_bboxes[col]
            self.objects[obj_id]['last_seen'] = time.time()
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)
            assignments[col] = obj_id

        # 阶段1：优先匹配高IoU目标
        high_pairs: List[Tuple[int, int, float]] = []
        for row in range(iou_matrix.shape[0]):
            for col in range(iou_matrix.shape[1]):
                iou_val = float(iou_matrix[row, col])
                if iou_val >= self.iou_high_threshold:
                    high_pairs.append((row, col, iou_val))
        high_pairs.sort(key=lambda item: item[2], reverse=True)
        for row, col, _ in high_pairs:
            if row in used_rows or col in used_cols:
                continue
            _assign(row, col)

        # 阶段2：结合距离 + 低IoU阈值兜底
        row_order = np.argsort(distance_matrix.min(axis=1))
        for row in row_order:
            if row in used_rows:
                continue
            col = distance_matrix[row].argmin()
            if col in used_cols:
                continue
            distance = float(distance_matrix[row, col])
            iou_val = float(iou_matrix[row, col])
            if distance > self.max_distance and iou_val < self.iou_low_threshold:
                continue
            _assign(row, col)

        unused_rows = set(range(distance_matrix.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(distance_matrix.shape[1])).difference(used_cols)
        for col in unused_cols:
            new_id = self.register(input_centroids[col], input_bboxes[col])
            assignments[col] = new_id

        self.last_assignments = assignments
        return self.objects

class VideoRelicTracker:
    def __init__(
        self,
        model,
        device,
        confidence_threshold: float = 0.1,
        window_name: Optional[str] = None,
        *,
        create_window: bool = True,
    ):
        self.model = model
        self.device = device
        self.tracker = SimpleTracker(max_disappeared=10)
        self.selected_relics = set()  # 选中的文物ID
        self.relic_detections = []  # 当前帧的文物检测结果
        self.tracked_objects = {}  # 跟踪的目标
        self.manual_fences: Dict[int, Dict[str, object]] = {}
        self.last_frame_shape: Tuple[int, int, int] | None = None
        self.window_name = (
            window_name
            if window_name is not None
            else "文物跟踪系统 - 点击选择文物，按Enter确认，按ESC退出"
        )
        self.confidence_threshold = confidence_threshold
        self._create_window = create_window

        # 创建窗口
        if self._create_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def _prepare_image(self, frame: np.ndarray) -> torch.Tensor:
        """预处理输入图像以便进行模型推理"""
        resized = cv2.resize(frame, (640, 640))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb_image.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device).float()

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
            self.handle_click(x, y)

    def get_clicked_relic(self, x, y):
        """获取点击的文物ID"""
        for detection in self.relic_detections:
            x1, y1, x2, y2 = detection['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return detection.get('track_id', None)
        return None

    def toggle_relic_selection(self, relic_id: Optional[int]) -> None:
        """根据给定的ID切换文物选中状态。"""
        if relic_id is None:
            return
        if relic_id in self.selected_relics:
            self.selected_relics.remove(relic_id)
            self.manual_fences.pop(relic_id, None)
            print(f"取消选择文物 {relic_id}")
        else:
            self.selected_relics.add(relic_id)
            print(f"选择文物 {relic_id}")

    def clear_selection(self) -> None:
        """取消所有选中的文物。"""
        if self.selected_relics:
            self.selected_relics.clear()
            print("已清空所有选中文物")
        if self.manual_fences:
            self.manual_fences.clear()

    def handle_click(self, x: int, y: int) -> None:
        """用于交互界面处理点击事件。"""
        clicked_relic = self.get_clicked_relic(x, y)
        if clicked_relic is not None:
            self.toggle_relic_selection(clicked_relic)
    
    def _detect_all_objects(self, frame: np.ndarray) -> List[Dict[str, object]]:
        """运行一次YOLO检测并返回所有检测结果。"""
        image_tensor = self._prepare_image(frame)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        detections = non_max_suppression(
            predictions,
            conf_thres=self.confidence_threshold,
        )

        detections_tensor = detections[0]
        if detections_tensor is None or len(detections_tensor) == 0:
            return []

        detections_tensor = detections_tensor.clone()
        detections_tensor[:, :4] = scale_coords(
            image_tensor.shape[2:],
            detections_tensor[:, :4],
            frame.shape,
        ).round()

        all_detections: List[Dict[str, object]] = []
        for *xyxy, conf, cls in detections_tensor:
            class_id = int(cls)
            confidence = float(conf)
            x1, y1, x2, y2 = map(int, xyxy)

            if class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_id]
            else:
                class_name = f'class_{class_id}'

            all_detections.append(
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'area': max(0, (x2 - x1) * (y2 - y1)),
                }
            )

        return all_detections

    def detect_relics(
        self,
        frame: np.ndarray,
        detections: Optional[Sequence[Dict[str, object]]] = None,
    ) -> List[Dict[str, object]]:
        """检测文物"""
        h, w = frame.shape[:2]

        # 分析图片特征
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bronze_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([30, 255, 255]))
        bronze_ratio = float(np.count_nonzero(bronze_mask)) / (h * w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (h * w)

        relic_detections: List[Dict[str, object]] = []
        if detections is None:
            detections = self._detect_all_objects(frame)

        for detection in detections:
            class_name = str(detection['class_name'])
            if class_name in EXCLUDED_CLASSES:
                continue

            class_id = int(detection['class_id'])
            confidence = float(detection['confidence'])
            x1, y1, x2, y2 = map(int, detection['bbox'])
            bbox_area = float(detection.get('area', max(0, (x2 - x1) * (y2 - y1))))
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
        assignments = self.tracker.get_last_assignments()
        
        if assignments:
            for idx, detection in enumerate(detections):
                detection['track_id'] = assignments.get(idx)
            return
        
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
    
    def draw_detections(self, frame, *, show_labels: bool = True):
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
            if show_labels and track_id is not None:
                put_text(
                    result_frame,
                    f"ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            
            # 如果被选中，绘制红色电子栅栏
            if is_selected:
                fence_info = self._resolve_fence_info(
                    track_id,
                    [x1, y1, x2, y2],
                    frame.shape,
                )
                fx1, fy1, fx2, fy2 = fence_info['fence_bbox']
                fence_color = (0, 0, 255)
                if fence_info.get('manual'):
                    fence_color = (0, 215, 255)
                cv2.rectangle(result_frame, (fx1, fy1), (fx2, fy2), fence_color, 3)

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

    # ------------------------------------------------------------------
    # 电子栅栏辅助方法
    # ------------------------------------------------------------------
    def _minimum_fence_span(self, frame_shape: Tuple[int, int, int]) -> int:
        h, w = frame_shape[:2]
        base = max(8, int(min(h, w) * 0.01))
        return max(8, base)

    def _normalize_bbox(
        self,
        bbox: Sequence[int],
        frame_shape: Tuple[int, int, int],
    ) -> List[int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        min_span = self._minimum_fence_span(frame_shape)
        if x2 - x1 < min_span:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - min_span // 2)
            x2 = min(w, x1 + min_span)
            if x2 - x1 < min_span:
                x1 = max(0, x2 - min_span)
        if y2 - y1 < min_span:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - min_span // 2)
            y2 = min(h, y1 + min_span)
            if y2 - y1 < min_span:
                y1 = max(0, y2 - min_span)

        return [x1, y1, x2, y2]

    def _resolve_fence_info(
        self,
        track_id: Optional[int],
        bbox: Sequence[int],
        frame_shape: Tuple[int, int, int],
    ) -> Dict[str, object]:
        fence_info = self.calculate_safety_fence(bbox, frame_shape)
        fence_info['manual'] = False
        if track_id is None:
            return fence_info

        manual_entry = self.manual_fences.get(track_id)
        if not manual_entry:
            return fence_info

        manual_bbox = list(map(int, manual_entry['bbox']))
        fence_info['fence_bbox'] = manual_bbox
        center_x = (manual_bbox[0] + manual_bbox[2]) // 2
        center_y = (manual_bbox[1] + manual_bbox[3]) // 2
        fence_info['relic_center'] = [center_x, center_y]
        fence_info['fence_area'] = max(
            0, (manual_bbox[2] - manual_bbox[0]) * (manual_bbox[3] - manual_bbox[1])
        )
        fence_info['manual'] = True
        return fence_info

    def get_detection_fence_info(
        self,
        detection: Dict[str, object],
        frame_shape: Tuple[int, int, int],
    ) -> Dict[str, object]:
        track_id = detection.get('track_id')
        return self._resolve_fence_info(track_id, detection['bbox'], frame_shape)

    def update_manual_fence(
        self,
        track_id: int,
        bbox: Sequence[int],
        *,
        frame_shape: Tuple[int, int, int] | None = None,
    ) -> List[int] | None:
        if track_id not in self.selected_relics:
            return None
        target_shape = frame_shape or self.last_frame_shape
        if target_shape is None:
            return None

        sanitized = self._normalize_bbox(bbox, target_shape)
        self.manual_fences[track_id] = {
            'bbox': sanitized,
            'updated_at': time.time(),
        }
        return list(sanitized)

    def get_selected_fences(
        self,
        frame_shape: Tuple[int, int, int] | None = None,
    ) -> List[Dict[str, object]]:
        target_shape = frame_shape or self.last_frame_shape
        if target_shape is None:
            return []

        fences: List[Dict[str, object]] = []
        for detection in self.relic_detections:
            track_id = detection.get('track_id')
            if track_id is None or track_id not in self.selected_relics:
                continue
            fence_info = self.get_detection_fence_info(detection, target_shape)
            fences.append(
                {
                    'track_id': track_id,
                    'bbox': list(fence_info['fence_bbox']),
                    'label': detection.get('class_name', 'relic'),
                    'manual': fence_info.get('manual', False),
                }
            )
        return fences

    @staticmethod
    def _hit_test_bbox(
        bbox: Sequence[int],
        x: int,
        y: int,
        tolerance: int,
    ) -> Dict[str, str] | None:
        x1, y1, x2, y2 = map(int, bbox)
        corners = {
            'top_left': (x1, y1),
            'top_right': (x2, y1),
            'bottom_left': (x1, y2),
            'bottom_right': (x2, y2),
        }
        for name, (cx, cy) in corners.items():
            if abs(x - cx) <= tolerance and abs(y - cy) <= tolerance:
                return {'kind': 'corner', 'name': name}

        if y1 <= y <= y2 and abs(x - x1) <= tolerance:
            return {'kind': 'edge', 'name': 'left'}
        if y1 <= y <= y2 and abs(x - x2) <= tolerance:
            return {'kind': 'edge', 'name': 'right'}
        if x1 <= x <= x2 and abs(y - y1) <= tolerance:
            return {'kind': 'edge', 'name': 'top'}
        if x1 <= x <= x2 and abs(y - y2) <= tolerance:
            return {'kind': 'edge', 'name': 'bottom'}

        return None

    def find_fence_handle(
        self,
        x: int,
        y: int,
        *,
        tolerance: int = 12,
        frame_shape: Tuple[int, int, int] | None = None,
    ) -> Dict[str, object] | None:
        target_shape = frame_shape or self.last_frame_shape
        if target_shape is None:
            return None

        for fence in self.get_selected_fences(target_shape):
            hit = self._hit_test_bbox(fence['bbox'], x, y, tolerance)
            if hit:
                return {**fence, 'hit': hit}
        return None
    
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
            self.last_frame_shape = frame.shape
            
            # 检测文物
            all_detections = self._detect_all_objects(frame)
            self.relic_detections = self.detect_relics(frame, all_detections)

            # 更新跟踪
            self.update_tracking(self.relic_detections)
            
            # 绘制检测结果
            result_frame = self.draw_detections(frame, show_labels=True)
            
            # 显示状态信息
            status_text = f"已选择: {len(self.selected_relics)} 个文物"
            put_text(
                result_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            
            # 显示帧数
            put_text(
                result_frame,
                f"Frame: {frame_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
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

def download_yolov7_tiny(destination: Path = DEFAULT_YOLO_MODEL_PATH) -> Optional[Path]:
    """下载YOLOv7-tiny预训练模型到``models/``目录。"""

    model_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)

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

def _torch_load_kwargs() -> Dict[str, object]:
    """Return compatibility kwargs for torch.load."""

    try:
        signature = inspect.signature(torch.load)
        if "weights_only" in signature.parameters:
            return {"weights_only": False}
    except (TypeError, ValueError):
        pass
    return {}


def load_model(model_path: Path):
    """加载YOLOv7模型（仅 CPU）"""
    device = torch.device('cpu')
    print("使用设备: CPU")

    try:
        checkpoint = torch.load(model_path, map_location=device, **_torch_load_kwargs())
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
    parser.add_argument('--yolo-model', type=str, default=str(DEFAULT_YOLO_MODEL_PATH), help='YOLO 模型路径')
    
    args = parser.parse_args()
    
    print("=== 视频文物跟踪系统 ===")
    print("实时检测、选择和跟踪文物")
    
    # 下载模型
    model_path = download_yolov7_tiny(Path(args.yolo_model))
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
