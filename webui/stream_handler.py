import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import time
from threading import Lock

object_protection_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'object_protection')
yolov7_path = os.path.join(object_protection_path, 'yolov7')
sys.path.insert(0, object_protection_path)
sys.path.insert(0, yolov7_path)

class RelicStreamHandler:
    def __init__(self, model_path=None, device='cuda', video_source=0):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.video_source = video_source
        self.cap = None
        self.is_running = False
        self.is_video_file = isinstance(video_source, str)
        self.lock = Lock()
        
        self.selected_relics = set()
        self.relic_detections = []
        self.tracked_objects = {}
        self.next_object_id = 0
        self.disappeared = {}
        self.max_disappeared = 10
        
        self.alerts = []
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'object_protection', 'yolov7-tiny.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model = checkpoint['model'].float()
        else:
            self.model = checkpoint.float()
        
        self.model.to(self.device).eval()
        if self.device.type != 'cpu':
            self.model.half()
        
        print(f"Model loaded successfully on {self.device}")
    
    def start_capture(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.video_source}")
            
            if not self.is_video_file:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
    
    def stop_capture(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def detect_relics(self, frame):
        h, w = frame.shape[:2]
        
        image = cv2.resize(frame, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        if self.device.type != 'cpu':
            image = image.half()
        
        with torch.no_grad():
            predictions = self.model(image)
        
        try:
            from utils.general import non_max_suppression, scale_coords
        except ImportError as e:
            print(f"Warning: Failed to import yolov7 utils: {e}")
            print(f"sys.path: {sys.path[:3]}")
            raise ImportError("Cannot import yolov7 utilities. Make sure yolov7 directory exists in object_protection/")
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = non_max_suppression(predictions, conf_thres=0.1)
        
        all_detections = []
        if predictions[0] is not None and len(predictions[0]) > 0:
            predictions[0][:, :4] = scale_coords(image.shape[2:], predictions[0][:, :4], frame.shape).round()
            
            for *xyxy, conf, cls in predictions[0]:
                class_id = int(cls)
                confidence = float(conf)
                x1, y1, x2, y2 = map(int, xyxy)
                bbox_area = (x2 - x1) * (y2 - y1)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'area': bbox_area
                }
                all_detections.append(detection)
        
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                      'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                      'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                      'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                      'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                      'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        excluded_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                          'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'tv', 'laptop', 'mouse',
                          'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                          'hair drier', 'toothbrush']
        
        relic_detections = []
        for detection in all_detections:
            class_id = detection['class_id']
            class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
            
            if class_name in excluded_classes:
                continue
            
            confidence = detection['confidence']
            area = detection['area']
            antiquity_score = 0.0
            
            if area > 100000:
                antiquity_score += 0.8
            elif area > 50000:
                antiquity_score += 0.6
            elif area > 10000:
                antiquity_score += 0.4
            else:
                antiquity_score += 0.3
            
            if confidence > 0.8:
                antiquity_score += 0.3
            elif confidence > 0.6:
                antiquity_score += 0.2
            elif confidence > 0.4:
                antiquity_score += 0.1
            
            high_antiquity_classes = ['bottle', 'wine glass', 'cup', 'bowl', 'vase', 'book', 'clock', 'scissors']
            if class_name in high_antiquity_classes:
                antiquity_score += 0.4
            else:
                antiquity_score += 0.1
            
            if antiquity_score >= 0.1:
                detection['antiquity_score'] = antiquity_score
                detection['class_name'] = class_name
                relic_detections.append(detection)
        
        return relic_detections
    
    def update_tracking(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    if object_id in self.tracked_objects:
                        del self.tracked_objects[object_id]
                    del self.disappeared[object_id]
            return
        
        input_centroids = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            input_centroids.append(centroid)
        
        if len(self.tracked_objects) == 0:
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                self.tracked_objects[self.next_object_id] = {
                    'centroid': centroid,
                    'bbox': [x1, y1, x2, y2],
                    'last_seen': time.time()
                }
                self.disappeared[self.next_object_id] = 0
                detection['track_id'] = self.next_object_id
                self.next_object_id += 1
        else:
            object_centroids = [self.tracked_objects[obj_id]['centroid'] for obj_id in self.tracked_objects.keys()]
            object_ids = list(self.tracked_objects.keys())
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] < 100:
                    object_id = object_ids[row]
                    x1, y1, x2, y2 = detections[col]['bbox']
                    self.tracked_objects[object_id]['centroid'] = input_centroids[col]
                    self.tracked_objects[object_id]['bbox'] = [x1, y1, x2, y2]
                    self.tracked_objects[object_id]['last_seen'] = time.time()
                    self.disappeared[object_id] = 0
                    detections[col]['track_id'] = object_id
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.tracked_objects[object_id]
                    del self.disappeared[object_id]
            
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                x1, y1, x2, y2 = detections[col]['bbox']
                centroid = input_centroids[col]
                self.tracked_objects[self.next_object_id] = {
                    'centroid': centroid,
                    'bbox': [x1, y1, x2, y2],
                    'last_seen': time.time()
                }
                self.disappeared[self.next_object_id] = 0
                detections[col]['track_id'] = self.next_object_id
                self.next_object_id += 1
    
    def check_alerts(self, frame_shape):
        new_alerts = []
        for track_id in self.selected_relics:
            if track_id in self.tracked_objects:
                obj = self.tracked_objects[track_id]
                bbox = obj['bbox']
                x1, y1, x2, y2 = bbox
                
                fence = self.calculate_safety_fence(bbox, frame_shape)
                fx1, fy1, fx2, fy2 = fence['fence_bbox']
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if not (fx1 < center_x < fx2 and fy1 < center_y < fy2):
                    alert = {
                        'type': 'fence_breach',
                        'track_id': track_id,
                        'message': f'文物 ID:{track_id} 移出安全区',
                        'timestamp': time.time()
                    }
                    new_alerts.append(alert)
        
        if new_alerts:
            with self.lock:
                self.alerts.extend(new_alerts)
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[-50:]
        
        return new_alerts
    
    def calculate_safety_fence(self, relic_bbox, frame_shape, safety_margin=0.3):
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
            'safety_margin': safety_margin
        }
    
    def draw_detections(self, frame):
        result_frame = frame.copy()
        
        for detection in self.relic_detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection.get('track_id', None)
            
            if track_id is None:
                continue
            
            is_selected = track_id in self.selected_relics
            
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            color = (0, 255, 0) if is_selected else (0, 0, 255)
            thickness = 4 if is_selected else 2
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"ID:{track_id}"
            cv2.putText(result_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if is_selected:
                fence_info = self.calculate_safety_fence([x1, y1, x2, y2], frame.shape)
                fx1, fy1, fx2, fy2 = fence_info['fence_bbox']
                cv2.rectangle(result_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
        
        status_text = f"Selected: {len(self.selected_relics)}"
        cv2.putText(result_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return result_frame
    
    def get_frame(self):
        if not self.is_running or self.cap is None:
            return None, []
        
        ret, frame = self.cap.read()
        if not ret:
            if self.is_video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None, []
            else:
                return None, []
        
        with self.lock:
            self.relic_detections = self.detect_relics(frame)
            self.update_tracking(self.relic_detections)
            self.check_alerts(frame.shape)
            result_frame = self.draw_detections(frame)
        
        return result_frame, self.get_relic_list()
    
    def get_relic_list(self):
        relic_list = []
        for detection in self.relic_detections:
            track_id = detection.get('track_id', None)
            if track_id is not None:
                relic_list.append({
                    'track_id': track_id,
                    'confidence': round(detection['confidence'] * 100, 1),
                    'class_name': detection.get('class_name', 'unknown'),
                    'selected': track_id in self.selected_relics,
                    'bbox': detection['bbox']
                })
        return relic_list
    
    def toggle_selection(self, track_id):
        with self.lock:
            if track_id in self.selected_relics:
                self.selected_relics.remove(track_id)
                return False
            else:
                self.selected_relics.add(track_id)
                return True
    
    def clear_selection(self):
        with self.lock:
            self.selected_relics.clear()
    
    def get_alerts(self):
        with self.lock:
            return self.alerts.copy()
