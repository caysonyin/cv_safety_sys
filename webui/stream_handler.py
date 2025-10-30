import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
import time
from threading import Lock
from collections import deque

object_protection_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'object_protection')
if object_protection_path not in sys.path:
    sys.path.insert(0, object_protection_path)

from general import non_max_suppression, scale_coords

try:
    import mediapipe as mp
    from mediapipe import Image as MPImage, ImageFormat
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Pose detection disabled.")

POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS) if MEDIAPIPE_AVAILABLE else []
DANGEROUS_CLASSES = {'knife', 'scissors', 'baseball bat'}

class IntegratedStreamHandler:
    
    def __init__(self, model_path=None, pose_model_path=None, device='cuda', video_source=0):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.video_source = video_source
        self.cap = None
        self.is_running = False
        self.is_video_file = isinstance(video_source, str)
        self.lock = Lock()
        
        # 文物检测
        self.selected_relics = set()
        self.relic_detections = []
        self.tracked_objects = {}
        self.next_object_id = 0
        self.disappeared = {}
        self.max_disappeared = 10
        
        # 姿态检测
        self.person_detections = []
        self.person_tracks = {}
        self.person_tracker_id = 0
        self.person_disappeared = {}
        self.pose_entries = []
        self.dangerous_detections = []
        
        self.alerts = []
        self.alert_history = deque(maxlen=50)
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'object_protection', 'yolov7-tiny.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading YOLO model from: {model_path}")
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
        print(f"YOLO model loaded on {self.device}")
        
        # 加载姿态检测模型
        self.pose_landmarker = None
        self.pose_timestamp_ms = 0
        
        if MEDIAPIPE_AVAILABLE and pose_model_path:
            if not os.path.exists(pose_model_path):
                print(f"Warning: Pose model not found: {pose_model_path}")
            else:
                try:
                    base_options = mp_python.BaseOptions(model_asset_path=pose_model_path)
                    options = mp_vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=mp_vision.RunningMode.VIDEO,
                        min_pose_detection_confidence=0.3,
                        min_pose_presence_confidence=0.3,
                        min_tracking_confidence=0.3,
                        num_poses=5,
                    )
                    self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                    print("Pose detection model loaded")
                except Exception as e:
                    print(f"Failed to load pose model: {e}")
    
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
    
    def detect_all_objects(self, frame):
        h, w = frame.shape[:2]
        
        image = cv2.resize(frame, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        if self.device.type != 'cpu':
            image = image.half()
        
        with torch.no_grad():
            predictions = self.model(image)
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = non_max_suppression(predictions, conf_thres=0.1)
        
        all_detections = []
        if predictions[0] is not None and len(predictions[0]) > 0:
            predictions[0][:, :4] = scale_coords(image.shape[2:], predictions[0][:, :4], frame.shape).round()
            
            class_names = [
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
            ]
            
            for *xyxy, conf, cls in predictions[0]:
                class_id = int(cls)
                confidence = float(conf)
                x1, y1, x2, y2 = map(int, xyxy)
                bbox_area = (x2 - x1) * (y2 - y1)
                class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'area': bbox_area
                }
                all_detections.append(detection)
        
        return all_detections
    
    def filter_relics(self, all_detections):
        excluded_classes = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                          'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'tv', 'laptop', 'mouse',
                          'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                          'hair drier', 'toothbrush'}
        
        high_antiquity_classes = {'bottle', 'wine glass', 'cup', 'bowl', 'vase', 'book', 'clock', 'scissors'}
        
        relic_detections = []
        for detection in all_detections:
            class_name = detection['class_name']
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
            
            if class_name in high_antiquity_classes:
                antiquity_score += 0.4
            
            if antiquity_score >= 0.1:
                detection['antiquity_score'] = antiquity_score
                relic_detections.append(detection)
        
        return relic_detections
    
    def filter_persons_and_dangers(self, all_detections):
        persons = []
        dangers = []
        
        for det in all_detections:
            if det['class_name'] == 'person':
                persons.append(det)
            elif det['class_name'] in DANGEROUS_CLASSES:
                dangers.append(det)
        
        return persons, dangers
    
    def detect_pose(self, frame):
        if not self.pose_landmarker:
            return []
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
            result = self.pose_landmarker.detect_for_video(mp_image, self.pose_timestamp_ms)
            self.pose_timestamp_ms += 1
            
            if not result.pose_landmarks:
                return []
            
            h, w = frame.shape[:2]
            pose_entries = []
            
            for landmarks in result.pose_landmarks:
                points = []
                for landmark in landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append((x, y))
                
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    pose_entries.append({'bbox': bbox, 'points': points})
            
            return pose_entries
        except Exception as e:
            print(f"Pose detection error: {e}")
            return []
    
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
            object_ids = list(self.tracked_objects.keys())
            object_centroids = np.array([self.tracked_objects[obj_id]['centroid'] for obj_id in object_ids])
            
            D = np.linalg.norm(object_centroids[:, np.newaxis] - np.array(input_centroids), axis=2)
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
    
    def match_pose_to_persons(self):
        for person in self.person_detections:
            person['pose_points'] = []
        
        for pose_entry in self.pose_entries:
            best_person = None
            best_iou = 0.0
            
            pose_bbox = pose_entry['bbox']
            for person in self.person_detections:
                person_bbox = person['bbox']
                iou = self.calculate_iou(pose_bbox, person_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_person = person
            
            if best_person and best_iou > 0.05:
                best_person['pose_points'] = pose_entry['points']
    
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def check_risks(self, frame_shape):
        new_alerts = []
        
        for person in self.person_detections:
            person['is_risky'] = False
            person['risk_messages'] = []
            
            person_bbox = person['bbox']
            person_id = person.get('track_id', 'Unknown')
            label = f"人员 {person_id}"
            
            for danger in self.dangerous_detections:
                danger_bbox = danger['bbox']
                if self.calculate_iou(person_bbox, danger_bbox) > 0.1:
                    message = f"{label} 携带 {danger['class_name']}"
                    person['is_risky'] = True
                    person['risk_messages'].append(message)
                    new_alerts.append({'type': 'danger', 'message': message, 'person_id': person_id})
            
            pose_points = person.get('pose_points', [])
            if pose_points:
                for track_id in self.selected_relics:
                    if track_id in self.tracked_objects:
                        relic_bbox = self.tracked_objects[track_id]['bbox']
                        fence = self.calculate_safety_fence(relic_bbox, frame_shape)
                        fence_bbox = fence['fence_bbox']
                        
                        for point in pose_points:
                            if self.point_in_bbox(point, fence_bbox):
                                message = f"{label} 入侵文物安全区"
                                person['is_risky'] = True
                                if message not in person['risk_messages']:
                                    person['risk_messages'].append(message)
                                    new_alerts.append({'type': 'fence', 'message': message, 'person_id': person_id})
                                break
        
        if new_alerts:
            with self.lock:
                self.alerts.extend(new_alerts)
                for alert in new_alerts:
                    self.alert_history.append({'timestamp': time.time(), **alert})
                
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[-50:]
        
        return new_alerts
    
    def point_in_bbox(self, point, bbox):
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
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
    
    def draw_frame(self, frame):
        result_frame = frame.copy()
        
        for detection in self.relic_detections:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection.get('track_id', None)
            
            if track_id is None:
                continue
            
            is_selected = track_id in self.selected_relics
            color = (0, 255, 0) if is_selected else (0, 0, 255)
            thickness = 4 if is_selected else 2
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            label = f"Relic ID:{track_id}"
            cv2.putText(result_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if is_selected:
                fence_info = self.calculate_safety_fence([x1, y1, x2, y2], frame.shape)
                fx1, fy1, fx2, fy2 = fence_info['fence_bbox']
                cv2.rectangle(result_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
        
        for pose_entry in self.pose_entries:
            points = pose_entry['points']
            for a, b in POSE_CONNECTIONS:
                if a < len(points) and b < len(points):
                    pa, pb = points[a], points[b]
                    cv2.line(result_frame, pa, pb, (0, 200, 0), 2)
            
            for x, y in points:
                cv2.circle(result_frame, (x, y), 3, (0, 255, 255), -1)
        
        for person in self.person_detections:
            x1, y1, x2, y2 = person['bbox']
            is_risky = person.get('is_risky', False)
            color = (0, 0, 255) if is_risky else (0, 128, 255)
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            person_id = person.get('track_id', '?')
            label = f"Person {person_id}"
            cv2.putText(result_frame, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            for idx, msg in enumerate(person.get('risk_messages', [])):
                cv2.putText(result_frame, msg, (x1, y2 + 20 + idx * 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        for danger in self.dangerous_detections:
            x1, y1, x2, y2 = danger['bbox']
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = f"{danger['class_name']} {danger['confidence']:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # 状态信息
        status_text = f"Relics: {len(self.selected_relics)} | Persons: {len(self.person_detections)}"
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
            all_detections = self.detect_all_objects(frame)
            
            self.relic_detections = self.filter_relics(all_detections)
            self.person_detections, self.dangerous_detections = self.filter_persons_and_dangers(all_detections)
            
            self.update_tracking(self.relic_detections)
            
            self.pose_entries = self.detect_pose(frame)
            self.match_pose_to_persons()
            
            self.check_risks(frame.shape)
            
            result_frame = self.draw_frame(frame)
        
        return result_frame, self.get_status()
    
    def get_status(self):
        return {
            'relics': [{
                'track_id': det.get('track_id'),
                'confidence': round(det['confidence'] * 100, 1),
                'class_name': det.get('class_name', 'unknown'),
                'selected': det.get('track_id') in self.selected_relics,
                'bbox': det['bbox']
            } for det in self.relic_detections if det.get('track_id') is not None],
            'persons': [{
                'track_id': p.get('track_id'),
                'is_risky': p.get('is_risky', False),
                'risk_messages': p.get('risk_messages', []),
                'has_pose': len(p.get('pose_points', [])) > 0
            } for p in self.person_detections],
            'dangers': [{
                'class_name': d['class_name'],
                'confidence': round(d['confidence'] * 100, 1)
            } for d in self.dangerous_detections]
        }
    
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
            return list(self.alert_history)
