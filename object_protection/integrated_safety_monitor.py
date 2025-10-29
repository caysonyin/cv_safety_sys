#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文物安全协同监控系统

将文物检测/电子栅栏、人体姿态识别与危险物品检测整合在同一视频流中，
实现风险人物联动报警与可视化提示。
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from mediapipe import Image as MPImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from object_protection.video_relic_tracking import (
    SimpleTracker,
    VideoRelicTracker,
    download_yolov7_tiny,
    load_model,
)
from WebcamPoseDetection.download_model import download_model as download_pose_model


POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
DANGEROUS_CLASSES = {
    'knife',
    'scissors',
    'baseball bat',
}


def point_in_bbox(point: Tuple[int, int], bbox: Sequence[int]) -> bool:
    """判断像素点是否落在边界框内。"""

    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    """计算两个边界框的IoU。"""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class PoseEntry:
    bbox: List[int]
    points: List[Tuple[int, int]]


class PoseLandmarkHelper:
    """轻量封装的MediaPipe姿态检测器，返回关节点坐标。"""

    def __init__(self, model_path: str):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            num_poses=5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> List[PoseEntry]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        self.timestamp_ms += 1

        if not result.pose_landmarks:
            return []

        h, w = frame.shape[:2]
        pose_entries: List[PoseEntry] = []

        for landmarks in result.pose_landmarks:
            xs: List[int] = []
            ys: List[int] = []
            points: List[Tuple[int, int]] = []

            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                xs.append(x)
                ys.append(y)
                points.append((x, y))

            if not xs or not ys:
                continue

            bbox = [min(xs), min(ys), max(xs), max(ys)]
            pose_entries.append(PoseEntry(bbox=bbox, points=points))

        return pose_entries

    def close(self):
        self.landmarker.close()


class IntegratedSafetyMonitor(VideoRelicTracker):
    """同时处理文物、人员与危险物品的协同安全监控器。"""

    def __init__(
        self,
        model,
        device,
        *,
        pose_model_path: str,
        confidence_threshold: float = 0.1,
    ):
        super().__init__(
            model,
            device,
            confidence_threshold=confidence_threshold,
            window_name="文物安全协同防护系统",
        )

        self.pose_helper = PoseLandmarkHelper(pose_model_path)
        self.person_tracker = SimpleTracker(max_disappeared=15, max_distance=120.0)
        self.person_tracks: Dict[int, Dict[str, object]] = {}
        self.person_detections: List[Dict[str, object]] = []
        self.dangerous_detections: List[Dict[str, object]] = []
        self.active_fences: List[Dict[str, object]] = []
        self.alert_history: Deque[Tuple[float, str]] = deque(maxlen=12)

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    def _update_person_detections(self, detections: Iterable[Dict[str, object]]) -> None:
        persons: List[Dict[str, object]] = []
        dangers: List[Dict[str, object]] = []

        for det in detections:
            class_name = str(det['class_name'])
            if class_name == 'person':
                persons.append({**det})
            elif class_name in DANGEROUS_CLASSES:
                dangers.append({**det})

        self.person_detections = persons
        self.dangerous_detections = dangers

        self.person_tracks = self.person_tracker.update(self.person_detections)

        if not self.person_tracks:
            for det in self.person_detections:
                det['track_id'] = None
            return

        tracked_ids = list(self.person_tracks.keys())
        tracked_centroids = np.array(
            [self.person_tracks[idx]['centroid'] for idx in tracked_ids],
            dtype=np.float32,
        )

        for det in self.person_detections:
            x1, y1, x2, y2 = det['bbox']
            centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            distances = np.linalg.norm(tracked_centroids - centroid, axis=1)
            best_index = distances.argmin()
            if distances[best_index] < 80:
                det['track_id'] = tracked_ids[best_index]
            else:
                det['track_id'] = None

    def _match_pose_to_persons(self, pose_entries: Sequence[PoseEntry]) -> None:
        for det in self.person_detections:
            det['pose_points'] = []

        for entry in pose_entries:
            best_person = None
            best_score = 0.0

            for det in self.person_detections:
                score = bbox_iou(det['bbox'], entry.bbox)
                if score > best_score:
                    best_score = score
                    best_person = det

            if best_person is not None and best_score > 0.05:
                best_person['pose_points'] = entry.points

    def _build_active_fences(self, frame_shape: Tuple[int, int, int]) -> None:
        fences: List[Dict[str, object]] = []
        for detection in self.relic_detections:
            track_id = detection.get('track_id')
            if track_id is None or track_id not in self.selected_relics:
                continue

            fence_info = self.calculate_safety_fence(detection['bbox'], frame_shape)
            detection['fence_info'] = fence_info
            fences.append(
                {
                    'bbox': fence_info['fence_bbox'],
                    'track_id': track_id,
                    'label': detection.get('class_name', 'relic'),
                }
            )

        self.active_fences = fences

    # ------------------------------------------------------------------
    # 风险分析
    # ------------------------------------------------------------------
    def _analyse_risks(self) -> List[str]:
        alerts: List[str] = []

        for person in self.person_detections:
            person['is_risky'] = False
            person['risk_messages'] = []

        for person in self.person_detections:
            points = person.get('pose_points', [])
            bbox = person['bbox']
            person_id = person.get('track_id')
            label = f"人员 {person_id}" if person_id is not None else "人员"

            # 危险物品绑定
            for danger in self.dangerous_detections:
                danger_bbox = danger['bbox']
                overlap = bbox_iou(bbox, danger_bbox)
                keypoint_overlap = any(point_in_bbox(pt, danger_bbox) for pt in points)

                if overlap > 0.05 or keypoint_overlap:
                    message = f"{label} 携带疑似 {danger['class_name']}"
                    person['is_risky'] = True
                    person['risk_messages'].append(message)
                    alerts.append(message)

            # 电子栅栏入侵
            for fence in self.active_fences:
                if not points:
                    continue
                if any(point_in_bbox(pt, fence['bbox']) for pt in points):
                    message = f"{label} 侵入 {fence['label']} 安全栅栏"
                    person['is_risky'] = True
                    person['risk_messages'].append(message)
                    alerts.append(message)

        return alerts

    # ------------------------------------------------------------------
    # 绘制与展示
    # ------------------------------------------------------------------
    def _draw_pose(self, frame: np.ndarray, pose_entries: Sequence[PoseEntry]) -> None:
        for entry in pose_entries:
            points = entry.points
            if not points:
                continue

            for a, b in POSE_CONNECTIONS:
                if a < len(points) and b < len(points):
                    pa = points[a]
                    pb = points[b]
                    cv2.line(frame, pa, pb, (0, 200, 0), 2)

            for x, y in points:
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    def _draw_persons(self, frame: np.ndarray) -> None:
        for person in self.person_detections:
            x1, y1, x2, y2 = map(int, person['bbox'])
            color = (0, 128, 255)
            if person.get('is_risky'):
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = person.get('track_id')
            if label is not None:
                text = f"Person {label}"
            else:
                text = "Person"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            for idx, message in enumerate(person.get('risk_messages', [])):
                cv2.putText(
                    frame,
                    message,
                    (x1, y2 + 20 + idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

    def _draw_dangerous_items(self, frame: np.ndarray) -> None:
        for danger in self.dangerous_detections:
            x1, y1, x2, y2 = map(int, danger['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = f"{danger['class_name']} {danger['confidence']:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
            )

    def _draw_alert_panel(self, frame: np.ndarray, alerts: Sequence[str]) -> None:
        timestamp = time.time()

        for alert in alerts:
            self.alert_history.append((timestamp, alert))

        h, w = frame.shape[:2]
        panel_width = min(320, w // 3)
        panel_x1 = w - panel_width - 10
        panel_y1 = 10
        panel_x2 = w - 10
        panel_y2 = panel_y1 + 160

        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), -1)
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (200, 200, 200), 2)

        cv2.putText(
            frame,
            "安全状态面板",
            (panel_x1 + 10, panel_y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        info_lines = [
            f"选中文物: {len(self.selected_relics)}",
            f"检测到人员: {len(self.person_detections)}",
            f"危险物品: {len(self.dangerous_detections)}",
        ]

        for idx, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (panel_x1 + 10, panel_y1 + 55 + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        recent_alerts = [msg for _, msg in reversed(self.alert_history)]
        recent_alerts = recent_alerts[:4]

        base_y = panel_y1 + 120
        if recent_alerts:
            cv2.putText(
                frame,
                "实时报警:",
                (panel_x1 + 10, base_y - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
            )

        for idx, msg in enumerate(recent_alerts):
            cv2.putText(
                frame,
                msg,
                (panel_x1 + 10, base_y + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self, video_source: int | str = 0) -> None:
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return

        frame_count = 0

        print("=== 文物安全协同防护系统 ===")
        print("操作提示: 点击选中文物，Enter确认，ESC退出，S保存当前帧")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                all_detections = self._detect_all_objects(frame)
                self.relic_detections = self.detect_relics(frame, all_detections)
                self.update_tracking(self.relic_detections)
                self._update_person_detections(all_detections)

                pose_entries = self.pose_helper.detect(frame)
                self._match_pose_to_persons(pose_entries)
                self._build_active_fences(frame.shape)
                alerts = self._analyse_risks()

                canvas = self.draw_detections(frame)
                self._draw_pose(canvas, pose_entries)
                self._draw_persons(canvas)
                self._draw_dangerous_items(canvas)
                self._draw_alert_panel(canvas, alerts)

                cv2.putText(
                    canvas,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == 13:
                    print(f"确认选择 {len(self.selected_relics)} 个文物")
                if key in (ord('s'), ord('S')):
                    filename = f"integrated_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, canvas)
                    print(f"保存帧到: {filename}")

        finally:
            cap.release()
            self.pose_helper.close()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文物安全协同防护系统")
    parser.add_argument('--source', type=str, default='0', help='视频源(0=摄像头或视频文件路径)')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO置信度阈值')
    parser.add_argument('--pose-model', type=str, default='models/pose_landmarker_full.task', help='姿态模型路径')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 准备姿态模型
    pose_model_path = Path(args.pose_model)
    if not pose_model_path.exists():
        downloaded = download_pose_model()
        if downloaded is None:
            print("无法准备姿态模型，请先运行 WebcamPoseDetection/download_model.py")
            return
        pose_model_path = Path(downloaded)

    # 准备YOLO模型
    model_path = download_yolov7_tiny()
    if model_path is None:
        return

    model, device = load_model(model_path)
    if model is None or device is None:
        return

    video_source: int | str = int(args.source) if args.source.isdigit() else args.source

    monitor = IntegratedSafetyMonitor(
        model,
        device,
        pose_model_path=str(pose_model_path),
        confidence_threshold=args.conf,
    )

    try:
        monitor.run(video_source)
    except KeyboardInterrupt:
        print("\n程序已中断")


if __name__ == '__main__':
    main()

