"""Real-time safety monitoring pipeline that merges detection, pose estimation and analytics."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from mediapipe import Image as MPImage, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .models import ensure_models, load_detection_model

COCO_DANGEROUS = {"knife", "scissors", "baseball bat"}
COCO_PERSON_LABEL = "person"
COCO_RELIC_CLASSES = {
    "vase",
    "clock",
    "book",
    "bottle",
    "wine glass",
    "cup",
    "bowl",
    "potted plant",
    "chair",
    "couch",
    "teddy bear",
    "tv",
    "laptop",
    "cell phone",
    "remote",
}


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    score: float
    label: str


@dataclass
class PoseEntry:
    bbox: Tuple[int, int, int, int]
    points: List[Tuple[int, int]]


def bbox_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
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
    if area_a <= 0 or area_b <= 0:
        return 0.0
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def point_in_bbox(point: Tuple[int, int], bbox: Sequence[int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


class PoseLandmarker:
    def __init__(self, model_path: Path):
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            num_poses=4,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> List[PoseEntry]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += 1

        pose_entries: List[PoseEntry] = []
        if not result.pose_landmarks:
            return pose_entries

        h, w = frame.shape[:2]
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
            bbox = (min(xs), min(ys), max(xs), max(ys))
            pose_entries.append(PoseEntry(bbox=bbox, points=points))
        return pose_entries

    def close(self) -> None:
        self._landmarker.close()


class SafetyMonitoringPipeline:
    def __init__(self, video_source: str | int = 0):
        pose_path, detection_path = ensure_models()
        self._pose_helper = PoseLandmarker(pose_path)
        self._detector, self._categories = load_detection_model(detection_path)

        self._video_source = video_source
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        now = time.time()
        self._latest_state: Dict[str, object] = {
            "persons": 0,
            "relics": 0,
            "dangerous": 0,
            "alerts": [],
            "active_intrusions": 0,
            "active_dangerous": 0,
            "fps": 0.0,
            "timestamp": now,
            "session": {
                "start": now,
                "elapsed": 0.0,
                "total_alerts": 0,
                "total_intrusions": 0,
                "total_dangerous": 0,
                "recent_alerts": [],
            },
        }

        self._fps_history: Deque[float] = deque(maxlen=30)
        self._session_start = time.time()
        self._total_alerts = 0
        self._total_intrusions = 0
        self._total_dangerous = 0
        self._alert_history: Deque[Tuple[float, str]] = deque(maxlen=50)
        self._alert_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        test_capture = cv2.VideoCapture(self._video_source)
        if not test_capture.isOpened():
            test_capture.release()
            raise RuntimeError(f"Unable to open video source: {self._video_source}")
        test_capture.release()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._pose_helper.close()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            if self._latest_frame is None:
                return None
            success, buffer = cv2.imencode(".jpg", self._latest_frame)
            if not success:
                return None
            return buffer.tobytes()

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._latest_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        capture = cv2.VideoCapture(self._video_source)
        if not capture.isOpened():
            return

        last_time = time.time()
        while not self._stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.1)
                continue

            start = time.time()
            state, annotated = self._process_frame(frame)
            duration = time.time() - start
            fps = 1.0 / duration if duration > 0 else 0.0
            self._fps_history.append(fps)

            state["fps"] = self._compute_fps()
            state["timestamp"] = time.time()
            state["session"] = {
                "start": self._session_start,
                "elapsed": time.time() - self._session_start,
                "total_alerts": self._total_alerts,
                "total_intrusions": self._total_intrusions,
                "total_dangerous": self._total_dangerous,
                "recent_alerts": list(self._alert_history),
            }

            with self._lock:
                self._latest_frame = annotated
                self._latest_state = state

            elapsed = time.time() - last_time
            if elapsed < 0.03:
                time.sleep(0.03 - elapsed)
            last_time = time.time()

        capture.release()

    def _compute_fps(self) -> float:
        if not self._fps_history:
            return 0.0
        return float(sum(self._fps_history) / len(self._fps_history))

    def _process_frame(self, frame: np.ndarray) -> Tuple[Dict[str, object], np.ndarray]:
        detections = self._run_detection(frame)
        pose_entries = self._pose_helper.detect(frame)

        persons = [det for det in detections if det.label == COCO_PERSON_LABEL]
        dangerous = [det for det in detections if det.label in COCO_DANGEROUS]
        relics = [det for det in detections if det.label in COCO_RELIC_CLASSES]

        live_alerts: List[str] = []
        new_alerts: List[str] = []
        active_intrusions = 0
        active_dangerous = 0

        def register_alert(message: str, *, category: str) -> None:
            nonlocal new_alerts
            live_alerts.append(message)
            now = time.time()
            last = self._alert_cache.get(message, 0.0)
            if now - last > 2.0:
                self._alert_cache[message] = now
                new_alerts.append(message)
                self._alert_history.append((now, message))
                if category == "intrusion":
                    self._total_intrusions += 1
                elif category == "danger":
                    self._total_dangerous += 1

        for person in persons:
            bbox = person.bbox
            intruded = False
            for relic in relics:
                if bbox_iou(bbox, relic.bbox) > 0.15:
                    message = f"人员侵入 {relic.label} 的安全区"
                    register_alert(message, category="intrusion")
                    intruded = True
            for entry in pose_entries:
                if entry.points and any(point_in_bbox(point, bbox) for point in entry.points):
                    for relic in relics:
                        expanded = self._expand_bbox(relic.bbox, frame.shape, 40)
                        if any(point_in_bbox(point, expanded) for point in entry.points):
                            message = f"人员接近 {relic.label}"
                            register_alert(message, category="intrusion")
                            intruded = True
            if intruded:
                active_intrusions += 1

            for danger in dangerous:
                if bbox_iou(bbox, danger.bbox) > 0.05:
                    message = f"检测到疑似危险物品：{danger.label}"
                    register_alert(message, category="danger")
                    active_dangerous += 1

        self._total_alerts += len(new_alerts)

        annotated = self._draw_overlay(frame, persons, relics, dangerous, pose_entries, live_alerts)
        state = {
            "persons": len(persons),
            "relics": len(relics),
            "dangerous": len(dangerous),
            "alerts": live_alerts,
            "active_intrusions": active_intrusions,
            "active_dangerous": active_dangerous,
        }
        return state, annotated

    def _run_detection(self, frame: np.ndarray) -> List[Detection]:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            outputs = self._detector([tensor])[0]

        detections: List[Detection] = []
        boxes = outputs["boxes"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy().astype(int)

        h, w = frame.shape[:2]
        for bbox, score, label_id in zip(boxes, scores, labels):
            if score < 0.6:
                continue
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            label = self._categories[label_id - 1] if 0 < label_id <= len(self._categories) else f"class_{label_id}"
            detections.append(Detection(bbox=(x1, y1, x2, y2), score=float(score), label=label))
        return detections

    def _expand_bbox(self, bbox: Tuple[int, int, int, int], shape: Tuple[int, int, int], margin: int) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        x1, y1, x2, y2 = bbox
        return (
            max(0, x1 - margin),
            max(0, y1 - margin),
            min(w - 1, x2 + margin),
            min(h - 1, y2 + margin),
        )

    def _draw_overlay(
        self,
        frame: np.ndarray,
        persons: Sequence[Detection],
        relics: Sequence[Detection],
        dangerous: Sequence[Detection],
        poses: Sequence[PoseEntry],
        alerts: Sequence[str],
    ) -> np.ndarray:
        canvas = frame.copy()

        for relic in relics:
            x1, y1, x2, y2 = relic.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(canvas, f"Relic: {relic.label}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        for person in persons:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 200, 120), 2)
            cv2.putText(canvas, "Person", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 2)

        for danger in dangerous:
            x1, y1, x2, y2 = danger.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(canvas, f"Danger: {danger.label}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for pose in poses:
            for point in pose.points:
                cv2.circle(canvas, point, 2, (255, 255, 255), -1)

        if alerts:
            panel_height = 20 * len(alerts) + 20
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (canvas.shape[1], panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            for index, message in enumerate(alerts):
                cv2.putText(
                    canvas,
                    message,
                    (10, 20 + index * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        return canvas
