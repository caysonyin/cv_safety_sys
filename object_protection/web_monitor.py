#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""面向Web仪表盘的协同安全监控器封装。"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Optional

import cv2
import numpy as np

from object_protection.integrated_safety_monitor import IntegratedSafetyMonitor


@dataclass
class AlertEvent:
    """表示一次报警事件。"""

    event_id: str
    message: str
    category: str
    timestamp: datetime

    def to_payload(self) -> Dict[str, object]:
        return {
            "id": self.event_id,
            "message": self.message,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
        }


class WebSafetyMonitor(IntegratedSafetyMonitor):
    """为Web端提供流媒体帧与状态数据的协同安全监控器。"""

    def __init__(
        self,
        model,
        device,
        *,
        pose_model_path: str,
        confidence_threshold: float = 0.1,
    ) -> None:
        super().__init__(
            model,
            device,
            pose_model_path=pose_model_path,
            confidence_threshold=confidence_threshold,
            display_window=False,
        )

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._video_source: int | str = 0
        self._placeholder_frame = self._build_placeholder_frame()
        self._latest_frame: bytes = self._placeholder_frame
        self._latest_frame_time: float = 0.0
        self._frame_timestamps: Deque[float] = deque(maxlen=120)
        self._event_log: Deque[AlertEvent] = deque(maxlen=500)
        self._live_alerts: List[Dict[str, object]] = []
        self._relic_snapshot: List[Dict[str, object]] = []
        self._people_snapshot: Dict[str, int] = {"total": 0, "risky": 0}
        self._dangerous_count = 0
        self._status_message: Optional[str] = None

    # ------------------------------------------------------------------
    # 启动与停止
    # ------------------------------------------------------------------
    def start(self, video_source: int | str = 0) -> None:
        """启动后台监控线程。"""

        with self._lock:
            if self._running:
                return
            self._running = True
            self._video_source = video_source

        self._thread = threading.Thread(target=self._loop, name="web-monitor", daemon=True)
        self._thread.start()

    def stop(self, *, close_pose_helper: bool = False) -> None:
        with self._lock:
            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if close_pose_helper:
            self.pose_helper.close()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_latest_frame(self) -> bytes:
        with self._lock:
            return self._latest_frame

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            now = datetime.now(timezone.utc)
            daily_cutoff = now - timedelta(days=1)
            weekly_cutoff = now - timedelta(days=7)

            daily_events = [event for event in self._event_log if event.timestamp >= daily_cutoff]
            weekly_events = [event for event in self._event_log if event.timestamp >= weekly_cutoff]

            def summarise(events: List[AlertEvent]) -> Dict[str, int]:
                summary = {"total": len(events), "intrusions": 0, "dangerous": 0, "other": 0}
                for evt in events:
                    if evt.category == "intrusion":
                        summary["intrusions"] += 1
                    elif evt.category == "danger":
                        summary["dangerous"] += 1
                    else:
                        summary["other"] += 1
                return summary

            fps = 0.0
            if len(self._frame_timestamps) > 1:
                duration = self._frame_timestamps[-1] - self._frame_timestamps[0]
                if duration > 0:
                    fps = (len(self._frame_timestamps) - 1) / duration

            session_elapsed = 0.0
            if self.monitoring_active:
                session_elapsed = time.time() - self.session_start_time

            alert_history_payload = [event.to_payload() for event in list(self._event_log)[-12:]][::-1]

            return {
                "stage": self.workflow_stage,
                "monitoring_active": self.monitoring_active,
                "session_seconds": session_elapsed,
                "session_duration": self._format_duration(session_elapsed),
                "status_message": self._status_message,
                "totals": {
                    "alerts": self.total_alerts,
                    "intrusions": self.total_intrusions,
                    "dangerous_flags": self.total_dangerous_flags,
                },
                "live_alerts": list(self._live_alerts),
                "alert_history": alert_history_payload,
                "relics": list(self._relic_snapshot),
                "people": dict(self._people_snapshot),
                "dangerous_items": self._dangerous_count,
                "daily_summary": summarise(daily_events),
                "weekly_summary": summarise(weekly_events),
                "frame_fps": fps,
                "last_frame_time": self._latest_frame_time,
                "selected_relics": sorted(int(rid) for rid in self.selected_relics if rid is not None),
            }

    def set_selected_relics(self, relic_ids: Iterable[int]) -> List[int]:
        valid_ids = {int(rid) for rid in relic_ids}
        with self._lock:
            self.selected_relics = valid_ids
            return sorted(valid_ids)

    def select_all_relics(self) -> List[int]:
        with self._lock:
            ids = {int(item["id"]) for item in self._relic_snapshot if item["id"] is not None}
            self.selected_relics = ids
            return sorted(ids)

    # ------------------------------------------------------------------
    # 内部逻辑
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.is_running():
            source = self._video_source
            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                self._update_status_message(f"无法打开视频源: {source}")
                self._publish_placeholder()
                time.sleep(1.0)
                continue

            self._update_status_message(None)
            if self.workflow_stage != "monitoring":
                self._change_stage("monitoring")

            frame_count = 0

            while self.is_running():
                ret, frame = cap.read()
                if not ret:
                    self._update_status_message("视频源中断，正在尝试重新连接…")
                    self._publish_placeholder()
                    time.sleep(0.5)
                    break

                frame_count += 1
                start_time = time.time()

                all_detections = self._detect_all_objects(frame)
                self.relic_detections = self.detect_relics(frame, all_detections)
                self.update_tracking(self.relic_detections)
                self._update_person_detections(all_detections)

                pose_entries = self.pose_helper.detect(frame)
                self._match_pose_to_persons(pose_entries)
                self._build_active_fences(frame.shape)

                alerts = self._analyse_risks()

                canvas = self.draw_detections(frame)
                self._draw_active_fence_overlay(canvas)
                self._draw_pose(canvas, pose_entries)
                self._draw_persons(canvas)
                self._draw_dangerous_items(canvas)
                self._draw_alert_panel(canvas, alerts)

                if alerts:
                    self.total_alerts += len(alerts)
                    for alert in alerts:
                        if "侵入" in alert:
                            self.total_intrusions += 1
                        if "携带" in alert:
                            self.total_dangerous_flags += 1

                cv2.putText(
                    canvas,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                self._draw_header(canvas, self.workflow_stage)
                self._draw_monitoring_summary(canvas)
                self._render_toast(canvas)

                success, buffer = cv2.imencode(".jpg", canvas)
                if success:
                    frame_bytes = buffer.tobytes()
                else:
                    frame_bytes = self._placeholder_frame

                frame_ready_time = time.time()
                self._frame_timestamps.append(frame_ready_time)
                while (
                    len(self._frame_timestamps) > 1
                    and frame_ready_time - self._frame_timestamps[0] > 5.0
                ):
                    self._frame_timestamps.popleft()

                self._publish_frame(
                    frame_bytes,
                    frame_ready_time,
                    alerts,
                )

                inference_time = time.time() - start_time
                if inference_time < 0.03:
                    time.sleep(0.03 - inference_time)

            cap.release()

    def _publish_placeholder(self) -> None:
        with self._lock:
            self._latest_frame = self._placeholder_frame
            self._latest_frame_time = time.time()
            self._live_alerts = []

    def _publish_frame(
        self,
        frame_bytes: bytes,
        timestamp: float,
        alerts: List[str],
    ) -> None:
        now = datetime.now(timezone.utc)
        live_alerts: List[Dict[str, object]] = []

        for index, message in enumerate(alerts):
            category = self._classify_alert(message)
            event_id = f"{int(timestamp * 1000)}-{index}"
            event = AlertEvent(event_id=event_id, message=message, category=category, timestamp=now)
            self._event_log.append(event)
            live_alerts.append(event.to_payload())

        relic_snapshot: List[Dict[str, object]] = []
        selected_ids = {int(rid) for rid in self.selected_relics if rid is not None}
        for detection in self.relic_detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            relic_snapshot.append(
                {
                    "id": int(track_id),
                    "label": str(detection.get("class_name", "relic")),
                    "confidence": round(float(detection.get("confidence", 0.0)), 3),
                    "score": round(float(detection.get("antiquity_score", 0.0)), 3),
                    "is_selected": int(track_id) in selected_ids,
                }
            )

        risky_people = sum(1 for person in self.person_detections if person.get("is_risky"))
        people_snapshot = {"total": len(self.person_detections), "risky": risky_people}

        with self._lock:
            self._latest_frame = frame_bytes
            self._latest_frame_time = timestamp
            self._live_alerts = live_alerts
            self._relic_snapshot = relic_snapshot
            self._people_snapshot = people_snapshot
            self._dangerous_count = len(self.dangerous_detections)

    def _update_status_message(self, message: Optional[str]) -> None:
        with self._lock:
            self._status_message = message

    @staticmethod
    def _classify_alert(message: str) -> str:
        if "侵入" in message:
            return "intrusion"
        if "携带" in message or "危险" in message:
            return "danger"
        return "notice"

    @staticmethod
    def _build_placeholder_frame() -> bytes:
        canvas = np.zeros((480, 720, 3), dtype=np.uint8)
        canvas[:] = (28, 32, 45)
        cv2.putText(
            canvas,
            "等待视频流...",
            (140, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 170, 255),
            2,
        )
        cv2.putText(
            canvas,
            "启动后将自动显示实时画面",
            (70, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
        success, buffer = cv2.imencode(".jpg", canvas)
        if not success:
            raise RuntimeError("无法创建占位帧")
        return buffer.tobytes()

