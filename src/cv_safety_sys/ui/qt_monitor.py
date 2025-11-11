#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 PySide6 的本地可视化客户端，用于展示文物安全协同防护系统。"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl
from PySide6.QtGui import QImage, QPixmap, QColor, QPalette
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cv_safety_sys.monitoring.integrated_monitor import IntegratedSafetyMonitor
from cv_safety_sys.pose.model_downloader import (
    DEFAULT_MODEL_PATH as DEFAULT_POSE_MODEL_PATH,
    download_model as download_pose_model,
)
from cv_safety_sys.detection.yolov7_tracker import (
    DEFAULT_YOLO_MODEL_PATH,
    download_yolov7_tiny,
    load_model,
)


class VideoLabel(QLabel):
    """自适应缩放的视频显示组件，并支持点击映射。"""

    clicked = Signal(int, int)
    pressed = Signal(int, int)
    dragged = Signal(int, int)
    released = Signal(int, int, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._scaled_size = QSize()
        self._mouse_pressed = False
        self._dragging = False
        self._drag_threshold = 5
        self._press_pos: tuple[int, int] | None = None
        self._suppress_click = False
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(
            "background-color: #141821; border: 1px solid #222836; border-radius: 6px;"
        )

    def setImage(self, image: QImage) -> None:
        self._image = image
        self._update_pixmap()

    def clearFrame(self) -> None:
        self._image = None
        self._scaled_size = QSize()
        self.clear()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._image is not None:
            self._update_pixmap()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.LeftButton:
            return

        mapped = self._map_to_image(event.pos())
        if mapped is None:
            return
        self._mouse_pressed = True
        self._dragging = False
        self._press_pos = mapped
        self._suppress_click = False
        self.pressed.emit(*mapped)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not self._mouse_pressed or not (event.buttons() & Qt.LeftButton):
            return
        mapped = self._map_to_image(event.pos())
        if mapped is None:
            return
        if not self._dragging and self._press_pos is not None:
            dx = abs(mapped[0] - self._press_pos[0])
            dy = abs(mapped[1] - self._press_pos[1])
            if max(dx, dy) >= self._drag_threshold:
                self._dragging = True
        if self._dragging:
            self.dragged.emit(*mapped)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if not self._mouse_pressed:
            return
        mapped = self._map_to_image(event.pos())
        if mapped is None:
            mapped = self._press_pos
        was_dragging = self._dragging
        if mapped is not None:
            self.released.emit(mapped[0], mapped[1], was_dragging)
            if not was_dragging and not self._suppress_click:
                self.clicked.emit(mapped[0], mapped[1])
        self._mouse_pressed = False
        self._dragging = False
        self._press_pos = None
        self._suppress_click = False

    def suppress_next_click(self) -> None:
        self._suppress_click = True

    def _map_to_image(self, pos) -> tuple[int, int] | None:
        if self._image is None:
            return None
        label_width = self.width()
        label_height = self.height()
        scaled_width = self._scaled_size.width()
        scaled_height = self._scaled_size.height()
        offset_x = (label_width - scaled_width) / 2
        offset_y = (label_height - scaled_height) / 2

        x = pos.x()
        y = pos.y()
        if not (
            offset_x <= x <= offset_x + scaled_width
            and offset_y <= y <= offset_y + scaled_height
        ):
            return None

        rel_x = (x - offset_x) / max(1.0, scaled_width)
        rel_y = (y - offset_y) / max(1.0, scaled_height)

        img_w = self._image.width()
        img_h = self._image.height()
        mapped_x = int(rel_x * img_w)
        mapped_y = int(rel_y * img_h)
        return mapped_x, mapped_y
    def _update_pixmap(self) -> None:
        if self._image is None:
            self.clear()
            return
        pixmap = QPixmap.fromImage(self._image)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_size = scaled.size()
        self.setPixmap(scaled)


class AlertBannerWidget(QWidget):
    """可视化报警区，展示安全/报警状态与详细列表。"""

    alert_selected = Signal(int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.icon_label = QLabel("SAFE")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedSize(72, 72)
        self.icon_label.setStyleSheet(
            "background-color: #143027; color: #63f5a8; border-radius: 12px; font-weight: 600;"
        )

        self.headline_label = QLabel("环境安全，持续守护")
        headline_font = self.headline_label.font()
        headline_font.setPointSize(18)
        headline_font.setBold(True)
        self.headline_label.setFont(headline_font)
        self.headline_label.setStyleSheet("color: #63f5a8;")

        self.details_list = QListWidget()
        self.details_list.setFixedHeight(140)
        self.details_list.setStyleSheet(
            "background-color: #11141e; border: 1px solid #1f2433; border-radius: 6px; color: #f8b4b4;"
        )

        text_layout = QVBoxLayout()
        text_layout.addWidget(self.headline_label)
        text_layout.addWidget(self.details_list)

        root = QHBoxLayout()
        root.setSpacing(16)
        root.addWidget(self.icon_label)
        root.addLayout(text_layout, stretch=1)
        self.setLayout(root)
        self.details_list.itemClicked.connect(self._on_item_clicked)
        self._current_alerts: List[Dict[str, object]] = []

    def update_alerts(self, alerts: Sequence[Dict[str, object]]) -> None:
        self._current_alerts = list(alerts)
        if self._current_alerts:
            self._set_alert_state(self._current_alerts)
        else:
            self._set_safe_state()

    def _set_safe_state(self) -> None:
        self.icon_label.setText("SAFE")
        self.icon_label.setStyleSheet(
            "background-color: #143027; color: #63f5a8; border-radius: 12px; font-weight: 600;"
        )
        self.headline_label.setText("环境安全，持续守护")
        self.headline_label.setStyleSheet("color: #63f5a8;")
        self.details_list.clear()
        item = QListWidgetItem("暂无报警事件。")
        item.setForeground(QColor("#63f5a8"))
        self.details_list.addItem(item)

    def _set_alert_state(self, alerts: Sequence[Dict[str, object]]) -> None:
        self.icon_label.setText("ALERT")
        self.icon_label.setStyleSheet(
            "background-color: #4c0b12; color: #ff7b8a; border-radius: 12px; font-weight: 600;"
        )
        self.headline_label.setText("警报触发，请立即处理")
        self.headline_label.setStyleSheet("color: #ff7b8a;")
        self.details_list.clear()
        for alert in alerts:
            messages = alert.get('messages', [])
            label = alert.get('label', '报警')
            summary = label
            if messages:
                summary += f" - {messages[0]}"
                if len(messages) > 1:
                    summary += f" (+{len(messages) - 1})"
            item = QListWidgetItem(summary)
            severity = alert.get('severity', 'intrusion')
            color = "#ff7b8a" if severity == 'danger' else "#ffc16b"
            item.setForeground(QColor(color))
            item.setData(Qt.UserRole, alert.get('track_id'))
            self.details_list.addItem(item)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        track_id = item.data(Qt.UserRole)
        if track_id is None:
            return
        self.alert_selected.emit(int(track_id), item.text())


class MonitorWorker(QThread):
    frame_ready = Signal(np.ndarray, dict)
    alerts_emitted = Signal(list)
    error_occurred = Signal(str)

    def __init__(
        self,
        monitor: IntegratedSafetyMonitor,
        video_source: int | str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.video_source = video_source
        self.monitor_lock = threading.Lock()
        self._running = False

    def run(self) -> None:  # type: ignore[override]
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.error_occurred.emit(f"无法打开视频源: {self.video_source}")
            return

        self._running = True
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                with self.monitor_lock:
                    result = self.monitor.process_frame(frame)
                rgb_frame = cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb_frame, result['status'])
                self.alerts_emitted.emit(result['alerts'])
                self.msleep(10)
        except Exception as exc:  # pragma: no cover - UI runtime safeguard
            self.error_occurred.emit(str(exc))
        finally:
            cap.release()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


class SafetyMonitorWindow(QMainWindow):
    """PySide6 主窗口，展示实时视频与安全状态。"""

    def __init__(
        self,
        monitor: IntegratedSafetyMonitor,
        video_source: int | str,
        alert_sound: Path | None = None,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.alert_sound_path = alert_sound
        self.worker = MonitorWorker(self.monitor, video_source)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.alerts_emitted.connect(self.on_alerts_emitted)
        self.worker.error_occurred.connect(self.on_worker_error)

        self.video_label = VideoLabel()
        self.video_label.clicked.connect(self.on_video_clicked)
        self.video_label.pressed.connect(self.on_video_pressed)
        self.video_label.dragged.connect(self.on_video_dragged)
        self.video_label.released.connect(self.on_video_released)

        self.stage_value = QLabel("文物选择阶段")
        self.session_value = QLabel("00:00")
        self.person_value = QLabel("0")
        self.relic_value = QLabel("0")
        self.relic_ids_value = QLabel("-")
        self.alert_total_value = QLabel("0")
        self.intrusion_value = QLabel("0")
        self.dangerous_value = QLabel("0")
        self.fence_value = QLabel("0")
        self.toast_label = QLabel()
        self.toast_label.setWordWrap(True)
        self.toast_label.hide()

        self.alerts_list = QListWidget()
        self.alerts_list.setStyleSheet("background-color: #1e2431; border: none; color: #ff9c9c;")
        self.alert_banner = AlertBannerWidget()
        self.alert_banner.alert_selected.connect(self.on_alert_item_clicked)

        self.latest_status: Dict[str, object] = {}
        self.current_frame: np.ndarray | None = None
        self.last_alert_sound_time = 0.0
        self._fence_drag_state: Dict[str, object] | None = None

        self._setup_palette()
        central = QWidget()
        central.setLayout(self._build_layout())
        self.setCentralWidget(central)
        self.setWindowTitle("文物安全协同防护 - 本地客户端")
        self.resize(1280, 720)

        self.sound_effect: QSoundEffect | None = None
        self._prepare_sound()
        self.worker.start()

    def _setup_palette(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(18, 22, 30))
        palette.setColor(QPalette.Base, QColor(25, 31, 43))
        palette.setColor(QPalette.AlternateBase, QColor(28, 34, 45))
        palette.setColor(QPalette.WindowText, QColor(230, 233, 240))
        self.setPalette(palette)

    def _build_layout(self) -> QHBoxLayout:
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        root_layout.addWidget(self.video_label, stretch=3)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(16)

        status_group = QGroupBox("实时状态")
        status_group.setStyleSheet("QGroupBox { color: #cfd8ff; font-weight: bold; }")
        status_form = QFormLayout()
        status_form.addRow("当前阶段", self.stage_value)
        status_form.addRow("监控时长", self.session_value)
        status_form.addRow("在场人员", self.person_value)
        status_form.addRow("监控文物", self.relic_value)
        status_form.addRow("文物编号", self.relic_ids_value)
        status_form.addRow("活动栅栏", self.fence_value)
        status_form.addRow("累计报警", self.alert_total_value)
        status_form.addRow("栅栏入侵", self.intrusion_value)
        status_form.addRow("危险携带", self.dangerous_value)
        status_group.setLayout(status_form)

        alert_group = QGroupBox("最新报警")
        alert_group.setStyleSheet("QGroupBox { color: #ffb0b0; font-weight: bold; }")
        alert_layout = QVBoxLayout()
        alert_layout.addWidget(self.alerts_list)
        alert_group.setLayout(alert_layout)

        banner_group = QGroupBox("报警提醒")
        banner_group.setStyleSheet("QGroupBox { color: #f7d27b; font-weight: bold; }")
        banner_layout = QVBoxLayout()
        banner_layout.addWidget(self.alert_banner)
        banner_group.setLayout(banner_layout)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("开始监控")
        self.start_button.clicked.connect(self.on_start_monitoring)
        self.reset_button = QPushButton("返回选择")
        self.reset_button.clicked.connect(self.on_reset_selection)
        self.clear_button = QPushButton("清空选中")
        self.clear_button.clicked.connect(self.on_clear_selection)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.clear_button)

        snapshot_row = QHBoxLayout()
        self.snapshot_button = QPushButton("保存快照")
        self.snapshot_button.clicked.connect(self.on_save_snapshot)
        self.exit_button = QPushButton("退出")
        self.exit_button.clicked.connect(self.close)
        snapshot_row.addWidget(self.snapshot_button)
        snapshot_row.addWidget(self.exit_button)

        sidebar.addWidget(status_group)
        sidebar.addWidget(alert_group)
        sidebar.addWidget(banner_group)
        sidebar.addWidget(self.toast_label)
        sidebar.addLayout(button_row)
        sidebar.addLayout(snapshot_row)
        sidebar.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        root_layout.addLayout(sidebar, stretch=2)
        return root_layout

    def _prepare_sound(self) -> None:
        if self.alert_sound_path and self.alert_sound_path.exists():
            sound_effect = QSoundEffect(self)
            sound_effect.setSource(
                QUrl.fromLocalFile(str(self.alert_sound_path.resolve()))
            )
            sound_effect.setVolume(0.6)
            self.sound_effect = sound_effect
        else:  # fallback to simple beep when no custom sound file provided
            self.sound_effect = None

    def on_frame_ready(self, rgb_frame: np.ndarray, status: Dict[str, object]) -> None:
        self.latest_status = status
        self.current_frame = rgb_frame
        image = QImage(
            rgb_frame.data,
            rgb_frame.shape[1],
            rgb_frame.shape[0],
            rgb_frame.strides[0],
            QImage.Format_RGB888,
        )
        self.video_label.setImage(image)
        self._update_status_panel(status)

    def on_alerts_emitted(self, alerts: List[str]) -> None:
        if not alerts:
            return
        now = time.time()
        if now - self.last_alert_sound_time < 0.8:
            return
        self.last_alert_sound_time = now
        if self.sound_effect is not None:
            self.sound_effect.play()
        else:
            QApplication.beep()

    def on_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "运行错误", message)

    def on_video_pressed(self, x: int, y: int) -> None:
        if self.latest_status.get('stage') != 'selection':
            self._fence_drag_state = None
            return
        with self.worker.monitor_lock:
            handle = self.monitor.pick_fence_handle(x, y)
        if handle:
            self._fence_drag_state = {
                'track_id': handle['track_id'],
                'hit': handle.get('hit', {}),
                'start_bbox': list(handle['bbox']),
            }
            self.video_label.suppress_next_click()
        else:
            self._fence_drag_state = None

    def on_video_dragged(self, x: int, y: int) -> None:
        if not self._fence_drag_state:
            return
        new_bbox = self._compose_drag_bbox(x, y)
        if new_bbox is None:
            return
        with self.worker.monitor_lock:
            updated = self.monitor.adjust_fence_bbox(
                self._fence_drag_state['track_id'],
                new_bbox,
            )
        if updated is not None:
            self._fence_drag_state['latest_bbox'] = updated

    def on_video_released(self, x: int, y: int, was_dragging: bool) -> None:
        if self._fence_drag_state:
            self._fence_drag_state = None

    def on_video_clicked(self, x: int, y: int) -> None:
        if self.latest_status.get('stage') != 'selection':
            QMessageBox.information(self, "提示", "请先返回文物选择阶段再调整选中目标。")
            return
        with self.worker.monitor_lock:
            self.monitor.handle_click(x, y)

    def on_alert_item_clicked(self, track_id: int, summary: str) -> None:
        reply = QMessageBox.question(
            self,
            "处理警报",
            f"{summary}\n\n是否解除该警报并恢复常规跟踪？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        with self.worker.monitor_lock:
            cleared = self.monitor.acknowledge_alert(track_id)
        if cleared:
            QMessageBox.information(self, "完成", "已解除该警报。")
        else:
            QMessageBox.information(self, "提示", "未找到对应警报，可能已被解除。")

    def _compose_drag_bbox(self, x: int, y: int) -> List[int] | None:
        state = self._fence_drag_state
        if not state:
            return None
        bbox = list(state.get('start_bbox', []))
        if len(bbox) != 4:
            return None
        hit = state.get('hit', {})
        handle_name = str(hit.get('name', ''))
        handle_kind = hit.get('kind')

        if handle_kind == 'corner':
            if 'left' in handle_name:
                bbox[0] = x
            if 'right' in handle_name:
                bbox[2] = x
            if 'top' in handle_name:
                bbox[1] = y
            if 'bottom' in handle_name:
                bbox[3] = y
        elif handle_kind == 'edge':
            if handle_name == 'left':
                bbox[0] = x
            elif handle_name == 'right':
                bbox[2] = x
            elif handle_name == 'top':
                bbox[1] = y
            elif handle_name == 'bottom':
                bbox[3] = y
        else:
            return None
        return [int(v) for v in bbox]

    def on_start_monitoring(self) -> None:
        with self.worker.monitor_lock:
            started = self.monitor.start_monitoring()
        if started:
            self.start_button.setEnabled(False)

    def on_reset_selection(self) -> None:
        with self.worker.monitor_lock:
            self.monitor.enter_selection_mode()
        self.start_button.setEnabled(True)

    def on_clear_selection(self) -> None:
        with self.worker.monitor_lock:
            self.monitor.clear_selection()

    def on_save_snapshot(self) -> None:
        if self.current_frame is None:
            QMessageBox.information(self, "提示", "暂无可保存的画面。")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("snapshots")
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"snapshot_{timestamp}.jpg"
        bgr_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filename), bgr_frame)
        QMessageBox.information(self, "保存成功", f"已保存到 {filename}")

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        mins, secs = divmod(seconds, 60)
        return f"{mins:02d}:{secs:02d}"

    def _update_status_panel(self, status: Dict[str, object]) -> None:
        stage_map = {
            'selection': "文物选择阶段",
            'monitoring': "实时监控阶段",
        }
        stage = status.get('stage', 'selection')
        self.stage_value.setText(stage_map.get(stage, str(stage)))
        duration = float(status.get('session_duration', 0.0))
        self.session_value.setText(self._format_duration(duration))
        self.person_value.setText(str(status.get('person_count', 0)))
        selected = status.get('selected_relics', [])
        self.relic_value.setText(str(len(selected)))
        self.relic_ids_value.setText(
            ", ".join(str(idx) for idx in selected) if selected else "-"
        )
        self.fence_value.setText(str(status.get('fence_count', 0)))
        self.alert_total_value.setText(str(status.get('total_alerts', 0)))
        self.intrusion_value.setText(str(status.get('total_intrusions', 0)))
        self.dangerous_value.setText(str(status.get('total_dangerous_flags', 0)))

        self.start_button.setEnabled(stage != 'monitoring')

        self.alerts_list.clear()
        for message in status.get('recent_alerts', []) or []:
            item = QListWidgetItem(message)
            self.alerts_list.addItem(item)

        self.alert_banner.update_alerts(status.get('active_alerts', []))

        toast = status.get('toast')
        if toast and isinstance(toast, dict) and time.time() < float(toast.get('expire', 0.0)):
            message = toast.get('message', '')
            color = toast.get('color', (0, 170, 255))
            r, g, b = color
            self.toast_label.setStyleSheet(
                f"background-color: rgba({r}, {g}, {b}, 60); color: #e8ecff; padding: 8px; border-radius: 6px;"
            )
            self.toast_label.setText(message)
            self.toast_label.show()
        else:
            self.toast_label.hide()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.worker.stop()
        with self.worker.monitor_lock:
            self.monitor.pose_helper.close()
        super().closeEvent(event)


def prepare_monitor(
    confidence: float,
    pose_model: Path | None,
    yolo_model: Path | None = None,
) -> IntegratedSafetyMonitor:
    pose_model_path = pose_model
    if pose_model_path is None or not pose_model_path.exists():
        downloaded = download_pose_model(pose_model_path)
        if downloaded is None:
            raise RuntimeError("无法下载姿态模型，请检查网络连接或手动放置模型到 models/ 目录")
        pose_model_path = Path(downloaded)

    model_path = yolo_model if yolo_model and yolo_model.exists() else None
    if model_path is not None:
        model_path = Path(model_path)

    target_model = model_path or DEFAULT_YOLO_MODEL_PATH
    model_path = download_yolov7_tiny(target_model)
    if model_path is None:
        raise RuntimeError("无法准备YOLO模型")

    model, device = load_model(model_path)
    if model is None or device is None:
        raise RuntimeError("模型加载失败")

    monitor = IntegratedSafetyMonitor(
        model,
        device,
        pose_model_path=str(pose_model_path),
        confidence_threshold=confidence,
        create_window=False,
    )
    return monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文物安全协同防护 PySide6 客户端")
    parser.add_argument('--source', type=str, default='0', help='视频源(0=摄像头或视频路径)')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO 置信度阈值')
    parser.add_argument('--pose-model', type=str, default=str(DEFAULT_POSE_MODEL_PATH), help='姿态模型路径')
    parser.add_argument('--yolo-model', type=str, default=str(DEFAULT_YOLO_MODEL_PATH), help='YOLO 模型路径')
    parser.add_argument('--alert-sound', type=str, default=None, help='报警提示音文件路径（可选）')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_model_path = Path(args.pose_model)
    yolo_model_path = Path(args.yolo_model)
    monitor = prepare_monitor(
        args.conf,
        pose_model_path if pose_model_path.exists() else None,
        yolo_model_path if yolo_model_path.exists() else None,
    )

    video_source: int | str = int(args.source) if args.source.isdigit() else args.source
    alert_sound = Path(args.alert_sound) if args.alert_sound else None

    app = QApplication(sys.argv)
    window = SafetyMonitorWindow(
        monitor,
        video_source,
        alert_sound if alert_sound and alert_sound.exists() else None,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
