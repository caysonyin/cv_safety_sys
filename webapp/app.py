from __future__ import annotations

import atexit
import os
import time
from pathlib import Path
from typing import Iterable

from flask import Flask, Response, jsonify, render_template, request

from WebcamPoseDetection.download_model import download_model as download_pose_model
from object_protection.video_relic_tracking import download_yolov7_tiny, load_model
from object_protection.web_monitor import WebSafetyMonitor


def _parse_video_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def create_app() -> Flask:
    """创建并初始化Flask应用。"""

    app = Flask(__name__)
    app.config.update(JSON_AS_ASCII=False)

    model_path = download_yolov7_tiny(Path("object_protection/yolov7-tiny.pt"))
    if model_path is None:
        raise RuntimeError("无法准备YOLOv7-tiny模型文件")

    model, device = load_model(model_path)
    if model is None or device is None:
        raise RuntimeError("YOLOv7模型加载失败")

    pose_model_path = Path("models/pose_landmarker_full.task")
    if not pose_model_path.exists():
        downloaded = download_pose_model()
        if downloaded is None:
            raise RuntimeError("无法下载MediaPipe姿态模型")
        pose_model_path = Path(downloaded)

    confidence_env = os.getenv("CONFIDENCE_THRESHOLD", "0.25")
    try:
        confidence = float(confidence_env)
    except ValueError:
        confidence = 0.25

    video_source_env = os.getenv("VIDEO_SOURCE", "0")
    video_source = _parse_video_source(video_source_env)

    monitor = WebSafetyMonitor(
        model,
        device,
        pose_model_path=str(pose_model_path),
        confidence_threshold=confidence,
    )
    monitor.start(video_source=video_source)

    app.config["monitor"] = monitor

    @app.route("/")
    def index() -> str:
        return render_template("dashboard.html")

    @app.route("/stream")
    def stream() -> Response:
        def generate():
            boundary = b"--frame\r\n"
            content_type = b"Content-Type: image/jpeg\r\n\r\n"
            while True:
                frame = monitor.get_latest_frame()
                yield boundary + content_type + frame + b"\r\n"
                time.sleep(0.04)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def status() -> Response:
        return jsonify(monitor.get_status())

    @app.route("/api/relics/select", methods=["POST"])
    def select_relics() -> Response:
        payload = request.get_json(silent=True) or {}
        selected = payload.get("selected", [])

        if selected == "all":
            selected_ids = monitor.select_all_relics()
        else:
            if isinstance(selected, str) or not isinstance(selected, Iterable):
                return jsonify({"error": "selected 字段必须是列表或'all'"}), 400
            try:
                selected_ids = monitor.set_selected_relics(int(rid) for rid in selected)  # type: ignore[arg-type]
            except ValueError:
                return jsonify({"error": "selected 字段包含非法ID"}), 400

        return jsonify({"selected": selected_ids})

    @app.route("/api/relics/clear", methods=["POST"])
    def clear_relics() -> Response:
        monitor.set_selected_relics([])
        return jsonify({"selected": []})

    atexit.register(lambda: monitor.stop(close_pose_helper=True))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
