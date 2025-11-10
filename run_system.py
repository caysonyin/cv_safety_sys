#!/usr/bin/env python3
"""Entry point for the unified cultural relic safety monitoring system."""
from __future__ import annotations

import argparse
import logging

import uvicorn

from safety_monitor.pipeline import SafetyMonitoringPipeline
from safety_monitor.web import create_app


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cultural relic safety monitoring server")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="视频源（默认0表示本地摄像头，也可填写视频文件路径）",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: str | int
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    logging.info("Starting safety monitoring pipeline (source=%s)", source)
    pipeline = SafetyMonitoringPipeline(video_source=source)
    app = create_app(pipeline)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
