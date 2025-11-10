"""FastAPI application exposing the safety monitoring pipeline via a web dashboard."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import SafetyMonitoringPipeline


def create_app(pipeline: SafetyMonitoringPipeline) -> FastAPI:
    app = FastAPI(title="Cultural Relic Safety Monitor")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover
        pipeline.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover
        pipeline.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="Missing index.html")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/video_feed")
    async def video_feed() -> StreamingResponse:
        async def frame_iterator() -> AsyncIterator[bytes]:
            while True:
                frame = pipeline.get_frame()
                if frame is not None:
                    yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(frame_iterator(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    async def status() -> dict:
        return pipeline.get_state()

    return app
