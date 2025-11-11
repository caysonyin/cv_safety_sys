#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unicode-aware text rendering helpers for OpenCV frames."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Color = Tuple[int, int, int]

_FONT_ENV = "CV_SAFETY_FONT"
_FONT_CANDIDATES: Sequence[str] = (
    os.environ.get(_FONT_ENV) or "",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhl.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def _has_wide_characters(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


@lru_cache(maxsize=1)
def _detect_font_path() -> Path | None:
    for candidate in _FONT_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


@lru_cache(maxsize=12)
def _load_font(pixel_size: int) -> ImageFont.FreeTypeFont | None:
    font_path = _detect_font_path()
    if not font_path:
        return None
    try:
        return ImageFont.truetype(str(font_path), pixel_size)
    except OSError:
        return None


def _bgr_to_rgba(color: Color) -> Tuple[int, int, int, int]:
    b, g, r = color
    return (r, g, b, 255)


def _render_text_bitmap(
    text: str,
    font: ImageFont.FreeTypeFont,
    color: Color,
    thickness: int,
) -> np.ndarray:
    stroke = max(0, thickness - 1)
    bbox = font.getbbox(text)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    pad = max(2, thickness + 1)
    image = Image.new("RGBA", (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text(
        (pad - bbox[0], pad - bbox[1]),
        text,
        fill=_bgr_to_rgba(color),
        font=font,
        stroke_width=stroke,
        stroke_fill=_bgr_to_rgba(color),
    )
    return np.array(image)


def _overlay_bitmap(
    frame: np.ndarray,
    bitmap: np.ndarray,
    top_left: Tuple[int, int],
) -> None:
    x, y = top_left
    h, w = bitmap.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    if x >= frame_w or y >= frame_h:
        return

    clip_x1 = max(0, x)
    clip_y1 = max(0, y)
    clip_x2 = min(frame_w, x + w)
    clip_y2 = min(frame_h, y + h)
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return

    bmp_x1 = clip_x1 - x
    bmp_y1 = clip_y1 - y
    bmp_x2 = bmp_x1 + (clip_x2 - clip_x1)
    bmp_y2 = bmp_y1 + (clip_y2 - clip_y1)

    roi = frame[clip_y1:clip_y2, clip_x1:clip_x2]
    text_rgba = bitmap[bmp_y1:bmp_y2, bmp_x1:bmp_x2]
    if text_rgba.shape[2] == 3:
        text_rgba = np.concatenate(
            [text_rgba, np.full(text_rgba.shape[:2] + (1,), 255, dtype=np.uint8)],
            axis=2,
        )

    alpha = (text_rgba[:, :, 3:4].astype(np.float32)) / 255.0
    text_bgr = text_rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    roi[:] = (1.0 - alpha) * roi.astype(np.float32) + alpha * text_bgr
    roi[:] = np.clip(roi, 0, 255).astype(np.uint8)


def put_text(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_face: int,
    font_scale: float,
    color: Color,
    thickness: int = 1,
    line_type: int = cv2.LINE_AA,
) -> None:
    """Drop-in replacement for cv2.putText with Unicode support."""

    if not text:
        return

    if not _has_wide_characters(text):
        cv2.putText(frame, text, org, font_face, font_scale, color, thickness, line_type)
        return

    font_size = max(12, int(round(32 * font_scale)))
    font = _load_font(font_size)
    if font is None:
        cv2.putText(frame, text, org, font_face, font_scale, color, thickness, line_type)
        return

    bitmap = _render_text_bitmap(text, font, color, thickness)
    top_left = (int(org[0]), int(org[1]) - bitmap.shape[0] + max(2, thickness))
    _overlay_bitmap(frame, bitmap, top_left)
