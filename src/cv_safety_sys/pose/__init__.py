# Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen
# File purpose:   init   module.
"""Pose-related helpers for the CV Safety System."""

from .model_downloader import DEFAULT_MODEL_PATH, download_model

__all__ = ["DEFAULT_MODEL_PATH", "download_model"]
