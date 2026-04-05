# Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen
# File purpose: Cloud transport abstractions and Huawei IoT adapter exports.
"""Cloud integration helpers for CV safety monitoring."""

from .huawei_cloud_adapter import (
    CloudDataPublisher,
    HuaweiMQTTConfig,
    HuaweiMQTTPublisher,
    NoOpCloudPublisher,
    build_alert_payload,
    build_snapshot_payload,
)

__all__ = [
    'CloudDataPublisher',
    'NoOpCloudPublisher',
    'HuaweiMQTTConfig',
    'HuaweiMQTTPublisher',
    'build_snapshot_payload',
    'build_alert_payload',
]
