#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen
# File purpose: One-file Huawei Cloud MQTT configuration for quick project integration.
"""Editable Huawei Cloud MQTT settings used by the existing monitor app.

How to use:
1. Set `ENABLED = True`.
2. Fill the device credentials below.
3. Run the project normally; monitor data will be uploaded in real time.
"""

from __future__ import annotations

from typing import Tuple

from .huawei_cloud_adapter import (
    CloudDataPublisher,
    HuaweiMQTTConfig,
    HuaweiMQTTPublisher,
    NoOpCloudPublisher,
)

# ---------------------------------------------------------------------------
# Quick-start switch
# ---------------------------------------------------------------------------
ENABLED = False

# ---------------------------------------------------------------------------
# Fill your Huawei Cloud IoT device information here.
# Replace all placeholders after creating a device and key in Huawei Cloud.
# ---------------------------------------------------------------------------
DEVICE_ID = "your_device_id"
MQTT_HOST = "your_iotda_mqtt_host"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "your_mqtt_client_id"
MQTT_USERNAME = "your_mqtt_username"
MQTT_PASSWORD = "your_mqtt_password_or_secret"

# Topics can include {device_id}, which will be replaced automatically.
STATUS_TOPIC_TEMPLATE = "cv_safety/{device_id}/status"
ALERT_TOPIC_TEMPLATE = "cv_safety/{device_id}/alerts"

# Optional publish parameters.
MQTT_QOS = 1
MQTT_KEEPALIVE_SECONDS = 60


def build_cloud_runtime() -> Tuple[CloudDataPublisher, str]:
    """Build cloud publisher + device id directly from this settings file."""
    if not ENABLED:
        return NoOpCloudPublisher(), DEVICE_ID

    required_fields = {
        'DEVICE_ID': DEVICE_ID,
        'MQTT_HOST': MQTT_HOST,
        'MQTT_CLIENT_ID': MQTT_CLIENT_ID,
        'MQTT_USERNAME': MQTT_USERNAME,
        'MQTT_PASSWORD': MQTT_PASSWORD,
    }
    missing = [name for name, value in required_fields.items() if not str(value).strip()]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "Huawei cloud upload is enabled, but required settings are empty: "
            f"{missing_text}. Please update src/cv_safety_sys/cloud/huawei_cloud_settings.py."
        )

    status_topic = STATUS_TOPIC_TEMPLATE.format(device_id=DEVICE_ID)
    alert_topic = ALERT_TOPIC_TEMPLATE.format(device_id=DEVICE_ID)

    config = HuaweiMQTTConfig(
        host=MQTT_HOST,
        port=MQTT_PORT,
        client_id=MQTT_CLIENT_ID,
        username=MQTT_USERNAME,
        password=MQTT_PASSWORD,
        status_topic=status_topic,
        alert_topic=alert_topic,
        keepalive_seconds=MQTT_KEEPALIVE_SECONDS,
        qos=MQTT_QOS,
    )
    return HuaweiMQTTPublisher(config), DEVICE_ID
