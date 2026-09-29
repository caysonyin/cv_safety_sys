#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Linsheng Yin, Heng Quan, Bojin Li, Yunpeng Din, Penghan Chen
# File purpose: Huawei Cloud IoT data publishing adapter and payload schema.
"""Cloud transport adapter for exporting monitoring data to Huawei Cloud IoT.

This module keeps the monitor side decoupled from a specific cloud implementation.
It provides:

- A lightweight protocol (`CloudDataPublisher`) for pluggable transports.
- A no-op implementation that allows local runs without cloud dependencies.
- A Huawei MQTT publisher with lazy import for `paho-mqtt`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Protocol


class CloudDataPublisher(Protocol):
    """Common interface used by monitor pipeline to push cloud payloads."""

    def publish_monitoring_snapshot(self, payload: Mapping[str, object]) -> None:
        """Publish a frame-level status snapshot."""

    def publish_alert_event(self, payload: Mapping[str, object]) -> None:
        """Publish a new alert event."""

    def close(self) -> None:
        """Release network resources if needed."""


class NoOpCloudPublisher:
    """Default placeholder publisher used when cloud transport is not configured."""

    def publish_monitoring_snapshot(self, payload: Mapping[str, object]) -> None:
        _ = payload

    def publish_alert_event(self, payload: Mapping[str, object]) -> None:
        _ = payload

    def close(self) -> None:
        return


@dataclass
class HuaweiMQTTConfig:
    """Connection parameters for Huawei Cloud IoT over MQTT."""

    host: str
    port: int
    client_id: str
    username: str
    password: str
    status_topic: str
    alert_topic: str
    keepalive_seconds: int = 60
    qos: int = 1


class HuaweiMQTTPublisher:
    """MQTT transport implementation for Huawei Cloud IoT device access."""

    def __init__(self, config: HuaweiMQTTConfig):
        self.config = config
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Huawei MQTT publisher requires 'paho-mqtt'. Install it before enabling cloud upload."
            ) from exc

        self._mqtt_module = mqtt
        self.client = mqtt.Client(client_id=config.client_id)
        self.client.username_pw_set(config.username, config.password)
        self.client.connect(config.host, config.port, keepalive=config.keepalive_seconds)
        self.client.loop_start()

    def publish_monitoring_snapshot(self, payload: Mapping[str, object]) -> None:
        self._publish(self.config.status_topic, payload)

    def publish_alert_event(self, payload: Mapping[str, object]) -> None:
        self._publish(self.config.alert_topic, payload)

    def _publish(self, topic: str, payload: Mapping[str, object]) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False)
        self.client.publish(topic, body, qos=self.config.qos)

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


def build_snapshot_payload(
    *,
    device_id: str,
    status: Mapping[str, object],
) -> Dict[str, object]:
    """Build normalized status payload for cloud forwarding."""

    return {
        'version': '1.0',
        'event_type': 'monitoring_snapshot',
        'timestamp': time.time(),
        'device_id': device_id,
        'status': dict(status),
    }


def build_alert_payload(
    *,
    device_id: str,
    alert_message: str,
    severity: str,
    status: Mapping[str, object],
) -> Dict[str, object]:
    """Build normalized alert payload for cloud forwarding."""

    return {
        'version': '1.0',
        'event_type': 'alert',
        'timestamp': time.time(),
        'device_id': device_id,
        'severity': severity,
        'message': alert_message,
        'status': dict(status),
    }
