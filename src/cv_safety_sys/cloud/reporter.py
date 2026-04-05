"""报警数据上报协议定义与华为云 IoT 预留实现。"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Sequence


LOGGER = logging.getLogger(__name__)


class AlertReporter(ABC):
    """报警上报器接口。"""

    @abstractmethod
    def report(self, payload: Dict[str, Any]) -> None:
        """上报一条结构化报警消息。"""


class NullAlertReporter(AlertReporter):
    """默认空实现：不执行任何上传动作。"""

    def report(self, payload: Dict[str, Any]) -> None:  # noqa: ARG002
        return


class ConsoleAlertReporter(AlertReporter):
    """调试实现：将报警数据输出到日志。"""

    def report(self, payload: Dict[str, Any]) -> None:
        LOGGER.info("alert_payload=%s", json.dumps(payload, ensure_ascii=False))


@dataclass
class HuaweiCloudIotReporter(AlertReporter):
    """华为云 IoT 上报占位实现（预留 MQTT 对接点）。"""

    endpoint: str
    port: int
    device_id: str
    topic: str = "$oc/devices/{device_id}/sys/properties/report"

    def report(self, payload: Dict[str, Any]) -> None:
        """预留 MQTT 发布入口。

        后续可接入 paho-mqtt 并在此实现真正的 TLS 鉴权与 publish。
        """

        resolved_topic = self.topic.format(device_id=self.device_id)
        LOGGER.debug(
            "HuaweiCloudIotReporter prepared payload for %s:%s topic=%s payload=%s",
            self.endpoint,
            self.port,
            resolved_topic,
            json.dumps(payload, ensure_ascii=False),
        )

    @staticmethod
    def build_alarm_payload(
        *,
        alerts: Sequence[str],
        status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构建符合设备属性上报思路的结构化负载。"""

        return {
            "event_time": datetime.now(timezone.utc).isoformat(),
            "services": [
                {
                    "service_id": "cv_safety_monitor",
                    "properties": {
                        "alerts": list(alerts),
                        "total_alerts": int(status.get("total_alerts", 0)),
                        "total_intrusions": int(status.get("total_intrusions", 0)),
                        "total_dangerous_flags": int(status.get("total_dangerous_flags", 0)),
                        "person_count": int(status.get("person_count", 0)),
                        "fence_count": int(status.get("fence_count", 0)),
                    },
                }
            ],
        }
