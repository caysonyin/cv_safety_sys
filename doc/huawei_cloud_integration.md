# Huawei Cloud IoT Integration Guide

This project now reserves a clean cloud transport interface for alarm/status upload.
You can integrate Huawei Cloud IoT by plugging an MQTT publisher into the monitor runtime.

## 1. What was added

- A **pluggable cloud transport interface** (`CloudDataPublisher`).
- A **default no-op publisher** (`NoOpCloudPublisher`) so local runs work without cloud.
- A **Huawei MQTT publisher** (`HuaweiMQTTPublisher`) with config (`HuaweiMQTTConfig`).
- Two standardized payload builders:
  - `build_snapshot_payload(...)`
  - `build_alert_payload(...)`
- Automatic data hook in `IntegratedSafetyMonitor.process_frame(...)`:
  - publishes one snapshot each frame.
  - publishes one alert event for each new alert message.

## 2. Install dependency for MQTT

```bash
pip install paho-mqtt
```

> `paho-mqtt` is imported lazily. If you do not enable Huawei MQTT, the dependency is optional.

## 3. Huawei Cloud preparation (concept)

On Huawei Cloud IoTDA, typically you need:

1. Create a product.
2. Create/register a device.
3. Get device credentials (device ID/username/password or access secret depending on your setup).
4. Get MQTT endpoint and port.
5. Plan two topics:
   - status topic (monitoring snapshots)
   - alert topic (alarm events)

Use your actual tenant/project topic naming strategy in production.

## 4. Quick integration example

```python
from cv_safety_sys.cloud import HuaweiMQTTConfig, HuaweiMQTTPublisher
from cv_safety_sys.monitoring.integrated_monitor import IntegratedSafetyMonitor

mqtt_config = HuaweiMQTTConfig(
    host="your-iotda-endpoint",
    port=1883,
    client_id="device_client_id",
    username="device_username",
    password="device_password",
    status_topic="devices/your-device-id/status",
    alert_topic="devices/your-device-id/alerts",
    keepalive_seconds=60,
    qos=1,
)

publisher = HuaweiMQTTPublisher(mqtt_config)

monitor = IntegratedSafetyMonitor(
    model=model,
    device=device,
    pose_model_path="models/pose_landmarker_lite.task",
    confidence_threshold=0.25,
    cloud_publisher=publisher,
    device_id="your-device-id",
)
```

When you close the monitor, cloud resources are released automatically via `monitor.close()`.

## 5. Payload schema

### 5.1 Monitoring snapshot payload

```json
{
  "version": "1.0",
  "event_type": "monitoring_snapshot",
  "timestamp": 1710000000.0,
  "device_id": "camera-gate-01",
  "status": {
    "stage": "monitoring",
    "person_count": 2,
    "total_alerts": 3
  }
}
```

### 5.2 Alert payload

```json
{
  "version": "1.0",
  "event_type": "alert",
  "timestamp": 1710000001.0,
  "device_id": "camera-gate-01",
  "severity": "intrusion",
  "message": "Person ID:8 intruded into relic safety fence",
  "status": {
    "stage": "monitoring",
    "person_count": 2,
    "total_alerts": 4
  }
}
```

## 6. Recommended next step

For production-grade Huawei IoTDA integration, consider adding:

- TLS (`8883`) and certificate validation.
- reconnect/backoff and offline message queue.
- stricter topic/QoS policy.
- message signing strategy required by your tenant setup.
- optional compression/rate limiting for high FPS uploads.
