# 华为云 IoT 接入指南（预留接口版）

本文说明如何把系统中的报警信息（侵入、危险物携带等）通过 MQTT 路径接入华为云 IoTDA。

## 1. 当前代码已预留的能力

项目新增了统一上报接口 `AlertReporter`，并预置：

- `NullAlertReporter`：默认空实现（不上传）。
- `ConsoleAlertReporter`：本地日志调试。
- `HuaweiCloudIotReporter`：华为云上报占位实现（已预留 topic 与 payload 构建，便于你补全 MQTT 发布代码）。

报警事件在 `IntegratedSafetyMonitor.process_frame()` 和 `run()` 流程中触发；一旦出现报警，会调用 `_report_alerts()` 生成 payload 并交给 `alert_reporter.report()`。

## 2. 华为云侧准备（你已有思路，补充细节）

1. 在华为云 IoTDA 创建设备，记录：
   - `device_id`
   - 设备密钥（或对应认证信息）
   - MQTT 接入地址（`endpoint`）与端口（通常 TLS 端口）
2. 规划设备上报 topic（示例默认使用）：
   - `$oc/devices/{device_id}/sys/properties/report`
3. 在产品模型里创建服务（示例使用 `cv_safety_monitor`）及属性：
   - `alerts`（字符串数组）
   - `total_alerts`
   - `total_intrusions`
   - `total_dangerous_flags`
   - `person_count`
   - `fence_count`

## 3. 代码接入方式

### 3.1 初始化上报器

在创建 `IntegratedSafetyMonitor` 时注入：

```python
from cv_safety_sys.cloud.reporter import HuaweiCloudIotReporter
from cv_safety_sys.monitoring.integrated_monitor import IntegratedSafetyMonitor

reporter = HuaweiCloudIotReporter(
    endpoint="your-iotda-mqtt-endpoint",
    port=8883,
    device_id="your-device-id",
)

monitor = IntegratedSafetyMonitor(
    model=model,
    device=device,
    pose_model_path="models/pose_landmarker_lite.task",
    alert_reporter=reporter,
)
```

### 3.2 补全 MQTT 发布逻辑

编辑 `src/cv_safety_sys/cloud/reporter.py` 中 `HuaweiCloudIotReporter.report()`：

- 初始化 MQTT 客户端（建议 `paho-mqtt`）。
- 配置 TLS 与设备认证（按华为云 IoTDA 文档）。
- 连接 `endpoint:port`。
- `publish(resolved_topic, payload_json, qos=1)`。
- 增加重连、异常日志与离线缓存（可选）。

## 4. 当前上报 payload 结构

系统内部会构造如下结构（示意）：

```json
{
  "event_time": "2026-04-01T00:00:00+00:00",
  "services": [
    {
      "service_id": "cv_safety_monitor",
      "properties": {
        "alerts": ["人员 ID:3 侵入 cup 安全栅栏"],
        "total_alerts": 10,
        "total_intrusions": 7,
        "total_dangerous_flags": 3,
        "person_count": 2,
        "fence_count": 1
      }
    }
  ]
}
```

## 5. 联调建议

1. 先用 `ConsoleAlertReporter` 验证本地报警数据结构。
2. 再替换为 `HuaweiCloudIotReporter` 并补全 MQTT 代码。
3. 在 IoTDA 控制台检查设备在线状态、消息到达与属性上报记录。
