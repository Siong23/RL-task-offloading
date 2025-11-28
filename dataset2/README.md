# IoT–Edge Telemetry Dataset (Kubernetes + Prometheus)

## 📘 Description
This dataset contains time-series telemetry data collected from a Kubernetes-based IoT–Edge computing environment instrumented with **Prometheus** and **Node Exporter**. The setup consists of **two IoT nodes** and **two edge nodes**, forming a compact cluster that reflects a realistic IoT–Edge–Cloud continuum.

- IoT (Raspberry Pi) nodes with cameras connect via Wi-Fi through a **5G CPE device**, streaming continuous real-time video to the edge nodes.
- Edge nodes operate in a **Multi-access Edge Computing (MEC)** environment to emulate near-edge processing and inference capabilities.
- Telemetry includes detailed system-level metrics such as **CPU utilization**, **memory usage**, **disk I/O**, and **network throughput**, collected periodically from each node.

This dataset supports research in:
- Task offloading
- AI-driven resource management
- Intelligent workload placement  
across distributed IoT–Edge–Cloud infrastructures.

It is particularly valuable for training and validating **reinforcement learning (RL)** or other data-driven models targeting optimization of performance, latency, and energy efficiency in heterogeneous systems.

---

## 📊 PromQL Queries Used

**CPU Utilization (%)**
```
100 * (1 - avg by (instance) (
    irate(node_cpu_seconds_total{job="node-exporter", mode=~"idle|iowait|steal", instance="<IP>:9100"}[1m])
))
```

**Memory Load (%)**
```
100 * (1 - (node_memory_MemAvailable_bytes{instance="<IP>:9100"} 
      / node_memory_MemTotal_bytes{instance="<IP>:9100"}))
```

**Disk I/O Load (bytes/s)**
```
sum by (instance) (
    irate(node_disk_reads_completed_total{instance="<IP>:9100"}[1m]) +
    irate(node_disk_writes_completed_total{instance="<IP>:9100"}[1m])
)
```

**Network Download Speed (bytes/s)**
```
sum by (instance) (
    irate(node_network_receive_bytes_total{instance="<IP>:9100", device!~"lo"}[1m])
)
```

**Network Upload Speed (bytes/s)**
```
sum by (instance) (
    irate(node_network_transmit_bytes_total{instance="<IP>:9100", device!~"lo"}[1m])
)
```

---

## 🔑 Key Features
- Four-node Kubernetes cluster  
  (2 IoT nodes + 2 Edge nodes)
- Prometheus-based metric collection using **Node Exporter**
- **5G-enabled IoT connectivity** via CPE and Wi-Fi
- Multi-dimensional telemetry:
  - CPU
  - Memory
  - Disk I/O
  - Network throughput
- Time-aligned sampling suitable for **machine learning** and **reinforcement learning**
- Contains both **stress** and **non-stress** workload scenarios

---

## 🚀 Potential Use Cases
- Reinforcement learning for **dynamic task offloading**
- Optimizing **edge–cloud collaboration**
- AI-driven **workload orchestration** in MEC environments
- **Energy-aware** IoT–Edge computing
- Benchmarking telemetry-driven **scheduling algorithms**

---

## 📁 Format
- Dataset is stored in **CSV format**, containing timestamped metrics for each node.
- Column structure:

```
[time_slot, <node>_cpu, <node>_memory, <node>_disk_io, <node>_net_rx, <node>_net_tx]
```

