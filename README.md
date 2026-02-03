# Time-Series Anomaly Detection Benchmark

This project benchmarks multiple anomaly detection algorithms on synthetic time-series data under **base** and **stress** conditions.

The goal is to evaluate how different models trade off **precision, recall, false alarms, and inference latency** when data quality degrades — a common real-world scenario in monitoring and observability systems.

---

## Why this project exists

In production systems (sensors, logs, metrics, finance, infrastructure), anomaly detection models often behave very differently when:

- the data distribution slowly drifts
- values go missing
- anomalies are subtle rather than obvious

This project explicitly separates:
- **Base cases** → clean, stable data
- **Stress cases** → drift, missing data, subtle anomalies

This ensures **fair, reproducible, side-by-side benchmarking**.

---

## Dataset Design

Sy
