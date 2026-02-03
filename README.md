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

Synthetic time-series data is generated with the following components:

### Base Dataset
- Seasonal signal + noise
- No drift
- No missing values
- Clear point anomalies

### Stress Dataset
- Slow drift over time
- Increased noise
- Missing values (sensor dropouts)
- Subtle anomalies (hard to detect)

Ground truth anomaly labels are known, enabling objective evaluation.

---

## Models Benchmarked

The following anomaly detectors are evaluated:

- **Z-score**
  - Simple statistical baseline
  - High precision, low recall

- **Isolation Forest**
  - Tree-based anomaly detector
  - Strong balance between precision and recall

- **One-Class SVM**
  - Sensitive boundary-based detector
  - Higher false alarm rates

All models are evaluated on identical datasets.

---

## Evaluation Metrics

For each model and dataset, the following metrics are reported:

- Precision
- Recall
- F1 score
- False alarm rate
- Fit time (ms)
- Inference time (ms)

Results are automatically written to CSV and Markdown summary reports.

---

## Running the benchmark

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
