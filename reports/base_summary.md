# Anomaly Detection Benchmark Summary

This report compares multiple anomaly detectors on base and stress synthetic time-series datasets.

## Dataset: base

| model            |   precision |   recall |       f1 |   false_alarm_rate |   fit_time_ms |   infer_time_ms |
|:-----------------|------------:|---------:|---------:|-------------------:|--------------:|----------------:|
| isolation_forest |    0.65     |     0.65 | 0.65     |         0.00714286 | 192.31        |         14.7224 |
| zscore           |    1        |     0.2  | 0.333333 |         0          |   0.000100001 |          9.9335 |
| oneclass_svm     |    0.104651 |     0.45 | 0.169811 |         0.0785714  |   2.3348      |          2.3607 |

**Top performer**: `isolation_forest` (F1=0.650, Recall=0.650, Precision=0.650)
