# Anomaly Detection Benchmark Summary

This report compares multiple anomaly detectors on base and stress synthetic time-series datasets.

## Dataset: stress

| model            |   precision |   recall |       f1 |   false_alarm_rate |   fit_time_ms |   infer_time_ms |
|:-----------------|------------:|---------:|---------:|-------------------:|--------------:|----------------:|
| isolation_forest |   0.5       |     0.25 | 0.333333 |         0.0104167  |  163.161      |         16.2701 |
| zscore           |   0.666667  |     0.1  | 0.173913 |         0.00208333 |    0.00540004 |          9.8069 |
| oneclass_svm     |   0.0939597 |     0.35 | 0.148148 |         0.140625   |    4.168      |          3.6496 |

**Top performer**: `isolation_forest` (F1=0.333, Recall=0.250, Precision=0.500)
