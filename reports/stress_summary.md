# Anomaly Detection Benchmark Summary

This report compares multiple anomaly detectors on base and stress synthetic time-series datasets.

## Dataset: stress

| model            |   precision |   recall |       f1 |   false_alarm_rate |   fit_time_ms |   infer_time_ms |
|:-----------------|------------:|---------:|---------:|-------------------:|--------------:|----------------:|
| isolation_forest |   0.5       |     0.25 | 0.333333 |         0.0104167  | 142.337       |         14.2105 |
| zscore           |   0.666667  |     0.1  | 0.173913 |         0.00208333 |   0.000200002 |          9.7277 |
| oneclass_svm     |   0.0939597 |     0.35 | 0.148148 |         0.140625   |   3.6906      |          3.7268 |

**Top performer**: `isolation_forest` (F1=0.333, Recall=0.250, Precision=0.500)
