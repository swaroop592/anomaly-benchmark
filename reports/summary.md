# Anomaly Detection Benchmark Summary

This report compares multiple anomaly detectors on base and stress synthetic time-series datasets.

## Dataset: base

| model            |   precision |   recall |       f1 |   false_alarm_rate |   fit_time_ms |   infer_time_ms |
|:-----------------|------------:|---------:|---------:|-------------------:|--------------:|----------------:|
| isolation_forest |   0.65      |     0.65 | 0.65     |         0.00714286 | 136.944       |         14.8416 |
| zscore           |   1         |     0.2  | 0.333333 |         0          |   0.000100001 |         10.0858 |
| oneclass_svm     |   0.0721649 |     0.35 | 0.119658 |         0.0918367  |   3.0853      |          2.436  |

**Top performer**: `isolation_forest` (F1=0.650, Recall=0.650, Precision=0.650)

## Dataset: stress

| model            |   precision |   recall |       f1 |   false_alarm_rate |   fit_time_ms |   infer_time_ms |
|:-----------------|------------:|---------:|---------:|-------------------:|--------------:|----------------:|
| isolation_forest |   0.55      |     0.55 | 0.55     |         0.00918367 | 145.065       |         16.2226 |
| zscore           |   0.714286  |     0.25 | 0.37037  |         0.00204082 |   0.000200002 |          9.5953 |
| oneclass_svm     |   0.0823529 |     0.35 | 0.133333 |         0.0795918  |   3.5974      |          6.2186 |

**Top performer**: `isolation_forest` (F1=0.550, Recall=0.550, Precision=0.550)
