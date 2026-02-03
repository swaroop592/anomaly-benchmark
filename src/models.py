import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


@dataclass
class ModelResult:
    y_pred: np.ndarray
    fit_time_ms: float
    infer_time_ms: float
    details: Dict[str, Any]


def rolling_zscore_detector(x: np.ndarray, window: int = 50, threshold: float = 3.0) -> np.ndarray:
    """
    Simple baseline:
    - compute rolling mean/std
    - flag points where |z| > threshold
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros(n, dtype=int)

    for i in range(n):
        start = max(0, i - window)
        hist = x[start:i]  # past window only
        if len(hist) < max(10, window // 5):
            continue
        mu = np.mean(hist)
        sd = np.std(hist) + 1e-8
        z = (x[i] - mu) / sd
        if abs(z) > threshold:
            y[i] = 1
    return y


def run_zscore(x: np.ndarray, window: int = 50, threshold: float = 3.0) -> ModelResult:
    t0 = time.perf_counter()
    # no fitting; treat as 0
    fit_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    y_pred = rolling_zscore_detector(x, window=window, threshold=threshold)
    infer_ms = (time.perf_counter() - t1) * 1000

    return ModelResult(
        y_pred=y_pred,
        fit_time_ms=fit_ms,
        infer_time_ms=infer_ms,
        details={"window": window, "threshold": threshold},
    )


def run_isolation_forest(x: np.ndarray, contamination: float = 0.02, random_state: int = 42) -> ModelResult:
    """
    Isolation Forest flags anomalies with -1. We'll convert to {0,1}.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )

    t0 = time.perf_counter()
    model.fit(x)
    fit_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    preds = model.predict(x)  # -1 anomaly, 1 normal
    infer_ms = (time.perf_counter() - t1) * 1000

    y_pred = (preds == -1).astype(int)

    return ModelResult(
        y_pred=y_pred,
        fit_time_ms=fit_ms,
        infer_time_ms=infer_ms,
        details={"contamination": contamination, "n_estimators": 200},
    )


def run_oneclass_svm(x: np.ndarray, nu: float = 0.02, gamma: str = "scale") -> ModelResult:
    """
    One-Class SVM flags anomalies with -1. We'll convert to {0,1}.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)

    model = OneClassSVM(nu=nu, gamma=gamma)

    t0 = time.perf_counter()
    model.fit(x)
    fit_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    preds = model.predict(x)
    infer_ms = (time.perf_counter() - t1) * 1000

    y_pred = (preds == -1).astype(int)

    return ModelResult(
        y_pred=y_pred,
        fit_time_ms=fit_ms,
        infer_time_ms=infer_ms,
        details={"nu": nu, "gamma": gamma},
    )
