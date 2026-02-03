import numpy as np
import pandas as pd


def generate_time_series(
    n_points: int,
    anomaly_frac: float,
    noise_std: float,
    drift_per_step: float,
    missing_frac: float,
    subtle_anomaly_frac: float,
    seed: int,
):
    """
    Synthetic time-series generator with:
      - seasonality + noise
      - point anomalies (big spikes)
      - subtle anomalies (small spikes)
      - drift (slow shift over time)
      - missing values

    Returns df with columns:
      - time
      - value
      - is_anomaly (1 for either point or subtle)
      - anomaly_type: {"none","point","subtle"}
      - is_missing (1 if value is missing)
    """
    rng = np.random.default_rng(seed)
    time = np.arange(n_points)

    # Base signal: sinusoidal + drift + noise
    seasonal = 5 * np.sin(2 * np.pi * time / 50)
    drift = drift_per_step * time
    noise = rng.normal(0, noise_std, size=n_points)

    value = seasonal + drift + noise

    is_anomaly = np.zeros(n_points, dtype=int)
    anomaly_type = np.array(["none"] * n_points, dtype=object)

    # Point anomalies (large spikes)
    n_point = int(n_points * anomaly_frac)
    point_idx = rng.choice(n_points, size=n_point, replace=False)
    value[point_idx] += rng.normal(10, 3, size=n_point)
    is_anomaly[point_idx] = 1
    anomaly_type[point_idx] = "point"

    # Subtle anomalies (small spikes) - choose indices that are not already point anomalies
    n_subtle = int(n_points * subtle_anomaly_frac)
    if n_subtle > 0:
        candidates = np.setdiff1d(np.arange(n_points), point_idx)
        subtle_idx = rng.choice(candidates, size=min(n_subtle, len(candidates)), replace=False)
        value[subtle_idx] += rng.normal(2.5, 0.8, size=len(subtle_idx))
        is_anomaly[subtle_idx] = 1
        anomaly_type[subtle_idx] = "subtle"

    # Missing values
    is_missing = np.zeros(n_points, dtype=int)
    n_miss = int(n_points * missing_frac)
    if n_miss > 0:
        miss_idx = rng.choice(n_points, size=n_miss, replace=False)
        value[miss_idx] = np.nan
        is_missing[miss_idx] = 1

    return pd.DataFrame(
        {
            "time": time,
            "value": value,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "is_missing": is_missing,
        }
    )


def from_config(cfg: dict) -> pd.DataFrame:
    d = cfg["dataset"]
    return generate_time_series(
        n_points=int(d["n_points"]),
        anomaly_frac=float(d["anomaly_frac"]),
        noise_std=float(d["noise_std"]),
        drift_per_step=float(d["drift_per_step"]),
        missing_frac=float(d["missing_frac"]),
        subtle_anomaly_frac=float(d["subtle_anomaly_frac"]),
        seed=int(d["seed"]),
    )
