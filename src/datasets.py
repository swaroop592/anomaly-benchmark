import numpy as np
import pandas as pd


def generate_time_series(
    n_points: int = 1000,
    anomaly_frac: float = 0.02,
    noise_std: float = 0.5,
    seed: int = 42,
):
    """
    Generate a synthetic time series with seasonality + noise + point anomalies.

    Returns:
        df with columns:
          - time
          - value
          - is_anomaly (ground truth)
    """
    rng = np.random.default_rng(seed)

    time = np.arange(n_points)

    # Base signal: sinusoidal + trend
    seasonal = 5 * np.sin(2 * np.pi * time / 50)
    trend = 0.001 * time
    noise = rng.normal(0, noise_std, size=n_points)

    value = seasonal + trend + noise

    is_anomaly = np.zeros(n_points, dtype=int)

    # Inject point anomalies
    n_anoms = int(n_points * anomaly_frac)
    anomaly_idx = rng.choice(n_points, size=n_anoms, replace=False)

    value[anomaly_idx] += rng.normal(10, 3, size=n_anoms)
    is_anomaly[anomaly_idx] = 1

    return pd.DataFrame(
        {
            "time": time,
            "value": value,
            "is_anomaly": is_anomaly,
        }
    )


def generate_base_dataset():
    """
    Easy case:
    - low noise
    - clear spikes
    """
    return generate_time_series(
        n_points=1000,
        anomaly_frac=0.02,
        noise_std=0.3,
        seed=1,
    )


def generate_stress_dataset():
    """
    Hard case:
    - higher noise
    - more subtle anomalies
    """
    return generate_time_series(
        n_points=1000,
        anomaly_frac=0.02,
        noise_std=1.2,
        seed=2,
    )
