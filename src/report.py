import os
import pandas as pd
import matplotlib.pyplot as plt


def write_summary_markdown(df: pd.DataFrame, out_path: str):
    lines = []
    lines.append("# Anomaly Detection Benchmark Summary\n")
    lines.append("This report compares multiple anomaly detectors on base and stress synthetic time-series datasets.\n")

    for dataset_name in df["dataset"].unique():
        lines.append(f"## Dataset: {dataset_name}\n")
        sub = df[df["dataset"] == dataset_name].copy()
        sub = sub.sort_values(["f1", "recall", "precision"], ascending=False)

        lines.append(sub[[
            "model", "precision", "recall", "f1", "false_alarm_rate",
            "fit_time_ms", "infer_time_ms"
        ]].to_markdown(index=False))

        lines.append("")

        # Recommendation
        best = sub.iloc[0]
        lines.append(f"**Top performer**: `{best['model']}` (F1={best['f1']:.3f}, Recall={best['recall']:.3f}, Precision={best['precision']:.3f})\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_series_with_predictions(df_series: pd.DataFrame, y_pred: pd.Series, out_path: str, title: str):
    """
    Plot value vs time with anomalies highlighted.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(df_series["time"], df_series["value"], linewidth=1)
    idx = df_series.loc[y_pred == 1, "time"]
    vals = df_series.loc[y_pred == 1, "value"]
    plt.scatter(idx, vals, marker="x")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
