import argparse
import yaml
import pandas as pd

from src.datasets import from_config
from src.models import run_zscore, run_isolation_forest, run_oneclass_svm
from src.metrics import prf1, false_alarm_rate
from src.report import write_summary_markdown, plot_series_with_predictions


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_all_models(df: pd.DataFrame, dataset_name: str, cfg: dict):
    # Handle missing values for modeling
    x = df["value"].copy()
    x = x.interpolate(limit_direction="both")
    x = x.bfill().ffill()
    x = x.values

    y_true = df["is_anomaly"].values

    m = cfg["models"]

    model_runs = [
        (
            "zscore",
            lambda: run_zscore(
                x,
                window=int(m["zscore"]["window"]),
                threshold=float(m["zscore"]["threshold"]),
            ),
        ),
        (
            "isolation_forest",
            lambda: run_isolation_forest(
                x,
                contamination=float(m["isolation_forest"]["contamination"]),
                random_state=int(m["isolation_forest"]["random_state"]),
            ),
        ),
        (
            "oneclass_svm",
            lambda: run_oneclass_svm(
                x,
                nu=float(m["oneclass_svm"]["nu"]),
                gamma=str(m["oneclass_svm"]["gamma"]),
            ),
        ),
    ]

    rows = []
    for model_name, fn in model_runs:
        out = fn()
        metrics = prf1(y_true, out.y_pred)
        far = false_alarm_rate(y_true, out.y_pred)

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "false_alarm_rate": far,
                "fit_time_ms": out.fit_time_ms,
                "infer_time_ms": out.infer_time_ms,
                "details": str(out.details),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out_csv", default="results/results.csv")
    parser.add_argument("--report_md", default="reports/summary.md")
    parser.add_argument("--plot_dir", default="reports")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df_series = from_config(cfg)
    dataset_name = cfg["dataset"]["name"]

    rows = run_all_models(df_series, dataset_name, cfg)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    write_summary_markdown(df, args.report_md)

    # Plot: best model for this dataset
    best_model = df.sort_values("f1", ascending=False).iloc[0]["model"]

    # Re-run best model to get predictions for plotting
    x_plot = df_series["value"].copy().interpolate(limit_direction="both").bfill().ffill().values

    if best_model == "zscore":
        pred = run_zscore(
            x_plot,
            window=int(cfg["models"]["zscore"]["window"]),
            threshold=float(cfg["models"]["zscore"]["threshold"]),
        ).y_pred
    elif best_model == "isolation_forest":
        pred = run_isolation_forest(
            x_plot,
            contamination=float(cfg["models"]["isolation_forest"]["contamination"]),
            random_state=int(cfg["models"]["isolation_forest"]["random_state"]),
        ).y_pred
    else:
        pred = run_oneclass_svm(
            x_plot,
            nu=float(cfg["models"]["oneclass_svm"]["nu"]),
            gamma=str(cfg["models"]["oneclass_svm"]["gamma"]),
        ).y_pred

    plot_series_with_predictions(
        df_series,
        pd.Series(pred),
        out_path=f"{args.plot_dir}/{dataset_name}_best_model.png",
        title=f"{dataset_name} dataset anomalies predicted by {best_model}",
    )

    print("Benchmark complete.")
    print(df[["dataset", "model", "precision", "recall", "f1", "infer_time_ms"]])


if __name__ == "__main__":
    main()
