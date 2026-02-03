import argparse
import pandas as pd

from src.datasets import generate_base_dataset, generate_stress_dataset
from src.models import run_zscore, run_isolation_forest, run_oneclass_svm
from src.metrics import prf1, false_alarm_rate
from src.report import write_summary_markdown, plot_series_with_predictions


def run_all_models(df, dataset_name: str):
    x = df["value"].values
    y_true = df["is_anomaly"].values

    results = []

    model_runs = [
        ("zscore", lambda: run_zscore(x, window=50, threshold=3.0)),
        ("isolation_forest", lambda: run_isolation_forest(x, contamination=0.02)),
        ("oneclass_svm", lambda: run_oneclass_svm(x, nu=0.02)),
    ]

    for model_name, fn in model_runs:
        out = fn()
        metrics = prf1(y_true, out.y_pred)
        far = false_alarm_rate(y_true, out.y_pred)

        results.append({
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
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", default="results/results.csv")
    parser.add_argument("--report_md", default="reports/summary.md")
    parser.add_argument("--plot_dir", default="reports")
    args = parser.parse_args()

    base = generate_base_dataset()
    stress = generate_stress_dataset()

    rows = []
    rows += run_all_models(base, "base")
    rows += run_all_models(stress, "stress")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    write_summary_markdown(df, args.report_md)

    # One plot example: base dataset with best model's predictions
    best_base = df[df["dataset"] == "base"].sort_values("f1", ascending=False).iloc[0]["model"]
    if best_base == "zscore":
        pred = run_zscore(base["value"].values).y_pred
    elif best_base == "isolation_forest":
        pred = run_isolation_forest(base["value"].values).y_pred
    else:
        pred = run_oneclass_svm(base["value"].values).y_pred

    plot_series_with_predictions(
        base,
        pd.Series(pred),
        out_path=f"{args.plot_dir}/base_best_model.png",
        title=f"Base dataset anomalies predicted by {best_base}",
    )

    print("Benchmark complete.")
    print(df[["dataset", "model", "precision", "recall", "f1", "infer_time_ms"]])


if __name__ == "__main__":
    main()
