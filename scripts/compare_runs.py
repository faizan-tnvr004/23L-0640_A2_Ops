"""Task 3.5: pull all runs of the experiment from MLflow and build one comparison table + chart."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from src import config  # noqa: F401,E402

COLS = ["tags.mlflow.runName", "run_id", "metrics.test_accuracy", "metrics.test_precision",
        "metrics.test_recall", "metrics.test_f1", "metrics.test_roc_auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=os.getenv("PROJECT_NAME", "23L-0640_A2_Ops"))
    args = ap.parse_args()
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    runs = mlflow.search_runs(experiment_names=[args.experiment])
    runs = runs[[c for c in COLS if c in runs.columns]].dropna(subset=["metrics.test_f1"])
    runs.columns = [c.split(".")[-1] for c in runs.columns]
    runs = runs.sort_values("test_f1", ascending=False)
    print(runs.round(4).to_markdown(index=False))
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    runs.to_csv(out / "run_comparison.csv", index=False)
    ax = runs.set_index("runName")[["test_accuracy", "test_f1", "test_recall", "test_roc_auc"]].plot.bar(
        figsize=(10, 5), rot=30)
    ax.set_title("Run comparison (test set, calibrated threshold)")
    plt.tight_layout()
    plt.savefig(out / "run_comparison.png", dpi=150)
    print("saved", out / "run_comparison.csv", "and run_comparison.png")


if __name__ == "__main__":
    main()
