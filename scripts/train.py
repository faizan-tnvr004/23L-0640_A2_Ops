"""Train a VisionGuard model and track it with MLflow (Tasks 2.3, 2.4, 2.5, 3.3, 3.4, 3.5).

Examples (see docs/RUNBOOK.md for the six required runs):
  python scripts/train.py --model baseline  --run-name baseline_cnn_manual
  python scripts/train.py --model mobilenet --run-name mobilenetv2_transfer_manual --unfreeze-layers 20
  python scripts/train.py --model mobilenet --run-name mobilenetv2_transfer_autolog --logging autolog

Test data is used exactly once, at the end of the run, with a threshold chosen on validation data.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from src import evaluation as ev  # noqa: E402
from src.config import IMG_SIZE  # noqa: E402
from src.data_pipeline import load_split, make_dataset  # noqa: E402
from src.models import (build_baseline_cnn, build_mobilenet, compile_model, layer_table,  # noqa: E402
                        set_unfrozen_layers, trainable_table)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def dvc_data_hash() -> str:
    f = ROOT / "data" / "raw.dvc"
    if f.exists():
        m = re.search(r"md5:\s*(\S+)", f.read_text())
        if m:
            return m.group(1)
    return "unknown"


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual logging: one MLflow metric point per epoch (epoch index is already global)."""

    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: float(v) for k, v in (logs or {}).items()}
        metrics["learning_rate"] = float(np.array(self.model.optimizer.learning_rate))
        mlflow.log_metrics(metrics, step=epoch + 1)


def fit_stage(model, train_ds, val_ds, lr, epochs, start_epoch, ckpt_path, manual):
    compile_model(model, lr)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    if manual:
        cbs.append(MlflowEpochLogger())
    hist = model.fit(train_ds, validation_data=val_ds, epochs=start_epoch + epochs,
                     initial_epoch=start_epoch, callbacks=cbs, verbose=2).history
    best_epoch = start_epoch + int(np.argmin(hist["val_loss"])) + 1
    return hist, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "mobilenet"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--logging", choices=["manual", "autolog"], default="manual")
    ap.add_argument("--experiment", default=os.getenv("PROJECT_NAME", "23L-0640_A2_Ops"))
    ap.add_argument("--lr", type=float, default=1e-3, help="baseline learning rate")
    ap.add_argument("--epochs", type=int, default=10, help="baseline max epochs")
    ap.add_argument("--stage1-lr", type=float, default=1e-4)
    ap.add_argument("--stage1-epochs", type=int, default=8)
    ap.add_argument("--stage2-lr", type=float, default=1e-5)
    ap.add_argument("--stage2-epochs", type=int, default=6)
    ap.add_argument("--unfreeze-layers", type=int, default=20, help="0 = backbone stays frozen")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits-dir", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--no-mlflow", action="store_true", help="skip tracking (debugging)")
    args = ap.parse_args()

    set_seed(args.seed)
    run_dir = ROOT / "artifacts" / "runs" / args.run_name
    for sub in ("plots", "checkpoints"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    splits = Path(args.splits_dir)
    dfs = {k: load_split(splits / f"{k}_split.csv") for k in ("train", "val", "test")}
    train_ds = make_dataset(dfs["train"], True, args.batch_size, args.img_size, not args.no_augment, args.seed)
    val_ds = make_dataset(dfs["val"], False, args.batch_size, args.img_size)
    test_ds = make_dataset(dfs["test"], False, args.batch_size, args.img_size)

    if args.model == "baseline":
        model = build_baseline_cnn(args.img_size)
    else:
        model = build_mobilenet(args.img_size, args.dropout)

    manual = args.logging == "manual"
    if not args.no_mlflow:
        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(args.experiment)
        if manual:
            mlflow.tensorflow.autolog(disable=True)
        else:
            mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

    run_ctx = (mlflow.start_run(run_name=args.run_name) if not args.no_mlflow
               else _NullRun())
    with run_ctx as run:
        tracking = not args.no_mlflow
        if tracking:
            mlflow.set_tags({"git_commit_hash": git_commit(), "dvc_data_hash": dvc_data_hash(),
                             "random_seed": str(args.seed), "logging_mode": args.logging,
                             "model_type": "baseline_cnn" if args.model == "baseline"
                             else "transfer_learning_mobilenetv2"})
            if manual:
                mlflow.log_params({
                    "backbone": "custom_4block_cnn" if args.model == "baseline" else "MobileNetV2",
                    "weights": "random" if args.model == "baseline" else "imagenet",
                    "img_size": args.img_size, "batch_size": args.batch_size,
                    "augmentation": not args.no_augment, "loss": "binary_crossentropy",
                    "optimizer": "Adam", "seed": args.seed,
                    **({"lr": args.lr, "epochs": args.epochs} if args.model == "baseline" else
                       {"stage1_lr": args.stage1_lr, "stage1_epochs": args.stage1_epochs,
                        "stage2_lr": args.stage2_lr, "stage2_epochs": args.stage2_epochs,
                        "unfreeze_layers": args.unfreeze_layers, "dropout": args.dropout}),
                })

        history_parts = []  # (title, history, start_epoch, best_epoch)
        specs = []
        if args.model == "baseline":
            specs.append(("Layer table", layer_table(model)))
            hist, best = fit_stage(model, train_ds, val_ds, args.lr, args.epochs, 0,
                                   run_dir / "checkpoints" / "best.keras", manual)
            history_parts.append((args.run_name, hist, 0, best))
        else:
            set_unfrozen_layers(model, 0)
            specs.append(("Stage 1 trainable layers", trainable_table(model)))
            hist1, best1 = fit_stage(model, train_ds, val_ds, args.stage1_lr, args.stage1_epochs, 0,
                                     run_dir / "checkpoints" / "stage1.keras", manual)
            history_parts.append((f"{args.run_name} stage1", hist1, 0, best1))
            if args.unfreeze_layers > 0:
                set_unfrozen_layers(model, args.unfreeze_layers)
                specs.append(("Stage 2 trainable layers", trainable_table(model)))
                start = len(hist1["loss"])
                hist2, best2 = fit_stage(model, train_ds, val_ds, args.stage2_lr, args.stage2_epochs,
                                         start, run_dir / "checkpoints" / "stage2.keras", manual)
                history_parts.append((f"{args.run_name} stage2", hist2, start, best2))

        # ------------------------------------------------ evaluation (test used once)
        y_val, y_test = dfs["val"].label.values, dfs["test"].label.values
        p_val, p_test = ev.predict_probs(model, val_ds), ev.predict_probs(model, test_ds)
        thr, curve = ev.best_f1_threshold(y_val, p_val)
        m50, mthr = ev.compute_metrics(y_test, p_test, 0.5), ev.compute_metrics(y_test, p_test, thr)

        plots = run_dir / "plots"
        for i, (title, hist, start, best) in enumerate(history_parts, start=1):
            ev.plot_curves(hist, start, best, title, plots / f"training_curves_{i}.png")
        ev.plot_confusion(mthr["confusion_matrix"], f"Confusion matrix (threshold {thr:.3f})",
                          plots / "confusion_matrix.png")
        ev.plot_roc(y_test, p_test, f"ROC curve: {args.run_name}", plots / "roc_curve.png")
        ev.plot_threshold_curve(curve, thr, plots / "validation_threshold_curve.png")

        pd.DataFrame({"path": dfs["test"].path, "label": y_test, "p_ai": p_test}).to_csv(
            run_dir / "test_predictions.csv", index=False)
        summary = {"run_name": args.run_name, "val_threshold": thr, "test_at_0.5": m50,
                   "test_at_threshold": mthr, "best_epochs": [b for *_, b in history_parts]}
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        with open(run_dir / "model_specifications.txt", "w", encoding="utf-8") as fh:
            model.summary(print_fn=lambda s, **k: fh.write(s + "\n"))
            for title, rows in specs:
                fh.write(f"\n== {title} ==\n")
                for row in rows:
                    fh.write(" | ".join(str(c) for c in row) + "\n")
        model_path = run_dir / "model.keras"
        model.save(model_path)
        env_txt = run_dir / "package_list.txt"
        env_txt.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode())

        print(json.dumps(summary, indent=2))
        for title, rows in specs:
            print(f"\n== {title} ==")
            for row in rows:
                print(" | ".join(str(c) for c in row))

        if tracking:
            mlflow.log_metrics({f"test_{k}": v for k, v in mthr.items()
                                if k not in ("confusion_matrix", "threshold")})
            mlflow.log_metrics({f"test_{k}_at_0_5": v for k, v in m50.items()
                                if k not in ("confusion_matrix", "threshold")})
            mlflow.log_metric("val_best_threshold", thr)
            mlflow.log_artifacts(str(plots), "evaluation_plots")
            for f in splits.glob("*_split.csv"):
                mlflow.log_artifact(str(f), "splits")
            mlflow.log_artifact(str(env_txt), "environment")
            mlflow.log_artifact(str(run_dir / "model_specifications.txt"), "model_specifications")
            mlflow.log_artifact(str(run_dir / "test_predictions.csv"), "evaluation_plots")
            mlflow.log_artifacts(str(run_dir / "checkpoints"), "checkpoints")
            mlflow.log_artifact(str(model_path), "model")
            print("MLflow run id:", run.info.run_id)


class _NullRun:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
