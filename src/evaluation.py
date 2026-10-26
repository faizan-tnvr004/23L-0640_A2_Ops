"""Metrics, threshold calibration and plots (Tasks 2.4 / 2.5)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score, roc_curve)


def predict_probs(model, ds) -> np.ndarray:
    return model.predict(ds, verbose=0).ravel()


def compute_metrics(y, p, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def best_f1_threshold(y, p, lo=0.05, hi=0.95, step=0.005):
    """Pick the threshold that maximises F1 on VALIDATION data (never on test)."""
    rows = []
    for t in np.arange(lo, hi + 1e-9, step):
        yhat = (p >= t).astype(int)
        rows.append({"threshold": float(t),
                     "precision": precision_score(y, yhat, zero_division=0),
                     "recall": recall_score(y, yhat, zero_division=0),
                     "f1": f1_score(y, yhat, zero_division=0)})
    curve = pd.DataFrame(rows)
    best = curve.loc[(curve.f1 - curve.f1.max()).abs().lt(1e-12)]
    # tie-break: the candidate closest to 0.5 is the least extreme choice
    thr = float(best.iloc[(best.threshold - 0.5).abs().argsort().iloc[0]].threshold)
    return thr, curve


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Real (0)", "AI (1)"], yticklabels=["Real (0)", "AI (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y, p, title, path):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="navy", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_curve(curve: pd.DataFrame, thr: float, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve.threshold, curve.precision, "--", label="Precision")
    ax.plot(curve.threshold, curve.recall, "-", label="Recall")
    ax.plot(curve.threshold, curve.f1, "-.", label="F1-score")
    ax.axvline(thr, color="k", ls=":", label=f"Chosen threshold = {thr:.3f}")
    ax.set_xlabel("Decision threshold (validation set)")
    ax.set_ylabel("Score")
    ax.set_title("Validation threshold calibration")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history: dict, start_epoch: int, best_epoch: int, title: str, path):
    """Loss / accuracy curves; best_epoch is the GLOBAL (1-based) epoch to mark."""
    n = len(history["loss"])
    epochs = list(range(start_epoch + 1, start_epoch + n + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, history[key], "o-", label=f"Train {label}")
        ax.plot(epochs, history[f"val_{key}"], "s-", label=f"Val {label}")
        ax.axvline(best_epoch, color="red", ls="--", label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} curves: {title}")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
