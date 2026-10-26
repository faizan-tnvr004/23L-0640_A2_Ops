"""Task 3.1 / 3.2 toy lab: tiny Keras model on Fashion-MNIST tracked with MLflow.

Local:   python toy_labs/mlflow_toy.py            (then run `mlflow ui` and open http://127.0.0.1:5000)
DagsHub: python toy_labs/mlflow_toy.py --remote   (needs MLFLOW_TRACKING_* in .env)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import tensorflow as tf  # noqa: E402

from src import config  # noqa: F401,E402  (loads .env)


def train(lr: float, run_name: str):
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    x_tr, x_te = x_tr / 255.0, x_te / 255.0
    model = tf.keras.Sequential([tf.keras.layers.Input((28, 28)), tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(64, activation="relu"),
                                 tf.keras.layers.Dense(10, activation="softmax")])
    model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"learning_rate": lr, "batch_size": 64, "epochs": 5, "optimizer": "adam",
                           "architecture": "2-layer-dense"})
        hist = model.fit(x_tr, y_tr, validation_split=0.2, epochs=5, batch_size=64, verbose=2).history
        for epoch in range(5):
            mlflow.log_metrics({k: v[epoch] for k, v in hist.items()}, step=epoch + 1)
        fig, ax = plt.subplots()
        ax.plot(hist["loss"], label="train")
        ax.plot(hist["val_loss"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)
        model.save("toy_model.keras")
        mlflow.log_artifact("toy_model.keras", "model")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true")
    args = ap.parse_args()
    if args.remote:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])  # from .env, never hard-coded
        mlflow.set_experiment("Toy-Experiment-Remote")
        train(0.001, "toy_remote_run")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Toy-Experiment-Local")
        for lr in (0.001, 0.01):
            train(lr, f"toy_run_lr_{lr}")
