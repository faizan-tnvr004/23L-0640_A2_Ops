"""Task 2.7 toy lab: random search over units and learning rate on Fashion-MNIST."""
import keras_tuner as kt
import pandas as pd
import tensorflow as tf


def build(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)), tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("units", 32, 128, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    model.compile(tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                  "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    (x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
    x = x / 255.0
    tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=5, seed=42,
                            directory="toy_labs/tuner_out", project_name="fmnist", overwrite=True)
    tuner.search(x, y, validation_split=0.2, epochs=3, batch_size=128)
    rows = [{"trial_id": t.trial_id, **t.hyperparameters.values, "best_val_accuracy": t.score}
            for t in tuner.oracle.get_best_trials(5)]
    print(pd.DataFrame(rows).sort_values("best_val_accuracy", ascending=False).to_markdown(index=False))
    print("Best hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)
