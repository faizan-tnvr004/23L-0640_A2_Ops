"""Tiny stand-in model used by CI and tests (the real model is DVC-tracked)."""
import tensorflow as tf

from .config import IMG_SIZE


def build_stub_model(path: str) -> str:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output_classification")(x)
    model = tf.keras.Model(inputs, out, name="stub_model")
    model.save(path)
    return path
