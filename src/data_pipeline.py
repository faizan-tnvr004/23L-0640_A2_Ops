"""Task 2.1: tf.data input pipeline.

read split CSV -> from_tensor_slices(path, label) -> read + decode (RGB, 3 channels)
-> bilinear resize to IMG_SIZE x IMG_SIZE -> [train only] shuffle + random augmentation
-> batch -> prefetch(AUTOTUNE).
tf.image.resize does not crop or pad, so non-square images are stretched to a square.
"""
from pathlib import Path

import pandas as pd
import tensorflow as tf

from .config import IMG_SIZE, ROOT

AUTOTUNE = tf.data.AUTOTUNE


def load_split(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _abs(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)


def _load(path, label, img_size):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size])  # float32, still 0..255
    return img, tf.cast(label, tf.float32)


def build_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),      # +/- 5% of a full turn (~18 degrees)
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")


def make_dataset(df: pd.DataFrame, training: bool, batch_size: int = 32, img_size: int = IMG_SIZE,
                 augment: bool = True, seed: int = 42) -> tf.data.Dataset:
    paths = [_abs(p) for p in df["path"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, df["label"].values.astype("float32")))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _load(p, y, img_size), num_parallel_calls=AUTOTUNE)
    if training and augment:
        aug = build_augmenter()
        ds = ds.batch(batch_size).map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.batch(batch_size)
    return ds.prefetch(AUTOTUNE)
