"""Preprocessing shared by training and serving.

Training pipeline: tf.io.decode_image(channels=3) -> tf.image.resize (bilinear).
Serving: PIL decode -> RGB -> the SAME tf.image.resize.  The model itself starts
with a Rescaling layer (1/255 for the CNN, x/127.5 - 1 for MobileNetV2, which is
exactly mobilenet_v2.preprocess_input), so raw 0-255 pixels go in at both stages
and train/serve skew is impossible by construction.
"""
import io

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import IMG_SIZE


def decode_to_rgb_array(data: bytes) -> np.ndarray:
    """Decode image bytes to an HxWx3 uint8 array (alpha dropped, like TF)."""
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def resize_for_model(arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Bilinear resize to size x size (no crop/pad) and add a batch axis."""
    resized = tf.image.resize(arr, [size, size])
    return tf.expand_dims(tf.cast(resized, tf.float32), 0).numpy()
