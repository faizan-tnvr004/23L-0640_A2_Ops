"""Task 2.2 / 2.3: baseline CNN and MobileNetV2 transfer model.

Both models take RAW 0-255 pixels and rescale internally:
  * CNN:         Rescaling(1/255)
  * MobileNetV2: Rescaling(1/127.5, offset=-1)  == mobilenet_v2.preprocess_input
"""
import tensorflow as tf
from tensorflow.keras import layers

from .config import IMG_SIZE


def compile_model(model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_baseline_cnn(img_size: int = IMG_SIZE, dropout_block: float = 0.3, dropout_head: float = 0.4):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_tensor")
    x = layers.Rescaling(1.0 / 255, name="rescaling")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=True, name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool_{i}")(x)
    x = layers.Dropout(dropout_block, name="dropout_block4")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(dropout_head, name="dropout_head")(x)
    out = layers.Dense(1, activation="sigmoid", name="output_classification")(x)
    return tf.keras.Model(inputs, out, name="Baseline_CNN")


def build_mobilenet(img_size: int = IMG_SIZE, dropout: float = 0.3, pretrained: bool = True):
    base = tf.keras.applications.MobileNetV2(weights="imagenet" if pretrained else None,
                                             include_top=False, input_shape=(img_size, img_size, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_tensor")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)  # BatchNorm stays in inference mode, also during fine-tuning
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_head")(x)
    out = layers.Dense(1, activation="sigmoid", name="output_classification")(x)
    return tf.keras.Model(inputs, out, name="MobileNetV2_Transfer")


def get_backbone(model):
    return next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))


def set_unfrozen_layers(model, n: int):
    """Freeze the whole backbone (n == 0) or unfreeze only its last n layers.

    BatchNormalization layers always stay frozen: updating their statistics on a small
    dataset is a common cause of fine-tuning collapse.
    """
    base = get_backbone(model)
    if n <= 0:
        base.trainable = False
        return model
    base.trainable = True
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-n:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model


def trainable_table(model):
    """Rows (scope, layer name, type, trainable params) for the report (Task 2.3)."""
    rows = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub in layer.layers:
                n = sum(int(w.shape.num_elements()) for w in sub.trainable_weights)
                if n:
                    rows.append((layer.name, sub.name, type(sub).__name__, n))
        else:
            n = sum(int(w.shape.num_elements()) for w in layer.trainable_weights)
            if n:
                rows.append(("Head", layer.name, type(layer).__name__, n))
    return rows


def layer_table(model):
    """Rows (name, type, output shape, params) for the Task 2.2 table."""
    rows = []
    for layer in model.layers:
        shape = getattr(layer, "output", None)
        rows.append((layer.name, type(layer).__name__, str(getattr(shape, "shape", "")), layer.count_params()))
    return rows
