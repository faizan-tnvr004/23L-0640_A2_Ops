"""Standalone inference module: the model is loaded ONCE and reused."""
import threading
import time

import numpy as np
import tensorflow as tf

from .config import load_settings
from .preprocessing import decode_to_rgb_array, resize_for_model


def calibrated_confidence(p: float, threshold: float) -> float:
    """Confidence in the verdict, with the decision threshold mapped to 0.5.

    A raw score of 0.30 with threshold 0.279 is an (uncertain) AI verdict; piecewise
    rescaling makes 'confidence' consistent with the calibrated threshold.
    """
    t = min(max(threshold, 1e-6), 1 - 1e-6)
    p_cal = 0.5 * p / t if p < t else 0.5 + 0.5 * (p - t) / (1 - t)
    return float(max(p_cal, 1 - p_cal))


class VisionGuardClassifier:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_path: str, threshold: float, model_version: str):
        started = time.perf_counter()
        self.model_path = model_path
        self.threshold = threshold
        self.model_version = model_version
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self._predict_lock = threading.Lock()
        # warm-up so the first real request is not slow
        self.model(np.zeros((1, 224, 224, 3), dtype="float32"), training=False)
        self.load_ms = (time.perf_counter() - started) * 1000

    @classmethod
    def get(cls, model_path=None, threshold=None, model_version=None):
        settings = load_settings()
        model_path = model_path or settings.model_path
        threshold = settings.decision_threshold if threshold is None else threshold
        model_version = model_version or settings.model_version
        with cls._lock:
            inst = cls._instance
            if inst is None or inst.model_path != model_path or inst.threshold != threshold:
                cls._instance = cls(model_path, threshold, model_version)
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    def predict(self, image_bytes: bytes) -> dict:
        started = time.perf_counter()
        x = resize_for_model(decode_to_rgb_array(image_bytes))
        with self._predict_lock:
            raw = float(self.model(x, training=False).numpy().ravel()[0])
        is_ai = raw >= self.threshold
        return {
            "label": "AI-Generated" if is_ai else "Authentic (Real)",
            "is_ai": bool(is_ai),
            "confidence": calibrated_confidence(raw, self.threshold),
            "raw_score": raw,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }


if __name__ == "__main__":
    import sys

    clf = VisionGuardClassifier.get()
    print(f"[VisionGuard] model loaded in {clf.load_ms:.1f} ms, threshold={clf.threshold}")
    for path in sys.argv[1:]:
        with open(path, "rb") as fh:
            print(path, clf.predict(fh.read()))
