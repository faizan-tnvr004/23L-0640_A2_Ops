"""Task 3.6: load models:/VisionGuard-Classifier@champion and classify one image.

python scripts/load_champion_by_alias.py path/to/image.jpg [--threshold 0.279]

Resolves the alias in the registry, downloads that run's model artifact, then predicts.
If the remote registry cannot be reached it falls back to the local DVC copy and says so.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
import tensorflow as tf  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src import config  # noqa: F401,E402
from src.inference import calibrated_confidence  # noqa: E402
from src.preprocessing import decode_to_rgb_array, resize_for_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--name", default="VisionGuard-Classifier")
    ap.add_argument("--threshold", type=float, default=float(os.getenv("DECISION_THRESHOLD", "0.5")))
    args = ap.parse_args()

    print(f"Loading champion model from URI: 'models:/{args.name}@champion' ...")
    model_file = None
    try:
        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        version = MlflowClient().get_model_version_by_alias(args.name, "champion")
        print(f"[OK] alias @champion -> version {version.version} (run {version.run_id})")
        model_file = mlflow.artifacts.download_artifacts(run_id=version.run_id,
                                                         artifact_path="model/model.keras")
    except Exception as exc:
        print(f"[NOTE] registry load failed ({type(exc).__name__}: {exc})")
        model_file = str(ROOT / "models" / "champion_model.keras")
        print(f"Falling back to local DVC artifact '{model_file}'")

    model = tf.keras.models.load_model(model_file, compile=False)
    print(f"[OK] Loaded '{model_file}'")
    data = Path(args.image).read_bytes()
    raw = float(model(resize_for_model(decode_to_rgb_array(data)), training=False).numpy().ravel()[0])
    is_ai = raw >= args.threshold
    print("\nPREDICTION RESULT")
    print(f"  Image path        : {args.image}")
    print(f"  Raw sigmoid output: {raw:.4f}")
    print(f"  Decision threshold: {args.threshold}")
    print(f"  Predicted class   : {'AI-Generated (Fake)' if is_ai else 'Authentic (Real)'}")
    print(f"  Confidence        : {calibrated_confidence(raw, args.threshold) * 100:.2f}%")


if __name__ == "__main__":
    main()
