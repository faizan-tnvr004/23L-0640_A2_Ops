"""Task 2 bonus: Grad-CAM for 4 correct and 4 wrong test predictions (MobileNetV2 champion)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from src.models import get_backbone  # noqa: E402
from src.preprocessing import decode_to_rgb_array, resize_for_model  # noqa: E402


def gradcam(model, x):
    """Heat map of the last conv feature map (backbone output) w.r.t. the AI score."""
    base = get_backbone(model)
    idx = model.layers.index(base)
    with tf.GradientTape() as tape:
        h = tf.convert_to_tensor(x)
        for layer in model.layers[1:idx]:
            h = layer(h)
        feats = base(h, training=False)
        tape.watch(feats)
        z = feats
        for layer in model.layers[idx + 1:]:
            z = layer(z, training=False) if "dropout" in layer.name else layer(z)
        score = z[:, 0]
    grads = tape.gradient(score, feats)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * feats, axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-9)


def overlay(img_arr, cam):
    heat = np.array(Image.fromarray((cam * 255).astype("uint8")).resize(img_arr.shape[1::-1]))
    return heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mobilenetv2_transfer_manual")
    ap.add_argument("--model", default=str(ROOT / "models" / "champion_model.keras"))
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    df = pd.read_csv(ROOT / "artifacts" / "runs" / args.run / "test_predictions.csv")
    df["pred"] = (df.p_ai >= args.threshold).astype(int)
    right = df[df.pred == df.label].sample(4, random_state=1)
    wrong = df[df.pred != df.label].sample(min(4, (df.pred != df.label).sum()), random_state=1)
    picks = [("Correct", r) for _, r in right.iterrows()] + [("Misclassified", r) for _, r in wrong.iterrows()]

    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
    for col, (kind, row) in enumerate(picks):
        p = Path(row.path)
        data = (p if p.is_absolute() else ROOT / p).read_bytes()
        arr = decode_to_rgb_array(data)
        cam = gradcam(model, resize_for_model(arr))
        small = np.asarray(Image.fromarray(arr).resize((224, 224)))
        axes[0, col].imshow(small)
        axes[0, col].set_title(f"{kind}\ntruth={'AI' if row.label else 'Real'} P(AI)={row.p_ai:.2f}", fontsize=8)
        axes[1, col].imshow(small)
        axes[1, col].imshow(overlay(small, cam), cmap="jet", alpha=0.45)
        for r in (0, 1):
            axes[r, col].axis("off")
    fig.suptitle("Bonus: Grad-CAM, 4 correct vs 4 misclassified predictions")
    fig.tight_layout()
    out = ROOT / "artifacts" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_2_19_gradcam.png", dpi=150)
    print("saved", out / "fig_2_19_gradcam.png")


if __name__ == "__main__":
    main()
