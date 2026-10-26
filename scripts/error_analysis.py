"""Task 2.6: show 10 misclassified test images (5 false positives, 5 false negatives).

Reads artifacts/runs/<run>/test_predictions.csv written by train.py.
The 'hypothesis' column of the output CSV is left EMPTY: look at each image and write your own.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mobilenetv2_transfer_manual")
    ap.add_argument("--threshold", type=float, required=True, help="validation-chosen threshold")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "artifacts" / "runs" / args.run / "test_predictions.csv")
    df["pred"] = (df.p_ai >= args.threshold).astype(int)
    fp = df[(df.label == 0) & (df.pred == 1)].sort_values("p_ai", ascending=False).head(args.n)
    fn = df[(df.label == 1) & (df.pred == 0)].sort_values("p_ai").head(args.n)
    fp = fp.assign(error="False Positive")
    fn = fn.assign(error="False Negative")
    show = pd.concat([fp, fn]).reset_index(drop=True)

    fig, axes = plt.subplots(2, args.n, figsize=(3.4 * args.n, 7.4))
    for ax, (i, row) in zip(axes.ravel(), show.iterrows()):
        p = Path(row.path)
        with Image.open(p if p.is_absolute() else ROOT / p) as img:
            ax.imshow(img.convert("RGB"))
        truth = "AI" if row.label == 1 else "Real"
        ax.set_title(f"#{i + 1} {row.error}\ntruth={truth}  P(AI)={row.p_ai:.2f}", fontsize=8, color="red")
        ax.axis("off")
    fig.suptitle("Task 2.6: visual error analysis (top: false positives, bottom: false negatives)")
    fig.tight_layout()
    out = ROOT / "artifacts" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_2_15_error_grid.png", dpi=150)
    show.assign(hypothesis="").to_csv(out / "error_analysis_table.csv", index=False)
    print(show[["path", "label", "p_ai", "error"]].to_string())
    print("saved figure and table (fill in the 'hypothesis' column yourself) to", out)


if __name__ == "__main__":
    main()
