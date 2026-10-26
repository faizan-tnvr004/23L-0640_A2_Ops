"""Task 1.3: sample grid + per-class histograms (width, height, file size)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(ROOT / "artifacts" / "dataset_audit_full.csv"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "figures"))
    ap.add_argument("--per-class", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.audit)
    df = df[(~df.corrupt) & (~df.is_duplicate)]
    sns.set_theme(style="whitegrid", palette="Set2")

    fig, axes = plt.subplots(2, args.per_class, figsize=(3.2 * args.per_class, 6.6))
    for r, cls in enumerate(["AI", "Real"]):
        sample = df[df["class"] == cls].sample(args.per_class, random_state=42)
        for ax, (_, row) in zip(axes[r], sample.iterrows()):
            p = Path(row.path)
            with Image.open(p if p.is_absolute() else ROOT / p) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{cls} ({row.width:.0f}x{row.height:.0f})", fontsize=9)
            ax.axis("off")
    fig.suptitle("Sample image grid: AI (top) vs Real (bottom)")
    fig.tight_layout()
    fig.savefig(out / "fig_1_3_sample_grid.png", dpi=150)
    plt.close(fig)

    for col, label, name in [("width", "Width (pixels)", "fig_1_4_width_hist.png"),
                             ("height", "Height (pixels)", "fig_1_5_height_hist.png"),
                             ("file_size_kb", "File size (KB)", "fig_1_6_filesize_hist.png")]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(data=df, x=col, hue="class", bins=20, kde=True, ax=ax)
        ax.set_title(f"Image {label.split(' (')[0].lower()} distribution by class")
        ax.set_xlabel(label)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
    print("Figures saved to", out)


if __name__ == "__main__":
    main()
