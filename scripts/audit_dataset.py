"""Task 1.2: audit the dataset (own script, no dataset helper libraries).

Expected layout:   data/raw/ai/*.jpg   data/raw/real/*.jpg
(Kaggle folders are usually named AiArtData and RealArt: rename them or pass --ai-dir/--real-dir.)

For every image: class, file size, width, height, colour mode, format, SHA-256.
Unreadable files and exact duplicates (same SHA-256) are flagged.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--ai-dir", default="ai")
    ap.add_argument("--real-dir", default="real")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "dataset_audit_full.csv"))
    args = ap.parse_args()

    root = Path(args.root)
    print(f"Scanning dataset in '{root}' ...")
    rows = []
    for cls, sub in (("AI", args.ai_dir), ("Real", args.real_dir)):
        folder = root / sub
        if not folder.is_dir():
            sys.exit(f"Folder not found: {folder}")
        for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT):
            rec = {"path": path.relative_to(ROOT).as_posix() if ROOT in path.resolve().parents
                   else path.as_posix(),
                   "class": cls, "file_size_kb": path.stat().st_size / 1024, "sha256": sha256_of(path),
                   "width": None, "height": None, "mode": None, "format": None, "corrupt": False}
            try:
                with Image.open(path) as img:
                    rec.update(width=img.width, height=img.height, mode=img.mode, format=img.format)
                    img.verify()
                with Image.open(path) as img:
                    img.load()  # a full decode catches truncated files that verify() misses
            except Exception:
                rec["corrupt"] = True
            rows.append(rec)

    df = pd.DataFrame(rows)
    df["is_duplicate"] = df.duplicated(subset="sha256", keep="first") & ~df["corrupt"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    ok = df[~df["corrupt"]]
    summary = []
    for name, sub in [*ok.groupby("class"), ("Overall (all)", ok)]:
        summary.append({
            "class": name, "count": len(sub),
            "width min/med/max": f"{sub.width.min():.0f} / {sub.width.median():.0f} / {sub.width.max():.0f}",
            "height min/med/max": f"{sub.height.min():.0f} / {sub.height.median():.0f} / {sub.height.max():.0f}",
            "size KB min/med/max": f"{sub.file_size_kb.min():.1f} / {sub.file_size_kb.median():.1f} / {sub.file_size_kb.max():.1f}",
            "formats": ",".join(sorted(sub.format.unique())), "modes": ",".join(sorted(sub["mode"].unique())),
        })
    table = pd.DataFrame(summary).to_markdown(index=False)
    text = (f"TASK 1.2 DATASET AUDIT SUMMARY\n{table}\n\n"
            f"Corrupted/unreadable files : {int(df['corrupt'].sum())}\n"
            f"Exact byte-level duplicates: {int(df['is_duplicate'].sum())}\n")
    print(text)
    (Path(args.out).parent / "audit_summary.md").write_text(text, encoding="utf-8")
    if df["is_duplicate"].any():
        print("Duplicate files (later copies are excluded from the splits):")
        print(df[df["is_duplicate"]][["path", "class"]].to_string(index=False))
    print(f"Full per-image table saved to '{args.out}'")


if __name__ == "__main__":
    main()
