"""Create stratified 70/15/15 train/val/test CSVs from the audit table.

Corrupt files and exact duplicates are excluded first so no image leaks across splits.
Label convention: AI = 1, Real = 0 (the model outputs P(AI)).
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(ROOT / "artifacts" / "dataset_audit_full.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.audit)
    df = df[(~df.corrupt) & (~df.is_duplicate)].copy()
    df["label"] = (df["class"] == "AI").astype(int)
    df = df[["path", "label", "class", "sha256"]]

    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}_split.csv", index=False)
        print(f"{name:<5}: {len(part):4d} images  (AI={int(part.label.sum())}, Real={int((1 - part.label).sum())})")
    assert not (set(train.sha256) & set(test.sha256)) and not (set(train.sha256) & set(val.sha256))
    print("Splits written to", out)


if __name__ == "__main__":
    main()
