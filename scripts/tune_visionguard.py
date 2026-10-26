"""Task 2.7 (apply): Keras Tuner random search over lr, dropout and number of unfrozen layers.

python scripts/tune_visionguard.py --trials 8 --epochs 5
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import keras_tuner as kt  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.data_pipeline import load_split, make_dataset  # noqa: E402
from src.models import build_mobilenet, compile_model, set_unfrozen_layers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits-dir", default=str(ROOT / "data" / "splits"))
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    splits = Path(args.splits_dir)
    train_ds = make_dataset(load_split(splits / "train_split.csv"), True, seed=args.seed)
    val_ds = make_dataset(load_split(splits / "val_split.csv"), False)

    def hypermodel(hp):
        lr = hp.Choice("learning_rate", [1e-5, 5e-4, 1e-3])
        dropout = hp.Choice("dropout", [0.2, 0.3, 0.4])
        unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 30])
        model = build_mobilenet(dropout=dropout)
        set_unfrozen_layers(model, unfrozen)
        return compile_model(model, lr)

    tuner = kt.RandomSearch(hypermodel, objective="val_accuracy", max_trials=args.trials,
                            seed=args.seed, directory=str(ROOT / "artifacts" / "tuner"),
                            project_name="visionguard", overwrite=True)
    tuner.search(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])

    rows = []
    for trial in tuner.oracle.get_best_trials(args.trials):
        rows.append({"trial_id": trial.trial_id, **trial.hyperparameters.values,
                     "best_val_accuracy": trial.score})
    table = pd.DataFrame(rows).sort_values("best_val_accuracy", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    print("\nHYPERPARAMETER TUNING SUMMARY\n" + table.to_markdown(index=False))
    out = ROOT / "artifacts" / "tuner_results.csv"
    table.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
