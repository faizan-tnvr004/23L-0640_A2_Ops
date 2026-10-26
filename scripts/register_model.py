"""Task 3.6: register two model versions and alias the best one as 'champion'.

python scripts/register_model.py --baseline-run <run_id> --champion-run <run_id>

Also copies the champion file to models/champion_model.keras (the file the serving app loads,
so the web app never calls DagsHub per request). Track it with: dvc add models/champion_model.keras
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src import config  # noqa: F401,E402  (loads .env)

NAME = "VisionGuard-Classifier"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run", required=True)
    ap.add_argument("--champion-run", required=True)
    ap.add_argument("--name", default=NAME)
    args = ap.parse_args()

    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    v1 = mlflow.register_model(f"runs:/{args.baseline_run}/model", args.name)
    client.update_model_version(args.name, v1.version, description="Version 1: best baseline custom CNN")
    v2 = mlflow.register_model(f"runs:/{args.champion_run}/model", args.name)
    client.update_model_version(args.name, v2.version, description="Version 2: champion MobileNetV2")
    client.set_registered_model_alias(args.name, "champion", v2.version)
    print(f"Registered {args.name}: v{v1.version} (baseline), v{v2.version} (champion, alias @champion)")

    dest = ROOT / "models" / "champion_model.keras"
    dest.parent.mkdir(exist_ok=True)
    local = mlflow.artifacts.download_artifacts(run_id=args.champion_run, artifact_path="model/model.keras")
    shutil.copy(local, dest)
    print("Champion model copied to", dest)


if __name__ == "__main__":
    main()
