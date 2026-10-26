# VisionGuard runbook (commands in the order you need them)

Replace `ROLLNO` with your roll number everywhere (project / experiment / repo / cloud names).
Use Python 3.10 or 3.11. Commands are PowerShell/bash neutral unless noted.

## 0. Setup (Task 1.1)
```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_env.py      # screenshot this
```
Download the Kaggle dataset, put the two class folders at `data/raw/ai/` and `data/raw/real/`
(rename `AiArtData` -> `ai`, `RealArt` -> `real`, or pass `--ai-dir/--real-dir`).

## 1. Data (Task 1)
```
python scripts/audit_dataset.py        # 1.2: counts, min/median/max, corrupt, duplicates
python scripts/eda.py                  # 1.3: figures in artifacts/figures/  (write 2-3 sentences each)
python scripts/make_splits.py          # stratified 70/15/15 CSVs in data/splits/
```
### 1.4 DVC toy part (separate folder)
```
mkdir dvc_toy && cd dvc_toy
git init && dvc init
# create toy_data.csv with a few rows
dvc add toy_data.csv && git add . && git commit -m "Track v1"
# edit the CSV (add rows)
dvc add toy_data.csv && git add . && git commit -m "Track v2"
dvc remote add -d localremote ../dvc_remote_toy && dvc push
git checkout HEAD~1 toy_data.csv.dvc && dvc checkout       # back to v1
del toy_data.csv && dvc pull                                # delete + restore
git log --oneline                                           # screenshot
```
### 1.4 Apply to the project
```
dvc init
dvc add data/raw
dvc remote add -d origin https://dagshub.com/<user>/23L-0640_A2_Ops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-user>
dvc remote modify origin --local password <dagshub-token>
git add data/raw.dvc data/.gitignore .dvc && git commit -m "Track dataset with DVC" && dvc push && git push
```
Explain in the report: `data/raw.dvc` stores the directory's md5, size, file count and path; Git is built for
text diffs, so binaries bloat history, slow clones and hit GitHub's 100 MB file limit.

## 2. Tracking setup (Task 3.1 / 3.2)
```
python toy_labs/mlflow_toy.py && mlflow ui          # two local runs (lr 0.001 and 0.01), compare them
python toy_labs/tuner_toy.py                        # Task 2.7 toy
```
Create the DagsHub project from your GitHub repo, copy the MLflow URL + token into `.env`
(see `.env.example`), then `python toy_labs/mlflow_toy.py --remote`. Prove `.env` is ignored with
`git status --ignored`.

## 3. Training: the six controlled runs (Tasks 2.x, 3.3-3.5)
Each run changes ONE thing relative to a reference run. Suggested set:
```
python scripts/train.py --model baseline  --run-name baseline_cnn_manual
python scripts/train.py --model baseline  --run-name baseline_cnn_low_lr --lr 1e-4
python scripts/train.py --model baseline  --run-name baseline_cnn_no_augmentation --no-augment
python scripts/train.py --model mobilenet --run-name mobilenetv2_transfer_manual --unfreeze-layers 20
python scripts/train.py --model mobilenet --run-name mobilenetv2_fully_frozen --unfreeze-layers 0
python scripts/train.py --model mobilenet --run-name mobilenetv2_high_lr --stage2-lr 1e-3
# Task 3.3 autologging comparison (same settings as a manual run):
python scripts/train.py --model mobilenet --run-name mobilenetv2_transfer_autolog --logging autolog
python scripts/train.py --model baseline  --run-name baseline_cnn_autolog --logging autolog
python scripts/tune_visionguard.py --trials 8      # Task 2.7
python scripts/compare_runs.py                     # Task 3.5 table + chart
```
Your numbers will differ from anyone else's: report what YOUR runs produce and choose the champion
from them (the validation threshold is printed per run in `artifacts/runs/<run>/metrics.json`).

## 4. Analysis (Tasks 2.5, 2.6, bonus)
```
python scripts/error_analysis.py --run <champion_run_name> --threshold <val_threshold>
python scripts/gradcam.py        --run <champion_run_name> --threshold <val_threshold>
```
Fill the `hypothesis` column of `artifacts/figures/error_analysis_table.csv` yourself after looking at each image.

## 5. Registry (Task 3.6)
```
python scripts/register_model.py --baseline-run <run_id> --champion-run <run_id>
python scripts/load_champion_by_alias.py data/raw/ai/<some image>.jpg --threshold <val_threshold>
dvc add models/champion_model.keras && git add models/champion_model.keras.dvc && git commit && dvc push
```
Set `DECISION_THRESHOLD` in `.env` to your calibrated value.

## 6. App (Tasks 4 and 5)
```
python src/inference.py data/raw/real/<x>.jpg data/raw/ai/<y>.jpg   # 4.1 console output
python app.py                                                       # http://127.0.0.1:5000
python scripts/capture_screenshots.py --real <real.jpg> --ai <ai.jpg>
pytest -q
```

## 7. Docker (Task 6)
```
copy .env.example .env      # edit passwords, SECRET_KEY, threshold
docker compose up -d --build
docker compose ps           # both healthy
curl http://localhost:5000/health
Test-NetConnection localhost -Port 5432     # must FAIL: DB port is not published
docker run --rm visionguard-web:latest whoami      # appuser
docker images visionguard-web ; docker history visionguard-web:latest
docker compose logs web | docker compose restart web | docker compose exec web sh | docker compose exec db sh
```
Task 6.4: record the image size, apply one optimisation (e.g. build once with `tensorflow` + `python:3.10`
and once with `tensorflow-cpu` + `slim`) and report both numbers.
Task 6.5: break it on purpose (rename the model file, or use a wrong DB password), diagnose from
`docker compose logs`, fix it.
Task 6.6: do the toy matrix workflow in a throwaway repo first; then push this repo and add the badge:
`![CI](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)`.

## 8. Cloud API (Task 7, pathway C: AWS EC2)
1. Launch Ubuntu 22.04 `t3.micro`; open inbound TCP 22 and 8000 in the Security Group; add a 2 GB swap file.
2. `sudo apt-get update && sudo apt-get install -y docker.io`
3. Build + push from your PC: `docker build -t <dockerhub-user>/visionguard-api:v1 . && docker push ...`
   (copy the real `champion_model.keras` into `models/` first so it is baked into the image).
4. On the VM: `sudo docker run -d -p 8000:5000 --name visionguard-api -e API_KEY=<secret> -e DECISION_THRESHOLD=<thr> -e SECRET_KEY=<x> <dockerhub-user>/visionguard-api:v1`
5. `curl http://localhost:8000/health` on the VM, then Postman: POST `http://<ip>:8000/predict`, body form-data,
   key `image` (File), header `X-API-Key`. Test once from a phone on mobile data. Screenshot each step.
Note: the free tier changes; check the AWS console for current limits and remember to stop the instance afterwards.

## 9. Integration (Task 8)
In `.env` on your PC: `CLOUD_API_URL=http://<ip>:8000` and `CLOUD_API_KEY=<secret>`, then `python app.py`.
Upload several real and AI images, screenshot each result page, then the History page.
