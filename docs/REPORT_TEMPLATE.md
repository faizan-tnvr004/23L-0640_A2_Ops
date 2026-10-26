# <RollNumber>_VisionGuard.pdf: report skeleton

Front matter: name, roll number, GitHub / DagsHub / live-URL links, graded commit hash
(`git rev-parse HEAD`), AI/tutorial disclosure (say honestly which parts an AI assistant helped with).

For every sub-task write: **what you did, why, how it works in your own words, what went wrong.**
Every screenshot needs a caption, a reference in the text, and visible context (terminal prompt with your
username, date/time, project name). Every number must come from YOUR runs.

| Task | Evidence needed (from the assignment) | Produced by | Done |
|---|---|---|---|
| 1.1 | terminal versions + GPU check | `scripts/verify_env.py` | [ ] |
| 1.2 | per-class counts, min/median/max, corrupt, duplicates | `scripts/audit_dataset.py` | [ ] |
| 1.3 | sample grid + 3 histograms, 2-3 sentences each | `scripts/eda.py` | [ ] |
| 1.4 | toy DVC git log; project `data/raw.dvc`; pointer-file + why-not-Git explanation | RUNBOOK 1.4 | [ ] |
| 2.1 | pipeline description, augmented batch figure, image size + aspect ratio discussion | `src/data_pipeline.py` | [ ] |
| 2.2 | model summary, per-layer table, loss/optimizer/metric justification, why each layer | `src/models.py`, `model_specifications.txt` | [ ] |
| 2.3 | two sets of curves, trainable layers per stage, preprocessing explanation | `train.py` output | [ ] |
| 2.4 | curves with best epoch marked; seeds + callbacks | `plots/training_curves_*.png` | [ ] |
| 2.5 | metric table (both models), confusion matrix, ROC, threshold justification, which error is worse | `metrics.json`, plots | [ ] |
| 2.6 | 10 misclassified images + YOUR hypothesis for each | `scripts/error_analysis.py` | [ ] |
| 2.7 | toy tuner table, 8-trial table, which setting mattered, why best-val-score overfits val set | tuner scripts | [ ] |
| Bonus | 4 correct + 4 wrong Grad-CAM, comment | `scripts/gradcam.py` | [ ] |
| 3.1 | both local MLflow runs, comparison view, 6 definitions | `toy_labs/mlflow_toy.py` | [ ] |
| 3.2 | DagsHub project, remote toy run, `.env` ignored | RUNBOOK 2 | [ ] |
| 3.3 | run page (params, curves, tags) for manual AND autolog; what each captures/misses | `train.py --logging` | [ ] |
| 3.4 | Artifacts tab | MLflow UI | [ ] |
| 3.5 | >= 6 runs, one change each, one table, chosen best + justification | `scripts/compare_runs.py` | [ ] |
| 3.6 | registry with 2 versions + `@champion`; alias-load terminal output | `register_model.py`, `load_champion_by_alias.py` | [ ] |
| 4.1 | console output on 2 known images; why load once; why preprocessing must match | `python src/inference.py ...` | [ ] |
| 4.2 | 3 failure-case screenshots (413, 415, undecodable 400) | `scripts/capture_screenshots.py` | [ ] |
| 4.3 | upload page + result page + history menu | same | [ ] |
| 4.4 | `/health` JSON; why debug mode off | same | [ ] |
| 5 | history page showing decision, time, model, confidence | same | [ ] |
| 6.1 | each Docker action + 5 definitions | RUNBOOK 7 | [ ] |
| 6.2 | Dockerfile link, instruction-order explanation, `whoami` in container | | [ ] |
| 6.3 | compose link, services healthy, DB port closed, model-delivery decision | | [ ] |
| 6.4 | two size numbers + largest layers + conclusion | `docker images/history` | [ ] |
| 6.5 | lifecycle commands, deliberate break, log excerpt, fix | | [ ] |
| 6.6 | red + green toy run, green project run, README badge, secrets explanation | | [ ] |
| 7 | step-by-step deployment with screenshots, Postman test, API URL, phone test from another network | RUNBOOK 8 | [ ] |
| 8 | local Flask -> cloud API with several inputs, history page | RUNBOOK 9 | [ ] |

## Design decisions worth explaining in your own words
- Model files contain their own Rescaling layer, so raw 0-255 pixels enter at training and serving (no skew);
  serving resizes with the same `tf.image.resize` as training.
- Label AI = 1; the model outputs P(AI). Threshold is chosen on validation F1, then evaluated once on test.
- "Confidence" shown in the UI is rescaled so the calibrated threshold maps to 0.5 (`calibrated_confidence`).
- Uploads are read into memory, validated by extension AND real decoded format, never written to disk.
- Gateway mode: the local Flask app forwards to the cloud API and logs to its own database too.
- API_KEY protects the cloud `/predict` endpoint (`X-API-Key`).
- Fine-tuning keeps BatchNorm frozen (deliberate deviation from "unfreeze everything in the block"; justify it).
