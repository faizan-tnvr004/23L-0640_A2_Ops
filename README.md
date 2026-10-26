# VisionGuard: AI vs Real image classification & deployment

![CI](https://github.com/faizan-tnvr004/23L-0640_A2_Ops/actions/workflows/ci.yml/badge.svg)

Flask web service that classifies an uploaded image as AI-generated or real (TensorFlow/Keras),
with MLflow + DagsHub tracking, DVC data versioning, a database audit trail, Docker Compose and cloud deployment.

```
app.py                  Flask app (local mode, or gateway mode when CLOUD_API_URL is set)
src/                    config, preprocessing, data pipeline, models, evaluation, inference, database
scripts/                audit, EDA, splits, training, tuning, registry, error analysis, Grad-CAM, screenshots
toy_labs/               MLflow + Keras Tuner toy labs
templates/, static/     HTML/CSS front end
tests/                  pytest suite (uses a stub model)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
docs/RUNBOOK.md         every command, in order
docs/REPORT_TEMPLATE.md evidence checklist for the PDF report
```
Start with `docs/RUNBOOK.md`. Copy `.env.example` to `.env` (never commit it).
