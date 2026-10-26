"""VisionGuard Flask application.

Two modes, chosen purely by configuration:
  * local mode   - CLOUD_API_URL empty: the champion model runs inside this process
  * gateway mode - CLOUD_API_URL set:  uploads are forwarded to the cloud API
Both modes log every prediction to the database and render the same pages.
"""
import base64
import hmac
import io
import os
import time

import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src.config import ALLOWED_EXTENSIONS, ALLOWED_PIL_FORMATS, load_settings
from src.database import Database


class UploadError(Exception):
    def __init__(self, status: int, title: str, message: str):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


def wants_json() -> bool:
    if request.args.get("format") == "json":
        return True
    best = request.accept_mimetypes
    return best.accept_json and not best.accept_html


def make_preview(data: bytes) -> str:
    """Small base64 JPEG preview so the upload never has to be stored on disk."""
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        img.thumbnail((640, 640))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def validate_upload(file_storage, max_bytes: int):
    if file_storage is None or not file_storage.filename:
        raise UploadError(400, "No Image Provided", "Choose an image file (form field 'image').")
    ext = os.path.splitext(file_storage.filename)[1].lower().lstrip(".")
    safe_name = secure_filename(file_storage.filename) or f"upload.{ext or 'bin'}"
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(415, "Invalid Image Upload",
                          f"Invalid file extension '.{ext}'. Only JPEG, PNG, and WEBP are accepted.")
    data = file_storage.read()  # kept in memory only; never written to disk
    if len(data) > max_bytes:
        raise UploadError(413, "File Size Limit Exceeded",
                          f"The uploaded file exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    if not data:
        raise UploadError(400, "Empty File", "The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        with Image.open(io.BytesIO(data)) as full:
            full.load()  # force a real decode: catches truncated/corrupt payloads
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise UploadError(400, "Corrupted Image",
                          "The file looks like an image but could not be decoded.")
    if fmt not in ALLOWED_PIL_FORMATS:
        raise UploadError(415, "Invalid Image Upload",
                          f"Image content is {fmt}; only JPEG, PNG, and WEBP are accepted.")
    return safe_name, data


def create_app(settings=None) -> Flask:
    settings = settings or load_settings()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=settings.secret_key,
        MAX_CONTENT_LENGTH=settings.max_content_length_mb * 1024 * 1024 + 64 * 1024,
        DEBUG=False,
    )
    max_bytes = settings.max_content_length_mb * 1024 * 1024
    gateway = bool(settings.cloud_api_url)

    db = Database(settings.database_url)
    db.init()

    classifier = None
    if not gateway:
        from src.inference import VisionGuardClassifier  # TF only needed in local mode
        classifier = VisionGuardClassifier.get(settings.model_path, settings.decision_threshold,
                                               settings.model_version)

    ctx = {"gateway": gateway, "cloud_url": settings.cloud_api_url,
           "project": settings.project_name}

    @app.context_processor
    def inject():
        return ctx

    def check_api_key():
        if settings.api_key and not hmac.compare_digest(
                request.headers.get("X-API-Key", ""), settings.api_key):
            raise UploadError(401, "Unauthorized", "A valid X-API-Key header is required.")

    def predict_via_cloud(name, data):
        headers = {"Accept": "application/json"}
        if settings.cloud_api_key:
            headers["X-API-Key"] = settings.cloud_api_key
        started = time.perf_counter()
        try:
            resp = requests.post(f"{settings.cloud_api_url}/predict?format=json",
                                 files={"image": (name, data, "application/octet-stream")},
                                 headers=headers, timeout=120)
        except requests.RequestException as exc:
            raise UploadError(502, "Cloud API Unreachable", f"Could not reach the cloud API: {exc}")
        roundtrip = (time.perf_counter() - started) * 1000
        if resp.status_code != 200:
            try:
                msg = resp.json().get("message", resp.text[:200])
            except ValueError:
                msg = resp.text[:200]
            raise UploadError(resp.status_code if resp.status_code in (400, 401, 413, 415) else 502,
                              "Cloud API Error", msg)
        out = resp.json()
        out["roundtrip_ms"] = roundtrip
        return out

    # ---------------------------------------------------------------- routes
    @app.get("/")
    def index():
        return render_template("index.html", max_mb=settings.max_content_length_mb)

    @app.post("/predict")
    def predict():
        check_api_key()
        name, data = validate_upload(request.files.get("image"), max_bytes)
        if gateway:
            result = predict_via_cloud(name, data)
            source = "cloud"
        else:
            result = classifier.predict(data)
            source = "local"
        record_id = db.log(name, result, source=source)
        payload = {**result, "filename": name, "record_id": record_id}
        if wants_json():
            return jsonify(payload)
        return render_template("result.html", r=payload, preview=make_preview(data),
                               source=source)

    @app.get("/history")
    def history():
        rows = db.history(limit=min(int(request.args.get("limit", 200)), 1000))
        if wants_json():
            return jsonify([{**r, "timestamp": r["timestamp"].isoformat()} for r in rows])
        return render_template("history.html", rows=rows, total=db.count())

    @app.get("/health")
    def health():
        body = {
            "status": "healthy",
            "service": "VisionGuard",
            "project": settings.project_name,
            "student_roll_number": settings.student_roll_number,
            "environment": settings.environment,
            "mode": "gateway" if gateway else "local",
            "model_loaded": bool(classifier) if not gateway else None,
            "model_version": settings.model_version,
            "decision_threshold": settings.decision_threshold,
            "max_upload_size_mb": settings.max_content_length_mb,
            "database_connected": True,
            "database_engine": db.engine_name,
            "database_records_logged": 0,
        }
        try:
            body["database_records_logged"] = db.count()
        except Exception:
            body["database_connected"] = False
            body["status"] = "degraded"
        if gateway:
            try:
                body["cloud_health"] = requests.get(f"{settings.cloud_api_url}/health",
                                                    timeout=10).json()
            except Exception as exc:
                body["cloud_health"] = {"status": "unreachable", "error": str(exc)}
        return jsonify(body), (200 if body["status"] == "healthy" else 503)

    # -------------------------------------------------------- error handling
    def render_error(status, title, message):
        if wants_json():
            return jsonify({"error": title, "message": message, "status": status}), status
        return render_template("error.html", status=status, title=title, message=message), status

    @app.errorhandler(UploadError)
    def on_upload_error(exc):
        return render_error(exc.status, exc.title, exc.message)

    @app.errorhandler(RequestEntityTooLarge)
    def on_too_large(_exc):
        return render_error(413, "File Size Limit Exceeded",
                            f"The uploaded file exceeds the {settings.max_content_length_mb} MB limit.")

    @app.errorhandler(HTTPException)
    def on_http_error(exc):
        return render_error(exc.code or 500, exc.name, exc.description or "")

    @app.errorhandler(Exception)
    def on_unexpected(exc):
        app.logger.exception("Unhandled error: %s", exc)
        return render_error(500, "Server Error", "Something went wrong while processing the image.")

    return app


if __name__ == "__main__":
    # Built-in Flask server, debug explicitly OFF (the Werkzeug debugger allows remote code execution).
    create_app().run(host=os.getenv("FLASK_HOST", "127.0.0.1"),
                     port=int(os.getenv("FLASK_PORT", "5000")), debug=False)
