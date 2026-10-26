import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.inference import VisionGuardClassifier  # noqa: E402
from src.stub_model import build_stub_model  # noqa: E402


@pytest.fixture(scope="session")
def stub_model_path(tmp_path_factory):
    return build_stub_model(str(tmp_path_factory.mktemp("model") / "stub.keras"))


@pytest.fixture()
def client(monkeypatch, tmp_path, stub_model_path):
    monkeypatch.setenv("MODEL_PATH", stub_model_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MAX_CONTENT_LENGTH_MB", "1")
    monkeypatch.setenv("DECISION_THRESHOLD", "0.279")
    monkeypatch.setenv("CLOUD_API_URL", "")
    monkeypatch.setenv("API_KEY", "")
    VisionGuardClassifier.reset()
    app = create_app(load_settings())
    app.config["TESTING"] = True
    return app.test_client()


def png_bytes(size=(64, 48), color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()
