"""Central configuration.

Every runtime setting comes from environment variables (optionally loaded from
an untracked .env file), so no secret or address is ever written in code.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

IMG_SIZE = 224
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True)
class Settings:
    model_path: str
    decision_threshold: float
    database_url: str
    max_content_length_mb: int
    secret_key: str
    cloud_api_url: str   # non-empty => local Flask acts as a client of the cloud API
    api_key: str         # if set on the server, /predict requires X-API-Key
    cloud_api_key: str   # key the gateway sends to the cloud API
    model_version: str
    project_name: str
    student_roll_number: str
    environment: str


def load_settings() -> Settings:
    """Read settings at call time so tests can change the environment first."""
    return Settings(
        model_path=os.getenv("MODEL_PATH", str(ROOT / "models" / "champion_model.keras")),
        decision_threshold=float(os.getenv("DECISION_THRESHOLD", "0.5")),
        database_url=os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
        max_content_length_mb=int(os.getenv("MAX_CONTENT_LENGTH_MB", "5")),
        secret_key=os.getenv("SECRET_KEY", "change-me"),
        cloud_api_url=os.getenv("CLOUD_API_URL", "").rstrip("/"),
        api_key=os.getenv("API_KEY", ""),
        cloud_api_key=os.getenv("CLOUD_API_KEY", ""),
        model_version=os.getenv("MODEL_VERSION", "v2-champion-mobilenetv2"),
        project_name=os.getenv("PROJECT_NAME", "23L-0640_A2_Ops"),
        student_roll_number=os.getenv("STUDENT_ROLL_NUMBER", "23L-0640"),
        environment=os.getenv("APP_ENV", "production"),
    )
