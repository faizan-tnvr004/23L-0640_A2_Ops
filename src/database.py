"""Prediction audit trail. SQLite locally, PostgreSQL in Docker Compose."""
import os
import time
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, func, insert, select)
from sqlalchemy.engine import make_url


class Database:
    def __init__(self, url: str):
        self.url = url
        parsed = make_url(url)
        kwargs = {}
        if parsed.get_backend_name() == "sqlite":
            kwargs["connect_args"] = {"check_same_thread": False}
            if parsed.database and parsed.database != ":memory:":
                parent = os.path.dirname(os.path.abspath(parsed.database))
                os.makedirs(parent, exist_ok=True)
        self.engine = create_engine(url, pool_pre_ping=True, **kwargs)
        self.engine_name = parsed.get_backend_name()
        self.metadata = MetaData()
        self.predictions = Table(
            "predictions", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("filename", String(255), nullable=False),
            Column("predicted_label", String(64), nullable=False),
            Column("confidence", Float, nullable=False),
            Column("raw_score", Float, nullable=False),
            Column("decision_threshold", Float, nullable=False),
            Column("model_version", String(128), nullable=False),
            Column("latency_ms", Float, nullable=False),
            Column("source", String(16), nullable=False, default="local"),
        )

    def init(self, retries: int = 10, delay: float = 2.0):
        last = None
        for _ in range(retries):
            try:
                self.metadata.create_all(self.engine)
                return
            except Exception as exc:  # database may still be starting
                last = exc
                time.sleep(delay)
        raise RuntimeError(f"Could not initialise database: {last}")

    def log(self, filename, result: dict, source: str = "local") -> int:
        row = {
            "timestamp": datetime.now(timezone.utc),
            "filename": filename,
            "predicted_label": result["label"],
            "confidence": result["confidence"],
            "raw_score": result["raw_score"],
            "decision_threshold": result["threshold"],
            "model_version": result["model_version"],
            "latency_ms": result["latency_ms"],
            "source": source,
        }
        with self.engine.begin() as conn:
            res = conn.execute(insert(self.predictions).values(**row))
            return int(res.inserted_primary_key[0])

    def history(self, limit: int = 200):
        stmt = select(self.predictions).order_by(self.predictions.c.id.desc()).limit(limit)
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(self.predictions)).scalar())
