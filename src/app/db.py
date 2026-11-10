"""SQLite-журнал прогнозів і фідбеку. З'єднання — на кожен запит (def-ендпоінти працюють у різних потоках)."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.data import PROJECT_ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY, ts TEXT NOT NULL, features TEXT NOT NULL, score REAL NOT NULL, label INTEGER NOT NULL,
    uncertainty REAL NOT NULL, model_version TEXT NOT NULL, fallback_applied INTEGER NOT NULL, ood_log_density REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY REFERENCES predictions(request_id), ts TEXT NOT NULL, y_true INTEGER NOT NULL
);
"""


def db_path() -> Path:
    return Path(os.environ.get("FEEDBACK_DB", PROJECT_ROOT / "feedback.db"))


def init_db() -> None:
    with sqlite3.connect(db_path()) as con:
        con.executescript(SCHEMA)


def log_prediction(request_id: str, features: dict, res: dict, model_version: str) -> None:
    with sqlite3.connect(db_path()) as con:
        con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)",
                    (request_id, datetime.now(timezone.utc).isoformat(), json.dumps(features), res["score"], res["label"],
                     res["uncertainty"], model_version, int(res["fallback_applied"]), res["ood_log_density"]))


def store_feedback(request_id: str, y_true: int) -> bool:
    """False — невідомий request_id."""
    with sqlite3.connect(db_path()) as con:
        if con.execute("SELECT 1 FROM predictions WHERE request_id = ?", (request_id,)).fetchone() is None:
            return False
        con.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?)",
                    (request_id, datetime.now(timezone.utc).isoformat(), int(y_true)))
        return True
