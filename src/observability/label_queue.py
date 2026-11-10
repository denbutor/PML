"""Трек 4A — черга розмітки з журналу прогнозів.

    python -m src.observability.label_queue        # → reports/label_queue.csv, reports/label_queue_ood.csv

Правила (ТЗ ЛР5, розділ 3, 4A):
* кандидати — запити з таблиці ``predictions`` **без фідбеку**;
* OOD-запити (``fallback_applied = 1``) у чергу НЕ потрапляють — окремий список: вони сигналізують про нову зону даних,
  а не про межу класів;
* стратегія — з ЛР4A (reports/lab4a_strategy.json): **entropy + diversity** — топ-100 за ентропією → KMeans на 20
  кластерів у просторі моделі → найближчий до центру кожного; якщо кандидатів < 100 або немає моделі — чиста ентропія;
* черга — не більше 20 записів, **відсортованих за спаданням невизначеності**.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT

DEFAULT_STRATEGY = "entropy_diversity"
DIVERSITY_CANDIDATES = 100


def strategy_config() -> dict:
    p = PROJECT_ROOT / "reports" / "lab4a_strategy.json"
    cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {"strategy": cfg.get("query_strategy_recommended", DEFAULT_STRATEGY), "measure": cfg.get("uncertainty_measure", "entropy")}


def load_log(db_path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(
            "SELECT p.*, f.y_true FROM predictions p LEFT JOIN feedback f ON f.request_id = p.request_id", con)


def _diverse(cand: pd.DataFrame, size: int, predictor, seed: int = 42) -> pd.DataFrame:
    from sklearn.cluster import KMeans

    Z = predictor.features([json.loads(f) for f in cand["features"]])
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    km = KMeans(n_clusters=size, n_init=3, random_state=seed).fit(Zs)
    pick = []
    for c in range(size):
        members = np.flatnonzero(km.labels_ == c)
        pick.append(members[np.argmin(np.linalg.norm(Zs[members] - km.cluster_centers_[c], axis=1))])
    return cand.iloc[pick]


def build_queue(db_path, size: int = 20, predictor=None, strategy: str | None = None) -> dict:
    log = load_log(db_path)
    strategy = strategy or strategy_config()["strategy"]
    pending = log[log["y_true"].isna()]
    ood = pending[pending["fallback_applied"] == 1].sort_values("ood_log_density")
    cand = pending[pending["fallback_applied"] == 0].sort_values("uncertainty", ascending=False, kind="stable")
    used = "entropy"
    if strategy == "entropy_diversity" and predictor is not None and len(cand) >= max(DIVERSITY_CANDIDATES, size):
        queue = _diverse(cand.head(DIVERSITY_CANDIDATES), size, predictor)
        used = "entropy_diversity"
    else:
        queue = cand.head(size)
    queue = queue.sort_values("uncertainty", ascending=False, kind="stable").head(size)
    cols = ["request_id", "ts", "uncertainty", "score", "label", "model_version", "features"]
    return {"strategy": used, "size": int(len(queue)), "n_labeled": int(log["y_true"].notna().sum()),
            "n_pending": int(len(pending)), "queue": queue[cols].to_dict("records"),
            "ood": ood[cols + ["ood_log_density"]].to_dict("records")}


def main():
    from src.app.db import db_path
    from src.serving.predictor import Predictor

    pred = Predictor.from_dir(PROJECT_ROOT / "models", with_explain=False)
    q = build_queue(db_path(), predictor=pred)
    pd.DataFrame(q["queue"]).to_csv(PROJECT_ROOT / "reports" / "label_queue.csv", index=False)
    pd.DataFrame(q["ood"]).to_csv(PROJECT_ROOT / "reports" / "label_queue_ood.csv", index=False)
    print(f"черга: {q['size']} ({q['strategy']}), OOD окремо: {len(q['ood'])}, розмічено: {q['n_labeled']}, очікують: {q['n_pending']}")


if __name__ == "__main__":
    main()
