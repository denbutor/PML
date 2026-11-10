"""Трек 4A — раунд дорозмітки: потік пулу → черга → «оракул» → @challenger → порівняння з random.

    python -m src.observability.label_round --stream    # 1. пул ЛР4A (1 015) через /v1/predict, без міток
    python -m src.observability.label_round --label     # 2. черга (20) → мітки «оракула» через /v1/feedback
    python -m src.observability.label_round --retrain   # 3. challenger (train + 20 з черги) і random (train + 20 випадкових)
                                                        #    → ONNX + паритет → MLflow @challenger → парний бутстреп на test

Пул не перетинається ні з train v1, ні з test ЛР5 (src/serving/data.py). Розмітку симульовано: мітки —
приховані Churn пулу (data/processed/pool_oracle.csv, не в git). @champion автоматично НЕ змінюється.
"""
from __future__ import annotations

import argparse
import json
import shutil
import warnings

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT

ORACLE = PROJECT_ROOT / "data" / "processed" / "pool_oracle.csv"
REPORT = PROJECT_ROOT / "reports" / "label_round.json"
CHALLENGER_DIR = PROJECT_ROOT / "models" / "challenger"
RANDOM_DIR = PROJECT_ROOT / "models" / "random_baseline"
BUDGET = 20


def _client():
    from fastapi.testclient import TestClient

    from src.app.main import app
    return TestClient(app)


def stream():
    from src.serving.data import load_lab5
    from src.serving.features import REQUEST_FIELDS

    data = load_lab5()
    pool = data["pool"]
    rows = []
    with _client() as c:
        for _, r in pool.iterrows():
            payload = {k: (None if pd.isna(r[k]) else r[k]) for k in REQUEST_FIELDS}
            payload = _normalize(payload)
            resp = c.post("/v1/predict", json=payload)
            resp.raise_for_status()
            rows.append({"request_id": resp.json()["request_id"], data["id"]: int(r[data["id"]]),
                         "y_oracle": int(r[data["target"]])})
    pd.DataFrame(rows).to_csv(ORACLE, index=False)
    print(f"потік: {len(rows)} запитів пулу записано в журнал; оракул → {ORACLE.relative_to(PROJECT_ROOT)}")


ALIASES = {"PreferredLoginDevice": {"Phone": "Mobile Phone"}, "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
           "PreferedOrderCat": {"Mobile": "Mobile Phone"}}
INT_FIELDS = ["NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress", "Complain"]


def _normalize(payload: dict) -> dict:
    """Сирі значення набору → контракт API: синоніми категорій (як ValueAliasMapper) і типи int / float."""
    out = dict(payload)
    for k, m in ALIASES.items():
        if out.get(k) is not None:
            out[k] = m.get(out[k], out[k])
    for k, v in out.items():
        if v is None:
            continue
        out[k] = int(v) if k in INT_FIELDS else (float(v) if not isinstance(v, str) else v)
    return out


def label():
    from src.app.db import db_path
    from src.observability.label_queue import build_queue
    from src.serving.predictor import Predictor

    oracle = pd.read_csv(ORACLE).set_index("request_id")
    pred = Predictor.from_dir(PROJECT_ROOT / "models", with_explain=False)
    q = build_queue(db_path(), size=BUDGET, predictor=pred)
    with _client() as c:
        for item in q["queue"]:
            r = c.post("/v1/feedback", json={"request_id": item["request_id"], "y_true": int(oracle.loc[item["request_id"], "y_oracle"])})
            r.raise_for_status()
    pd.DataFrame(q["queue"]).to_csv(PROJECT_ROOT / "reports" / "label_queue_round1.csv", index=False)
    print(f"розмічено {len(q['queue'])} запитів ({q['strategy']}); OOD поза чергою: {len(q['ood'])}")


def _labeled_frame(request_ids, oracle, log) -> pd.DataFrame:
    sub = log.set_index("request_id").loc[list(request_ids)]
    rows = [{**json.loads(f), "Churn": int(oracle.loc[rid, "y_oracle"])} for rid, f in zip(sub.index, sub["features"])]
    return pd.DataFrame(rows)


def _export_variant(extra: pd.DataFrame, out_dir, tag: str, data: dict):
    from src.export.export_onnx import export, fit_v1

    pipe, final = fit_v1(data["train"], data["target"], extra=extra)
    res = export(pipe, final, data, out_dir=out_dir, tag=tag)
    champ_meta = json.loads((PROJECT_ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))
    meta["ood"] = champ_meta.get("ood")                       # OOD-компонент той самий (простір ознак той самий)
    meta["train_rows"] = int(len(data["train"]) + len(extra))
    meta["n_new_labels"] = int(len(extra))
    (out_dir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if (PROJECT_ROOT / "models" / "ood").exists():
        shutil.copytree(PROJECT_ROOT / "models" / "ood", out_dir / "ood", dirs_exist_ok=True)
    return res


def paired_bootstrap(y, probas: dict, thr_by: dict, cost_matrix, n_boot=1000, seed=42) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    from src.evaluation import total_cost

    def metrics(name, idx):
        p = probas[name][idx]
        yh = (p >= thr_by[name]).astype(int)
        return np.array([roc_auc_score(y[idx], p), f1_score(y[idx], yh), total_cost(y[idx], yh, cost_matrix) / len(idx)])

    names = list(probas)
    full = {n: metrics(n, np.arange(len(y))) for n in names}
    rng = np.random.default_rng(seed)
    boots = {n: [] for n in names}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))            # однакові індекси для всіх моделей
        for n in names:
            boots[n].append(metrics(n, idx))
    boots = {n: np.array(v) for n, v in boots.items()}
    keys = ["roc_auc", "f1", "cost_per_customer"]
    out = {"point": {n: dict(zip(keys, map(float, full[n]))) for n in names}, "diff_vs_champion": {}}
    for n in names:
        if n == "champion":
            continue
        d = boots[n] - boots["champion"]
        out["diff_vs_champion"][f"{n} − champion"] = {
            k: {"mean": float((full[n] - full["champion"])[j]), "ci_low": float(np.percentile(d[:, j], 2.5)),
                "ci_high": float(np.percentile(d[:, j], 97.5)),
                "contains_zero": bool(np.percentile(d[:, j], 2.5) <= 0 <= np.percentile(d[:, j], 97.5))}
            for j, k in enumerate(keys)}
    return out


def retrain():
    from mlflow import MlflowClient

    from src.app.db import db_path
    from src.evaluation import load_yaml
    from src.export.register_model import NAME, log_release, setup, stamp_report
    from src.observability.label_queue import load_log
    from src.serving.data import load_lab5
    from src.serving.predictor import Predictor

    warnings.filterwarnings("ignore")
    data = load_lab5()
    oracle = pd.read_csv(ORACLE).set_index("request_id")
    log = load_log(db_path())
    labeled = log[log["y_true"].notna()]["request_id"].tolist()
    rng = np.random.default_rng(42)
    rand_ids = rng.choice(oracle.index.to_numpy(), size=BUDGET, replace=False).tolist()   # той самий пул і бюджет
    _export_variant(_labeled_frame(labeled, oracle, log), CHALLENGER_DIR, "challenger", data)
    _export_variant(_labeled_frame(rand_ids, oracle, log), RANDOM_DIR, "random", data)
    stamp_report(PROJECT_ROOT / "reports" / "onnx_parity_random.json", None, None)   # random-модель не реєструється
    reg = log_release("challenger", model_dir=CHALLENGER_DIR, alias="challenger",
                      extra_tags={"strategy": "entropy_diversity", "n_new_labels": str(len(labeled)),
                                  "parent_champion": str(MlflowClient().get_model_version_by_alias(NAME, "champion").version)})

    test = data["test"]
    y = test[data["target"]].to_numpy()
    preds = {"champion": Predictor.from_dir(PROJECT_ROOT / "models", with_explain=False),
             "challenger": Predictor.from_dir(CHALLENGER_DIR, with_explain=False),
             "random": Predictor.from_dir(RANDOM_DIR, with_explain=False)}
    probas = {n: p.proba_frame(test) for n, p in preds.items()}
    thr = {n: float(p.meta["threshold"]) for n, p in preds.items()}
    boot = paired_bootstrap(y, probas, thr, load_yaml("config/costs.yaml")["matrix"])
    champ = MlflowClient().get_model_version_by_alias(NAME, "champion")
    d = boot["diff_vs_champion"]["challenger − champion"]
    proven = (not d["roc_auc"]["contains_zero"] and d["roc_auc"]["mean"] > 0) or \
             (not d["cost_per_customer"]["contains_zero"] and d["cost_per_customer"]["mean"] < 0)
    report = {"model_version": str(reg["version"]), "run_id": reg["run_id"],      # звіт належить run @challenger
              "champion": {"model_version": champ.version, "run_id": champ.run_id},
              "challenger": {"model_version": reg["version"], "run_id": reg["run_id"], "n_new_labels": len(labeled),
                             "strategy": "entropy_diversity"},
              "random": {"n_new_labels": BUDGET, "registered": False},
              "test": {"n": int(len(y)), "positives": int(y.sum()), "dataset": "holdout ЛР3"},
              "bootstrap": {"n_boot": 1000, "paired": True, **boot},
              "promotion_decision": ("рекомендовано просунути @challenger після ручної перевірки" if proven else
                                     "НЕ просувати: інтервал різниці challenger − champion містить 0 (перевагу не доведено)"),
              "champion_changed_automatically": False}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    setup()
    import mlflow
    with mlflow.start_run(run_id=reg["run_id"]):
        mlflow.log_artifact(str(REPORT), artifact_path="reports")
    print(json.dumps({"point": boot["point"], "diff": boot["diff_vs_champion"], "decision": report["promotion_decision"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stream", action="store_true")
    g.add_argument("--label", action="store_true")
    g.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    stream() if a.stream else label() if a.label else retrain()
