"""ЛР5, Етапи 2 і 4 — реєстрація версій у MLflow Model Registry (лише аліаси, без Stages).

    python -m src.export.register_model --release v1          # v1 → @champion
    python -m src.export.register_model --release v2          # + OOD-артефакти; @champion → v2, v1: superseded_by=2
    python -m src.export.register_model --challenger DIR --tags strategy=entropy_diversity n_new_labels=20

В одному run: гіперпараметри; метрики ЛР3 (зовнішні фолди вкладеної CV) з 95% інтервалами і метрики на test
з бутстреп-інтервалами; хеш DVC-знімка; ONNX (для сервінгу) + skops-артефакти й фон SHAP (для пояснень);
model_meta.json і deploy/requirements.txt.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import PROJECT_ROOT

NAME = "ChurnGuard"
TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "churnguard-lab5"


def setup():
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def dvc_hashes() -> dict:
    raw = yaml.safe_load((PROJECT_ROOT / "data/raw/E Commerce Dataset.xlsx.dvc").read_text(encoding="utf-8"))["outs"][0]["md5"]
    lock = yaml.safe_load((PROJECT_ROOT / "dvc.lock").read_text(encoding="utf-8"))["stages"]
    outs = {o["path"]: o["md5"] for st in lock.values() for o in st.get("outs", [])}
    return {"dvc_raw_md5": raw, "dvc_train_md5": outs.get("data/processed/train.csv"),
            "dvc_holdout_md5": outs.get("data/processed/holdout.csv"),
            "dvc_lab4a_split_md5": outs.get("data/processed/lab4a_split.csv"), "dvc_reference_rev": "lab3-v1.0"}


def cv_metrics_with_ci() -> dict:
    """Метрики фінального підходу ЛР3 (RF, class_weight + T) по 5 зовнішніх фолдах, 95% t-інтервал."""
    from scipy import stats

    from src.evaluation import load_yaml, summarize_fold_metrics

    t = summarize_fold_metrics(pd.read_csv(PROJECT_ROOT / "reports" / "lab3_nested_RF_predictions.csv"),
                               load_yaml("config/costs.yaml"))
    t = t[t.approach == "Cost-sensitive навчання + оптимальний поріг T"]
    out = {}
    for key, col in [("cv_f1", "F1"), ("cv_roc_auc", "ROC_AUC"), ("cv_cost_loss", "cost_per_customer")]:
        v = t[col].to_numpy()
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        out.update({key: v.mean(), f"{key}_ci_low": v.mean() - h, f"{key}_ci_high": v.mean() + h})
    return out


def test_metrics_with_ci(model_dir: Path, n_boot: int = 1000, seed: int = 42) -> dict:
    """Метрики моделі на test ЛР5 (holdout ЛР3) + 95% бутстреп-інтервали (1000 повторень)."""
    from sklearn.metrics import f1_score, roc_auc_score

    from src.evaluation import load_yaml, total_cost
    from src.serving.predictor import Predictor
    from src.serving.data import load_lab5

    data = load_lab5()
    pred = Predictor.from_dir(model_dir)
    y = data["test"][data["target"]].to_numpy()
    p = pred.proba_frame(data["test"])
    thr = pred.meta["threshold"]
    cm = load_yaml("config/costs.yaml")["matrix"]

    def m(idx):
        yh = (p[idx] >= thr).astype(int)
        return f1_score(y[idx], yh), roc_auc_score(y[idx], p[idx]), total_cost(y[idx], yh, cm) / len(idx)

    point = m(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boots = np.array([m(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    out = {}
    for j, key in enumerate(["test_f1", "test_roc_auc", "test_cost_loss"]):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        out.update({key: point[j], f"{key}_ci_low": lo, f"{key}_ci_high": hi})
    out["f1"], out["roc_auc"], out["cost_loss"] = point
    return out


def stamp_report(path: Path, version, run_id) -> None:
    """Дописує у JSON-звіт версію реєстру й run, для яких його пораховано (вимога ЛР5: кожен звіт має run_id і model_version).
    Попередні версії з тим самим артефактом зберігаються в ``registered_as``."""
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    hist = d.get("registered_as", [])
    if version is not None and not any(h["model_version"] == str(version) for h in hist):
        hist.append({"model_version": str(version), "run_id": run_id})
    d.update({"model_version": None if version is None else str(version), "run_id": run_id, "registered_as": hist})
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


# які звіти описують артефакти релізу: v2 — той самий ONNX-граф, що й v1, плюс OOD
RELEASE_REPORTS = {"v1": ["onnx_parity_v1.json"], "v2": ["onnx_parity_v1.json", "ood_report.json"],
                   "challenger": ["onnx_parity_challenger.json"]}


def log_release(tag: str, model_dir: Path = PROJECT_ROOT / "models", alias: str = "champion",
                extra_tags: dict | None = None) -> dict:
    import mlflow.onnx
    import onnx
    from mlflow import MlflowClient

    mlflow = setup()
    client = MlflowClient()
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    metrics = {**cv_metrics_with_ci(), **test_metrics_with_ci(model_dir)}
    with mlflow.start_run(run_name=f"release-{tag}") as run:
        params = dict(meta["hyperparameters"])
        params.update({"threshold": meta["threshold"], "class_weight": json.dumps(meta["class_weight"]),
                       "n_features": len(meta["features"]), "opset": meta["opset"],
                       "preprocessing_version": meta["preprocessing_version"], "train_rows": meta["train_rows"]})
        if "ood" in meta:
            params.update({"ood_bandwidth": meta["ood"]["bandwidth"], "ood_threshold": meta["ood"]["threshold"],
                           "ood_pca_components": meta["ood"]["n_components"]})
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tags({**dvc_hashes(), "release": tag, "model_family": meta["model_family"], **(extra_tags or {})})
        for f in ["model_meta.json", "preprocessor.skops", "pipeline.skops", "shap_background.npy"]:
            mlflow.log_artifact(str(model_dir / f), artifact_path="serving")
        if (model_dir / "ood").exists():
            mlflow.log_artifacts(str(model_dir / "ood"), artifact_path="serving/ood")
        mlflow.log_artifact(str(PROJECT_ROOT / "deploy" / "requirements.txt"), artifact_path="serving")
        info = mlflow.onnx.log_model(onnx.load(str(model_dir / "pipeline.onnx")), name="onnx_model",
                                     registered_model_name=NAME)
        run_id = run.info.run_id
    version = int(info.registered_model_version)
    client.set_registered_model_alias(NAME, alias, version)
    for k, v in {"release": tag, "run_id": run_id, **(extra_tags or {})}.items():
        client.set_model_version_tag(NAME, str(version), k, str(v))
    for name in RELEASE_REPORTS.get(tag, []):
        stamp_report(PROJECT_ROOT / "reports" / name, version, run_id)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(PROJECT_ROOT / "reports" / name), artifact_path="reports")
    return {"version": version, "run_id": run_id, "alias": alias, "metrics": metrics}


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--release", choices=["v1", "v2"])
    g.add_argument("--challenger", metavar="MODEL_DIR")
    ap.add_argument("--tags", nargs="*", default=[])
    args = ap.parse_args()
    tags = dict(t.split("=", 1) for t in args.tags)
    from mlflow import MlflowClient
    if args.release:
        res = log_release(args.release, alias="champion", extra_tags=tags)
        if args.release == "v2":
            client = MlflowClient()
            for mv in client.search_model_versions(f"name='{NAME}'"):
                if mv.tags.get("release") == "v1":
                    client.set_model_version_tag(NAME, mv.version, "superseded_by", str(res["version"]))
    else:
        res = log_release("challenger", model_dir=Path(args.challenger), alias="challenger", extra_tags=tags)
    print(json.dumps({k: v for k, v in res.items() if k != "metrics"}, ensure_ascii=False))
    print(json.dumps({k: round(float(v), 4) for k, v in res["metrics"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
