"""Docker Compose: первинне (одноразове) завантаження поточної моделі з директорії models/ (яка має тег-аліас @champion) до реєстру розгорнутого MLflow-сервера.
Оскільки локальний файл mlflow.db зберігає зафіксовані абсолютні шляхи операційної системи Windows до файлів артефактів, усередині Docker-контейнера база даних реєстру ініціалізується з нуля. Вона наповнюється на основі аналогічного набору компонентів (структура ONNX, метадані skops, фонові дані, OOD та конфіг model_meta.json), а також показників ефективності, взятих із файлу reports/test_evaluation.json.
У разі, якщо модель із міткою @champion уже зареєстрована в системі, процес автоматично завершується без повторного внесення змін (реалізовано принцип ідемпотентності).
"""
import json
import os
from pathlib import Path

import mlflow
import mlflow.onnx
import onnx
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

ROOT = Path(__file__).resolve().parents[1]
NAME = "ChurnGuard"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()
try:
    mv = client.get_model_version_by_alias(NAME, "champion")
    print(f"@champion уже існує: v{mv.version}")
    raise SystemExit(0)
except MlflowException:
    pass

meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
ev = json.loads((ROOT / "reports" / "test_evaluation.json").read_text(encoding="utf-8"))
mlflow.set_experiment("churnguard-docker")
with mlflow.start_run(run_name=f"release-{meta.get('tag', 'v2')}-docker") as run:
    mlflow.log_params({**meta["hyperparameters"], "threshold": meta["threshold"]})
    q = ev["quality"]["model"]
    mlflow.log_metrics({"test_f1": q["f1"], "test_roc_auc": q["roc_auc"], "test_recall": q["recall"]})
    mlflow.set_tags({"source": "seeded from repository artifacts (models/)", "release": meta.get("tag", "v2"),
                     "original_model_version": str(ev["model_version"]), "original_run_id": str(ev["run_id"])})
    for f in ["model_meta.json", "preprocessor.skops", "pipeline.skops", "shap_background.npy"]:
        mlflow.log_artifact(str(ROOT / "models" / f), artifact_path="serving")
    if (ROOT / "models" / "ood").exists():
        mlflow.log_artifacts(str(ROOT / "models" / "ood"), artifact_path="serving/ood")
    info = mlflow.onnx.log_model(onnx.load(str(ROOT / "models" / "pipeline.onnx")), name="onnx_model",
                                 registered_model_name=NAME)
client.set_registered_model_alias(NAME, "champion", info.registered_model_version)
print(f"зареєстровано v{info.registered_model_version} → @champion (run {run.info.run_id})")