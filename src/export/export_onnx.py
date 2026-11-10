"""ЛР5, Етап 1 — навчання v1 і експорт у ONNX.

    python -m src.export.export_onnx

1. train ЛР5 (src/serving/data.py) → таблиця входу з полів запиту (src/serving/features.py);
2. герметичний конвеєр ЛР3 (imblearn: препроцесор → Filter → SFS → SMOTE → RF) з гіперпараметрами
   фінальної моделі ЛР3 (reports/lab3_final_model.json), конфігурація без ознак з відгуків;
3. SMOTE прибирається (діє лише у fit), навчений SFS замінюється ``IndexSelector`` з тими самими індексами;
4. межа графа: **препроцесор поза ONNX** (models/preprocessor.skops), **у ONNX — RandomForest** на 15 ознаках;
5. артефакти: models/pipeline.onnx, models/pipeline.skops (sklearn-модель для паритету й SHAP),
   models/preprocessor.skops, models/shap_background.npy, models/model_meta.json; reports/onnx_parity.json.
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.serving.features import REQUEST_FIELDS, IndexSelector, add_client_dfs, client_dfs_names
from src.serving.data import load_lab5

MODELS = PROJECT_ROOT / "models"
SERVING_CONFIG = PROJECT_ROOT / "config" / "generated" / "config_rf_serving.yaml"
OPSET = 17
PREPROCESSING_VERSION = "lab5-serving-1"


def write_serving_config() -> Path:
    from src.pipeline import write_config_with_groups

    write_config_with_groups(
        PROJECT_ROOT / "config" / "config_rf_fe.yaml", SERVING_CONFIG,
        {"client_dfs": {"columns": client_dfs_names(), "kind": "numeric",
                        "policy": {"additive": ["yeo_johnson"],
                                   "partitioning": [{"winsorize": {"lower": 0.01, "upper": 0.99}}]}}},
        header="ЗГЕНЕРОВАНО src/export/export_onnx.py — не редагувати вручну.\n"
               "ЛР5 (сервіс): config_rf_fe.yaml + 64 клієнтські DFS-ознаки; без ознак з відгуків (недоступні в CRM на запиті).")
    return SERVING_CONFIG


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Рівно те, що побачить сервіс: лише поля запиту + похідні DFS-ознаки."""
    return add_client_dfs(df[REQUEST_FIELDS].reset_index(drop=True))


def fit_v1(train: pd.DataFrame, target: str, extra: pd.DataFrame | None = None):
    """Навчання конвеєра з гіперпараметрами ЛР3. ``extra`` — дорозмічені запити (раунд 4A)."""
    from src.evaluation import build_hermetic_pipeline, load_yaml
    from src.prepare_data import load_hermetic_config

    final = json.loads((PROJECT_ROOT / "reports" / "lab3_final_model.json").read_text(encoding="utf-8"))
    params = dict(final["best_params"])
    k = params.pop("k_neighbors")
    cw = {int(a): float(b) for a, b in load_yaml("config/costs.yaml")["class_weight"].items()}
    X, y = to_frame(train), train[target].astype(int).reset_index(drop=True)
    if extra is not None and len(extra):
        X = pd.concat([X, to_frame(extra)], ignore_index=True)
        y = pd.concat([y, extra[target].astype(int).reset_index(drop=True)], ignore_index=True)
    pipe = build_hermetic_pipeline(SERVING_CONFIG, "RF", load_hermetic_config(), model_params=params,
                                   k_neighbors=k, class_weight=cw).fit(X, y)
    return pipe, final


def split_for_export(pipe):
    """imblearn-конвеєр → (препроцесор sklearn.Pipeline, export_pipe = Pipeline([model]), назви ознак)."""
    from sklearn.pipeline import Pipeline

    names = list(pipe.named_steps["columns"].get_feature_names_out())
    names = list(np.asarray(names)[pipe.named_steps["filter"].get_support()])
    idx = list(pipe.named_steps["sfs"].k_feature_idx_)
    selected = [str(names[i]) for i in idx]
    steps = []
    for name, step in pipe.steps:
        if name in ("resample", "model"):
            continue                                       # семплер діє лише у fit; модель — окремо
        steps.append((name, IndexSelector(indices=idx, names=selected).fit(np.zeros((1, len(names))))
                      if name == "sfs" else step))
    preprocessor = Pipeline(steps)
    export_pipe = Pipeline([("model", pipe.named_steps["model"])])
    return preprocessor, export_pipe, selected


def convert(export_pipe, n_features: int):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    clf = export_pipe.steps[-1][1]
    return convert_sklearn(export_pipe, initial_types=[("float_input", FloatTensorType([None, n_features]))],
                           options={id(clf): {"zipmap": False}}, target_opset=OPSET)


def onnx_proba(onx_bytes: bytes, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(onx_bytes, providers=["CPUExecutionProvider"])
    return sess.run(None, {"float_input": X.astype(np.float32)})[1]


def library_versions() -> dict:
    import importlib.metadata as md
    pk = ["scikit-learn", "imbalanced-learn", "numpy", "pandas", "onnx", "onnxruntime", "skl2onnx", "skops", "mlflow",
          "fastapi", "pydantic", "shap", "lime"]
    out = {"python": platform.python_version()}
    for p in pk:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def export(pipe, final: dict, data: dict, out_dir: Path = MODELS, tag: str = "v1", log=print) -> dict:
    import skops.io as sio

    out_dir.mkdir(parents=True, exist_ok=True)
    pre, export_pipe, selected = split_for_export(pipe)
    X_tr = to_frame(data["train"])
    X_te = to_frame(data["test"])
    # паритет препроцесора: IndexSelector ≡ навчений SFS
    assert np.allclose(pre.transform(X_te), pipe[:-2].transform(X_te), equal_nan=True)
    Z_tr = pre.transform(X_tr).astype(np.float32)
    Z_te = pre.transform(X_te).astype(np.float32)
    onx = convert(export_pipe, Z_te.shape[1])
    (out_dir / "pipeline.onnx").write_bytes(onx.SerializeToString())
    sio.dump(export_pipe, out_dir / "pipeline.skops")
    sio.dump(pre, out_dir / "preprocessor.skops")
    rng = np.random.default_rng(42)
    bg = Z_tr[rng.choice(len(Z_tr), size=min(300, len(Z_tr)), replace=False)]
    np.save(out_dir / "shap_background.npy", bg)

    thr = float(final["threshold"])
    p_skl = export_pipe.predict_proba(Z_te)
    p_onx = onnx_proba(onx.SerializeToString(), Z_te)
    diff = np.abs(p_skl - p_onx)
    flips = np.flatnonzero((p_skl[:, 1] >= thr) != (p_onx[:, 1] >= thr))
    parity = {"tag": tag, "n_test": int(len(Z_te)), "max_abs_diff": float(diff.max()), "atol": 1e-4,
              "label_flips_at_threshold": int(len(flips)),
              "flips": [{"row": int(i), "score_skl": float(p_skl[i, 1]), "score_onnx": float(p_onx[i, 1])} for i in flips]}
    meta = {
        "model_name": "ChurnGuard", "tag": tag,
        "request_fields": REQUEST_FIELDS,
        "features": selected,
        "dtypes": {f: "float32" for f in selected},
        "onnx_inputs": [{"name": "float_input", "shape": [None, len(selected)], "type": "tensor(float)"}],
        "onnx_outputs": ["label", "probabilities"],
        "classes": [0, 1], "positive_class": 1, "class_meaning": {"0": "залишиться", "1": "піде (відтік)"},
        "threshold": thr, "threshold_source": "ЛР3: OOF-прогнози вкладеної CV, матриця втрат config/costs.yaml",
        "model_family": "tree_ensemble", "opset": OPSET,
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing": "поза ONNX: models/preprocessor.skops (src/serving + DynamicPipelineBuilder ЛР1–3)",
        "uncertainty": {"measure": "entropy", "formula": "H(p) = -(p*log2 p + (1-p)*log2(1-p)), p = P(відтік)",
                        "source": "ЛР4A, reports/lab4a_strategy.json"},
        "protected_attributes": ["CityTier", "Gender"],
        "non_ml_rule": "Tenure <= 1 OR Complain == 1 → відтік (ЛР1)",
        "hyperparameters": final["best_params"], "class_weight": final["class_weight"],
        "train_rows": int(len(X_tr)), "library_versions": library_versions(),
        "data": {"train": "data/processed/train.csv @ lab3-v1.0, role=pool (lab4a_split.csv)",
                 "label_pool": "data/processed/train.csv @ lab3-v1.0, role=test (lab4a_split.csv)",
                 "test": "data/processed/holdout.csv @ lab3-v1.0"},
        "onnx_sha256": hashlib.sha256(onx.SerializeToString()).hexdigest(),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (PROJECT_ROOT / "reports" / f"onnx_parity_{tag}.json").write_text(json.dumps(parity, ensure_ascii=False, indent=1),
                                                                        encoding="utf-8")
    log(f"[{tag}] ознак {len(selected)}: {selected}")
    log(f"[{tag}] паритет: max|Δp| = {parity['max_abs_diff']:.2e}, змін класу на порозі {thr}: {len(flips)}")
    return {"meta": meta, "parity": parity, "preprocessor": pre, "export_pipe": export_pipe}


def main():
    warnings.filterwarnings("ignore")
    data = load_lab5()
    write_serving_config()
    pipe, final = fit_v1(data["train"], data["target"])
    export(pipe, final, data)


if __name__ == "__main__":
    main()
