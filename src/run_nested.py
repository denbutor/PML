"""ЛР3 — запуск вкладеної CV з командного рядка (важкий розрахунок поза Jupyter).

    python -m src.run_nested --model LR
    python -m src.run_nested --model RF

Результати → reports/lab3_nested_<MODEL>_{predictions,audit,trials}.csv, _folds.json.
Ноутбук lab3_robust_validation.ipynb читає їх (або перераховує тією самою функцією, якщо RECOMPUTE=True).
Причина винесення: вкладений паралелізм (SFS n_jobs=-1 усередині Optuna × RF) у ядрі Jupyter на Windows
може зависати; поза Jupyter той самий код відпрацьовує штатно.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

from joblib import Memory

from src.data import PROJECT_ROOT, load_data_config, read_processed
from src.evaluation import load_yaml, nested_cv
from src.prepare_data import load_hermetic_config

CONFIG = {"LR": "config/generated/config_lr_hermetic.yaml", "RF": "config/generated/config_rf_hermetic.yaml"}


def run(model: str) -> dict:
    warnings.filterwarnings("ignore")
    dc, hc, costs = load_data_config(), load_hermetic_config(), load_yaml("config/costs.yaml")
    train = read_processed("train")
    X, y = train.drop(columns=[dc["target"]]), train[dc["target"]].astype(int)
    memory = Memory(str(PROJECT_ROOT / hc["cache_dir"]), verbose=0)
    t0 = time.time()
    res = nested_cv(X, y, CONFIG[model], model, hc, costs, memory=memory,
                    log=lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True))
    rep = PROJECT_ROOT / "reports"
    res["predictions"].to_csv(rep / f"lab3_nested_{model}_predictions.csv", index=False)
    res["audit"].to_csv(rep / f"lab3_nested_{model}_audit.csv", index=False)
    res["trials"].to_csv(rep / f"lab3_nested_{model}_trials.csv", index=False)
    (rep / f"lab3_nested_{model}_folds.json").write_text(json.dumps(res["folds"], ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
    print(f"{model}: готово за {time.time() - t0:.0f} c", flush=True)
    return res


def final(model: str | None = None) -> dict:
    """Фінальна модель для ЛР4–5: обирається за мінімальними середніми втратами підходу
    «class_weight + поріг T» на зовнішніх фолдах (REFORMS 5d); далі Optuna на всьому train,
    поріг — на OOF-прогнозах усього train, навчання на всьому train. Holdout не використовується."""
    import joblib
    import pandas as pd
    from sklearn.model_selection import cross_val_predict

    from src.evaluation import (build_hermetic_pipeline, optimal_threshold, selected_feature_names,
                                summarize_fold_metrics, tune)

    warnings.filterwarnings("ignore")
    dc, hc, costs = load_data_config(), load_hermetic_config(), load_yaml("config/costs.yaml")
    rep = PROJECT_ROOT / "reports"
    if model is None:
        approach = "Cost-sensitive навчання + оптимальний поріг T"
        mean_cost = {}
        for m in CONFIG:
            t = summarize_fold_metrics(pd.read_csv(rep / f"lab3_nested_{m}_predictions.csv"), costs)
            mean_cost[m] = float(t.loc[t.approach == approach, "total_cost"].mean())
        model = min(mean_cost, key=mean_cost.get)
        print("Середні втрати (зовнішні фолди):", mean_cost, "→", model, flush=True)
    train = read_processed("train")
    X, y = train.drop(columns=[dc["target"]]), train[dc["target"]].astype(int)
    memory = Memory(str(PROJECT_ROOT / hc["cache_dir"]), verbose=0)
    cw = {int(k): float(v) for k, v in costs["class_weight"].items()}
    study, icv = tune(X, y, CONFIG[model], model, hc, memory=memory)
    best = dict(study.best_params)
    k = best.pop("k_neighbors", None)
    oof = cross_val_predict(build_hermetic_pipeline(CONFIG[model], model, hc, model_params=best, k_neighbors=k,
                                                    class_weight=cw, memory=memory),
                            X, y, cv=icv, method="predict_proba")[:, 1]
    t_final = optimal_threshold(y, oof, costs["matrix"])
    pipe = build_hermetic_pipeline(CONFIG[model], model, hc, model_params=best, k_neighbors=k, class_weight=cw).fit(X, y)
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipe, PROJECT_ROOT / "models" / "lab3_final_pipeline.joblib")
    meta = {"model": model, "config": CONFIG[model], "best_params": study.best_params,
            "inner_cv_roc_auc": float(study.best_value), "threshold": t_final, "class_weight": cw,
            "cost_matrix": costs["matrix"], "selected_features": selected_feature_names(pipe),
            "train_rows": int(len(X)), "data": "data/processed/train.csv (DVC, dvc.lock)",
            "artifact": "models/lab3_final_pipeline.joblib (не в git)"}
    (rep / "lab3_final_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(CONFIG))
    ap.add_argument("--final", action="store_true", help="навчити фінальну модель (після вкладеної CV обох моделей)")
    args = ap.parse_args()
    if args.final:
        final(args.model)
    elif args.model:
        run(args.model)
    else:
        ap.error("потрібно --model або --final")
