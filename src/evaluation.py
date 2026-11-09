"""Оцінювання: ROC-AUC, евристичний бейзлайн, метрики справедливості (DPR, EOR)."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def heuristic_predict(X: pd.DataFrame) -> pd.Series:
    """Правило ЛР1 (Non-ML Benchmark): Tenure ≤ 1 міс. АБО скарга минулого місяця → відтік.

    Правило — частина постановки задачі (ML Canvas), а не крок препроцесингу, тому його
    умови записані тут явно. Порівняння з NaN дає False.
    """
    return ((X["Tenure"] <= 1) | (X["Complain"] == 1)).astype(int)


def selection_rate(pred: np.ndarray) -> float:
    return float(np.mean(pred)) if len(pred) else np.nan


def true_positive_rate(y: np.ndarray, pred: np.ndarray) -> float:
    pos = y == 1
    return float(pred[pos].mean()) if pos.any() else np.nan


def fairness_report(y_true, y_pred, group, privileged, groups: Iterable | None = None) -> pd.DataFrame:
    """DPR і EOR кожної групи відносно привілейованої.

    DPR = P(ŷ=1 | група) / P(ŷ=1 | привілейована)        — паритет частки позитивних рішень
    EOR = TPR(група) / TPR(привілейована)                  — рівність можливостей серед справжніх позитивних
    Поріг-орієнтир 0.80 («правило 4/5») — евристика, а не математична межа.
    """
    y_true, y_pred, group = np.asarray(y_true), np.asarray(y_pred), np.asarray(group)
    groups = list(groups) if groups is not None else sorted(pd.unique(group))
    priv = group == privileged
    sr_p, tpr_p = selection_rate(y_pred[priv]), true_positive_rate(y_true[priv], y_pred[priv])
    rows = []
    for g in groups:
        m = group == g
        sr, tpr = selection_rate(y_pred[m]), true_positive_rate(y_true[m], y_pred[m])
        rows.append({"group": g, "n": int(m.sum()), "positives": int((y_true[m] == 1).sum()),
                     "selection_rate": sr, "TPR": tpr,
                     "DPR": sr / sr_p if sr_p else np.nan, "EOR": tpr / tpr_p if tpr_p else np.nan})
    return pd.DataFrame(rows)


def auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))


# =========================================================================== #
# ЛР3 — герметичний конвеєр, вкладена CV (Optuna), cost-sensitive, McNemar
# =========================================================================== #
from pathlib import Path

import yaml

from src.data import PROJECT_ROOT


def load_yaml(path) -> dict:
    path = Path(path)
    with open(path if path.is_absolute() else PROJECT_ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------- бізнес-втрати --------------------------------- #
def total_cost(y_true, y_pred, cost_matrix: dict) -> float:
    """TN·C00 + FP·C01 + FN·C10 + TP·C11 (ТЗ ЛР3, бонус 1)."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    c = cost_matrix
    return float(tn * c["C00"] + fp * c["C01"] + fn * c["C10"] + tp * c["C11"])


def cost_curve(y_true, proba, cost_matrix: dict, thresholds=None) -> pd.DataFrame:
    """Сумарні та середні (на клієнта) втрати для кожного порогу T: ŷ = 1 ⇔ P(Ŷ=1|X) ≥ T."""
    thresholds = np.round(np.linspace(0.0, 1.0, 101), 2) if thresholds is None else thresholds
    proba = np.asarray(proba)
    rows = [{"threshold": t, "total_cost": total_cost(y_true, (proba >= t).astype(int), cost_matrix)}
            for t in thresholds]
    out = pd.DataFrame(rows)
    out["cost_per_customer"] = out["total_cost"] / len(y_true)
    return out


def optimal_threshold(y_true, proba, cost_matrix: dict) -> float:
    curve = cost_curve(y_true, proba, cost_matrix)
    return float(curve.loc[curve.total_cost.idxmin(), "threshold"])


# --------------------------- герметичний конвеєр ---------------------------- #
def make_model(key: str, params: dict | None = None, class_weight=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    params = dict(params or {})
    if key == "LR":
        return LogisticRegression(max_iter=2000, solver="liblinear", penalty=params.get("penalty", "l2"),
                                  C=float(params.get("C", 1.0)), class_weight=class_weight)
    if key == "RF":
        depth = params.get("max_depth")
        return RandomForestClassifier(n_estimators=int(params.get("n_estimators", 200)),
                                      max_depth=None if depth is None else int(depth),
                                      class_weight=class_weight, random_state=42, n_jobs=-1)
    raise ValueError(f"Невідома модель {key!r}")


def build_hermetic_pipeline(config_path, model_key: str, hc: dict, model_params: dict | None = None,
                            k_neighbors: int | None = None, class_weight=None, resample: bool = True,
                            memory=None):
    """imblearn.Pipeline: препроцесор (ЛР1+ЛР2) → Filter → SFS → SMOTE → класифікатор.

    Кожна ланка, що вчиться з даних, навчається лише в ``fit`` усього конвеєра — тобто
    всередині того фолду, на якому конвеєр навчають. SMOTE (imblearn-семплер) працює
    тільки під час ``fit``: на тестовому фолді синтетичних прикладів не генерується.
    """
    from functools import partial

    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.pipeline import ExtendedPipelineBuilder

    sel = hc["selection"]
    score = f_classif if sel["filter_score"] == "f_classif" else partial(mutual_info_classif, random_state=42)
    sfs = SFS(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              k_features=int(sel["sfs_k_features"]), forward=True, floating=False,
              scoring=sel["sfs_scoring"], cv=int(sel["sfs_cv"]), n_jobs=-1, verbose=0)
    rs = hc["resampler"]
    sampler = (SMOTE(sampling_strategy=float(rs["sampling_strategy"]), random_state=int(rs["random_state"]),
                     k_neighbors=int(k_neighbors or rs["k_neighbors"])) if resample else "passthrough")
    return ImbPipeline(flatten_steps(ExtendedPipelineBuilder(config_path).build()) + [
        ("filter", SelectKBest(score, k=int(sel["filter_k"]))),
        ("sfs", sfs),
        ("resample", sampler),
        ("model", make_model(model_key, model_params, class_weight)),
    ], memory=memory)


def flatten_steps(prep) -> list:
    """imblearn.Pipeline не дозволяє вкладених Pipeline як проміжних кроків → розгортаємо препроцесор
    у плаский список ланок (синоніми → конструктори ознак → ColumnTransformer). Логіка не змінюється."""
    from sklearn.pipeline import Pipeline as SkPipeline

    if not isinstance(prep, SkPipeline):
        return [("columns", prep)]
    steps = []
    for name, step in prep.steps:
        steps.extend(flatten_steps(step) if isinstance(step, SkPipeline) else [(name, step)])
    return steps


def selected_feature_names(fitted_pipeline) -> list:
    """Назви ознак, що дійшли до моделі (після препроцесора → filter → SFS)."""
    names = np.asarray(fitted_pipeline.named_steps["columns"].get_feature_names_out())
    names = names[fitted_pipeline.named_steps["filter"].get_support()]
    return [str(names[i]) for i in fitted_pipeline.named_steps["sfs"].k_feature_idx_]


# --------------------------- вкладена CV ----------------------------------- #
def _suggest(trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            out[name] = trial.suggest_categorical(name, spec)
        elif isinstance(spec["low"], int) and isinstance(spec["high"], int):
            out[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log")))
    return out


def tune(X, y, config_path, model_key: str, hc: dict, memory=None, verbose: bool = False):
    """Внутрішній контур: Optuna (TPE) × StratifiedKFold(3) на переданій (тренувальній) частині."""
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    inner = hc["nested_cv"]["inner"]
    cv = StratifiedKFold(n_splits=int(inner["n_splits"]), shuffle=True, random_state=int(inner["random_state"]))
    space = hc["search_space"][model_key]

    def objective(trial):
        p = _suggest(trial, space)
        k = p.pop("k_neighbors", None)
        pipe = build_hermetic_pipeline(config_path, model_key, hc, model_params=p, k_neighbors=k, memory=memory)
        return cross_val_score(pipe, X, y, cv=cv, scoring=hc["nested_cv"]["objective"], n_jobs=1).mean()

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=int(hc["nested_cv"]["sampler_seed"])))
    study.optimize(objective, n_trials=int(hc["nested_cv"]["n_trials"]))
    return study, cv


def nested_cv(X, y, config_path, model_key: str, hc: dict, costs: dict, memory=None, log=print) -> dict:
    """Зовнішній StratifiedKFold(5) × внутрішній Optuna(3-fold). Для кожного зовнішнього фолду:

    1. підбір гіперпараметрів лише на його тренувальній частині;
    2. варіанти з найкращими параметрами: baseline (без ваг), weighted (class_weight з матриці втрат),
       no_resample (без SMOTE — для перевірки впливу ланки) → ймовірності на зовнішньому тесті;
    3. поріг T — на OOF-прогнозах ВНУТРІШНЬОЇ CV (тренувальна частина), не на зовнішньому тесті;
    4. аудиторський слід імпутера з препроцесора, навченого на цьому фолді.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from src.pipeline import ExtendedPipelineBuilder

    outer = hc["nested_cv"]["outer"]
    ocv = StratifiedKFold(n_splits=int(outer["n_splits"]), shuffle=bool(outer["shuffle"]),
                          random_state=int(outer["random_state"]))
    cw = {int(k): float(v) for k, v in costs["class_weight"].items()}
    preds, folds, audits, studies = [], [], [], []
    y = pd.Series(np.asarray(y), index=X.index)
    for fold, (tr, te) in enumerate(ocv.split(X, y)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        study, icv = tune(Xtr, ytr, config_path, model_key, hc, memory=memory)
        best = dict(study.best_params)
        k = best.pop("k_neighbors", None)

        def pipe(**kw):
            return build_hermetic_pipeline(config_path, model_key, hc, model_params=best, k_neighbors=k,
                                           memory=memory, **kw)

        variants = {"baseline": dict(class_weight=None), "weighted": dict(class_weight=cw),
                    "no_resample": dict(class_weight=None, resample=False)}
        out = pd.DataFrame({"row": X.index[te], "fold": fold, "y": yte.to_numpy()})
        fitted = {}
        for name, kw in variants.items():
            fitted[name] = pipe(**kw).fit(Xtr, ytr)
            out[f"p_{name}"] = fitted[name].predict_proba(Xte)[:, 1]
        # поріг — на внутрішніх OOF-прогнозах тренувальної частини, щоб не підглядати в зовнішній тест
        oof_w = cross_val_predict(pipe(class_weight=cw), Xtr, ytr, cv=icv, method="predict_proba")[:, 1]
        oof_b = cross_val_predict(pipe(class_weight=None), Xtr, ytr, cv=icv, method="predict_proba")[:, 1]
        t_w, t_b = optimal_threshold(ytr, oof_w, costs["matrix"]), optimal_threshold(ytr, oof_b, costs["matrix"])
        out["T_weighted"], out["T_baseline"] = t_w, t_b
        preds.append(out)
        audit = ExtendedPipelineBuilder(config_path).audit_trail(fitted["weighted"].named_steps["columns"])
        audit.insert(0, "fold", fold)
        audits.append(audit)
        folds.append({"fold": fold, "n_train": len(tr), "n_test": len(te), "best_params": study.best_params,
                      "inner_best_roc_auc": float(study.best_value), "T_weighted": t_w, "T_baseline": t_b,
                      "selected_features": selected_feature_names(fitted["weighted"])})
        trials = study.trials_dataframe()
        trials.insert(0, "fold", fold)
        studies.append(trials)
        log(f"[{model_key}] fold {fold}: inner AUC {study.best_value:.4f}, params {study.best_params}, "
            f"T_w={t_w:.2f}, outer AUC(w)={roc_auc_score(yte, out['p_weighted']):.4f}")
    return {"predictions": pd.concat(preds, ignore_index=True), "folds": folds,
            "audit": pd.concat(audits, ignore_index=True), "trials": pd.concat(studies, ignore_index=True)}


APPROACHES = {
    "Baseline (без зважування, поріг 0.5)": ("p_baseline", None),
    "Cost-sensitive навчання (class_weight), поріг 0.5": ("p_weighted", None),
    "Cost-sensitive навчання + оптимальний поріг T": ("p_weighted", "T_weighted"),
}


def summarize_fold_metrics(pred: pd.DataFrame, costs: dict) -> pd.DataFrame:
    """Метрики трьох підходів по кожному зовнішньому фолду (ТЗ, п. 4.5)."""
    from sklearn.metrics import f1_score

    rows = []
    for fold, g in pred.groupby("fold"):
        y = g["y"].to_numpy()
        for name, (pcol, tcol) in APPROACHES.items():
            t = 0.5 if tcol is None else float(g[tcol].iloc[0])
            p = g[pcol].to_numpy()
            yhat = (p >= t).astype(int)
            cost = total_cost(y, yhat, costs["matrix"])
            rows.append({"fold": fold, "approach": name, "threshold": t, "total_cost": cost,
                         "cost_per_customer": cost / len(y), "F1": f1_score(y, yhat), "ROC_AUC": roc_auc_score(y, p),
                         "recall": float(yhat[y == 1].mean()),
                         "precision": float(y[yhat == 1].mean()) if yhat.any() else np.nan})
    return pd.DataFrame(rows)


# --------------------------- статистика ------------------------------------ #
def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar (mlxtend): чи відрізняється структура помилок двох моделей на тих самих спостереженнях."""
    from mlxtend.evaluate import mcnemar, mcnemar_table

    tb = mcnemar_table(y_target=np.asarray(y_true), y_model1=np.asarray(pred_a), y_model2=np.asarray(pred_b))
    chi2, p = mcnemar(ary=tb, corrected=True)
    return {"table": tb, "chi2": float(chi2), "p_value": float(p),
            "only_a_correct": int(tb[0, 1]), "only_b_correct": int(tb[1, 0])}
