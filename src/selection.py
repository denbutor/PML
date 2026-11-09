"""ЛР2, Крок 5 — спершу базова статистика, потім чесний відбір ознак.

Рівні (за зростанням вартості): Filter → Hybrid (Filter + mlxtend SFS) → Embedded (L1) → Tree-based.
Усі функції отримують **лише тренувальну** матрицю (після препроцесора, навченого на train).
Параметри — з config/selection.yaml.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import PROJECT_ROOT


def load_selection_config(path: str | Path = "config/selection.yaml") -> dict:
    path = Path(path)
    with open(path if path.is_absolute() else PROJECT_ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Базова статистика
# --------------------------------------------------------------------------- #
def high_correlation_pairs(Z: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr = Z.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("abs_r").reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_r"]
    return pairs[pairs.abs_r > threshold].sort_values("abs_r", ascending=False).reset_index(drop=True)


def vif_table(Z: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor: VIF_j = 1 / (1 − R²_j), R²_j — регресія ознаки j на решту."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    Zc = add_constant(Z.astype(float), has_constant="add")
    vals = [variance_inflation_factor(Zc.values, i + 1) for i in range(Z.shape[1])]
    return pd.DataFrame({"feature": Z.columns, "VIF": vals}).sort_values("VIF", ascending=False).reset_index(drop=True)


def basic_stats(Z: pd.DataFrame, cfg: dict) -> Dict[str, pd.DataFrame]:
    """Описова статистика, константні ознаки, сильно корельовані пари."""
    desc = Z.describe().T
    desc["n_unique"] = Z.nunique()
    return {
        "describe": desc,
        "constant": desc[desc.n_unique <= 1],
        "high_corr_pairs": high_correlation_pairs(Z, cfg["basic_stats"]["high_corr_threshold"]),
    }


# --------------------------------------------------------------------------- #
# Відбір
# --------------------------------------------------------------------------- #
def make_filter(cfg: dict) -> SelectKBest:
    f = cfg["filter"]
    if f["score_func"] == "mutual_info":
        score = partial(mutual_info_classif, random_state=int(f["random_state"]))
    elif f["score_func"] == "f_classif":
        score = f_classif
    else:
        raise ValueError(f"filter.score_func: невідомо {f['score_func']!r}")
    return SelectKBest(score_func=score, k=int(f["k"]))


def _sfs_estimator():
    # SFS оцінює підмножини логістичною регресією; скейлер — бо вхід може бути нешкальованим (родина partitioning)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def make_sfs(cfg: dict):
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS

    h = cfg["hybrid"]
    return SFS(_sfs_estimator(), k_features=int(h["k_features"]), forward=bool(h["forward"]),
               floating=bool(h["floating"]), scoring=h["scoring"], cv=int(h["cv"]), n_jobs=-1, verbose=0)


def filter_select(Z: pd.DataFrame, y, cfg: dict) -> pd.DataFrame:
    sel = make_filter(cfg).fit(Z, y)
    return (pd.DataFrame({"feature": Z.columns, "score": sel.scores_, "selected": sel.get_support()})
            .sort_values("score", ascending=False).reset_index(drop=True))


def hybrid_select(Z: pd.DataFrame, y, cfg: dict) -> Dict[str, object]:
    filt = make_filter(cfg).fit(Z, y)
    Zf = Z.loc[:, filt.get_support()]
    sfs = make_sfs(cfg).fit(Zf.to_numpy(), np.asarray(y))
    chosen = [Zf.columns[i] for i in sfs.k_feature_idx_]
    path = pd.DataFrame([{"n_features": k, "cv_roc_auc": v["avg_score"]} for k, v in sfs.subsets_.items()])
    return {"filter_kept": list(Zf.columns), "selected": chosen, "cv_score": float(sfs.k_score_), "path": path}


def embedded_l1(Z: pd.DataFrame, y, cfg: dict) -> pd.DataFrame:
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(penalty="l1", solver="liblinear", C=float(cfg["embedded"]["C"]), max_iter=1000))
    model.fit(Z, y)
    coef = model[-1].coef_.ravel()
    return (pd.DataFrame({"feature": Z.columns, "coef": coef, "selected": coef != 0})
            .assign(abs_coef=lambda d: d.coef.abs()).sort_values("abs_coef", ascending=False).reset_index(drop=True))


def tree_importance(Z: pd.DataFrame, y, cfg: dict) -> pd.DataFrame:
    """Permutation importance RF на відкладеній частині train (не на тих рядках, де ліс вчився)."""
    t = cfg["tree"]
    Z_fit, Z_val, y_fit, y_val = train_test_split(Z, y, test_size=float(t["validation_share"]),
                                                  stratify=y, random_state=int(t["random_state"]))
    rf = RandomForestClassifier(n_estimators=int(t["n_estimators"]), random_state=int(t["random_state"]), n_jobs=-1)
    rf.fit(Z_fit, y_fit)
    perm = permutation_importance(rf, Z_val, y_val, scoring="roc_auc", n_repeats=int(t["n_repeats"]),
                                  random_state=int(t["random_state"]), n_jobs=-1)
    return (pd.DataFrame({"feature": Z.columns, "perm_importance": perm.importances_mean,
                          "perm_std": perm.importances_std, "gini_importance": rf.feature_importances_})
            .sort_values("perm_importance", ascending=False).reset_index(drop=True))


def compare_selections(sets: Dict[str, List[str]]) -> pd.DataFrame:
    """Матриця «ознака × метод» (1 = обрано) + скільки методів погодились."""
    feats = sorted(set().union(*sets.values()))
    table = pd.DataFrame({name: [int(f in s) for f in feats] for name, s in sets.items()}, index=feats)
    table["votes"] = table.sum(axis=1)
    return table.sort_values("votes", ascending=False)
