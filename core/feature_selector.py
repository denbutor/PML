"""Багатоступеневий відбір ознак: базова статистика, Filter, Hybrid, Embedded та Tree Importance."""
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

from core.data_loader import PROJECT_ROOT_DIR


def load_selection_config(config_path: str | Path = "config/feature_selection.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT_DIR / path
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def find_high_correlation_pairs(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("correlation").reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_r"]
    return pairs[pairs.abs_r > threshold].sort_values("abs_r", ascending=False).reset_index(drop=True)


def compute_vif_table(df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    df_const = add_constant(df.astype(float), has_constant="add")
    vif_vals = [variance_inflation_factor(df_const.values, i + 1) for i in range(df.shape[1])]
    return pd.DataFrame({"feature": df.columns, "VIF": vif_vals}).sort_values("VIF", ascending=False).reset_index(
        drop=True)


def compute_basic_statistics(df: pd.DataFrame, config: dict) -> Dict[str, pd.DataFrame]:
    desc = df.describe().T
    desc["unique_count"] = df.nunique()
    return {
        "describe": desc,
        "constants": desc[desc.unique_count <= 1],
        "high_corr": find_high_correlation_pairs(df, config["basic_stats"]["high_corr_threshold"]),
    }


def build_filter_selector(config: dict) -> SelectKBest:
    f_cfg = config["filter"]
    if f_cfg["score_func"] == "mutual_info":
        fn = partial(mutual_info_classif, random_state=int(f_cfg["random_state"]))
    else:
        fn = f_classif
    return SelectKBest(score_func=fn, k=int(f_cfg["k"]))


def apply_filter_selection(df: pd.DataFrame, y, config: dict) -> pd.DataFrame:
    selector = build_filter_selector(config).fit(df, y)
    return pd.DataFrame({
        "feature": df.columns,
        "score": selector.scores_,
        "selected": selector.get_support(),
    }).sort_values("score", ascending=False).reset_index(drop=True)


def apply_sequential_hybrid_selection(df: pd.DataFrame, y, config: dict) -> dict:
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS

    filter_sel = build_filter_selector(config).fit(df, y)
    filtered_df = df.loc[:, filter_sel.get_support()]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    h_cfg = config["hybrid"]
    sfs = SFS(
        model,
        k_features=int(h_cfg["k_features"]),
        forward=bool(h_cfg["forward"]),
        floating=bool(h_cfg["floating"]),
        scoring=h_cfg["scoring"],
        cv=int(h_cfg["cv"]),
        n_jobs=-1,
    ).fit(filtered_df.to_numpy(), np.asarray(y))

    selected_cols = [filtered_df.columns[i] for i in sfs.k_feature_idx_]
    return {"selected_features": selected_cols, "best_score": float(sfs.k_score_)}


def apply_l1_regularized_selection(df: pd.DataFrame, y, config: dict) -> pd.DataFrame:
    c_val = float(config["embedded"]["C"])
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(penalty="l1", solver="liblinear", C=c_val, max_iter=1000))
    model.fit(df, y)

    coefs = model[-1].coef_.ravel()
    return pd.DataFrame({
        "feature": df.columns,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
        "selected": coefs != 0,
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)


def compute_permutation_tree_importance(df: pd.DataFrame, y, config: dict) -> pd.DataFrame:
    t_cfg = config["tree"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        df, y, test_size=float(t_cfg["validation_share"]), stratify=y, random_state=int(t_cfg["random_state"])
    )
    rf = RandomForestClassifier(n_estimators=int(t_cfg["n_estimators"]), random_state=int(t_cfg["random_state"]),
                                n_jobs=-1)
    rf.fit(X_tr, y_tr)

    perm = permutation_importance(
        rf, X_val, y_val, scoring="roc_auc", n_repeats=int(t_cfg["n_repeats"]), random_state=int(t_cfg["random_state"]),
        n_jobs=-1
    )
    return pd.DataFrame({
        "feature": df.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)