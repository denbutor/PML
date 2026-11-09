"""Метрики якості, бейзлайн-правила та метрики справедливості."""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rule_based_churn_predict(df: pd.DataFrame) -> pd.Series:
    """Евристичний бейзлайн: Tenure <= 1 АБО Complain == 1 -> Churn."""
    return ((df["Tenure"] <= 1) | (df["Complain"] == 1)).astype(int)


def calculate_selection_rate(predictions: np.ndarray) -> float:
    return float(np.mean(predictions)) if len(predictions) else np.nan


def calculate_true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positives = (y_true == 1)
    return float(np.mean(y_pred[positives])) if np.sum(positives) > 0 else np.nan


def evaluate_demographic_fairness(y_true, y_pred, group_membership, privileged_group) -> pd.DataFrame:
    """Розрахунок метрик справедливості Demographic Parity Ratio (DPR) та Equal Opportunity Ratio (EOR)."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    groups = np.asarray(group_membership)

    priv_mask = (groups == privileged_group)
    priv_sr = calculate_selection_rate(y_pred_arr[priv_mask])
    priv_tpr = calculate_true_positive_rate(y_true_arr[priv_mask], y_pred_arr[priv_mask])

    records = []
    for g_val in sorted(pd.unique(groups)):
        mask = (groups == g_val)
        g_sr = calculate_selection_rate(y_pred_arr[mask])
        g_tpr = calculate_true_positive_rate(y_true_arr[mask], y_pred_arr[mask])

        records.append({
            "group": g_val,
            "n": int(np.sum(mask)),
            "positives": int(np.sum(y_true_arr[mask] == 1)),
            "selection_rate": g_sr,
            "TPR": g_tpr,
            "DPR": g_sr / priv_sr if priv_sr else np.nan,
            "EOR": g_tpr / priv_tpr if priv_tpr else np.nan,
        })
    return pd.DataFrame(records)


def calculate_auc(y_true, probabilities) -> float:
    return float(roc_auc_score(y_true, probabilities))