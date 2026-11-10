"""Ознаки, які сервіс рахує з полів запиту (без Featuretools, без відгуків).

Клієнтські DFS-ознаки ЛР2 — детерміновані функції полів одного клієнта:
  ``A / B`` (``divide_numeric``; ділення на 0 → NaN) і ``NATURAL_LOGARITHM(A)`` (log для A > 0, інакше NaN).
Формули побайтово відтворюють матрицю DFS у data/processed/train.csv (тест ``test_client_dfs_parity``).
Ознаки з відгуків Amazon у сервісі не використовуються: вони не доступні в робочій CRM на момент запиту
(прив'язка корпусу імітаційна, ЛР2) і не були інформативними (ЛР2–ЛР3).
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# колонки клієнта, що йшли в DFS (config/multimodal.yaml → dfs.client_columns; без захищених ознак)
DFS_CLIENT_COLUMNS = ["Tenure", "OrderCount", "CashbackAmount", "CouponUsed", "DaySinceLastOrder",
                      "HourSpendOnApp", "WarehouseToHome", "OrderAmountHikeFromlastYear"]

# поля запиту API = табличні ознаки ЛР1 без ідентифікатора, цілі й захищених ознак
REQUEST_FIELDS = ["Tenure", "PreferredLoginDevice", "WarehouseToHome", "PreferredPaymentMode", "HourSpendOnApp",
                  "NumberOfDeviceRegistered", "PreferedOrderCat", "SatisfactionScore", "MaritalStatus",
                  "NumberOfAddress", "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
                  "DaySinceLastOrder", "CashbackAmount"]


def client_dfs_names(columns: Sequence[str] = DFS_CLIENT_COLUMNS) -> List[str]:
    ratios = [f"{a} / {b}" for a in columns for b in columns if a != b]
    logs = [f"NATURAL_LOGARITHM({a})" for a in columns]
    return ratios + logs


def add_client_dfs(df: pd.DataFrame, columns: Sequence[str] = DFS_CLIENT_COLUMNS) -> pd.DataFrame:
    """Додає 64 клієнтські DFS-ознаки (56 відношень + 8 логарифмів) до таблиці з полями запиту."""
    out = df.copy()
    num = {c: pd.to_numeric(out[c], errors="coerce").astype(float) for c in columns}
    new = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for a in columns:
            for b in columns:
                if a != b:
                    new[f"{a} / {b}"] = (num[a] / num[b]).replace([np.inf, -np.inf], np.nan)
        for a in columns:
            new[f"NATURAL_LOGARITHM({a})"] = np.log(num[a].where(num[a] > 0))
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def requests_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Список запитів (dict полів REQUEST_FIELDS; None = пропуск) → таблиця входу препроцесора."""
    df = pd.DataFrame(list(records))
    missing = [c for c in REQUEST_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"У запиті бракує полів: {missing}")
    df = df[REQUEST_FIELDS].copy()
    for c in REQUEST_FIELDS:   # None → NaN (pandas сам вирішує, але явно для числових)
        if df[c].dtype == object and c not in ("PreferredLoginDevice", "PreferredPaymentMode", "PreferedOrderCat", "MaritalStatus"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return add_client_dfs(df)


class IndexSelector(BaseEstimator, TransformerMixin):
    """Вибір колонок за зафіксованими індексами — заміна навченого mlxtend SFS у експортованому конвеєрі.

    Поведінка ідентична ``SequentialFeatureSelector.transform`` (ті самі ``k_feature_idx_``), але об'єкт
    простий і серіалізується skops без сторонніх типів.
    """

    def __init__(self, indices: Sequence[int] = (), names: Sequence[str] = ()):
        self.indices = indices
        self.names = names

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, list(self.indices)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.names, dtype=object)
