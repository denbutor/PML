"""Спільні дрібні утиліти для трансформерів."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


def _as_frame(X, columns=None) -> pd.DataFrame:
    """Приводить вхід до DataFrame, зберігаючи назви колонок, якщо вони є."""
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if columns is None:
        columns = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=list(columns))


class _NamedColumnsMixin:
    """Запам'ятовування назв вхідних колонок (для one-to-one трансформерів)."""

    def _remember_columns(self, X) -> pd.DataFrame:
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)
