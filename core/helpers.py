"""Спільні допоміжні утиліти для маніпуляцій з даними та валідації колонок."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


def _to_dataframe(data: any, default_column_names: list[str] | None = None) -> pd.DataFrame:
    """Приводить довільні вхідні дані до типу pandas.DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    array_data = np.ndarray(data) if not isinstance(data, np.ndarray) else data
    if array_data.ndim == 1:
        array_data = array_data.reshape(-1, 1)
    if default_column_names is None:
        default_column_names = [f"col_{i}" for i in range(array_data.shape[1])]
    return pd.DataFrame(array_data, columns=list(default_column_names))


class _BaseColumnTracker:
    """Міксин для збереження назв вхідних ознак у sklearn-сумісних трансформерах."""

    def _store_input_columns(self, dataframe: any) -> pd.DataFrame:
        df = _to_dataframe(dataframe)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)