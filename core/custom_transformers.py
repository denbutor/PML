"""Атомарні трансформери препроцесингу та їх глобальний реєстр."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from core.feature_constructors import OutOfFoldWoEEncoder, WeightOfEvidenceEncoder
from core.helpers import _BaseColumnTracker, _to_dataframe


class CategoryAliasMapper(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Мапування та уніфікація синонімів категоріальних змінних."""

    def __init__(self, mapping: Mapping[str, Mapping[Any, Any]] | None = None):
        self.mapping = mapping

    def fit(self, X, y=None):
        self._store_input_columns(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        dataframe = _to_dataframe(X, self.feature_names_in_).copy()
        if self.mapping:
            for column_name, replacement_rules in self.mapping.items():
                if column_name in dataframe.columns:
                    dataframe[column_name] = dataframe[column_name].replace(dict(replacement_rules))
        return dataframe


class QuantileOutlierClipper(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Вінсоризація (кліпінг) екстремальних квантилів числового розподілу."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        dataframe = self._store_input_columns(X)
        if not (0.0 <= self.lower < self.upper <= 1.0):
            raise ValueError(f"Некоректні квантилі: lower={self.lower}, upper={self.upper}")

        matrix = dataframe.to_numpy(dtype=float)
        self.lower_bounds_ = np.nanquantile(matrix, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(matrix, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ("lower_bounds_", "upper_bounds_"))
        dataframe = _to_dataframe(X, self.feature_names_in_)
        clipped_matrix = np.clip(dataframe.to_numpy(dtype=float), self.lower_bounds_, self.upper_bounds_)
        return pd.DataFrame(clipped_matrix, columns=dataframe.columns, index=dataframe.index)


def _log_transform(**kwargs) -> FunctionTransformer:
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one", **kwargs)


def _one_hot_encode(**kwargs) -> OneHotEncoder:
    defaults = {"handle_unknown": "ignore", "sparse_output": False}
    defaults.update(kwargs)
    return OneHotEncoder(**defaults)


def _yeo_johnson_transform(**kwargs) -> PowerTransformer:
    defaults = {"method": "yeo-johnson", "standardize": True}
    defaults.update(kwargs)
    return PowerTransformer(**defaults)


def _ordinal_encode(**kwargs) -> OrdinalEncoder:
    defaults = {"handle_unknown": "use_encoded_value", "unknown_value": -1}
    defaults.update(kwargs)
    return OrdinalEncoder(**defaults)


TRANSFORMER_REGISTRY: Dict[str, Callable[..., BaseEstimator]] = {
    "log1p": _log_transform,
    "standard_scaler": StandardScaler,
    "winsorize": QuantileOutlierClipper,
    "woe": WeightOfEvidenceEncoder,
    "one_hot": _one_hot_encode,
    "ordinal_encoder": _ordinal_encode,
    "oof_woe": OutOfFoldWoEEncoder,
    "yeo_johnson": _yeo_johnson_transform,
}