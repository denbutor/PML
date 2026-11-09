from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _ensure_dataframe(X, column_names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, np.ndarray):
        cols = column_names if column_names is not None else [f"col_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
    raise TypeError(f"Очікувався DataFrame або ndarray, отримано {type(X)}")


class CategoryAliasStandardizer(BaseEstimator, TransformerMixin):
    """Приводить альтернативні позначення категорій до єдиного вигляду."""

    def __init__(self, alias_map: Dict[str, Dict[Any, Any]] | None = None):
        self.alias_map = alias_map or {}

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X_df = _ensure_dataframe(X, self.feature_names_in_)

        for col, replacements in self.alias_map.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(replacements)
        return X_df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


class Log1pFeatureTransformer(BaseEstimator, TransformerMixin):
    """Застосовує np.log1p до числових колонок для згладжування хвостів."""

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X_df = _ensure_dataframe(X, self.feature_names_in_)
        res = np.log1p(np.maximum(0, X_df.to_numpy(dtype=float)))
        return pd.DataFrame(res, columns=self.feature_names_in_, index=X_df.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


class WinsorizeTruncator(BaseEstimator, TransformerMixin):
    """Обрізає екстремальні викиди на межах 1-го та 99-го перцентилів."""

    def __init__(self, lower_percentile: float = 0.01, upper_percentile: float = 0.99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.bounds_: Dict[str, Tuple[float, float]] = {}

        for col in X_df.columns:
            low = float(X_df[col].quantile(self.lower_percentile))
            high = float(X_df[col].quantile(self.upper_percentile))
            self.bounds_[col] = (low, high)
        return self

    def transform(self, X):
        check_is_fitted(self, "bounds_")
        X_df = _ensure_dataframe(X, self.feature_names_in_)
        out = X_df.copy()

        for col, (low, high) in self.bounds_.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=low, upper=high)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "bounds_")
        return self.feature_names_in_


class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """Кодує категоріальні ознаки через логарифм відношення шансів (WoE)."""

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("WoE кодувальник потребує цільову змінну y у методі fit.")
        X_df = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)

        y_arr = np.asarray(y)
        total_pos = np.sum(y_arr == 1)
        total_neg = np.sum(y_arr == 0)

        self.woe_maps_: Dict[str, Dict[Any, float]] = {}
        self.global_fallback_: Dict[str, float] = {}

        for col in X_df.columns:
            column_series = X_df[col].astype(str)
            df_temp = pd.DataFrame({"feat": column_series, "target": y_arr})

            stats = df_temp.groupby("feat")["target"].agg(["count", "sum"])
            stats["pos"] = stats["sum"]
            stats["neg"] = stats["count"] - stats["sum"]

            pos_rate = (stats["pos"] + self.smoothing) / (total_pos + 2 * self.smoothing)
            neg_rate = (stats["neg"] + self.smoothing) / (total_neg + 2 * self.smoothing)

            woe_vals = np.log(pos_rate / neg_rate)
            self.woe_maps_[col] = woe_vals.to_dict()

            global_pos = (total_pos + self.smoothing) / (total_pos + 2 * self.smoothing)
            global_neg = (total_neg + self.smoothing) / (total_neg + 2 * self.smoothing)
            self.global_fallback_[col] = float(np.log(global_pos / global_neg))

        return self

    def transform(self, X):
        check_is_fitted(self, "woe_maps_")
        X_df = _ensure_dataframe(X, self.feature_names_in_)
        out = pd.DataFrame(index=X_df.index)

        for col in self.feature_names_in_:
            series_str = X_df[col].astype(str)
            mapping = self.woe_maps_[col]
            fallback = self.global_fallback_[col]
            out[col] = series_str.map(mapping).fillna(fallback).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "woe_maps_")
        return self.feature_names_in_


class IdentityPassthroughTransformer(BaseEstimator, TransformerMixin):
    """Трансформер-пустушка, який повертає дані без змін."""

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        return _ensure_dataframe(X, self.feature_names_in_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


def _get_sklearn_onehot(**kwargs):
    from sklearn.preprocessing import OneHotEncoder
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore", **kwargs)


def _get_sklearn_ordinal(**kwargs):
    from sklearn.preprocessing import OrdinalEncoder
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, **kwargs)


def _get_sklearn_scaler(**kwargs):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler(**kwargs)


TRANSFORMER_REGISTRY: Dict[str, Any] = {
    "log1p": Log1pFeatureTransformer,
    "winsorizer": WinsorizeTruncator,
    "standard_scaler": _get_sklearn_scaler,
    "woe_encoder": WeightOfEvidenceEncoder,
    "one_hot_encoder": _get_sklearn_onehot,
    "ordinal_encoder": _get_sklearn_ordinal,
    "passthrough": IdentityPassthroughTransformer,
}