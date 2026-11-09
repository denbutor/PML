"""Атомарні трансформери та реєстр «назва в конфігурації → фабрика трансформера».

Реєстр — це *механізм*: перелік операцій, які система вміє виконувати.
Яку операцію застосувати до якої групи ознак — *політика*, вона живе в YAML
(секція ``transform_policy``). Тому назви операцій тут є, а назв колонок,
порогів і ключів ``model_family`` — немає.
"""
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

# WoE-кодувальники винесено в src/features.py (ЛР2); тут лише реєстрація кроків.
from src.features import OOFWoEEncoder, WoEEncoder
from src.utils import _NamedColumnsMixin, _as_frame


class ValueAliasMapper(_NamedColumnsMixin, BaseEstimator, TransformerMixin):
    """Зводить синоніми категорій до канонічного значення (напр. "CC" → "Credit Card").

    Перетворення безстанове: таблиця замін задається в конфігурації, з даних
    нічого не оцінюється, тому витоку бути не може. Застосовується до імпутації,
    інакше мода рахувалася б по «розщеплених» категоріях.
    """

    def __init__(self, mapping: Mapping[str, Mapping[Any, Any]] | None = None):
        self.mapping = mapping

    def fit(self, X, y=None):
        self._remember_columns(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = _as_frame(X, self.feature_names_in_).copy()
        for col, aliases in (self.mapping or {}).items():
            if col in X.columns:
                X[col] = X[col].replace(dict(aliases))
        return X


class Winsorizer(_NamedColumnsMixin, BaseEstimator, TransformerMixin):
    """Вінсоризація: обрізання значень за перцентилями, оціненими на train.

    Межі (``lower_``/``upper_``) обчислюються лише у ``fit`` — на тестовій
    підвибірці вони тільки застосовуються.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = self._remember_columns(X)
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError(
                f"Winsorizer: потрібно 0 <= lower < upper <= 1, отримано "
                f"lower={self.lower}, upper={self.upper}"
            )
        values = X.to_numpy(dtype=float)
        self.lower_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ("lower_", "upper_"))
        X = _as_frame(X, self.feature_names_in_)
        clipped = np.clip(X.to_numpy(dtype=float), self.lower_, self.upper_)
        return pd.DataFrame(clipped, columns=X.columns, index=X.index)


def _log1p(**kwargs) -> FunctionTransformer:
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one", **kwargs)


def _one_hot(**kwargs) -> OneHotEncoder:
    params = {"handle_unknown": "ignore", "sparse_output": False}
    params.update(kwargs)
    return OneHotEncoder(**params)


def _yeo_johnson(**kwargs) -> PowerTransformer:
    params = {"method": "yeo-johnson", "standardize": True}
    params.update(kwargs)
    return PowerTransformer(**params)


def _review_text_svd(**kwargs):
    from src.extraction import ReviewTextSVD   # пізній імпорт: extraction тягне важкі залежності
    return ReviewTextSVD(**kwargs)


def _ordinal(**kwargs) -> OrdinalEncoder:
    params = {"handle_unknown": "use_encoded_value", "unknown_value": -1}
    params.update(kwargs)
    return OrdinalEncoder(**params)


# Назва кроку в transform_policy → фабрика (kwargs з конфігурації передаються як є).
TRANSFORMER_REGISTRY: Dict[str, Callable[..., BaseEstimator]] = {
    "log1p": _log1p,
    "standard_scaler": StandardScaler,
    "winsorize": Winsorizer,
    "woe": WoEEncoder,
    "one_hot": _one_hot,
    "ordinal_encoder": _ordinal,
    # --- ЛР2 ---
    "oof_woe": OOFWoEEncoder,
    "yeo_johnson": _yeo_johnson,
    # --- ЛР3 ---
    "review_text_svd": _review_text_svd,
}
