"""Кастомні кодувальники та конструктори нових ознак."""
from __future__ import annotations

import math
from typing import Any, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from core.helpers import _BaseColumnTracker, _to_dataframe


class BoundedRatioTransformer(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Обчислення відношення числових колонок із захистом від ділення на нуль."""

    def __init__(self, target_column: str, numerator: str, denominator: str,
                 scale_denominator: float = 1.0, epsilon: float = 1e-5):
        self.target_column = target_column
        self.numerator = numerator
        self.denominator = denominator
        self.scale_denominator = scale_denominator
        self.epsilon = epsilon

    def fit(self, X, y=None):
        self._store_input_columns(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        df = _to_dataframe(X, self.feature_names_in_)
        num_vals = df[self.numerator].astype(float)
        den_vals = df[self.denominator].astype(float)

        ratio = num_vals / (den_vals / self.scale_denominator + self.epsilon)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        result_df = df.copy()
        result_df[self.target_column] = ratio
        return result_df


class StageBinningEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Дискретизація неперервної часової або числової змінної по інтервалах."""

    def __init__(self, target_column: str, source_column: str, edges: Sequence[float]):
        self.target_column = target_column
        self.source_column = source_column
        self.edges = list(edges)

    def fit(self, X, y=None):
        self._store_input_columns(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        df = _to_dataframe(X, self.feature_names_in_)
        vals = df[self.source_column]
        binned = pd.cut(vals, bins=self.edges, labels=False, include_lowest=True)

        res_df = df.copy()
        res_df[self.target_column] = binned
        return res_df


class PeriodicTimeEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Циклічне кодування міток часу через синус та косинус."""

    def __init__(self, period_mapping: Mapping[str, float]):
        self.period_mapping = period_mapping

    def fit(self, X, y=None):
        self._store_input_columns(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        df = _to_dataframe(X, self.feature_names_in_).copy()

        for col_name, period_val in self.period_mapping.items():
            if col_name in df.columns:
                series = df[col_name].astype(float)
                df[f"{col_name}_sin"] = np.sin(2 * np.pi * series / period_val)
                df[f"{col_name}_cos"] = np.cos(2 * np.pi * series / period_val)
        return df


class FeatureCrossEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Комбінування (схрещування) кількох категоріальних ознак у єдиний сегмент."""

    def __init__(self, target_column: str, source_columns: List[str], separator: str = " | "):
        self.target_column = target_column
        self.source_columns = source_columns
        self.separator = separator

    def fit(self, X, y=None):
        self._store_input_columns(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        df = _to_dataframe(X, self.feature_names_in_)

        combined_series = None
        for col in self.source_columns:
            # Приведення до object запобігає TypeError при fillna у CategoricalDtype
            str_col = df[col].astype(object).fillna("NA").astype(str)
            if combined_series is None:
                combined_series = str_col
            else:
                combined_series = combined_series + self.separator + str_col

        res_df = df.copy()
        res_df[self.target_column] = combined_series
        return res_df


class WeightOfEvidenceEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Базове обчислення Weight of Evidence (WoE) зі згладжуванням Лапласа."""

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing

    def fit(self, X, y):
        if y is None:
            raise ValueError("WoEEncoder потребує цільову змінну y для навчального етапу (fit)")
        df = self._store_input_columns(X)
        y_arr = np.asarray(y)

        global_positives = np.sum(y_arr == 1)
        global_negatives = np.sum(y_arr == 0)

        self.encoding_maps_ = {}
        self.global_woe_ = {}

        for col in df.columns:
            column_series = df[col].astype(object).fillna("Missing").astype(str)
            stats = pd.DataFrame({"cat": column_series, "target": y_arr}).groupby("cat")["target"].agg(["count", "sum"])

            stats["pos"] = stats["sum"]
            stats["neg"] = stats["count"] - stats["sum"]

            num = (stats["pos"] + self.smoothing) / (global_positives + 2 * self.smoothing)
            den = (stats["neg"] + self.smoothing) / (global_negatives + 2 * self.smoothing)

            woe_values = np.log(num / den)
            self.encoding_maps_[col] = woe_values.to_dict()

            glob_num = (global_positives + self.smoothing) / (len(y_arr) + 2 * self.smoothing)
            glob_den = (global_negatives + self.smoothing) / (len(y_arr) + 2 * self.smoothing)
            self.global_woe_[col] = float(np.log(glob_num / glob_den))

        return self

    def transform(self, X):
        check_is_fitted(self, "encoding_maps_")
        df = _to_dataframe(X, self.feature_names_in_)
        transformed_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            string_series = df[col].astype(object).fillna("Missing").astype(str)
            mapping_dict = self.encoding_maps_[col]
            fallback_val = self.global_woe_[col]
            transformed_df[col] = string_series.map(mapping_dict).fillna(fallback_val).astype(float)

        return transformed_df


class OutOfFoldWoEEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Регуляризований Out-Of-Fold (OOF) WoE для запобігання витоку даних."""

    def __init__(self, n_splits: int = 5, smoothing: float = 0.5, random_state: int = 42):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    def fit(self, X, y):
        df = self._store_input_columns(X)
        self.base_encoder_ = WeightOfEvidenceEncoder(smoothing=self.smoothing).fit(df, y)
        return self

    def fit_transform(self, X, y, **fit_params):
        df = self._store_input_columns(X)
        y_arr = np.asarray(y)

        self.fit(df, y_arr)

        oof_transformed = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kfold.split(df, y_arr):
            train_sub_df, val_sub_df = df.iloc[train_idx], df.iloc[val_idx]
            y_train_sub = y_arr[train_idx]

            sub_encoder = WeightOfEvidenceEncoder(smoothing=self.smoothing).fit(train_sub_df, y_train_sub)
            oof_transformed.iloc[val_idx] = sub_encoder.transform(val_sub_df)

        return oof_transformed

    def transform(self, X):
        check_is_fitted(self, "base_encoder_")
        return self.base_encoder_.transform(X)


def build_constructors(feature_eng_config: List[dict]) -> List[tuple[str, BaseEstimator]]:
    """Фабрика для створення кроків конструювання ознак з конфігурації."""
    transformers_list = []
    for idx, spec in enumerate(feature_eng_config):
        ctype = spec["type"]
        name = spec.get("name", f"eng_{idx}")

        if ctype == "ratio":
            tr = BoundedRatioTransformer(
                target_column=name,
                numerator=spec["numerator"],
                denominator=spec["denominator"],
                scale_denominator=spec.get("denominator_scale", 1.0),
                epsilon=spec.get("epsilon", 1e-5),
            )
        elif ctype == "bins":
            tr = StageBinningEncoder(
                target_column=name,
                source_column=spec["source"],
                edges=spec["edges"],
            )
        elif ctype == "cross":
            tr = FeatureCrossEncoder(
                target_column=name,
                source_columns=spec["sources"],
            )
        else:
            raise ValueError(f"Невідомий тип конструктора: {ctype}")
        transformers_list.append((f"fe_{name}", tr))
    return transformers_list


def get_constructor_source_cols(feature_eng_config: List[dict]) -> List[str]:
    """Вилучає назви вхідних джерельних колонок із конфігурації конструкторів."""
    sources = []
    for spec in feature_eng_config or []:
        if "source" in spec:
            sources.append(spec["source"])
        if "sources" in spec:
            sources.extend(spec["sources"])
        if "numerator" in spec:
            sources.append(spec["numerator"])
        if "denominator" in spec:
            sources.append(spec["denominator"])
    return list(set(sources))


# """Кастомні кодувальники та конструктори нових ознак."""
# from __future__ import annotations
#
# import math
# from typing import Any, List, Mapping, Sequence
#
# import numpy as np
# import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.model_selection import StratifiedKFold
# from sklearn.utils.validation import check_is_fitted
#
# from core.helpers import _BaseColumnTracker, _to_dataframe
#
#
# class BoundedRatioTransformer(_BaseColumnTracker, BaseEstimator, TransformerMixin):
#     """Обчислення відношення числових колонок із захистом від ділення на нуль."""
#
#     def __init__(self, target_column: str, numerator: str, denominator: str,
#                  scale_denominator: float = 1.0, epsilon: float = 1e-5):
#         self.target_column = target_column
#         self.numerator = numerator
#         self.denominator = denominator
#         self.scale_denominator = scale_denominator
#         self.epsilon = epsilon
#
#     def fit(self, X, y=None):
#         self._store_input_columns(X)
#         return self
#
#     def transform(self, X):
#         check_is_fitted(self, "feature_names_in_")
#         df = _to_dataframe(X, self.feature_names_in_)
#         num_vals = df[self.numerator].astype(float)
#         den_vals = df[self.denominator].astype(float)
#
#         ratio = num_vals / (den_vals / self.scale_denominator + self.epsilon)
#         ratio = ratio.replace([np.inf, -np.inf], np.nan)
#
#         result_df = df.copy()
#         result_df[self.target_column] = ratio
#         return result_df
#
#
# class StageBinningEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
#     """Дискретизація неперервної часової або числової змінної по інтервалах."""
#
#     def __init__(self, target_column: str, source_column: str, edges: Sequence[float]):
#         self.target_column = target_column
#         self.source_column = source_column
#         self.edges = list(edges)
#
#     def fit(self, X, y=None):
#         self._store_input_columns(X)
#         return self
#
#     def transform(self, X):
#         check_is_fitted(self, "feature_names_in_")
#         df = _to_dataframe(X, self.feature_names_in_)
#         vals = df[self.source_column]
#         binned = pd.cut(vals, bins=self.edges, labels=False, include_lowest=True)
#
#         res_df = df.copy()
#         res_df[self.target_column] = binned
#         return res_df
#
#
# class PeriodicTimeEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
#     """Циклічне кодування міток часу через синус та косинус."""
#
#     def __init__(self, period_mapping: Mapping[str, float]):
#         self.period_mapping = period_mapping
#
#     def fit(self, X, y=None):
#         self._store_input_columns(X)
#         return self
#
#     def transform(self, X):
#         check_is_fitted(self, "feature_names_in_")
#         df = _to_dataframe(X, self.feature_names_in_).copy()
#
#         for col_name, period_val in self.period_mapping.items():
#             if col_name in df.columns:
#                 series = df[col_name].astype(float)
#                 df[f"{col_name}_sin"] = np.sin(2 * np.pi * series / period_val)
#                 df[f"{col_name}_cos"] = np.cos(2 * np.pi * series / period_val)
#         return df
#
#
# class FeatureCrossEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
#     """Комбінування (схрещування) кількох категоріальних ознак у єдиний сегмент."""
#
#     def __init__(self, target_column: str, source_columns: List[str], separator: str = " | "):
#         self.target_column = target_column
#         self.source_columns = source_columns
#         self.separator = separator
#
#     def fit(self, X, y=None):
#         self._store_input_columns(X)
#         return self
#
#     def transform(self, X):
#         check_is_fitted(self, "feature_names_in_")
#         df = _to_dataframe(X, self.feature_names_in_)
#
#         combined_series = None
#         for col in self.source_columns:
#             str_col = df[col].fillna("NA").astype(str)
#             if combined_series is None:
#                 combined_series = str_col
#             else:
#                 combined_series = combined_series + self.separator + str_col
#
#         res_df = df.copy()
#         res_df[self.target_column] = combined_series
#         return res_df
#
#
# class WeightOfEvidenceEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
#     """Базове обчислення Weight of Evidence (WoE) зі згладжуванням Лапласа."""
#
#     def __init__(self, smoothing: float = 0.5):
#         self.smoothing = smoothing
#
#     def fit(self, X, y):
#         if y is None:
#             raise ValueError("WoEEncoder потребує цільову змінну y для навчального етапу (fit)")
#         df = self._store_input_columns(X)
#         y_arr = np.asarray(y)
#
#         global_positives = np.sum(y_arr == 1)
#         global_negatives = np.sum(y_arr == 0)
#
#         self.encoding_maps_ = {}
#         self.global_woe_ = {}
#
#         for col in df.columns:
#             column_series = df[col].fillna("Missing").astype(str)
#             stats = pd.DataFrame({"cat": column_series, "target": y_arr}).groupby("cat")["target"].agg(["count", "sum"])
#
#             stats["pos"] = stats["sum"]
#             stats["neg"] = stats["count"] - stats["sum"]
#
#             num = (stats["pos"] + self.smoothing) / (global_positives + 2 * self.smoothing)
#             den = (stats["neg"] + self.smoothing) / (global_negatives + 2 * self.smoothing)
#
#             woe_values = np.log(num / den)
#             self.encoding_maps_[col] = woe_values.to_dict()
#
#             glob_num = (global_positives + self.smoothing) / (len(y_arr) + 2 * self.smoothing)
#             glob_den = (global_negatives + self.smoothing) / (len(y_arr) + 2 * self.smoothing)
#             self.global_woe_[col] = float(np.log(glob_num / glob_den))
#
#         return self
#
#     def transform(self, X):
#         check_is_fitted(self, "encoding_maps_")
#         df = _to_dataframe(X, self.feature_names_in_)
#         transformed_df = pd.DataFrame(index=df.index)
#
#         for col in df.columns:
#             string_series = df[col].fillna("Missing").astype(str)
#             mapping_dict = self.encoding_maps_[col]
#             fallback_val = self.global_woe_[col]
#             transformed_df[col] = string_series.map(mapping_dict).fillna(fallback_val).astype(float)
#
#         return transformed_df
#
#
# class OutOfFoldWoEEncoder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
#     """Регуляризований Out-Of-Fold (OOF) WoE для запобігання витоку даних."""
#
#     def __init__(self, n_splits: int = 5, smoothing: float = 0.5, random_state: int = 42):
#         self.n_splits = n_splits
#         self.smoothing = smoothing
#         self.random_state = random_state
#
#     def fit(self, X, y):
#         df = self._store_input_columns(X)
#         self.base_encoder_ = WeightOfEvidenceEncoder(smoothing=self.smoothing).fit(df, y)
#         return self
#
#     def fit_transform(self, X, y, **fit_params):
#         df = self._store_input_columns(X)
#         y_arr = np.asarray(y)
#
#         self.fit(df, y_arr)
#
#         oof_transformed = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
#         kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
#
#         for train_idx, val_idx in kfold.split(df, y_arr):
#             train_sub_df, val_sub_df = df.iloc[train_idx], df.iloc[val_idx]
#             y_train_sub = y_arr[train_idx]
#
#             sub_encoder = WeightOfEvidenceEncoder(smoothing=self.smoothing).fit(train_sub_df, y_train_sub)
#             oof_transformed.iloc[val_idx] = sub_encoder.transform(val_sub_df)
#
#         return oof_transformed
#
#     def transform(self, X):
#         check_is_fitted(self, "base_encoder_")
#         return self.base_encoder_.transform(X)
#
#
# def build_constructors(feature_eng_config: List[dict]) -> List[tuple[str, BaseEstimator]]:
#     """Фабрика для створення кроків конструювання ознак з конфігурації."""
#     transformers_list = []
#     for idx, spec in enumerate(feature_eng_config):
#         ctype = spec["type"]
#         name = spec.get("name", f"eng_{idx}")
#
#         if ctype == "ratio":
#             tr = BoundedRatioTransformer(
#                 target_column=name,
#                 numerator=spec["numerator"],
#                 denominator=spec["denominator"],
#                 scale_denominator=spec.get("denominator_scale", 1.0),
#                 epsilon=spec.get("epsilon", 1e-5),
#             )
#         elif ctype == "bins":
#             tr = StageBinningEncoder(
#                 target_column=name,
#                 source_column=spec["source"],
#                 edges=spec["edges"],
#             )
#         elif ctype == "cross":
#             tr = FeatureCrossEncoder(
#                 target_column=name,
#                 source_columns=spec["sources"],
#             )
#         else:
#             raise ValueError(f"Невідомий тип конструктора: {ctype}")
#         transformers_list.append((f"fe_{name}", tr))
#     return transformers_list
#
#
# def get_constructor_source_cols(feature_eng_config: List[dict]) -> List[str]:
#     """Вилучає назви вхідних джерельних колонок із конфігурації конструкторів."""
#     sources = []
#     for spec in feature_eng_config or []:
#         if "source" in spec:
#             sources.append(spec["source"])
#         if "sources" in spec:
#             sources.extend(spec["sources"])
#         if "numerator" in spec:
#             sources.append(spec["numerator"])
#         if "denominator" in spec:
#             sources.append(spec["denominator"])
#     return list(set(sources))