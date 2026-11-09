"""Динамічний конфігуратор препроцесингу на основі параметрів у YAML."""
from __future__ import annotations
from pathlib import Path
import copy
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from core.custom_transformers import TRANSFORMER_REGISTRY, CategoryAliasMapper
from core.helpers import _to_dataframe

MANDATORY_CONFIG_KEYS = (
    "model_family",
    "features",
    "feature_kinds",
    "protected_attributes",
    "imputation_policy",
    "transform_policy",
)

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]


def validate_pipeline_config(config: dict) -> None:
    """Перевіряє ізоляцію захищених атрибутів від списку моделей."""
    protected = set(config.get("protected_attributes", []))
    feature_set = set()
    for grp in config.get("features", {}).values():
        feature_set.update(grp)

    leaked = protected & feature_set
    if leaked:
        raise ValueError(f"Виявлено витік захищених атрибутів у список ознак: {leaked}")


def _parse_step_definition(spec: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping) and len(spec) == 1:
        step_name, step_args = next(iter(spec.items()))
        return str(step_name), dict(step_args or {})
    raise ValueError(f"Невалідний опис кроку: {spec}")


def validate_config_schema(config: dict) -> None:
    """Валідатор структури YAML конфігуратора."""
    missing = [key for key in MANDATORY_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"У конфігураторі відсутні обов'язкові ключі: {missing}")

    family = config["model_family"]
    if family not in config["transform_policy"]:
        raise ValueError(f"Родина моделей {family} відсутня у transform_policy")

    seen_cols: Dict[str, str] = {}
    for group_name, cols in config["features"].items():
        for col in cols or []:
            if col in seen_cols:
                raise ValueError(f"Колонка {col} дублюється в групах {seen_cols[col]} та {group_name}")
            seen_cols[col] = group_name

class SmartMissingImputer(BaseEstimator, TransformerMixin):
    """Динамічний імпутер пропусків за відсотковими порогами політики."""

    def __init__(self, policy_rules: List[Dict[str, Any]], indicator_suffix: str = "__missing"):
        self.policy_rules = policy_rules
        self.indicator_suffix = indicator_suffix

    def _match_rule(self, missing_share: float) -> Dict[str, Any]:
        for rule in self.policy_rules:
            if missing_share <= float(rule["max_share"]):
                return rule
        raise ValueError(f"Неможливо підібрати правило імпутації для частки {missing_share}")

    def fit(self, X, y=None):
        df = _to_dataframe(X)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.fill_values_ = {}
        self.audit_ = {}

        for col in df.columns:
            share = float(df[col].isna().mean())
            rule = self._match_rule(share)
            strat = rule["strategy"]

            if strat == "median":
                val = df[col].median()
            elif strat == "most_frequent":
                # Захист від Categorical при виклику mode()
                col_obj = df[col].astype(object) if isinstance(df[col].dtype, pd.CategoricalDtype) else df[col]
                modes = col_obj.mode(dropna=True)
                val = modes.iloc[0] if len(modes) else "Missing"
            elif strat == "constant":
                val = rule["fill_value"]
            else:
                val = df[col].median()

            self.fill_values_[col] = val
            self.audit_[col] = {
                "missing_share": round(share, 4),
                "rule_max_share": float(rule["max_share"]),
                "strategy": strat,
                "fill_value": val,
                "add_indicator": bool(rule.get("add_indicator", False)),
            }
        return self

    def transform(self, X):
        check_is_fitted(self, "fill_values_")
        df = _to_dataframe(X, self.feature_names_in_)
        out_dict = {}
        indicators = {}

        for col in df.columns:
            series = df[col]
            mask = series.isna()
            fill_val = self.fill_values_[col]

            # ОСНОВНЕ ВИПРАВЛЕННЯ: якщо колонка категоріальна або ми вставляємо рядок,
            # примусово конвертуємо її в object тип до застосування .where()
            if isinstance(series.dtype, pd.CategoricalDtype) or isinstance(fill_val, str):
                series = series.astype(object)

            out_dict[col] = series.where(~mask, fill_val)
            if self.audit_[col]["add_indicator"]:
                indicators[f"{col}{self.indicator_suffix}"] = mask.astype(float)

        out_dict.update(indicators)
        return pd.DataFrame(out_dict, index=df.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "audit_")
        base = list(self.feature_names_in_)
        inds = [f"{c}{self.indicator_suffix}" for c in self.feature_names_in_ if self.audit_[c]["add_indicator"]]
        return np.asarray(base + inds, dtype=object)


# class SmartMissingImputer(BaseEstimator, TransformerMixin):
#     """Динамічний імпутер пропусків за відсотковими порогами політики."""
#
#     def __init__(self, policy_rules: List[Dict[str, Any]], indicator_suffix: str = "__missing"):
#         self.policy_rules = policy_rules
#         self.indicator_suffix = indicator_suffix
#
#     def _match_rule(self, missing_share: float) -> Dict[str, Any]:
#         for rule in self.policy_rules:
#             if missing_share <= float(rule["max_share"]):
#                 return rule
#         raise ValueError(f"Неможливо підібрати правило імпутації для частки {missing_share}")
#
#     def fit(self, X, y=None):
#         df = _to_dataframe(X)
#         self.feature_names_in_ = np.asarray(df.columns, dtype=object)
#         self.fill_values_ = {}
#         self.audit_ = {}
#
#         for col in df.columns:
#             share = float(df[col].isna().mean())
#             rule = self._match_rule(share)
#             strat = rule["strategy"]
#
#             if strat == "median":
#                 val = df[col].median()
#             elif strat == "most_frequent":
#                 modes = df[col].mode(dropna=True)
#                 val = modes.iloc[0] if len(modes) else "Missing"
#             elif strat == "constant":
#                 val = rule["fill_value"]
#             else:
#                 val = df[col].median()
#
#             self.fill_values_[col] = val
#             self.audit_[col] = {
#                 "missing_share": round(share, 4),
#                 "rule_max_share": float(rule["max_share"]),
#                 "strategy": strat,
#                 "fill_value": val,
#                 "add_indicator": bool(rule.get("add_indicator", False)),
#             }
#         return self
#
#     def transform(self, X):
#         check_is_fitted(self, "fill_values_")
#         df = _to_dataframe(X, self.feature_names_in_)
#         out_dict = {}
#         indicators = {}
#
#         for col in df.columns:
#             series = df[col]
#             mask = series.isna()
#             fill_val = self.fill_values_[col]
#
#             if isinstance(fill_val, str) and not pd.api.types.is_string_dtype(series):
#                 series = series.map(lambda v: v if pd.isna(v) else str(v))
#
#             out_dict[col] = series.where(~mask, fill_val)
#             if self.audit_[col]["add_indicator"]:
#                 indicators[f"{col}{self.indicator_suffix}"] = mask.astype(float)
#
#         out_dict.update(indicators)
#         return pd.DataFrame(out_dict, index=df.index)
#
#     def get_feature_names_out(self, input_features=None):
#         check_is_fitted(self, "audit_")
#         base = list(self.feature_names_in_)
#         inds = [f"{c}{self.indicator_suffix}" for c in self.feature_names_in_ if self.audit_[c]["add_indicator"]]
#         return np.asarray(base + inds, dtype=object)


class ConfigurablePipelineFactory:
    """Конструює об'єкт ColumnTransformer відповідно до правил YAML."""

    def __init__(self, config_filepath: str):
        with open(config_filepath, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        validate_pipeline_config(self.config)
        validate_config_schema(self.config)

    def _assemble_group_pipeline(self, group_name: str, cols: List[str], family: str) -> List[Tuple[str, Any]]:
        cfg = self.config
        steps = []

        aliases = {c: cfg["value_aliases"][c] for c in cols if c in (cfg.get("value_aliases") or {})}
        if aliases:
            steps.append(("aliases", CategoryAliasMapper(mapping=copy.deepcopy(aliases))))

        kind = cfg["feature_kinds"][group_name]
        steps.append(("imputer", SmartMissingImputer(policy_rules=copy.deepcopy(cfg["imputation_policy"][kind]))))

        for spec in cfg["transform_policy"][family][group_name] or []:
            step_type, kwargs = _parse_step_definition(spec)
            step_obj = TRANSFORMER_REGISTRY[step_type](**copy.deepcopy(kwargs))
            steps.append((f"{step_type}_{len(steps)}", step_obj))

        return steps

    def build(self) -> ColumnTransformer:
        family = self.config["model_family"]
        transformers = []
        for grp, cols in self.config["features"].items():
            if not cols:
                continue
            pipe_steps = self._assemble_group_pipeline(grp, list(cols), family)
            transformers.append((grp, Pipeline(pipe_steps), list(cols)))
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    def policy_summary(self) -> pd.DataFrame:
        records = []
        for grp in self.config["features"]:
            rec = {"group": grp}
            for fam, pol in self.config["transform_policy"].items():
                chain = [_parse_step_definition(s)[0] for s in (pol.get(grp) or [])]
                rec[fam] = " -> ".join(chain) if chain else "passthrough"
            records.append(rec)
        return pd.DataFrame(records)

    class ConfigurablePipelineFactory:
        """Конструює об'єкт ColumnTransformer відповідно до правил YAML."""

        def __init__(self, config_filepath: str | Path):
            path = Path(config_filepath)
            if not path.is_absolute():
                path = PROJECT_ROOT_DIR / path

            if not path.exists():
                raise FileNotFoundError(f"Конфігураційний файл не знайдено: {path}")

            with open(path, encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            validate_pipeline_config(self.config)
            validate_config_schema(self.config)