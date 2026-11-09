from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from core.feature_transformers import (
    TRANSFORMER_REGISTRY,
    CategoryAliasStandardizer,
    _ensure_dataframe,
)

BASE_DIR = Path(__file__).resolve().parents[1]

REQUIRED_CONFIG_KEYS = (
    "model_family",
    "features",
    "protected_attributes",
    "imputation_policy",
    "transform_policy",
)


def guard_config_integrity(config: dict) -> None:
    """Перевіряє відсутність захищених ознак у списку робочих змінних."""
    protected_set = set(config.get("protected_attributes", []))
    feature_set = set()

    for cols in config.get("features", {}).values():
        feature_set.update(cols)

    leaked = protected_set & feature_set
    if leaked:
        raise ValueError(
            f"[Quality Gate Violation] Захищені ознаки потрапили в перелік features: {leaked}"
        )


def _parse_step_spec(spec: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping) and len(spec) == 1:
        step_name, params = next(iter(spec.items()))
        return str(step_name), dict(params or {})
    raise ValueError(f"Некоректний формат опису кроку трансформації: {spec!r}")


def _verify_imputation_rules(category_name: str, rules_list: Any) -> None:
    if not isinstance(rules_list, list) or not rules_list:
        raise ValueError(
            f"Секція imputation_policy.{category_name} має містити непорожній список."
        )

    thresholds = []
    for rule in rules_list:
        if "max_share" not in rule or "strategy" not in rule:
            raise ValueError(
                f"Правило в imputation_policy.{category_name} не містить max_share або strategy: {rule}"
            )
        if rule["strategy"] == "constant" and "fill_value" not in rule:
            raise ValueError(
                f"Стратегія 'constant' в imputation_policy.{category_name} вимагає fill_value: {rule}"
            )
        thresholds.append(float(rule["max_share"]))

    if thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError(
            f"Пороги max_share в imputation_policy.{category_name} повинні строго зростати: {thresholds}"
        )
    if thresholds[-1] < 1.0:
        raise ValueError(
            f"Останній поріг у imputation_policy.{category_name} має дорівнювати 1.0 (поточний: {thresholds[-1]})"
        )


def validate_full_schema(config: dict) -> None:
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"Відсутні обов'язкові конфігураційні секції: {missing}")

    chosen_family = config["model_family"]
    defined_families = config["transform_policy"]
    if chosen_family not in defined_families:
        raise ValueError(
            f"Модельна родина '{chosen_family}' відсутня в transform_policy. Наявні: {list(defined_families.keys())}"
        )

    feature_groups = config["features"]
    registered_cols: Dict[str, str] = {}

    for group, columns in feature_groups.items():
        for col in columns or []:
            if col in registered_cols:
                raise ValueError(
                    f"Колонка '{col}' входить одночасно у декілька груп: '{registered_cols[col]}' та '{group}'"
                )
            registered_cols[col] = group

    for family_name, ruleset in defined_families.items():
        missing_groups = set(feature_groups.keys()) - set(ruleset or {})
        if missing_groups:
            raise ValueError(
                f"У родинній політиці '{family_name}' відсутні групи: {sorted(missing_groups)}"
            )

        for group, steps in (ruleset or {}).items():
            for step_item in steps or []:
                name, _ = _parse_step_spec(step_item)
                if name not in TRANSFORMER_REGISTRY:
                    raise ValueError(
                        f"Невідомий трансформер '{name}' у transform_policy.{family_name}.{group}."
                    )

    for kind, rules in config["imputation_policy"].items():
        _verify_imputation_rules(kind, rules)

    unknown_aliases = set(config.get("value_aliases", {}).keys()) - set(registered_cols.keys())
    if unknown_aliases:
        raise ValueError(
            f"value_aliases містить колонки, яких немає у features: {sorted(unknown_aliases)}"
        )


class ThresholdBasedImputer(BaseEstimator, TransformerMixin):
    """Адаптивний імпутер: визначає стратегію за часткою пропусків на етапі fit."""

    def __init__(self, rules_policy: List[Dict[str, Any]], missing_suffix: str = "__missing"):
        self.rules_policy = rules_policy
        self.missing_suffix = missing_suffix

    def _match_rule(self, share: float) -> Dict[str, Any]:
        for rule in self.rules_policy:
            if share <= float(rule["max_share"]):
                return rule
        raise ValueError(f"Не знайдено правила для частки пропусків {share:.4f}")

    @staticmethod
    def _compute_fill_value(series: pd.Series, rule: Dict[str, Any]):
        strat = rule["strategy"]
        if strat == "median":
            val = series.median()
        elif strat == "mean":
            val = series.mean()
        elif strat == "most_frequent":
            modes = series.mode(dropna=True)
            val = modes.iloc[0] if len(modes) > 0 else np.nan
        elif strat == "constant":
            val = rule["fill_value"]
        else:
            raise ValueError(f"Невідома стратегія заповнення пропусків: '{strat}'")

        if pd.isna(val):
            raise ValueError(f"Неможливо обчислити значення для колонки '{series.name}' зі стратегією '{strat}'")
        return val.item() if hasattr(val, "item") else val

    def fit(self, X, y=None):
        _verify_imputation_rules("<ThresholdBasedImputer>", self.rules_policy)
        X_df = _ensure_dataframe(X)
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.fill_values_: Dict[str, Any] = {}
        self.audit_log_: Dict[str, Dict[str, Any]] = {}

        for col in X_df.columns:
            share = float(X_df[col].isna().mean())
            matched_rule = self._match_rule(share)
            learned_val = self._compute_fill_value(X_df[col], matched_rule)

            self.fill_values_[col] = learned_val
            self.audit_log_[col] = {
                "observed_missing_share": round(share, 4),
                "applied_threshold": float(matched_rule["max_share"]),
                "chosen_strategy": matched_rule["strategy"],
                "imputed_value": learned_val,
                "created_indicator": bool(matched_rule.get("add_indicator", False)),
            }
        return self

    def transform(self, X):
        check_is_fitted(self, "audit_log_")
        X_df = _ensure_dataframe(X, self.feature_names_in_)

        if list(X_df.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"Невідповідність колонок на вході transform. Очікувалися: {list(self.feature_names_in_)}"
            )

        processed_cols = {}
        indicator_cols = {}

        for col in X_df.columns:
            series = X_df[col]
            missing_mask = series.isna()
            fill_val = self.fill_values_[col]

            if isinstance(fill_val, str) and not pd.api.types.is_string_dtype(series):
                series = series.map(lambda item: item if pd.isna(item) else str(item))

            if not pd.api.types.is_numeric_dtype(series):
                series = series.astype(object)

            processed_cols[col] = series.where(~missing_mask, fill_val)

            if self.audit_log_[col]["created_indicator"]:
                indicator_cols[f"{col}{self.missing_suffix}"] = missing_mask.astype(float)

        processed_cols.update(indicator_cols)
        return pd.DataFrame(processed_cols, index=X_df.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "audit_log_")
        base_names = list(self.feature_names_in_)
        added_indicators = [
            f"{c}{self.missing_suffix}"
            for c in self.feature_names_in_
            if self.audit_log_[c]["created_indicator"]
        ]
        return np.asarray(base_names + added_indicators, dtype=object)


class PipelineSchemaComposer:
    """Збирає ненавчений ColumnTransformer за декларативною YAML-конфігурацією."""

    STEP_ALIASES = "alias_mapper"
    STEP_IMPUTER = "imputer"

    def __init__(self, config_filepath: str | Path):
        path = Path(config_filepath)
        if not path.is_absolute():
            path = BASE_DIR / path

        with open(path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        guard_config_integrity(self.config)
        validate_full_schema(self.config)

    def _determine_category(self, group_name: str) -> str:
        if group_name.startswith("numeric"):
            return "numeric"
        if group_name.startswith("categorical"):
            return "categorical"
        raise ValueError(f"Неможливо автоматично визначити категорію для групи: '{group_name}'")

    def _assemble_group_pipeline(
        self, group: str, columns: List[str], model_family: str
    ) -> List[Tuple[str, Any]]:
        cfg = self.config
        pipeline_steps: List[Tuple[str, Any]] = []

        aliases_subset = {
            col: cfg["value_aliases"][col]
            for col in columns
            if col in cfg.get("value_aliases", {})
        }
        if aliases_subset:
            pipeline_steps.append(
                (self.STEP_ALIASES, CategoryAliasStandardizer(alias_map=copy.deepcopy(aliases_subset)))
            )

        cat_type = self._determine_category(group)
        impute_rules = cfg["imputation_policy"][cat_type]
        pipeline_steps.append(
            (self.STEP_IMPUTER, ThresholdBasedImputer(rules_policy=copy.deepcopy(impute_rules)))
        )

        existing_step_names = {name for name, _ in pipeline_steps}
        transform_sequence = cfg["transform_policy"][model_family][group] or []

        for step_spec in transform_sequence:
            name, params = _parse_step_spec(step_spec)
            unique_name = (
                name
                if name not in existing_step_names
                else f"{name}_{len(pipeline_steps)}"
            )
            existing_step_names.add(unique_name)

            transformer_factory = TRANSFORMER_REGISTRY[name]
            instance = transformer_factory(**copy.deepcopy(params))
            pipeline_steps.append((unique_name, instance))

        return pipeline_steps

    def assemble(self) -> ColumnTransformer:
        family = self.config["model_family"]
        transformer_tuples = []

        for group_name, cols in self.config["features"].items():
            if not cols:
                continue
            group_pipe = Pipeline(
                self._assemble_group_pipeline(group_name, list(cols), family)
            )
            transformer_tuples.append((group_name, group_pipe, list(cols)))

        return ColumnTransformer(
            transformers=transformer_tuples,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def extract_audit_summary(self, fitted_ct: ColumnTransformer) -> pd.DataFrame:
        check_is_fitted(fitted_ct, "transformers_")
        records = []

        for group_name, pipe_obj, _ in fitted_ct.transformers_:
            if not isinstance(pipe_obj, Pipeline):
                continue
            imputer_obj = pipe_obj.named_steps[self.STEP_IMPUTER]
            for feat, info in imputer_obj.audit_log_.items():
                records.append({"feature": feat, "group": group_name, **info})

        return pd.DataFrame(records)


