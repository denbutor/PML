"""
* ``validate_config`` — готова функція з ТЗ: блокує потрапляння захищених ознак у features;
* ``validate_schema`` — додаткові вхідні ворота: структура конфігурації коректна;
* ``AdaptiveImputer`` — імпутер, що обирає правило за часткою пропусків, обчисленою у ``fit``;
* ``DynamicPipelineBuilder`` — збирає ненавчений ``ColumnTransformer`` за YAML-описом.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.transformers import TRANSFORMER_REGISTRY, ValueAliasMapper
from src.utils import _as_frame

REQUIRED_SECTIONS = (
    "model_family",
    "features",
    "feature_kinds",
    "protected_attributes",
    "imputation_policy",
    "transform_policy",
)

def validate_config(config: dict) -> None:
    """Блокує потрапляння protected_attributes до features."""
    protected = set(config.get("protected_attributes", []))
    features = set()
    for group in config.get("features", {}).values():
        features.update(group)
    leaked = protected & features
    if leaked:
        raise ValueError(f"Захищені ознаки потрапили у features: {leaked}")


def _parse_step(spec: Any) -> Tuple[str, Dict[str, Any]]:
    """Крок transform_policy: або рядок ``name``, або словник ``{name: {kwargs}}``."""
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, Mapping) and len(spec) == 1:
        name, kwargs = next(iter(spec.items()))
        return str(name), dict(kwargs or {})
    raise ValueError(f"Некоректний опис кроку transform_policy: {spec!r}")


def _validate_rules(kind: str, rules: Any) -> None:
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"imputation_policy.{kind}: очікується непорожній список правил")
    shares = []
    for rule in rules:
        if "max_share" not in rule or "strategy" not in rule:
            raise ValueError(f"imputation_policy.{kind}: правило без max_share/strategy: {rule}")
        if rule["strategy"] == "constant" and "fill_value" not in rule:
            raise ValueError(f"imputation_policy.{kind}: strategy=constant потребує fill_value: {rule}")
        shares.append(float(rule["max_share"]))
    if shares != sorted(shares) or len(set(shares)) != len(shares):
        raise ValueError(f"imputation_policy.{kind}: пороги max_share мають строго зростати: {shares}")
    if shares[-1] < 1.0:
        raise ValueError(
            f"imputation_policy.{kind}: останній поріг має бути 1.0, щоб покрити будь-яку частку, "
            f"зараз {shares[-1]}"
        )


def validate_schema(config: dict) -> None:
    """Структурна перевірка конфігурації до збирання конвеєра (fail fast, fail loud)."""
    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"У конфігурації відсутні секції: {missing}")

    family = config["model_family"]
    families = config["transform_policy"]
    if family not in families:
        raise ValueError(
            f"model_family={family!r} не описано в transform_policy (є: {sorted(families)})"
        )

    features = config["features"]
    seen: Dict[str, str] = {}
    for group, cols in features.items():
        for col in cols or []:
            if col in seen:
                raise ValueError(f"Колонка {col!r} входить одночасно до груп {seen[col]!r} і {group!r}")
            seen[col] = group

    kinds = config["feature_kinds"]
    for group in features:
        if group not in kinds:
            raise ValueError(f"Для групи {group!r} не задано feature_kinds")
        if kinds[group] not in config["imputation_policy"]:
            raise ValueError(
                f"Група {group!r} має тип {kinds[group]!r}, для якого немає imputation_policy"
            )
    for kind, rules in config["imputation_policy"].items():
        _validate_rules(kind, rules)

    # Обидві родини описують однаковий набір груп — інакше конфігурації непорівнювані.
    for fam, policy in families.items():
        absent = set(features) - set(policy or {})
        if absent:
            raise ValueError(f"transform_policy.{fam}: не описано групи {sorted(absent)}")
        for group, steps in (policy or {}).items():
            for spec in steps or []:
                name, _ = _parse_step(spec)
                if name not in TRANSFORMER_REGISTRY:
                    raise ValueError(
                        f"transform_policy.{fam}.{group}: невідомий крок {name!r}; "
                        f"доступні: {sorted(TRANSFORMER_REGISTRY)}"
                    )

    unknown_alias_cols = set(config.get("value_aliases") or {}) - set(seen)
    if unknown_alias_cols:
        raise ValueError(f"value_aliases посилається на колонки поза features: {sorted(unknown_alias_cols)}")


# --------------------------------------------------------------------------- #
# Адаптивний імпутер
# --------------------------------------------------------------------------- #
class AdaptiveImputer(BaseEstimator, TransformerMixin):
    """Маршрутизує колонки за часткою пропусків згідно з політикою з конфігурації.

    * Частка пропусків обчислюється **лише у fit** (тобто на тренувальній підвибірці / фолді).
    * Для колонки обирається перше правило політики, у якого ``share <= max_share``.
    * ``transform`` нічого не переобчислює: застосовує зафіксовані у fit значення.
    * ``audit_`` — аудиторський слід: частка, правило, стратегія, значення заповнення.
    * Індикатор пропуску створюється за правилом (а не за фактом пропусків у train),
      тому розмірність виходу не залежить від того, чи є пропуски в тесті.
    """

    def __init__(self, policy: List[Dict[str, Any]], indicator_suffix: str = "__missing"):
        self.policy = policy
        self.indicator_suffix = indicator_suffix

    def _select_rule(self, share: float) -> Dict[str, Any]:
        for rule in self.policy:
            if share <= float(rule["max_share"]):
                return rule
        raise ValueError(f"Жодне правило політики не покриває частку пропусків {share:.3f}")

    @staticmethod
    def _learn_value(series: pd.Series, rule: Dict[str, Any]):
        strategy = rule["strategy"]
        if strategy == "median":
            value = series.median()
        elif strategy == "mean":
            value = series.mean()
        elif strategy == "most_frequent":
            modes = series.mode(dropna=True)
            value = modes.iloc[0] if len(modes) else np.nan
        elif strategy == "constant":
            value = rule["fill_value"]
        else:
            raise ValueError(f"Невідома стратегія імпутації: {strategy!r}")
        if pd.isna(value):
            raise ValueError(
                f"Колонка {series.name!r}: стратегія {strategy!r} не може оцінити значення "
                f"(усі значення пропущені) — потрібне правило constant"
            )
        return value.item() if hasattr(value, "item") else value

    def fit(self, X, y=None):  # y — лише для сумісності з API sklearn, не використовується
        _validate_rules("<AdaptiveImputer>", self.policy)
        X = _as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.statistics_: Dict[str, Any] = {}
        self.audit_: Dict[str, Dict[str, Any]] = {}
        for col in X.columns:
            share = float(X[col].isna().mean())  # частка обчислюється тут і тільки тут
            rule = self._select_rule(share)
            value = self._learn_value(X[col], rule)
            self.statistics_[col] = value
            self.audit_[col] = {
                "missing_share": round(share, 4),
                "rule_max_share": float(rule["max_share"]),
                "strategy": rule["strategy"],
                "fill_value": value,
                "add_indicator": bool(rule.get("add_indicator", False)),
            }
        return self

    def transform(self, X):
        check_is_fitted(self, "audit_")
        X = _as_frame(X, self.feature_names_in_)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"AdaptiveImputer: очікувались колонки {list(self.feature_names_in_)}, "
                f"отримано {list(X.columns)}"
            )
        out = {}
        indicators = {}
        for col in X.columns:
            series = X[col]
            mask = series.isna()
            value = self.statistics_[col]
            if isinstance(value, str) and not pd.api.types.is_string_dtype(series):
                # числовий код категорії + текстова мітка "Missing" → усе як текст
                series = series.map(lambda v: v if pd.isna(v) else str(v))
            if not pd.api.types.is_numeric_dtype(series):
                series = series.astype(object)
            out[col] = series.where(~mask, value)
            if self.audit_[col]["add_indicator"]:
                indicators[f"{col}{self.indicator_suffix}"] = mask.astype(float)
        out.update(indicators)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "audit_")
        names = list(self.feature_names_in_)
        names += [f"{c}{self.indicator_suffix}" for c in self.feature_names_in_ if self.audit_[c]["add_indicator"]]
        return np.asarray(names, dtype=object)


# --------------------------------------------------------------------------- #
# Конфігуратор
# --------------------------------------------------------------------------- #
class DynamicPipelineBuilder:
    """Читає YAML-конфігурацію і збирає з неї ненавчений ColumnTransformer."""

    IMPUTER_STEP = "imputer"
    ALIAS_STEP = "aliases"

    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        validate_config(self.config)
        validate_schema(self.config)

    def _group_steps(self, group: str, columns: List[str], family: str) -> List[Tuple[str, Any]]:
        cfg = self.config
        steps: List[Tuple[str, Any]] = []

        aliases = {c: cfg["value_aliases"][c] for c in columns if c in (cfg.get("value_aliases") or {})}
        if aliases:
            steps.append((self.ALIAS_STEP, ValueAliasMapper(mapping=copy.deepcopy(aliases))))

        kind = cfg["feature_kinds"][group]
        steps.append((self.IMPUTER_STEP, AdaptiveImputer(policy=copy.deepcopy(cfg["imputation_policy"][kind]))))

        used = {name for name, _ in steps}
        for spec in cfg["transform_policy"][family][group] or []:
            name, kwargs = _parse_step(spec)
            step_name = name if name not in used else f"{name}_{len(steps)}"
            used.add(step_name)
            steps.append((step_name, TRANSFORMER_REGISTRY[name](**copy.deepcopy(kwargs))))
        return steps

    def build(self) -> ColumnTransformer:
        """Повертає ненавчений ColumnTransformer. До даних не звертається."""
        model_family = self.config["model_family"]
        transformers = []
        for group, columns in self.config["features"].items():
            if not columns:
                continue
            steps = self._group_steps(group, list(columns), model_family)
            transformers.append((group, Pipeline(steps), list(columns)))
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # усе, чого немає у features (ID, ціль, захищені ознаки), відкидається
            verbose_feature_names_out=False,
        )

    # -- звітність -------------------------------------------------------- #
    def policy_summary(self) -> pd.DataFrame:
        """Ланцюжок кроків для кожної групи за кожної родини моделей (з конфігурації, без даних)."""
        rows = []
        for group in self.config["features"]:
            row = {"group": group}
            for family, policy in self.config["transform_policy"].items():
                chain = [_parse_step(s)[0] for s in (policy.get(group) or [])]
                row[family] = " → ".join(chain) if chain else "passthrough"
            rows.append(row)
        return pd.DataFrame(rows)

    def audit_trail(self, fitted: ColumnTransformer) -> pd.DataFrame:
        """Аудиторський слід усіх AdaptiveImputer навченого ColumnTransformer."""
        check_is_fitted(fitted, "transformers_")
        rows = []
        for group, pipe, _ in fitted.transformers_:
            if not isinstance(pipe, Pipeline):
                continue
            imputer = pipe.named_steps[self.IMPUTER_STEP]
            for col, info in imputer.audit_.items():
                rows.append({"feature": col, "group": group, **info})
        return pd.DataFrame(rows)

    def metadata_table(self, fitted: ColumnTransformer) -> pd.DataFrame:
        """Таблиця п. 5.3: частка пропусків з аудиторського сліду + стратегії обох родин."""
        audit = self.audit_trail(fitted)
        policy = self.policy_summary().set_index("group")
        audit["rule"] = audit.apply(
            lambda r: f"max_share: {r.rule_max_share:.2f}, {r.strategy}"
            + (" + indicator" if r.add_indicator else ""),
            axis=1,
        )
        table = audit[["feature", "group", "missing_share", "rule"]].copy()
        for family in policy.columns:
            table[family] = table["group"].map(policy[family])
        return table
