"""Збирання розширеного препроцесора ЛР2 поверх DynamicPipelineBuilder з ЛР1.

Клас ЛР1 не переписується. Розширення — двома способами, обидва через YAML:

1. нові кроки в ``transform_policy`` (``oof_woe``, ``yeo_johnson``) — через реєстр трансформерів;
2. нова необов'язкова секція ``feature_engineering`` — список конструкторів ознак
   (``ratio`` / ``bins`` / ``cross``), що виконуються ДО ColumnTransformer і додають колонки.

Підсумковий об'єкт:  Pipeline([ ("construct", <конструктори>), ("columns", ColumnTransformer) ]).
"""
from __future__ import annotations

import copy

from sklearn.pipeline import Pipeline

from src.dynamic_pipeline_builder import DynamicPipelineBuilder
from src.features import build_constructors, constructor_sources
from src.transformers import ValueAliasMapper

FE_SECTION = "feature_engineering"


def validate_feature_engineering(config: dict) -> None:
    """Вхідні ворота ЛР2: конструктори не можуть використовувати захищені ознаки як вхід.

    Інакше похідна ознака (відношення, сегмент) перенесла б захищену ознаку в модель
    під іншою назвою — саме те, що перевірка ЛР1 за назвами не помітила б.
    """
    spec = config.get(FE_SECTION) or []
    protected = set(config.get("protected_attributes", []))
    leaked = protected & set(constructor_sources(spec))
    if leaked:
        raise ValueError(f"Захищені ознаки використано як вхід конструкторів ознак: {leaked}")
    names = [item.get("name") for item in spec]
    if len(names) != len(set(names)):
        raise ValueError(f"feature_engineering: назви нових ознак повторюються: {names}")


def write_extended_config(base_path, out_path, group: str, columns, kind: str,
                          policy: dict, replace_steps: dict | None = None, header: str = "") -> dict:
    """Генерує конфігурацію = базова + нова група ознак (напр. результат DFS).

    Назви DFS-ознак відомі лише після синтезу, тому конфігурація експериментів 3–4 генерується
    кодом і зберігається у config/generated/ — конвеєр і далі керується YAML, а протокол
    верифікації ЛР1 запускається на ній так само, як на ручних конфігураціях.
    ``policy`` — {родина: [кроки]} для нової групи; ``replace_steps`` — заміна назв кроків
    у всіх групах (напр. {"oof_woe": "woe"} для навмисно «дірявого» Експерименту 4).
    """
    import yaml
    from pathlib import Path

    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["features"][group] = list(columns)
    cfg["feature_kinds"][group] = kind
    for family, steps in policy.items():
        cfg["transform_policy"][family][group] = list(steps)
    if replace_steps:
        for family_policy in cfg["transform_policy"].values():
            for g, steps in family_policy.items():
                family_policy[g] = [replace_steps.get(s, s) if isinstance(s, str) else s for s in (steps or [])]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if header:
            f.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=200)
    return cfg


def write_config_with_groups(base_path, out_path, groups: dict, extra_imputation: dict | None = None,
                             header: str = "") -> dict:
    """Як ``write_extended_config``, але кілька нових груп за раз (ЛР3).

    ``groups`` = {назва: {"columns": [...], "kind": "...", "policy": {родина: [кроки]}}};
    ``extra_imputation`` = {kind: [правила]} — нові типи для imputation_policy (напр. "text").
    """
    import yaml
    from pathlib import Path

    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for kind, rules in (extra_imputation or {}).items():
        cfg["imputation_policy"][kind] = rules
    for name, spec in groups.items():
        cfg["features"][name] = list(spec["columns"])
        cfg["feature_kinds"][name] = spec["kind"]
        for family, steps in spec["policy"].items():
            cfg["transform_policy"][family][name] = list(steps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if header:
            f.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=200)
    return cfg


class ExtendedPipelineBuilder:
    """Обгортка над DynamicPipelineBuilder: ті самі вхідні ворота + валідація feature_engineering."""

    def __init__(self, config_path: str):
        self.base = DynamicPipelineBuilder(config_path)  # validate_config + validate_schema
        self.config = self.base.config
        validate_feature_engineering(self.config)

    def build(self):
        """Ненавчений препроцесор. Без секції feature_engineering — рівно ColumnTransformer ЛР1."""
        ct = self.base.build()
        spec = self.config.get(FE_SECTION) or []
        if not spec:
            return ct
        steps = build_constructors(spec)
        aliases = self.config.get("value_aliases") or {}
        if aliases:
            # синоніми зводяться ДО конструкторів, щоб сегмент не «розщеплювався» (Phone ≡ Mobile Phone)
            steps = [("aliases", ValueAliasMapper(mapping=copy.deepcopy(aliases)))] + steps
        return Pipeline([("construct", Pipeline(steps)), ("columns", ct)])

    # делегування звітності
    def column_transformer(self, fitted):
        return fitted.named_steps["columns"] if isinstance(fitted, Pipeline) else fitted

    def audit_trail(self, fitted):
        return self.base.audit_trail(self.column_transformer(fitted))

    def policy_summary(self):
        return self.base.policy_summary()
