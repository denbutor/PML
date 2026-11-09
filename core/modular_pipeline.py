"""Розширення конвеєра з підтримкою генерації кастомних ознак."""
from __future__ import annotations

import copy
from pathlib import Path
import yaml
from sklearn.pipeline import Pipeline

from core.custom_transformers import CategoryAliasMapper
from core.feature_constructors import build_constructors, get_constructor_source_cols
from core.pipeline_factory import ConfigurablePipelineFactory, PROJECT_ROOT_DIR


def validate_engineered_features(config: dict) -> None:
    spec = config.get("feature_engineering") or []
    protected = set(config.get("protected_attributes", []))
    used_sources = set(get_constructor_source_cols(spec))

    leaked = protected & used_sources
    if leaked:
        raise ValueError(f"Захищені атрибути використані при створенні кастомних ознак: {leaked}")


def generate_extended_config(base_cfg_path: str, output_path: str, feature_group: str,
                             new_columns: list, feature_kind: str, policy: dict,
                             replacement_steps: dict | None = None) -> dict:
    base_p = Path(base_cfg_path)
    if not base_p.is_absolute():
        base_p = PROJECT_ROOT_DIR / base_p

    with open(base_p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["features"][feature_group] = list(new_columns)
    cfg["feature_kinds"][feature_group] = feature_kind

    for family, steps in policy.items():
        cfg["transform_policy"][family][feature_group] = list(steps)

    if replacement_steps:
        for fam_pol in cfg["transform_policy"].values():
            for g, stps in fam_pol.items():
                fam_pol[g] = [replacement_steps.get(s, s) if isinstance(s, str) else s for s in (stps or [])]

    out_p = Path(output_path)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT_DIR / out_p

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return cfg


class CompositePipelineBuilder:
    """Обгортка для збірки повного конвеєра (конструктори + ColumnTransformer)."""

    def __init__(self, config_path: str | Path):
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT_DIR / path

        self.factory = ConfigurablePipelineFactory(path)
        self.config = self.factory.config
        validate_engineered_features(self.config)

    def build(self):
        col_transformer = self.factory.build()
        fe_specs = self.config.get("feature_engineering") or []
        if not fe_specs:
            return col_transformer

        fe_steps = build_constructors(fe_specs)
        aliases = self.config.get("value_aliases") or {}
        if aliases:
            fe_steps.insert(0, ("aliases", CategoryAliasMapper(mapping=copy.deepcopy(aliases))))

        return Pipeline([("constructors", Pipeline(fe_steps)), ("preprocessor", col_transformer)])

# """Розширення конвеєра з підтримкою генерації кастомних ознак."""
# from __future__ import annotations
#
# import copy
# from pathlib import Path
# import yaml
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
#
# from core.custom_transformers import CategoryAliasMapper
# from core.feature_constructors import build_constructors, get_constructor_source_cols
# from core.pipeline_factory import ConfigurablePipelineFactory
#
# PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
#
#
# def validate_engineered_features(config: dict) -> None:
#     spec = config.get("feature_engineering") or []
#     protected = set(config.get("protected_attributes", []))
#     used_sources = set(get_constructor_source_cols(spec))
#
#     leaked = protected & used_sources
#     if leaked:
#         raise ValueError(f"Захищені атрибути використані при створенні кастомних ознак: {leaked}")
#
#
# def generate_extended_config(base_cfg_path: str, output_path: str, feature_group: str,
#                              new_columns: list, feature_kind: str, policy: dict,
#                              replacement_steps: dict | None = None) -> dict:
#     with open(base_cfg_path, encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)
#
#     cfg["features"][feature_group] = list(new_columns)
#     cfg["feature_kinds"][feature_group] = feature_kind
#
#     for family, steps in policy.items():
#         cfg["transform_policy"][family][feature_group] = list(steps)
#
#     if replacement_steps:
#         for fam_pol in cfg["transform_policy"].values():
#             for g, stps in fam_pol.items():
#                 fam_pol[g] = [replacement_steps.get(s, s) if isinstance(s, str) else s for s in (stps or [])]
#
#     out_p = Path(output_path)
#     out_p.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_p, "w", encoding="utf-8") as f:
#         yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
#     return cfg
#
# class CompositePipelineBuilder:
#     """Обгортка для збірки повного конвеєра (конструктори + ColumnTransformer)."""
#
#     def __init__(self, config_path: str | Path):
#         path = Path(config_path)
#         if not path.is_absolute():
#             path = PROJECT_ROOT_DIR / path
#
#         self.factory = ConfigurablePipelineFactory(path)
#         self.config = self.factory.config
#         validate_engineered_features(self.config)
#
# # class CompositePipelineBuilder:
# #     """Обгортка для збірки повного конвеєра (конструктори + ColumnTransformer)."""
# #
# #     def __init__(self, config_path: str):
# #         self.factory = ConfigurablePipelineFactory(config_path)
# #         self.config = self.factory.config
# #         validate_engineered_features(self.config)
# #
# #     def build(self) -> Pipeline | ColumnTransformer:
# #         col_transformer = self.factory.build()
# #         fe_specs = self.config.get("feature_engineering") or []
# #         if not fe_specs:
# #             return col_transformer
# #
# #         fe_steps = build_constructors(fe_specs)
# #         aliases = self.config.get("value_aliases") or {}
# #         if aliases:
# #             fe_steps.insert(0, ("aliases", CategoryAliasMapper(mapping=copy.deepcopy(aliases))))
# #
# #         return Pipeline([("constructors", Pipeline(fe_steps)), ("preprocessor", col_transformer)])