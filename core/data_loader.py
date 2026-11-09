"""Завантаження, первинне очищення даних та імпутація штучних пропусків."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]


def fetch_dataset_config(config_path: str | Path = "config/dataset_specs.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT_DIR / path
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_raw_dataset(specs: dict) -> pd.DataFrame:
    src_info = specs["source"]
    file_path = Path(src_info["path"])
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT_DIR / file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Файл даних відсутній за шляхом: {file_path}")

    return pd.read_excel(file_path, sheet_name=src_info.get("sheet", 0))


def inject_mcar_missingness(df: pd.DataFrame, mask_ratios: Dict[str, float], random_seed: int) -> pd.DataFrame:
    df_copy = df.copy()
    rng = np.random.default_rng(random_seed)

    for col, ratio in mask_ratios.items():
        if col not in df_copy.columns:
            raise KeyError(f"Колонка {col} відсутня у датасеті")
        valid_indices = np.flatnonzero(df_copy[col].notna().to_numpy())
        sample_count = int(round(ratio * len(df_copy)))
        sample_count = min(sample_count, len(valid_indices))

        chosen_indices = rng.choice(valid_indices, size=sample_count, replace=False)
        df_copy.loc[df_copy.index[chosen_indices], col] = np.nan

    return df_copy


def remove_exact_duplicates(df: pd.DataFrame, id_col_name: str | None) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != id_col_name]
    return df.loc[~df.duplicated(subset=feature_cols, keep="first")].reset_index(drop=True)


def get_prepared_dataset(config: dict | None = None, apply_mcar: bool = True,
                         dedup: bool | None = None) -> pd.DataFrame:
    config = config or fetch_dataset_config()
    raw_df = read_raw_dataset(config)

    should_dedup = dedup if dedup is not None else bool(config.get("deduplicate", False))
    if should_dedup:
        raw_df = remove_exact_duplicates(raw_df, config.get("id_column"))

    mcar_spec = config.get("missing_injection") or {}
    if apply_mcar and mcar_spec.get("columns"):
        raw_df = inject_mcar_missingness(raw_df, mcar_spec["columns"], random_seed=int(mcar_spec["seed"]))

    return raw_df


def separate_features_target(df: pd.DataFrame, specs: dict) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = specs["target"]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y