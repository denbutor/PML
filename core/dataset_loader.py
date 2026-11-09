from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]


def read_data_config(cfg_path: str | Path = "config/dataset_config.yaml") -> dict:
    target_path = Path(cfg_path)
    if not target_path.is_absolute():
        target_path = BASE_DIR / target_path
    with open(target_path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def fetch_raw_dataframe(data_cfg: dict) -> pd.DataFrame:
    src = data_cfg["source"]
    file_path = Path(src["file_path"])
    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Файл {file_path} відсутній. Помістіть джерело даних за цим шляхом."
        )

    return pd.read_excel(file_path, sheet_name=src.get("sheet_name", 0))


def apply_mcar_mask(
    df: pd.DataFrame, target_shares: Dict[str, float], seed: int
) -> pd.DataFrame:
    df_masked = df.copy()
    generator = np.random.default_rng(seed)

    for feature_name, share in target_shares.items():
        if feature_name not in df_masked.columns:
            raise KeyError(f"Колонку '{feature_name}' не знайдено в наборі даних.")

        valid_indices = np.flatnonzero(df_masked[feature_name].notna().to_numpy())
        sample_size = min(int(round(share * len(df_masked))), len(valid_indices))

        selected_indices = generator.choice(
            valid_indices, size=sample_size, replace=False
        )
        df_masked.loc[df_masked.index[selected_indices], feature_name] = np.nan

    return df_masked


def get_prepared_dataset(
    data_cfg: dict | None = None, apply_injection: bool = True
) -> pd.DataFrame:
    config = data_cfg or read_data_config()
    raw_df = fetch_raw_dataframe(config)
    inj_spec = config.get("missing_data_injection", {})

    if apply_injection and inj_spec.get("target_shares"):
        raw_df = apply_mcar_mask(
            raw_df,
            inj_spec["target_shares"],
            seed=int(inj_spec.get("random_seed", 42)),
        )
    return raw_df


def extract_features_and_target(
    df: pd.DataFrame, data_cfg: dict
) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = data_cfg["target_field"]
    return df.drop(columns=[target_col]), df[target_col].astype(int)