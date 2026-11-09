"""Завантаження датасету та контрольоване інжектування пропусків.

ЛР3: усе читання файлів даних іде через DVC API (``open_data`` → ``dvc.api.open``), а не напряму
з диску: так код завжди бачить саме ту версію, на яку вказує .dvc-покажчик / dvc.lock (і будь-яку
попередню — через ``rev``).

Тут немає жодних статистик по даних: лише читання файла і детерміноване
(за seed) маскування випадково обраних клітинок (MCAR). Інжектування
імітує властивість джерела даних, а не крок препроцесингу, тому живе
поза конвеєром і описане в config/data.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import io

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = "data/processed"


def _rel(path: str | Path) -> str:
    """Шлях відносно кореня репозиторію (так його очікує dvc.api)."""
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(PROJECT_ROOT)
    return path.as_posix()


def open_data(path: str | Path, mode: str = "r", rev: str | None = None, encoding: str = "utf-8"):
    """Відкриває файл даних через DVC API (версіонований вміст), а не напряму з диску.

    ``rev`` — git-коміт / тег (напр. ``lab3-v1.0``); None — поточний робочий стан репозиторію.
    """
    import dvc.api

    kwargs = {} if "b" in mode else {"encoding": encoding}
    return dvc.api.open(_rel(path), repo=str(PROJECT_ROOT), rev=rev, mode=mode, **kwargs)


def read_bytes(path: str | Path, rev: str | None = None) -> bytes:
    with open_data(path, mode="rb", rev=rev) as f:
        return f.read()


def read_csv(path: str | Path, rev: str | None = None, **kwargs) -> pd.DataFrame:
    with open_data(path, rev=rev) as f:
        return pd.read_csv(f, **kwargs)


def read_processed(name: str, rev: str | None = None) -> pd.DataFrame:
    """Підготовлений набір ЛР3 (``train`` / ``holdout``) — лише через DVC."""
    return read_csv(f"{PROCESSED}/{name}.csv", rev=rev)


def load_data_config(path: str | Path = "config/data.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(data_config: dict, rev: str | None = None) -> pd.DataFrame:
    """Сирий датасет без жодних змін — через DVC (``data/raw/*.xlsx.dvc``)."""
    src = data_config["source"]
    try:
        content = read_bytes(src["path"], rev=rev)
    except Exception as exc:  # файл відсутній або не під DVC
        raise FileNotFoundError(
            f"Не вдалося прочитати {src['path']} через DVC ({exc}). Завантажте датасет за посиланням "
            f"{src.get('url')}, покладіть у data/raw/ і виконайте `dvc add`/`dvc pull`."
        ) from exc
    return pd.read_excel(io.BytesIO(content), sheet_name=src.get("sheet", 0))


def inject_mcar(df: pd.DataFrame, columns: Dict[str, float], seed: int) -> pd.DataFrame:
    """Додає пропуски типу MCAR: у колонці маскується рівно ``share·n`` випадкових клітинок.

    Клітинки обираються з усіх рядків рівноймовірно (незалежно від значень),
    тож підсумкова частка = природні пропуски + інжектовані (без перетину
    з уже пропущеними).
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col, share in columns.items():
        if col not in out.columns:
            raise KeyError(f"inject_mcar: колонки {col!r} немає в датасеті")
        candidates = np.flatnonzero(out[col].notna().to_numpy())
        k = int(round(share * len(out)))
        k = min(k, len(candidates))
        idx = rng.choice(candidates, size=k, replace=False)
        out.loc[out.index[idx], col] = np.nan
    return out


def drop_exact_duplicates(df: pd.DataFrame, id_column: str | None) -> pd.DataFrame:
    """Прибирає рядки, що повністю повторюють інший (без урахування ідентифікатора).

    Детерміноване очищення (залишається перше входження), без жодних статистик по даних.
    """
    subset = [c for c in df.columns if c != id_column]
    return df.loc[~df.duplicated(subset=subset, keep="first")].reset_index(drop=True)


def load_dataset(data_config: dict | None = None, inject: bool = True,
                 deduplicate: bool | None = None) -> pd.DataFrame:
    """Датасет, з яким працює конвеєр: сирі дані → [дедуплікація] → інжектовані пропуски."""
    data_config = data_config or load_data_config()
    df = load_raw(data_config)
    if deduplicate is None:
        deduplicate = bool(data_config.get("deduplicate", False))
    if deduplicate:
        df = drop_exact_duplicates(df, data_config.get("id_column"))
    spec = data_config.get("missing_injection") or {}
    if inject and spec.get("columns"):
        df = inject_mcar(df, spec["columns"], seed=int(spec["seed"]))
    return df


def split_xy(df: pd.DataFrame, data_config: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """X — усі колонки, крім цілі (ID і захищена ознака лишаються: їх відкидає ColumnTransformer)."""
    target = data_config["target"]
    return df.drop(columns=[target]), df[target].astype(int)
