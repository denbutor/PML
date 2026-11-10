"""Дані ЛР5 (через DVC).

* **train ЛР5** = train.csv @ lab3-v1.0, рядки з role == "pool" у lab4a_split.csv (3 044) — навчання v1;
* **пул дорозмітки** (трек 4A) = рядки з role == "test" у lab4a_split.csv (1 015, 25% train) — відкладено ДО етапу 1,
  у навчанні v1 не бере участі; мітки — «оракул»;
* **test ЛР5** = holdout.csv @ lab3-v1.0 (1 015) — не використовувався в ЛР3–ЛР4; лише фінальні метрики.
"""
from __future__ import annotations

import pandas as pd

from src.data import load_data_config, read_csv

REV = "lab3-v1.0"


def load_lab5() -> dict:
    dc = load_data_config()
    train_all = read_csv("data/processed/train.csv", rev=REV)
    roles = read_csv("data/processed/lab4a_split.csv")
    train_all = train_all.merge(roles, on=dc["id_column"], how="left", validate="one_to_one")
    assert train_all["role"].notna().all()
    test = read_csv("data/processed/holdout.csv", rev=REV)
    out = {"train": train_all[train_all.role == "pool"].drop(columns="role").reset_index(drop=True),
           "pool": train_all[train_all.role == "test"].drop(columns="role").reset_index(drop=True),
           "test": test.reset_index(drop=True), "target": dc["target"], "id": dc["id_column"]}
    ids = [set(out[k][dc["id_column"]]) for k in ("train", "pool", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "train / pool / test перетинаються"
    return out
