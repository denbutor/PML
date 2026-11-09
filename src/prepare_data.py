"""ЛР3 — стадія DVC ``prepare``: сирі дані → версіоновані набори для навчання.

Запуск: ``dvc repro`` (або ``python -m src.prepare_data``).

Що робить (усе безстанове — нічого не вчиться з даних, тож може виконуватись до розбиття):
1. сирий xlsx (через DVC) → дедуплікація → MCAR-пропуски (config/data.yaml);
2. прив'язка відгуків Amazon (корпус через DVC, data/raw/amazon.dvc) до клієнтів;
3. безстанові ознаки відгуків (оцінка, довжина, GLCM, циклічний час) → Featuretools DFS;
   **SVD-компоненти тексту сюди НЕ входять** — вони залежні від даних і рахуються всередині
   фолдів ланкою ``ReviewTextSVD`` (ЛР3: герметизація витоку №1);
4. колонка ``ReviewsText`` — тексти відгуків клієнта одним рядком (вхід для ReviewTextSVD);
5. стратифіковане розбиття: ``train.csv`` (розробка, вкладена CV) / ``holdout.csv`` (недоторканний).

Виходи (dvc.yaml → dvc.lock): data/processed/{train,holdout,reviews_linked}.csv, dfs_columns.json.
DFS-примітиви (mean, max, count, natural_logarithm, divide_numeric) не використовують ціль,
тому фіксація згенерованої матриці під DVC коректна (ТЗ ЛР3, п. 4.3).
"""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from src.data import PROCESSED, PROJECT_ROOT, load_data_config, load_dataset
from src.extraction import (ImageExtractor, customer_review_text, download_images, load_corpus,
                            load_multimodal_config, link_reviews_to_customers, review_features, run_dfs)


def load_hermetic_config(path: str | Path = "config/hermetic.yaml") -> dict:
    path = Path(path)
    with open(path if path.is_absolute() else PROJECT_ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    warnings.filterwarnings("ignore")
    dc, mm, hc = load_data_config(), load_multimodal_config(), load_hermetic_config()
    id_col, target = dc["id_column"], dc["target"]

    df = load_dataset(dc)                                                   # 1
    reviews = link_reviews_to_customers(df, load_corpus(mm), mm["linkage"])  # 2
    reviews["image_path"] = download_images(reviews, mm["corpus"]["images"])  # кеш у data/raw/amazon/images

    rev_feat = review_features(reviews, None, ImageExtractor(levels=mm["extraction"]["image"]["levels"]), mm)
    mm_stateless = copy.deepcopy(mm)                                          # 3: без text_svd_*
    mm_stateless["dfs"]["review_columns"] = [c for c in mm["dfs"]["review_columns"] if not c.startswith("text_svd")]
    fm, _ = run_dfs(df, rev_feat, mm_stateless)

    data = df.join(fm, on=id_col).join(customer_review_text(reviews, id_col), on=id_col)   # 4
    dev_idx, hold_idx = train_test_split(                                                  # 5
        data.index, test_size=float(hc["split"]["holdout_share"]), stratify=data[target],
        random_state=int(hc["split"]["seed"]))

    out = PROJECT_ROOT / PROCESSED
    out.mkdir(parents=True, exist_ok=True)
    data.loc[sorted(dev_idx)].to_csv(out / "train.csv", index=False)
    data.loc[sorted(hold_idx)].to_csv(out / "holdout.csv", index=False)
    reviews.to_csv(out / "reviews_linked.csv", index=False)
    (out / "dfs_columns.json").write_text(json.dumps(list(fm.columns), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"train {len(dev_idx)} | holdout {len(hold_idx)} | DFS-ознак {fm.shape[1]} | відгуків {len(reviews)}")


def write_hermetic_configs(hc: dict | None = None) -> list:
    """Конфігурації герметичного конвеєра ЛР3 = config_{lr,rf}_fe.yaml (ЛР2)
    + група ``multimodal_dfs`` (безстанові DFS-ознаки з data/processed)
    + група ``review_text`` (колонка ReviewsText → ReviewTextSVD усередині фолду).
    """
    from src.data import open_data
    from src.pipeline import write_config_with_groups

    hc = hc or load_hermetic_config()
    with open_data(f"{PROCESSED}/dfs_columns.json") as f:
        dfs_cols = json.load(f)
    tb = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in hc["text_block"].items()}
    groups = {
        "multimodal_dfs": {"columns": dfs_cols, "kind": "numeric",
                           "policy": {"additive": ["yeo_johnson"],
                                      "partitioning": [{"winsorize": {"lower": 0.01, "upper": 0.99}}]}},
        "review_text": {"columns": ["ReviewsText"], "kind": "text",
                        "policy": {"additive": [{"review_text_svd": tb}, "standard_scaler"],
                                   "partitioning": [{"review_text_svd": tb}]}},
    }
    text_rule = {"text": [{"max_share": 1.0, "strategy": "constant", "fill_value": ""}]}
    paths = []
    for f in ("lr", "rf"):
        out = PROJECT_ROOT / "config" / "generated" / f"config_{f}_hermetic.yaml"
        write_config_with_groups(PROJECT_ROOT / "config" / f"config_{f}_fe.yaml", out, groups, text_rule,
                                 header=f"ЗГЕНЕРОВАНО src/prepare_data.py::write_hermetic_configs — не редагувати вручну.\n"
                                        f"ЛР3: config_{f}_fe.yaml + безстанові DFS-ознаки + текстовий блок (SVD у фолдах).")
        paths.append(out)
    return paths


if __name__ == "__main__":
    main()
