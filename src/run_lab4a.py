"""ЛР4A — запуск з командного рядка.
    python -m src.run_lab4a --split   # стадія DVC lab4a_split: ролі test / pool (data/processed/lab4a_split.csv)
    python -m src.run_lab4a --ssl     # semi-supervised: частки міток × повтори × методи
    python -m src.run_lab4a --al      # active learning з оракулом: random / entropy / entropy + diversity
Результати → reports/lab4a_*.csv. Ноутбук notebooks/lab4a_semi_supervised.ipynb їх аналізує.
"""
from __future__ import annotations

import argparse
import time
import warnings

from src.data import PROCESSED, PROJECT_ROOT, load_data_config, read_csv
from src.evaluation import load_yaml
from src.semi_supervised import load_lab4a_config, load_train, make_split, run_active_learning, run_ssl_experiments

SPLIT_PATH = f"{PROCESSED}/lab4a_split.csv"


def load_xy():
    cfg, dc = load_lab4a_config(), load_data_config()
    df = load_train(cfg)
    roles = read_csv(SPLIT_PATH)
    df = df.merge(roles, on=dc["id_column"], how="left", validate="one_to_one")
    assert df["role"].notna().all(), "lab4a_split.csv не покриває train.csv — перезапустіть dvc repro"
    pool, test = df[df.role == "pool"].drop(columns="role"), df[df.role == "test"].drop(columns="role")
    t = dc["target"]
    return (pool.drop(columns=[t]), pool[t].astype(int), test.drop(columns=[t]), test[t].astype(int), cfg)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", action="store_true")
    g.add_argument("--ssl", action="store_true")
    g.add_argument("--al", action="store_true")
    args = ap.parse_args()
    rep = PROJECT_ROOT / "reports"
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    if args.split:
        cfg = load_lab4a_config()
        out = PROJECT_ROOT / SPLIT_PATH
        make_split(load_train(cfg), cfg).to_csv(out, index=False)
        log(f"→ {SPLIT_PATH}")
        return
    X_pool, y_pool, X_test, y_test, cfg = load_xy()
    costs = load_yaml("config/costs.yaml")
    log(f"pool {X_pool.shape}, test {X_test.shape}, відтік pool {y_pool.mean():.4f} / test {y_test.mean():.4f}")
    if args.ssl:
        metrics, pseudo, preds = run_ssl_experiments(X_pool, y_pool, X_test, y_test, cfg, costs, log=log)
        metrics.to_csv(rep / "lab4a_ssl_metrics.csv", index=False)
        pseudo.to_csv(rep / "lab4a_pseudo_labels.csv", index=False)
        preds.to_csv(rep / "lab4a_ssl_predictions.csv", index=False)
    if args.al:
        curves, queried = run_active_learning(X_pool, y_pool, X_test, y_test, cfg, costs, log=log)
        curves.to_csv(rep / "lab4a_al_curves.csv", index=False)
        queried.to_csv(rep / "lab4a_al_queries.csv", index=False)
    log("готово")


if __name__ == "__main__":
    main()