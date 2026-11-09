from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_data_config, load_dataset, read_csv, read_processed, split_xy
from src.extraction import (ImageExtractor, TextExtractor, load_multimodal_config, review_features, review_texts,
                            run_dfs)

LAB2_LEAKY_AUC_LR = 0.8599   # docs/lab2_report.md, §4


def lab2_split():
    dc = load_data_config()
    df = load_dataset(dc)
    X, y = split_xy(df, dc)
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.3, stratify=y, random_state=42)
    return dc, df, X, y, idx_tr, idx_te


def leaky_lr_proba():
    """Експеримент 4 ЛР2 для LR: TF-IDF/SVD, WoE, Filter і SFS навчені на ВСЬОМУ наборі (витік)."""
    from sklearn.linear_model import LogisticRegression

    from src.pipeline import ExtendedPipelineBuilder
    from src.selection import load_selection_config, make_filter, make_sfs

    warnings.filterwarnings("ignore")
    dc, df, X, y, idx_tr, idx_te = lab2_split()
    mm, sel_cfg = load_multimodal_config(), load_selection_config()
    reviews = read_csv("data/processed/reviews_linked.csv")
    text_ex = TextExtractor(**mm["extraction"]["text"]).fit(review_texts(reviews).to_frame())    # ВИТІК №1
    fm, _ = run_dfs(df, review_features(reviews, text_ex, ImageExtractor(levels=mm["extraction"]["image"]["levels"]), mm), mm)
    X4 = X.join(fm, on=dc["id_column"])
    prep = ExtendedPipelineBuilder("config/generated/config_lr_leaky.yaml").build()
    prep.fit(X4, y)                                                                               # ВИТІК №2
    Z = prep.transform(X4)
    flt = make_filter(sel_cfg).fit(Z, y)                                                          # ВИТІК №3
    Zf = flt.transform(Z)
    sfs = make_sfs(sel_cfg).fit(Zf, y.to_numpy())                                                 # ВИТІК №3
    Zs = sfs.transform(Zf)
    model = LogisticRegression(max_iter=1000).fit(Zs[idx_tr], y.iloc[idx_tr])
    return y.iloc[idx_te].to_numpy(), model.predict_proba(Zs[idx_te])[:, 1], df.iloc[idx_te][dc["id_column"]].to_numpy()


def lab3_full_table() -> pd.DataFrame:
    """Усі 5 074 клієнти у форматі ЛР3 (train.csv + holdout.csv через DVC)."""
    return pd.concat([read_processed("train"), read_processed("holdout")], ignore_index=True)
