"""Екстракція ознак із неструктурованих даних (текст і текстове середовище відгуків)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from core.helpers import _BaseColumnTracker, _to_dataframe


class ReviewTextEmbedder(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Екстрактор текстових ознак: TF-IDF з подальшою SVD декомпозицією."""

    def __init__(self, max_features: int = 500, n_components: int = 5, random_state: int = 42):
        self.max_features = max_features
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        df = self._store_input_columns(X)
        first_col = df.columns[0]
        text_series = df[first_col].fillna("").astype(str)

        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features, stop_words="english")
        tfidf_matrix = self.vectorizer_.fit_transform(text_series)

        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        if tfidf_matrix.shape[1] >= self.n_components:
            self.svd_.fit(tfidf_matrix)

        self.stats_ = {"fitted_samples": len(text_series)}
        return self

    def transform(self, X):
        check_is_fitted(self, ("vectorizer_", "svd_"))
        df = _to_dataframe(X, self.feature_names_in_)
        text_series = df[df.columns[0]].fillna("").astype(str)

        empty_mask = text_series.str.strip() == ""
        tfidf_matrix = self.vectorizer_.transform(text_series)

        if tfidf_matrix.shape[1] >= self.n_components:
            svd_features = self.svd_.transform(tfidf_matrix)
        else:
            svd_features = np.zeros((len(text_series), self.n_components))

        res_cols = [f"text_svd_{i}" for i in range(self.n_components)]
        result_df = pd.DataFrame(svd_features, columns=res_cols, index=df.index)

        result_df.loc[empty_mask, :] = np.nan
        self.stats_["transform_empty"] = int(empty_mask.sum())
        return result_df


class TextureFeatureExtractor(_BaseColumnTracker, BaseEstimator, TransformerMixin):
    """Заглушка/Екстрактор геометричних і текстурних ознак фотографій (GLCM)."""

    def __init__(self, levels: int = 32):
        self.levels = levels

    def fit(self, X, y=None):
        self._store_input_columns(X)
        self.stats_ = {"rows": len(X)}
        return self

    def transform(self, X):
        df = _to_dataframe(X, self.feature_names_in_)
        out = pd.DataFrame(index=df.index)

        out["img_contrast"] = np.nan
        out["img_correlation"] = np.nan
        out["img_energy"] = np.nan
        out["img_homogeneity"] = np.nan

        return out


# def link_reviews_to_customers(clients_df: pd.DataFrame, config: dict) -> pd.DataFrame:
#     """Імітація зв'язування відгуків із профілями клієнтів."""
#     records = []
#     for idx, row in clients_df.iterrows():
#         cid = row[config["linkage"]["id_column"]]
#         cnt = min(int(row.get(config["linkage"]["count_column"], 1)), config["linkage"]["max_reviews"])
#
#         for r_idx in range(max(1, cnt)):
#             records.append({
#                 config["linkage"]["id_column"]: cid,
#                 "rating": row.get(config["linkage"]["rating_column"], 3),
#                 "text_len": int(10 + (idx + r_idx) % 50),
#                 "text_svd_0": float(np.sin(idx)),
#                 "text_svd_1": float(np.cos(idx)),
#                 "text_svd_2": 0.0,
#                 "text_svd_3": 0.0,
#                 "text_svd_4": 0.0,
#                 "img_contrast": 0.5,
#                 "img_correlation": 0.8,
#                 "img_energy": 0.3,
#                 "img_homogeneity": 0.9,
#                 "has_image": 1 if r_idx == 0 else 0,
#                 "review_hour_sin": 0.1,
#                 "review_hour_cos": 0.9,
#                 "review_dow_sin": 0.2,
#                 "review_dow_cos": 0.8,
#             })
#     return pd.DataFrame(records)

def link_reviews_to_customers(clients_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Імітація зв'язування відгуків із профілями клієнтів із захистом від NaN."""
    records = []
    id_col = config["linkage"]["id_column"]
    count_col = config["linkage"]["count_column"]
    rating_col = config["linkage"]["rating_column"]
    max_revs = config["linkage"]["max_reviews"]

    for idx, row in clients_df.iterrows():
        cid = row[id_col]

        # Захист від NaN у колонці кількості замовлень
        raw_cnt = row.get(count_col)
        cnt_val = 1 if pd.isna(raw_cnt) else int(raw_cnt)
        cnt = min(max(1, cnt_val), max_revs)

        # Захист від NaN у оцінці задоволеності
        raw_rating = row.get(rating_col)
        rating_val = 3 if pd.isna(raw_rating) else raw_rating

        for r_idx in range(cnt):
            records.append({
                id_col: cid,
                "rating": rating_val,
                "text_len": int(10 + (idx + r_idx) % 50),
                "text_svd_0": float(np.sin(idx)),
                "text_svd_1": float(np.cos(idx)),
                "text_svd_2": 0.0,
                "text_svd_3": 0.0,
                "text_svd_4": 0.0,
                "img_contrast": 0.5,
                "img_correlation": 0.8,
                "img_energy": 0.3,
                "img_homogeneity": 0.9,
                "has_image": 1 if r_idx == 0 else 0,
                "review_hour_sin": 0.1,
                "review_hour_cos": 0.9,
                "review_dow_sin": 0.2,
                "review_dow_cos": 0.8,
            })

    return pd.DataFrame(records)

def run_dfs(clients_df: pd.DataFrame, reviews_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Виконання Deep Feature Synthesis поверх зв'язаних таблиць."""
    import featuretools as ft

    es = ft.EntitySet(id="churn_data")
    es.add_dataframe(
        dataframe_name="clients",
        dataframe=clients_df,
        index=config["linkage"]["id_column"],
    )
    es.add_dataframe(
        dataframe_name="reviews",
        dataframe=reviews_df,
        make_index=True,
        index="review_id",
    )
    es.add_relationship("clients", config["linkage"]["id_column"], "reviews", config["linkage"]["id_column"])

    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="clients",
        max_depth=config["dfs"]["max_depth"],
        agg_primitives=config["dfs"]["agg_primitives"],
        trans_primitives=config["dfs"]["trans_primitives"],
    )
    return feature_matrix