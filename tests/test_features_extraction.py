"""Тестування кастомних кодувальників та екстракторів неструктурованих даних."""
import numpy as np
import pandas as pd
import pytest

from src.feature_constructors import BoundedRatioTransformer, FeatureCrossEncoder, WeightOfEvidenceEncoder
from src.multimodal_extractor import ReviewTextEmbedder


def test_bounded_ratio_zero_division():
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [0.0, 5.0]})
    transformer = BoundedRatioTransformer("ratio", numerator="a", denominator="b", epsilon=1e-5)
    res = transformer.fit_transform(df)

    assert not np.isinf(res["ratio"].iloc[0])
    assert res["ratio"].iloc[1] == pytest.approx(4.0, rel=1e-3)


def test_feature_cross_encoder():
    df = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"]})
    encoder = FeatureCrossEncoder(target_column="segment", source_columns=["cat1", "cat2"])
    res = encoder.fit_transform(df)

    assert res["segment"].tolist() == ["A | X", "B | Y"]


def test_review_text_embedder():
    df = pd.DataFrame({"text": ["Excellent device", "Bad quality item", ""]})
    embedder = ReviewTextEmbedder(max_features=10, n_components=2)
    res = embedder.fit_transform(df)

    assert res.shape == (3, 2)
    assert res.iloc[2].isna().all()