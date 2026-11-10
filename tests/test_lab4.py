import numpy as np
import pandas as pd
import pytest

from src.semi_supervised import (UNLABELED, entropy, least_confidence, make_split, margin, sample_labeled,
                                 select_queries)


def test_entropy_bounds_and_peak():
    p = np.array([0.0, 0.5, 1.0, 0.2, 0.8])
    h = entropy(p)
    assert h[1] == pytest.approx(1.0)
    assert h[0] == pytest.approx(0.0, abs=1e-9) and h[2] == pytest.approx(0.0, abs=1e-9)
    assert h[3] == pytest.approx(h[4])


def test_binary_uncertainty_measures_rank_identically():
    p = np.random.default_rng(0).uniform(size=500)
    order = [np.argsort(-f(p), kind="stable") for f in (entropy, margin, least_confidence)]
    assert all((o == order[0]).all() for o in order[1:])


@pytest.mark.parametrize("strategy", ["random", "entropy", "entropy_diversity"])
def test_select_queries_returns_unique_batch(strategy):
    rng = np.random.default_rng(1)
    proba = rng.uniform(size=300)
    Z = rng.normal(size=(300, 5))
    idx = select_queries(strategy, proba, Z, batch=20, rng=np.random.default_rng(0), n_candidates=100)
    assert len(idx) == 20 and len(set(idx.tolist())) == 20 and idx.max() < 300


def test_entropy_strategy_picks_most_uncertain():
    proba = np.array([0.01, 0.49, 0.99, 0.55, 0.9])
    idx = select_queries("entropy", proba, np.zeros((5, 1)), batch=2, rng=np.random.default_rng(0))
    assert set(idx.tolist()) == {1, 3}


def test_split_and_labeled_sample_stratified():
    df = pd.DataFrame({"CustomerID": np.arange(1000), "Churn": (np.arange(1000) % 6 == 0).astype(int)})
    cfg = {"split": {"test_share": 0.25, "seed": 42}}
    roles = make_split(df, cfg)
    assert (roles.role == "test").sum() == 250
    test_ids = set(roles.loc[roles.role == "test", "CustomerID"])
    pool = df[~df.CustomerID.isin(test_ids)]
    assert not test_ids & set(pool.CustomerID)            # пул не перетинається з test
    mask = sample_labeled(pool.Churn.reset_index(drop=True), 0.1, seed=0)
    assert mask.sum() == 75 and abs(pool.Churn.to_numpy()[mask].mean() - pool.Churn.mean()) < 0.02


def test_unlabeled_marker():
    assert UNLABELED == -1
