"""Перевірки ЛР3: герметичність конвеєра, втрати, поріг, текстовий блок у фолдах, DVC.

Запуск: python -m pytest -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


# --- втрати і поріг ------------------------------------------------------------------ #
def test_total_cost_counts_each_cell():
    from src.evaluation import total_cost
    cm = {"C00": 0, "C01": 50, "C10": 480, "C11": 362}
    y = np.array([0, 0, 1, 1])
    yhat = np.array([0, 1, 0, 1])          # TN, FP, FN, TP
    assert total_cost(y, yhat, cm) == 0 + 50 + 480 + 362


def test_optimal_threshold_for_calibrated_scores_near_theory():
    """Для каліброваних імовірностей оптимальний поріг ≈ w0 / (w0 + w1)."""
    from src.evaluation import optimal_threshold
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)     # ідеальне калібрування
    cm = {"C00": 0, "C01": 50, "C10": 480, "C11": 362}
    t = optimal_threshold(y, p, cm)
    assert abs(t - 50 / (50 + 118)) <= 0.03


def test_costs_config_consistent():
    c = yaml.safe_load((ROOT / "config" / "costs.yaml").read_text(encoding="utf-8"))
    m, a, w = c["matrix"], c["assumptions"], c["class_weight"]
    clv = a["monthly_margin"] * a["retained_months"]
    assert m["C01"] == a["offer_cost"] and m["C10"] == clv
    assert m["C11"] == pytest.approx(a["offer_cost"] + (1 - a["offer_success_rate"]) * clv)
    assert w[0] == m["C01"] - m["C00"] and w[1] == pytest.approx(m["C10"] - m["C11"])


# --- текстовий блок ------------------------------------------------------------------ #
def test_review_text_svd_fits_only_on_given_rows():
    from src.extraction import REVIEW_SEP, ReviewTextSVD
    train = pd.DataFrame({"t": [REVIEW_SEP.join(["great phone fast", "battery ok"]), "bad case broke",
                                REVIEW_SEP.join(["love it", "great value"]), "terrible support", "nice screen bright"]})
    test = pd.DataFrame({"t": ["zzzunseen wordsxx", ""]})
    ex = ReviewTextSVD(max_features=50, n_components=2).fit(train)
    vocab = set(ex.extractor_.vectorizer_.vocabulary_)
    assert "zzzunseen" not in vocab                       # словник — лише з train
    out = ex.transform(test)
    assert out.shape == (2, 4) and np.isfinite(out.to_numpy()).all()   # невідомі слова / порожньо → 0, без NaN


# --- герметичний конвеєр -------------------------------------------------------------- #
def _hc():
    from src.prepare_data import load_hermetic_config
    return load_hermetic_config()


@pytest.mark.parametrize("family,model", [("lr", "LR"), ("rf", "RF")])
def test_hermetic_pipeline_structure(family, model):
    from imblearn.pipeline import Pipeline as ImbPipeline
    from src.evaluation import build_hermetic_pipeline
    path = ROOT / "config" / "generated" / f"config_{family}_hermetic.yaml"
    if not path.exists():
        pytest.skip("герметичні конфігурації ще не згенеровано")
    pipe = build_hermetic_pipeline(path, model, _hc())
    names = [n for n, _ in pipe.steps]
    assert isinstance(pipe, ImbPipeline)
    # відбір і ресемплер — ланки конвеєра, після препроцесора, перед моделлю
    assert names.index("columns") < names.index("filter") < names.index("sfs") < names.index("resample") < names.index("model")


def test_smote_not_applied_at_predict():
    """SMOTE змінює вибірку лише у fit; predict повертає рівно стільки рядків, скільки подали."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)); y = (rng.uniform(size=300) < 0.15).astype(int)
    pipe = ImbPipeline([("s", SMOTE(random_state=0)), ("m", LogisticRegression())]).fit(X, y)
    assert pipe.predict_proba(X[:37]).shape == (37, 2)


def test_dvc_pointer_matches_content():
    import hashlib
    from src.data import read_bytes
    ptr = ROOT / "data" / "raw" / "E Commerce Dataset.xlsx.dvc"
    if not ptr.exists():
        pytest.skip("DVC-покажчик відсутній")
    md5 = yaml.safe_load(ptr.read_text(encoding="utf-8"))["outs"][0]["md5"]
    try:
        content = read_bytes("data/raw/E Commerce Dataset.xlsx")
    except Exception:
        pytest.skip("дані не завантажено (dvc pull)")
    assert hashlib.md5(content).hexdigest() == md5
