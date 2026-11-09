"""Автотести для перевірки робочих контракту конфігуратора препроцесингу."""
from pathlib import Path
import pytest
import yaml

from src.pipeline_factory import ConfigurablePipelineFactory, SmartMissingImputer

ROOT_DIR = Path(__file__).resolve().parents[1]
LINEAR_CONFIG = ROOT_DIR / "config" / "config_linear.yaml"


def test_missing_imputer_routing():
    import pandas as pd
    import numpy as np

    data = pd.DataFrame({
        "low_missing": [1.0, 2.0, np.nan] + [1.0] * 97,
        "high_missing": [np.nan] * 50 + [2.0] * 50,
    })
    policy = [
        {"max_share": 0.10, "strategy": "median"},
        {"max_share": 1.00, "strategy": "constant", "fill_value": 0, "add_indicator": True},
    ]

    imputer = SmartMissingImputer(policy_rules=policy).fit(data)
    transformed = imputer.transform(data)

    assert "high_missing__missing" in transformed.columns
    assert "low_missing__missing" not in transformed.columns


def test_factory_builds_valid_pipeline():
    factory = ConfigurablePipelineFactory(str(LINEAR_CONFIG))
    ct = factory.build()
    assert ct is not None