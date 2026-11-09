import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from core.pipeline_factory import PipelineSchemaComposer, ThresholdBasedImputer

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATHS = [
    ROOT_DIR / "config" / "pipeline_linear.yaml",
    ROOT_DIR / "config" / "pipeline_tree.yaml",
]

SAMPLE_POLICY = [
    {"max_share": 0.10, "strategy": "median"},
    {"max_share": 0.40, "strategy": "median", "add_indicator": True},
    {"max_share": 1.00, "strategy": "constant", "fill_value": 0, "add_indicator": True},
]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def dump_yaml(tmp_path: Path, data: dict, name: str = "temp.yaml") -> Path:
    out = tmp_path / name
    out.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return out


def test_configs_differ_by_model_family_line_only():
    text_a = CONFIG_PATHS[0].read_text(encoding="utf-8").splitlines()
    text_b = CONFIG_PATHS[1].read_text(encoding="utf-8").splitlines()
    diffs = [
        line for line in difflib.unified_diff(text_a, text_b, lineterm="")
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert len(diffs) == 2
    assert all("model_family" in line for line in diffs)


def test_protected_attribute_leak_triggers_exception(tmp_path):
    cfg = load_yaml(CONFIG_PATHS[0])
    cfg["features"]["categorical_ordinal"].append(cfg["protected_attributes"][0])
    invalid_file = dump_yaml(tmp_path, cfg)

    with pytest.raises(ValueError, match="Quality Gate Violation"):
        PipelineSchemaComposer(str(invalid_file))


def test_unsupported_transformer_name_raises_error(tmp_path):
    cfg = load_yaml(CONFIG_PATHS[0])
    fam = cfg["model_family"]
    cfg["transform_policy"][fam]["numeric_continuous"] = ["non_existing_transformer"]
    invalid_file = dump_yaml(tmp_path, cfg)

    with pytest.raises(ValueError, match="Невідомий трансформер"):
        PipelineSchemaComposer(str(invalid_file))


@pytest.mark.parametrize("cfg_path", CONFIG_PATHS)
def test_assemble_returns_unfitted_transformer(cfg_path):
    ct = PipelineSchemaComposer(str(cfg_path)).assemble()
    with pytest.raises(NotFittedError):
        check_is_fitted(ct)


def test_zero_hardcoded_literals_in_source_code():
    code_text = "\n".join(
        (ROOT_DIR / "core" / file_name).read_text(encoding="utf-8")
        for file_name in ("pipeline_factory.py", "feature_transformers.py")
    )
    cfg = load_yaml(CONFIG_PATHS[0])
    literals = set(cfg["transform_policy"].keys()) | set(cfg["protected_attributes"])

    for cols in cfg["features"].values():
        literals |= set(cols)

    detected = {
        item for item in literals
        if re.search(rf"\b{re.escape(str(item))}\b", code_text)
    }
    assert not detected, f"У коді знайдено хардкод літералів з конфігурації: {detected}"


def test_threshold_based_imputer_routing():
    n = 500
    gen = np.random.default_rng(42)

    col_low = gen.normal(10, 2, n)
    col_low[gen.random(n) < 0.05] = np.nan

    col_mid = gen.normal(10, 2, n)
    col_mid[gen.random(n) < 0.25] = np.nan

    df = pd.DataFrame({"feat_low": col_low, "feat_mid": col_mid})
    imputer = ThresholdBasedImputer(SAMPLE_POLICY).fit(df)

    assert imputer.audit_log_["feat_low"]["chosen_strategy"] == "median"
    assert not imputer.audit_log_["feat_low"]["created_indicator"]

    assert imputer.audit_log_["feat_mid"]["chosen_strategy"] == "median"
    assert imputer.audit_log_["feat_mid"]["created_indicator"]

    transformed = imputer.transform(df)
    assert not transformed.isna().any().any()