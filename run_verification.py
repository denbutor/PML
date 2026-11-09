import yaml
from src.dynamic_pipeline_builder import DynamicPipelineBuilder


def test_idempotent(config_path):
    """Два послідовні збирання з тією самою конфігурацією дають ідентичну структуру."""
    a = DynamicPipelineBuilder(config_path).build()
    b = DynamicPipelineBuilder(config_path).build()
    assert repr(a) == repr(b), "Збирання не ідемпотентне"


def test_protected_excluded(config_path):
    """Жодна захищена ознака не потрапила до жодної гілки конвеєра."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ct = DynamicPipelineBuilder(config_path).build()
    used = {col for _, _, cols in ct.transformers for col in cols}
    leaked = used & set(cfg.get("protected_attributes", []))
    assert not leaked, f"Захищені ознаки в конвеєрі: {leaked}"


CONFIGS = (
    "config/config_lr.yaml", "config/config_rf.yaml",
    "config/config_lr_fe.yaml", "config/config_rf_fe.yaml",
    "config/generated/config_lr_full.yaml", "config/generated/config_rf_full.yaml",
    "config/generated/config_lr_leaky.yaml", "config/generated/config_rf_leaky.yaml",
    "config/generated/config_lr_hermetic.yaml", "config/generated/config_rf_hermetic.yaml",
)


if __name__ == "__main__":
    for path in CONFIGS:
        test_idempotent(path)
        test_protected_excluded(path)
        print(f"{path}: перевірки пройдено")
