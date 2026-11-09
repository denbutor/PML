import yaml
from core.pipeline_factory import ConfigurablePipelineFactory


def check_pipeline_idempotency(config_file_path: str) -> None:
    """Перевіряє, що дворазова збірка конвеєра з однакового конфігу дає ідентичний результат."""
    builder_a = ConfigurablePipelineFactory(config_file_path).build()
    builder_b = ConfigurablePipelineFactory(config_file_path).build()
    assert repr(builder_a) == repr(builder_b), f"Помилка ідемпотентності для {config_file_path}"


def check_protected_attributes_isolation(config_file_path: str) -> None:
    """Перевіряє відсутність захищених атрибутів у трансформерах конвеєра."""
    with open(config_file_path, encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    pipeline = ConfigurablePipelineFactory(config_file_path).build()
    extracted_features = {
        col for _, _, columns in pipeline.transformers for col in columns
    }
    protected_set = set(raw_config.get("protected_attributes", []))
    leakage = extracted_features & protected_set
    assert not leakage, f"Захищені атрибути виявлено у конвеєрі {config_file_path}: {leakage}"


CONFIG_TARGETS = (
    "config/config_linear.yaml",
    "config/config_tree.yaml",
    "config/config_linear_engineered.yaml",
    "config/config_tree_engineered.yaml",
    "config/generated/config_linear_full.yaml",
    "config/generated/config_tree_full.yaml",
    "config/generated/config_linear_leaky.yaml",
    "config/generated/config_tree_leaky.yaml",
)

if __name__ == "__main__":
    for config_path in CONFIG_TARGETS:
        check_pipeline_idempotency(config_path)
        check_protected_attributes_isolation(config_path)
        print(f"[OK] Конфігурація пройшла перевірку: {config_path}")