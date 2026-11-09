import yaml
from core.pipeline_factory import PipelineSchemaComposer


def verify_idempotency(config_path: str) -> None:
    """Перевіряє, що дворазове збирання створює тотожні за структурою об'єкти."""
    builder_1 = PipelineSchemaComposer(config_path).assemble()
    builder_2 = PipelineSchemaComposer(config_path).assemble()
    assert repr(builder_1) == repr(builder_2), f"Збирання не є ідемпотентним для {config_path}"


def verify_protected_isolation(config_path: str) -> None:
    """Перевіряє відсутність захищених атрибутів у трансформері."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ct = PipelineSchemaComposer(config_path).assemble()
    active_cols = {col for _, _, columns in ct.transformers for col in columns}
    leaked = active_cols & set(cfg.get("protected_attributes", []))
    assert not leaked, f"Витік захищених атрибутів у пайплайн: {leaked}"


if __name__ == "__main__":
    target_configs = [
        "configs/pipeline_linear.yaml",
        "configs/pipeline_tree.yaml",
    ]
    for path in target_configs:
        verify_idempotency(path)
        verify_protected_isolation(path)
        print(f"[{path}]: Усі перевірки верифікаційного протоколу успішно пройдені.")