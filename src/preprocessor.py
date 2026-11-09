"""Точка входу «препроцесор» за структурою репозиторію ТЗ ЛР3 (``src/preprocessor.py``).

Реалізація не дублюється: клас ЛР1 живе в ``src/dynamic_pipeline_builder.py``, розширення ЛР2 — у
``src/pipeline.py``. Тут лише реекспорт, щоб імпорт ``from src.preprocessor import ...`` працював.
"""
from src.dynamic_pipeline_builder import AdaptiveImputer, DynamicPipelineBuilder, validate_config, validate_schema
from src.pipeline import ExtendedPipelineBuilder

__all__ = ["AdaptiveImputer", "DynamicPipelineBuilder", "ExtendedPipelineBuilder", "validate_config", "validate_schema"]
