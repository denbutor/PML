"""Інференс поза HTTP: препроцесор (skops) → ONNX Runtime → ймовірність, мітка, невизначеність, OOD.

Використовується сервісом (src/app), реєстром (метрики на test), офлайн-звітом XAI і треком 4A.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.serving.artifacts import load_skops
from src.serving.features import REQUEST_FIELDS, requests_to_frame


def entropy(p) -> np.ndarray:
    """Міра невизначеності ЛР4A: H(p) = −[p·log₂p + (1−p)·log₂(1−p)] ∈ [0, 1]."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def non_ml_rule(row: dict) -> int:
    """Non-ML правило ЛР1: Tenure ≤ 1 міс. АБО скарга минулого місяця → відтік. Пропуск Tenure → умова не виконана."""
    t = row.get("Tenure")
    return int((t is not None and not (isinstance(t, float) and np.isnan(t)) and t <= 1) or row.get("Complain") == 1)


@dataclass
class Predictor:
    meta: dict
    preprocessor: Any
    session: Any
    export_pipe: Any = None
    background: Optional[np.ndarray] = None
    ood: Any = None
    version: str = "local"
    run_id: Optional[str] = None
    extras: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dir(cls, model_dir: str | Path, onnx_bytes: bytes | None = None, version: str = "local",
                 run_id: str | None = None, with_explain: bool = True) -> "Predictor":
        import onnxruntime as ort

        d = Path(model_dir)
        meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
        onnx_bytes = onnx_bytes or (d / "pipeline.onnx").read_bytes()
        sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        ood = load_skops(d / "ood" / "ood.skops") if (d / "ood" / "ood.skops").exists() else None
        return cls(meta=meta, preprocessor=load_skops(d / "preprocessor.skops"), session=sess,
                   export_pipe=load_skops(d / "pipeline.skops") if with_explain else None,
                   background=np.load(d / "shap_background.npy") if with_explain else None,
                   ood=ood, version=version, run_id=run_id)

    # ------------------------------------------------------------------ #
    def features(self, records) -> np.ndarray:
        """Поля запиту → вхід ONNX (float32, у порядку meta['features'])."""
        return self.preprocessor.transform(requests_to_frame(records)).astype(np.float32)

    def proba_from_features(self, Z: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.meta["onnx_inputs"][0]["name"]: Z.astype(np.float32)})[1][:, 1]

    def proba_frame(self, df: pd.DataFrame) -> np.ndarray:
        return self.proba_from_features(self.features(df[REQUEST_FIELDS].to_dict("records")))

    def ood_log_density(self, Z: np.ndarray) -> Optional[np.ndarray]:
        if self.ood is None:
            return None
        return self.ood["kde"].score_samples(self.ood["projection"].transform(Z))

    def predict(self, records: list[dict]) -> list[dict]:
        Z = self.features(records)
        p = self.proba_from_features(Z)
        dens = self.ood_log_density(Z)
        thr = float(self.meta["threshold"])
        ood_thr = self.meta.get("ood", {}).get("threshold")
        out = []
        for i, rec in enumerate(records):
            fallback = bool(dens is not None and ood_thr is not None and dens[i] < ood_thr)
            model_label = int(p[i] >= thr)
            label = non_ml_rule(rec) if fallback else model_label
            out.append({"score": float(p[i]), "label": label, "uncertainty": float(entropy(p[i])),
                        "model_label": model_label, "fallback_applied": fallback,
                        "ood_log_density": None if dens is None else float(dens[i])})
        return out
