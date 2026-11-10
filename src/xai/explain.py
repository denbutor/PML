"""Локальні пояснення одного запиту (для POST /v1/explain).

* Пояснюється **skops-артефакт із того самого run, що й ONNX** (models/pipeline.skops = Pipeline([RF])):
  модель — ``pipe[-1]``, дані — ``pipe[:-1].transform(x)`` (тут тотожність: вхід ONNX уже готовий).
* **SHAP:** ``TreeExplainer(model, data=фон, feature_perturbation="interventional", model_output="probability")`` —
  інтервенційна функція цінності з фоном models/shap_background.npy (300 рядків train у просторі входу ONNX);
  значення — для класу 1 (відтік), у одиницях імовірності; сума значень + base_value = P(відтік).
* **LIME:** ``LimeTabularExplainer(training_data=фон)``, 1500 збурень, функція прогнозу — увесь експортований конвеєр.
* **Збіг:** частка спільних ознак у топ-5 за |внеском| і рангова кореляція Спірмена |SHAP| vs |LIME| за всіма ознаками.
  Агрегація до вихідних ознак не потрібна: one-hot у моделі лише одна колонка (MaritalStatus_Single),
  векторизованих блоків (TF-IDF/GLCM) у моделі немає.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

FRIENDLY = {
    "Tenure": "Стаж, міс.", "Complain": "Скарга", "MaritalStatus_Single": "Неодружений(-а)",
    "OrdersPerTenureMonth": "Замовлень на міс. стажу", "TenureStage": "Етап клієнта", "Segment": "Сегмент (WoE)",
    "NATURAL_LOGARITHM(Tenure)": "log(стаж)",
}


def friendly(name: str) -> str:
    return FRIENDLY.get(name, name.replace("__missing", " — пропуск"))


class LocalExplainer:
    def __init__(self, export_pipe, background: np.ndarray, feature_names: list[str], lime_samples: int = 1500,
                 seed: int = 42):
        import shap
        from lime.lime_tabular import LimeTabularExplainer

        self.pipe = export_pipe
        self.model = export_pipe[-1]
        self.names = list(feature_names)
        self.bg = np.asarray(background, dtype=float)
        self.shap = shap.TreeExplainer(self.model, data=self.bg, feature_perturbation="interventional",
                                       model_output="probability")
        self.lime = LimeTabularExplainer(self.bg, feature_names=self.names, class_names=["залишиться", "відтік"],
                                         mode="classification", discretize_continuous=True, random_state=seed)
        self.lime_samples = lime_samples

    def shap_values(self, Z: np.ndarray) -> np.ndarray:
        sv = self.shap.shap_values(np.asarray(Z, dtype=float), check_additivity=False)
        sv = np.asarray(sv)
        if sv.ndim == 3:            # (n, features, classes)
            sv = sv[..., 1]
        return sv

    def base_value(self) -> float:
        ev = np.atleast_1d(self.shap.expected_value)
        return float(ev[1] if len(ev) > 1 else ev[0])

    def explain(self, z: np.ndarray) -> dict:
        z = np.asarray(z, dtype=float).ravel()
        sv = self.shap_values(z[None, :])[0]
        exp = self.lime.explain_instance(z, lambda A: self.pipe.predict_proba(np.asarray(A, dtype=np.float32)),
                                         num_features=len(self.names), num_samples=self.lime_samples, labels=(1,))
        lime_map = dict(exp.as_map()[1])                       # індекс ознаки → вага
        lime_vec = np.array([lime_map.get(i, 0.0) for i in range(len(self.names))])
        top_s = set(np.argsort(-np.abs(sv))[:5]); top_l = set(np.argsort(-np.abs(lime_vec))[:5])
        rho = spearmanr(np.abs(sv), np.abs(lime_vec)).statistic
        lime_list = sorted([(exp.domain_mapper.discretized_feature_names[i] if exp.domain_mapper.discretized_feature_names
                             else self.names[i], float(w)) for i, w in lime_map.items()], key=lambda t: -abs(t[1]))
        return {"shap": {"values": [float(v) for v in sv], "base_value": self.base_value(), "data": [float(v) for v in z],
                         "feature_names": [friendly(n) for n in self.names]},
                "lime": lime_list,
                "agreement": {"top5_overlap": len(top_s & top_l) / 5.0, "spearman": float(0.0 if np.isnan(rho) else rho)}}
