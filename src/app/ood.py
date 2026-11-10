"""ЛР5, Етап 4 — OOD-детектор (KDE) і fallback на Non-ML правило.

    python -m src.app.ood        # навчання → models/ood/ood.skops, models/model_meta.json["ood"], reports/ood_report.json

* Простір — вхід ONNX (15 ознак моделі), як вимагає ТЗ: ``export_pipe[:-1]`` тут тотожність.
* Проєкція: StandardScaler → PCA(whiten) з мінімальною кількістю компонент, що пояснює ≥ 90% дисперсії (≤ 10, правило ТЗ; розмірність не обирається за log-likelihood) —
  KDE деградує у високій розмірності.
* KDE (gaussian) на **80% train**; ширина вікна — GridSearchCV за log-likelihood, KFold(5, shuffle, rs=42);
  якщо оптимум на краю сітки — сітку розширюємо.
* Поріг — **1-й перцентиль** log-густини на **решті 20% train** (не на точках, де вчився KDE).
* Перевірка: частка fallback на test (хибні спрацювання на типових даних) і частка виявлених штучних OOD
  (проєкція test, зсунута на 3σ уздовж PC1 / PC1+PC2).
"""
from __future__ import annotations

import json
import warnings

import numpy as np

from src.data import PROJECT_ROOT

OOD_DIR = PROJECT_ROOT / "models" / "ood"


def fit_projection(Z, max_components: int = 10, target_evr: float = 0.90, seed: int = 42):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    evr = np.cumsum(PCA(random_state=seed).fit(StandardScaler().fit_transform(Z)).explained_variance_ratio_)
    k = int(min(max_components, np.searchsorted(evr, target_evr) + 1))
    proj = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=k, whiten=True, random_state=seed))]).fit(Z)
    return proj, k, float(evr[k - 1])


def fit_kde(P, seed: int = 42):
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neighbors import KernelDensity

    grid = np.logspace(-1.3, 0.7, 25)
    for _ in range(4):
        gs = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": grid},
                          cv=KFold(5, shuffle=True, random_state=seed)).fit(P)
        bw = gs.best_params_["bandwidth"]
        if bw == grid[0]:
            grid = np.logspace(np.log10(grid[0]) - 1, np.log10(grid[0]), 25)
        elif bw == grid[-1]:
            grid = np.logspace(np.log10(grid[-1]), np.log10(grid[-1]) + 1, 25)
        else:
            break
    return gs.best_estimator_, float(bw)


def main():
    import skops.io as sio
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from src.evaluation import load_yaml, total_cost
    from src.serving.data import load_lab5
    from src.serving.predictor import Predictor, non_ml_rule
    from src.serving.features import REQUEST_FIELDS

    warnings.filterwarnings("ignore")
    data = load_lab5()
    pred = Predictor.from_dir(PROJECT_ROOT / "models", with_explain=False)
    Z = pred.features(data["train"][REQUEST_FIELDS].to_dict("records"))
    Z_fit, Z_cal = train_test_split(Z, test_size=0.2, random_state=42)
    proj, k, evr = fit_projection(Z_fit)
    kde, bw = fit_kde(proj.transform(Z_fit))
    thr = float(np.percentile(kde.score_samples(proj.transform(Z_cal)), 1))

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    sio.dump({"projection": proj, "kde": kde}, OOD_DIR / "ood.skops")
    meta_path = PROJECT_ROOT / "models" / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["tag"] = "v2"
    meta["ood"] = {"method": "StandardScaler → PCA(whiten) → KDE(gaussian)", "space": "вхід ONNX (15 ознак)",
                   "n_components": k, "explained_variance": evr, "bandwidth": bw, "threshold": thr,
                   "threshold_rule": "1-й перцентиль log-густини на відкладених 20% train", "fit_rows": int(len(Z_fit)),
                   "calibration_rows": int(len(Z_cal)), "fallback": meta["non_ml_rule"]}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- перевірка на test ---
    test = data["test"]
    recs = test[REQUEST_FIELDS].to_dict("records")
    Zt = pred.features(recs)
    Pt = proj.transform(Zt)
    dens = kde.score_samples(Pt)
    fb = dens < thr
    shifted = {}
    for name, axes in {"PC1 +3σ": [0], "PC1+PC2 +3σ": [0, 1]}.items():
        S = Pt.copy()
        S[:, axes] += 3.0
        shifted[name] = float((kde.score_samples(S) < thr).mean())
    y = test[data["target"]].to_numpy()
    p = pred.proba_from_features(Zt)
    mthr = meta["threshold"]
    model_lab = (p >= mthr).astype(int)
    rule_lab = np.array([non_ml_rule(r) for r in recs])
    system_lab = np.where(fb, rule_lab, model_lab)
    cm = load_yaml("config/costs.yaml")["matrix"]

    def q(yy, lab, score=None):
        if len(yy) == 0:
            return None
        out = {"n": int(len(yy)), "positives": int(yy.sum()), "f1": float(f1_score(yy, lab, zero_division=0)),
               "cost_per_customer": float(total_cost(yy, lab, cm) / len(yy))}
        if score is not None and len(np.unique(yy)) == 2:
            out["roc_auc"] = float(roc_auc_score(yy, score))
        return out

    report = {"ood": meta["ood"],
              "fallback_share_test": float(fb.mean()), "n_fallback_test": int(fb.sum()),
              "synthetic_ood_detected": shifted,
              "model_all_test": q(y, model_lab, p),
              "model_without_fallback": q(y[~fb], model_lab[~fb], p[~fb]),
              "system_model_plus_rule": q(y, system_lab),
              "ood_subset": {"model": q(y[fb], model_lab[fb]), "rule": q(y[fb], rule_lab[fb]),
                             "note": "лише кілька об'єктів — порівняння описове"}}
    (PROJECT_ROOT / "reports" / "ood_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("fallback_share_test", "n_fallback_test", "synthetic_ood_detected")}, ensure_ascii=False))
    print("PCA k =", k, "EVR =", round(evr, 3), "| bandwidth =", round(bw, 4), "| threshold =", round(thr, 3))


if __name__ == "__main__":
    main()
