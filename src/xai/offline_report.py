"""ЛР5, Етап 5 — офлайн-звіт пояснень (те, що потребує моделі й усієї вибірки).

    python -m src.xai.offline_report      # → reports/xai/*.png, reports/xai/summary.json; лог у run @champion

* SHAP summary (beeswarm) на test — TreeExplainer, інтервенційна функція цінності з фоном models/shap_background.npy;
* для двох найважливіших ознак — PDP + ICE (kind="both") або **ALE**, якщо ознака має |ρ Спірмена| > 0.7 з будь-якою іншою;
* сурогатне дерево (глибина ≤ 4) на **ймовірностях моделі** train; fidelity — R² до ймовірностей моделі на test;
* стабільність: 30 об'єктів test (seed 42) × 20 збурень у вихідному просторі ознак (гаусів шум 3% std лише для
  неперервних полів, обрізання до меж Field, кожен рядок проходить валідацію PredictionRequest) → схожість SHAP.
"""
from __future__ import annotations

import json
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "xai"
CONTINUOUS = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
              "DaySinceLastOrder", "CashbackAmount"]
INK, MUTED, BLUE, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#fcfcfb"


def champion_ids():
    try:
        import mlflow
        from mlflow import MlflowClient

        from src.export.register_model import NAME, TRACKING_URI
        mlflow.set_tracking_uri(TRACKING_URI)
        mv = MlflowClient().get_model_version_by_alias(NAME, "champion")
        return str(mv.version), mv.run_id
    except Exception:
        return "local", None


def field_bounds():
    from src.app.schemas import PredictionRequest
    out = {}
    for name, f in PredictionRequest.model_fields.items():
        lo = next((m.ge for m in f.metadata if hasattr(m, "ge")), None)
        hi = next((m.le for m in f.metadata if hasattr(m, "le")), None)
        out[name] = (lo, hi)
    return out


def ale_1d(predict, Z, j, bins=20):
    """Accumulated Local Effects (Apley & Zhu, 2020) для ознаки j: середні локальні різниці прогнозу в квантильних бінах."""
    x = Z[:, j]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return edges, np.zeros(len(edges))
    idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)
    eff = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        m = idx == k
        if not m.any():
            continue
        lo, hi = Z[m].copy(), Z[m].copy()
        lo[:, j], hi[:, j] = edges[k], edges[k + 1]
        eff[k] = np.mean(predict(hi) - predict(lo))
    acc = np.concatenate([[0.0], np.cumsum(eff)])
    counts = np.bincount(idx, minlength=len(edges) - 1)
    centre = np.sum(((acc[:-1] + acc[1:]) / 2) * counts) / counts.sum()
    return edges, acc - centre


def main():
    import shap
    from scipy.stats import spearmanr
    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.metrics import r2_score
    from sklearn.tree import DecisionTreeRegressor, export_text

    from src.app.schemas import PredictionRequest
    from src.serving.data import load_lab5
    from src.serving.features import REQUEST_FIELDS
    from src.serving.predictor import Predictor
    from src.xai.explain import LocalExplainer, friendly

    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    version, run_id = champion_ids()
    pred = Predictor.from_dir(PROJECT_ROOT / "models")
    names = pred.meta["features"]
    fnames = [friendly(n) for n in names]
    data = load_lab5()
    Zte = pred.features(data["test"][REQUEST_FIELDS].to_dict("records")).astype(float)
    Ztr = pred.features(data["train"][REQUEST_FIELDS].to_dict("records")).astype(float)
    model = pred.export_pipe[-1]
    proba = lambda A: model.predict_proba(np.asarray(A, dtype=np.float32))[:, 1]
    ex = LocalExplainer(pred.export_pipe, pred.background, names)

    # 1. SHAP beeswarm
    sv = ex.shap_values(Zte)
    expl = shap.Explanation(values=sv, base_values=np.full(len(Zte), ex.base_value()), data=Zte, feature_names=fnames)
    plt.figure()
    shap.plots.beeswarm(expl, max_display=15, show=False)
    plt.title("SHAP: внески ознак у P(відтік) на test (одиниці ймовірності)", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT / "shap_beeswarm.png", dpi=130, bbox_inches="tight"); plt.close("all")
    mean_abs = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)

    # 2. PDP+ICE або ALE для двох найважливіших
    rho = spearmanr(Zte).statistic
    rho = np.nan_to_num(np.asarray(rho), nan=0.0)
    choices = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, feat in zip(axes, mean_abs.index[:2]):
        j = names.index(feat)
        others = np.abs(np.delete(rho[j], j))
        max_rho = float(others.max())
        partner = names[[k for k in range(len(names)) if k != j][int(others.argmax())]]
        if max_rho > 0.7:
            edges, ale = ale_1d(proba, Zte, j)
            ax.plot(edges, ale, color=BLUE, linewidth=2, marker="o", markersize=4)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_ylabel("ALE: зміна P(відтік)")
            method = "ALE"
        else:
            PartialDependenceDisplay.from_estimator(model, Zte.astype(np.float32), [j], kind="both", ax=ax,
                                                    subsample=200, random_state=42, feature_names=fnames)
            method = "PDP+ICE"
        ax.set_title(f"{method}: {friendly(feat)}  (max |ρ| = {max_rho:.2f} з «{friendly(partner)}»)", fontsize=9)
        ax.set_xlabel(friendly(feat))
        choices.append({"feature": feat, "method": method, "max_abs_spearman": max_rho, "most_correlated_with": partner})
    fig.tight_layout(); fig.savefig(OUT / "pdp_ale_top2.png", dpi=130, bbox_inches="tight"); plt.close("all")

    # 3. Сурогатне дерево на ймовірностях моделі
    sur = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Ztr, proba(Ztr))
    fidelity = float(r2_score(proba(Zte), sur.predict(Zte)))
    (OUT / "surrogate_tree.txt").write_text(export_text(sur, feature_names=fnames, decimals=3), encoding="utf-8")

    # 4. Стабільність пояснень
    rng = np.random.default_rng(42)
    bounds = field_bounds()
    train_std = data["train"][CONTINUOUS].std()
    test_rows = data["test"][REQUEST_FIELDS].reset_index(drop=True)
    picks = rng.choice(len(test_rows), size=30, replace=False)
    sims, overlaps, dps, invalid = [], [], [], 0
    from src.observability.label_round import _normalize
    for i in picks:
        base = _normalize({k: (None if pd.isna(v) else v) for k, v in test_rows.iloc[i].to_dict().items()})
        z0 = pred.features([base]); s0 = ex.shap_values(z0)[0]; p0 = proba(z0)[0]
        top0 = set(np.argsort(-np.abs(s0))[:5])
        for _ in range(20):
            row = dict(base)
            for c in CONTINUOUS:
                if row[c] is None:
                    continue
                lo, hi = bounds[c]
                row[c] = float(np.clip(row[c] + rng.normal(0, 0.03 * train_std[c]), lo, hi))
            if row["CouponUsed"] is not None and row["OrderCount"] is not None:
                row["CouponUsed"] = min(row["CouponUsed"], row["OrderCount"])
            try:
                PredictionRequest(**row)
            except Exception:
                invalid += 1
                continue
            z = pred.features([row]); s = ex.shap_values(z)[0]
            sims.append(float(np.nan_to_num(spearmanr(s0, s).statistic, nan=1.0)))
            overlaps.append(len(top0 & set(np.argsort(-np.abs(s))[:5])) / 5)
            dps.append(abs(float(proba(z)[0]) - p0))
    stability = {"n_objects": 30, "n_perturbations": 20, "noise": "N(0, (0.03·std_train)²), лише неперервні поля, обрізання до Field",
                 "invalid_skipped": invalid, "shap_spearman_mean": float(np.mean(sims)), "shap_spearman_p10": float(np.percentile(sims, 10)),
                 "top5_overlap_mean": float(np.mean(overlaps)), "abs_delta_p_mean": float(np.mean(dps)),
                 "abs_delta_p_p90": float(np.percentile(dps, 90))}

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(sims, bins=20, color=BLUE, edgecolor=SURFACE)
    ax.set_xlabel("ρ Спірмена: SHAP збуреного vs вихідного об'єкта"); ax.set_ylabel("збурень")
    ax.set_title(f"Стабільність пояснень (30 × 20), середнє ρ = {np.mean(sims):.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "stability.png", dpi=130, bbox_inches="tight"); plt.close("all")

    summary = {"model_version": version, "run_id": run_id, "explainer": "shap.TreeExplainer",
               "value_function": "interventional (feature_perturbation='interventional', фон models/shap_background.npy, 300 рядків train)",
               "model_output": "probability (клас 1 — відтік)", "base_value": ex.base_value(),
               "mean_abs_shap": {friendly(k): float(v) for k, v in mean_abs.items()},
               "pdp_or_ale": choices, "surrogate": {"max_depth": 4, "fit_on": "ймовірності моделі на train",
                                                    "fidelity_r2_test": fidelity},
               "stability": stability}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if run_id:
        import mlflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(OUT), artifact_path="reports/xai")
            mlflow.log_metrics({"xai_surrogate_fidelity_r2": fidelity, "xai_stability_spearman": stability["shap_spearman_mean"]})
    print(json.dumps({k: summary[k] for k in ("model_version", "pdp_or_ale", "surrogate", "stability")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
