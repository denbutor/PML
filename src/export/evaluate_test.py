"""ЛР5, Етапи 6–7 — фінальна оцінка @champion на test (holdout ЛР3) для Model Card і слайдів.

    python -m src.export.evaluate_test        # → reports/test_evaluation.json (з run_id і model_version)

* якість **моделі** і **системи** (модель + Non-ML правило для OOD-запитів) — рішення, які отримує користувач;
* DPR / EOR за групами CityTier і Gender (атрибути з test лише для аудиту, у модель не подаються) з 95% бутстреп-інтервалами
  (1000 повторень); група з < 30 об'єктами або < 10 позитивними — «недостатньо даних», без інтервалу;
* проксі захищеної ознаки (Cramér's V категорій і ε² числових ознак моделі з CityTier);
* чесний ROI: базова лінія — Non-ML правило ЛР1 на тому самому test; три сценарії припущень матриці втрат.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT

N_BOOT = 1000
MIN_N, MIN_POS = 30, 10


def costs_from(a: dict) -> dict:
    clv = a["monthly_margin"] * a["retained_months"]
    return {"C00": 0.0, "C01": float(a["offer_cost"]), "C10": float(clv),
            "C11": float(a["offer_cost"] + (1 - a["offer_success_rate"]) * clv)}


# Сценарії названо за ЕФЕКТОМ відносно Non-ML правила (а не за «добрими / поганими» бізнес-умовами):
# правило позначає ~44% клієнтів, тож що дешевші й успішніші пропозиції, то менше коштує його надлишок
# хибних тривог — і то МЕНША перевага моделі. Центральний = config/costs.yaml; межі свідомо широкі.
SCENARIOS = {
    "песимістичний": {"offer_cost": 35, "monthly_margin": 50, "retained_months": 15, "offer_success_rate": 0.50},
    "центральний": {"offer_cost": 50, "monthly_margin": 40, "retained_months": 12, "offer_success_rate": 0.35},
    "оптимістичний": {"offer_cost": 70, "monthly_margin": 30, "retained_months": 9, "offer_success_rate": 0.20},
}


def group_fairness(y, yhat, groups, privileged, rng) -> list:
    from src.evaluation import fairness_report

    base = fairness_report(y, yhat, groups, privileged=privileged)
    rows = []
    for _, r in base.iterrows():
        g = r["group"]
        row = {"group": str(g), "n": int(r["n"]), "positives": int(r["positives"]),
               "selection_rate": float(r["selection_rate"]), "TPR": None if pd.isna(r["TPR"]) else float(r["TPR"])}
        if g == privileged:
            row["role"] = "привілейована (референс)"
            rows.append(row)
            continue
        if r["n"] < MIN_N or r["positives"] < MIN_POS:
            row.update({"DPR": float(r["DPR"]), "EOR": None if pd.isna(r["EOR"]) else float(r["EOR"]),
                        "status": f"недостатньо даних (n < {MIN_N} або позитивних < {MIN_POS}) — без інтервалу"})
            rows.append(row)
            continue
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            b = fairness_report(y[idx], yhat[idx], groups[idx], privileged=privileged, groups=[g])
            boots.append([b["DPR"].iloc[0], b["EOR"].iloc[0]])
        boots = np.array(boots, dtype=float)
        row.update({"DPR": float(r["DPR"]), "DPR_ci": [float(np.nanpercentile(boots[:, 0], 2.5)), float(np.nanpercentile(boots[:, 0], 97.5))],
                    "EOR": float(r["EOR"]), "EOR_ci": [float(np.nanpercentile(boots[:, 1], 2.5)), float(np.nanpercentile(boots[:, 1], 97.5))]})
        rows.append(row)
    return rows


def main():
    from scipy.stats import kruskal
    from scipy.stats.contingency import association
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    from src.evaluation import total_cost
    from src.serving.data import load_lab5
    from src.serving.features import REQUEST_FIELDS
    from src.serving.predictor import Predictor
    from src.xai.offline_report import champion_ids

    warnings.filterwarnings("ignore")
    version, run_id = champion_ids()
    data = load_lab5()
    test = data["test"]
    y = test[data["target"]].to_numpy()
    from src.observability.label_round import _normalize
    recs = [_normalize({k: (None if pd.isna(v) else v) for k, v in r.items()}) for r in test[REQUEST_FIELDS].to_dict("records")]
    pred = Predictor.from_dir(PROJECT_ROOT / "models", with_explain=False)
    res = pred.predict(recs)
    score = np.array([r["score"] for r in res]); model_lab = np.array([r["model_label"] for r in res])
    system = np.array([r["label"] for r in res]); fb = np.array([r["fallback_applied"] for r in res])
    from src.serving.predictor import non_ml_rule
    rule = np.array([non_ml_rule(r) for r in recs])

    def block(lab, s=None):
        tn, fp, fn, tp = confusion_matrix(y, lab, labels=[0, 1]).ravel()
        out = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "recall": float(recall_score(y, lab)),
               "precision": float(precision_score(y, lab, zero_division=0)), "f1": float(f1_score(y, lab))}
        if s is not None:
            out["roc_auc"] = float(roc_auc_score(y, s))
        return out

    quality = {"model": block(model_lab, score), "system": block(system), "non_ml_rule": block(rule, rule),
               "fallback_share": float(fb.mean()), "n_fallback": int(fb.sum())}

    rng = np.random.default_rng(42)
    fairness = {"CityTier": group_fairness(y, system, test["CityTier"].to_numpy(), 1, rng),
                "Gender": group_fairness(y, system, test["Gender"].to_numpy(), "Male", rng),
                "decision": "система (модель з порогом ЛР3 + правило для OOD)", "n_boot": N_BOOT,
                "rule_insufficient": f"група з < {MIN_N} об'єктами або < {MIN_POS} позитивними"}

    proxies = {}
    for c in ["PreferredPaymentMode", "PreferedOrderCat", "PreferredLoginDevice", "MaritalStatus"]:
        x = test[c].fillna("NA").replace({"CC": "Credit Card", "COD": "Cash on Delivery", "Phone": "Mobile Phone", "Mobile": "Mobile Phone"})
        proxies[c] = float(association(pd.crosstab(x, test["CityTier"]).values, method="cramer"))
    Z = pred.features(recs)
    eps = {}
    for j, f in enumerate(pred.meta["features"]):
        parts = [Z[test["CityTier"].to_numpy() == t, j] for t in (1, 2, 3)]
        h, _ = kruskal(*parts)
        eps[f] = float((h - 2) / (len(Z) - 3))
    proxy = {"cramers_v_with_CityTier": proxies, "model_features_eps2_with_CityTier": eps,
             "note": "ε² ≥ 0.01 — малий ефект; ≥ 0.06 — середній"}

    roi = {}
    for name, a in SCENARIOS.items():
        cm = costs_from(a)
        loss_rule, loss_sys = total_cost(y, rule, cm), total_cost(y, system, cm)
        roi[name] = {"assumptions": a, "cost_matrix": cm, "loss_non_ml_rule": loss_rule, "loss_system": loss_sys,
                     "effect": loss_rule - loss_sys, "effect_per_1000_customers": (loss_rule - loss_sys) / len(y) * 1000,
                     "effect_share_of_rule_loss": (loss_rule - loss_sys) / loss_rule}
    out = {"model_version": version, "run_id": run_id, "test": {"n": int(len(y)), "positives": int(y.sum()),
           "dataset": "data/processed/holdout.csv @ lab3-v1.0"}, "threshold": pred.meta["threshold"],
           "quality": quality, "fairness": fairness, "proxy": proxy, "roi": roi, "currency": "у.о. (умовні одиниці, масштаб CashbackAmount)"}
    (PROJECT_ROOT / "reports" / "test_evaluation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"quality": quality, "fairness": fairness, "roi": {k: {kk: round(v, 2) if isinstance(v, float) else v
                      for kk, v in r.items() if kk in ("effect", "effect_per_1000_customers", "effect_share_of_rule_loss")} for k, r in roi.items()},
                      "proxy": proxies}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
