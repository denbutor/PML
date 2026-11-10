"""ChurnGuard — HITL-дашборд менеджера з утримання (ЛР5, Етап 5).

    streamlit run src/dashboard/app.py          (API: змінна API_URL, за замовчуванням http://localhost:8000)

Дашборд модель НЕ завантажує: прогнози й локальні пояснення — лише через HTTP (/v1/predict, /v1/explain,
/v1/feedback, /v1/label_queue); глобальні результати — з reports/ (офлайн-звіт src/xai/offline_report.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:          # streamlit run додає в sys.path лише src/dashboard, а не корінь проєкту
    sys.path.insert(0, str(ROOT))
API =os.environ.get("API_URL", "http://localhost:8000")
REPORTS = ROOT / "reports"

st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=60, **kw)
        return r.status_code, (r.json() if r.content else {})
    except requests.RequestException as exc:
        return 0, {"error": {"message": f"API недоступний: {exc}"}}


def read_json(name):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def waterfall(shap_block, max_display=10):
    import shap
    exp = shap.Explanation(values=np.array(shap_block["values"]), base_values=shap_block["base_value"],
                           data=np.array(shap_block["data"]), feature_names=shap_block["feature_names"])
    plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7, 4.5)
    return fig


def lime_chart(lime):
    df = pd.DataFrame(lime[:10], columns=["умова", "вага"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["умова"], df["вага"], color=["#2a78d6" if v > 0 else "#eb6834" for v in df["вага"]])
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_xlabel("внесок LIME у P(відтік)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def show_explanation(res: dict):
    pr = res["prediction"]
    meta = pr["meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ризик відтоку (модель)", f"{pr['score']:.1%}")
    c2.metric("Рекомендація", "Запропонувати утримання" if pr["label"] == 1 else "Не чіпати")
    c3.metric("Невизначеність (ентропія)", f"{pr['uncertainty']:.2f}")
    c4.metric("Версія моделі", meta["model_version"])
    if meta["fallback_applied"]:
        st.warning("⚠️ Нетиповий клієнт (OOD): рішення дало **Non-ML правило ЛР1** (стаж ≤ 1 міс. або скарга). "
                   "Прогноз моделі користувачу не віддано, SHAP і LIME нижче описують модель лише довідково.")
    a = res["agreement"]
    st.caption(f"Збіг SHAP і LIME: спільних у топ-5 — {a['top5_overlap']:.0%}, ρ Спірмена |внесків| — {a['spearman']:.2f}. "
               "Внески — у одиницях імовірності, це не відсотки й не причини.")
    l, r = st.columns(2)
    l.markdown("**SHAP (waterfall)**")
    l.pyplot(waterfall(res["shap"]), clear_figure=True)
    r.markdown("**LIME (локальна лінійна апроксимація)**")
    r.pyplot(lime_chart(res["lime"]), clear_figure=True)


def request_form(defaults: dict) -> dict | None:
    with st.form("case"):
        st.markdown("**Дані клієнта** (без захищених ознак: рівень міста й стать у формі відсутні)")
        c = st.columns(4)
        f = {}
        f["Tenure"] = c[0].number_input("Стаж, міс.", 0.0, 240.0, float(defaults["Tenure"]))
        f["OrderCount"] = c[1].number_input("Замовлень за місяць", 0.0, 200.0, float(defaults["OrderCount"]))
        f["CouponUsed"] = c[2].number_input("Купонів за місяць", 0.0, 200.0, float(defaults["CouponUsed"]))
        f["DaySinceLastOrder"] = c[3].number_input("Днів з останнього замовлення", 0.0, 3650.0, float(defaults["DaySinceLastOrder"]))
        f["CashbackAmount"] = c[0].number_input("Кешбек, у.о./міс.", 0.0, 10000.0, float(defaults["CashbackAmount"]))
        f["WarehouseToHome"] = c[1].number_input("Склад → дім, км", 0.0, 300.0, float(defaults["WarehouseToHome"]))
        f["HourSpendOnApp"] = c[2].number_input("Годин у застосунку", 0.0, 24.0, float(defaults["HourSpendOnApp"]))
        f["OrderAmountHikeFromlastYear"] = c[3].number_input("Зміна суми замовлень, %", -100.0, 1000.0,
                                                              float(defaults["OrderAmountHikeFromlastYear"]))
        f["NumberOfDeviceRegistered"] = int(c[0].number_input("Пристроїв", 1, 20, int(defaults["NumberOfDeviceRegistered"])))
        f["NumberOfAddress"] = int(c[1].number_input("Адрес", 1, 50, int(defaults["NumberOfAddress"])))
        f["SatisfactionScore"] = int(c[2].slider("Задоволеність", 1, 5, int(defaults["SatisfactionScore"])))
        f["Complain"] = 1 if c[3].checkbox("Скарга за місяць", bool(defaults["Complain"])) else 0
        cats = {"PreferredLoginDevice": ["Mobile Phone", "Computer"],
                "PreferredPaymentMode": ["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery"],
                "PreferedOrderCat": ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others", "(невідомо)"],
                "MaritalStatus": ["Single", "Married", "Divorced"]}
        cc = st.columns(4)
        for col, (k, opts) in zip(cc, cats.items()):
            cur = defaults.get(k) or "(невідомо)"
            v = col.selectbox(k, opts, index=opts.index(cur) if cur in opts else 0)
            f[k] = None if v == "(невідомо)" else v
        if st.form_submit_button("Оцінити й пояснити"):
            return f
    return None


st.title("ChurnGuard — підтримка рішень з утримання клієнтів")
code, health = api("GET", "/health")
if code != 200:
    st.error(f"Сервіс не готовий ({code}): {health}")
else:
    st.caption(f"API: {API} · модель v{health['model_version']} · OOD-детектор: {'так' if health['ood'] else 'ні'} · "
               "остаточне рішення ухвалює менеджер")

tab1, tab2, tab3, tab4 = st.tabs(["Інференс і пояснення", "Глобальні пояснення", "Довіра до пояснень", "Черга розмітки"])

with tab1:
    from src.app.schemas import PredictionRequest
    payload = request_form(PredictionRequest.model_config["json_schema_extra"]["example"])
    if payload is not None:
        code, res = api("POST", "/v1/explain", json=payload)
        if code == 200:
            st.session_state["last_explain"] = res
        else:
            st.error(f"{code}: {res.get('error', res)}")
    if "last_explain" in st.session_state:
        show_explanation(st.session_state["last_explain"])

with tab2:
    s = read_json("xai/summary.json")
    if s is None:
        st.info("Немає офлайн-звіту: python -m src.xai.offline_report")
    else:
        st.caption(f"Звіт для моделі v{s['model_version']} (run {s['run_id']}); explainer {s['explainer']}, функція цінності — {s['value_function']}.")
        st.image(str(REPORTS / "xai" / "shap_beeswarm.png"), caption="SHAP summary на test")
        st.image(str(REPORTS / "xai" / "pdp_ale_top2.png"), caption="Дві найважливіші ознаки")
        st.table(pd.DataFrame(s["pdp_or_ale"]).rename(columns={"feature": "ознака", "method": "метод",
                 "max_abs_spearman": "max |ρ| з іншою ознакою", "most_correlated_with": "з якою"}))
        st.caption("ALE замість PDP, якщо |ρ| > 0.7: PDP усереднює по нереалістичних комбінаціях корельованих ознак.")

with tab3:
    s = read_json("xai/summary.json")
    if s:
        c1, c2, c3 = st.columns(3)
        c1.metric("Fidelity сурогатного дерева (R², test)", f"{s['surrogate']['fidelity_r2_test']:.2f}")
        c2.metric("Стабільність SHAP (середнє ρ)", f"{s['stability']['shap_spearman_mean']:.2f}")
        c3.metric("Стабільність топ-5", f"{s['stability']['top5_overlap_mean']:.0%}")
        st.image(str(REPORTS / "xai" / "stability.png"))
        st.json(s["stability"])
        st.markdown("**Сурогатне дерево (глибина ≤ 4, навчене на ймовірностях моделі)**")
        st.code((REPORTS / "xai" / "surrogate_tree.txt").read_text(encoding="utf-8"))

with tab4:
    st.markdown("Які клієнти варто передати експертові: **найменш певні** прогнози (ентропія + різноманітність, ЛР4A). "
                "Нетипові (OOD) — окремим списком: це нова зона даних, а не межа класів. **Розмітку симульовано** — мітки з «оракула».")
    code, q = api("GET", "/v1/label_queue")
    if code != 200:
        st.error(f"{code}: {q}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Розмічено запитів", q["n_labeled"]); c2.metric("Очікують розмітки", q["n_pending"]); c3.metric("Стратегія", q["strategy"])
        queue = pd.DataFrame(q["queue"])
        if len(queue):
            st.dataframe(queue[["request_id", "uncertainty", "score", "label", "model_version"]], width="stretch")
            oracle_path = ROOT / "data" / "processed" / "pool_oracle.csv"
            oracle = pd.read_csv(oracle_path).set_index("request_id") if oracle_path.exists() else None
            rid = st.selectbox("Запит для розмітки", queue["request_id"].tolist())
            item = queue.set_index("request_id").loc[rid]
            feats = json.loads(item["features"])
            st.json(feats, expanded=False)
            # Streamlit перезапускає весь скрипт при кожній дії (і код усіх вкладок), тому:
            # log=false — запит уже в журналі, новий запис дав би копію клієнта в черзі;
            # кеш за request_id — SHAP + LIME рахуються один раз на запит, а не при кожному кліку.
            cache = st.session_state.setdefault("queue_explain", {})
            if rid not in cache:
                code, ex = api("POST", "/v1/explain", params={"log": "false"}, json=feats)
                if code == 200:
                    cache[rid] = ex
                else:
                    st.error(f"{code}: {ex.get('error', ex)}")
            if rid in cache:
                show_explanation(cache[rid])
            suggested = int(oracle.loc[rid, "y_oracle"]) if oracle is not None and rid in oracle.index else 0
            y_true = st.radio("Фактичний результат (оракул підказує: "
                              f"{'пішов' if suggested else 'залишився'})", [0, 1], index=suggested, horizontal=True,
                              format_func=lambda v: "1 — пішов" if v else "0 — залишився")
            if st.button("Надіслати y_true"):
                code, r = api("POST", "/v1/feedback", json={"request_id": rid, "y_true": int(y_true)})
                st.success("Збережено — запит зникне з наступної черги") if code == 200 else st.error(f"{code}: {r}")
        if q["ood"]:
            st.markdown(f"**OOD-запити поза чергою ({len(q['ood'])})**")
            st.dataframe(pd.DataFrame(q["ood"])[["request_id", "ood_log_density", "score", "label"]], width="stretch")
    lr = read_json("label_round.json")
    if lr:
        st.markdown("**Раунд дорозмітки: champion / challenger / random** (парний бутстреп, 1000 повторень, test)")
        rows = []
        for name, d in lr["bootstrap"]["diff_vs_champion"].items():
            for metric, v in d.items():
                rows.append({"порівняння": name, "метрика": metric, "різниця": round(v["mean"], 4),
                             "95% CI": f"[{v['ci_low']:.4f}; {v['ci_high']:.4f}]", "містить 0": v["contains_zero"]})
        st.table(pd.DataFrame(rows))
        st.info(f"Рішення: {lr['promotion_decision']}. @champion автоматично не змінюється.")
