"""ЛР4, трек 4A — semi-supervised learning і active learning на частково розмічених даних.

Контекст: інформація про відтік нових користувачів надходить із затримкою. Через це на етапі прийняття рішень розмічено лише частину даних (L), тоді як решта (U) залишається без маркування. Справжні класи для вибірки U приховані, але доступні «оракулу». Це дає змогу виміряти точність прогнозних псевдоміток та змоделювати роботу аналітика в межах активного навчання.
SupervisedOnLabeled — базовий підхід (бейзлайн): обробка даних та побудова моделі RF виключно на основі підмножини L (об'єкти U повністю ігноруються).
SelfTrainingModel — головний алгоритм дослідження (ТЗ 4A, варіант 1): використання sklearn-класу SelfTrainingClassifier поверх Random Forest. Метод покроково додає до навчального набору ті екземпляри з U, для яких впевненість класифікатора перевищує або дорівнює threshold.
LabelSpreadingModel — альтернатива для порівняння (варіант 3, на основі графів): побудова kNN-графа на об'єднаній множині L ∪ U.
Active learning (варіант 5, де роль експерта виконує оракул): тестування підходів random, entropy та entropy + diversity.
Сувора ізоляція даних: будь-які компоненти, що налаштовуються за допомогою цільової змінної (зокрема OOF-WoE, SelectKBest, сам класифікатор), навчаються виключно на розмічених записах. Валідаційний сет (test ЛР4A) взагалі не використовується під час оптимізації. Приховані мітки U задіяні лише для верифікації якості ітеративного розширення бази та як джерело відповідей оракула для AL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT, read_csv

UNLABELED = -1


# --------------------------------------------------------------------------- #
# Конфігурація, дані, розбиття
# --------------------------------------------------------------------------- #
def load_lab4a_config(path: str | Path = "config/lab4a.yaml") -> dict:
    path = Path(path)
    with open(path if path.is_absolute() else PROJECT_ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train(cfg: dict) -> pd.DataFrame:
    """train.csv з DVC-знімка ЛР3 (rev = git-тег) — не з диску напряму."""
    return read_csv(cfg["data"]["path"], rev=cfg["data"]["rev"])


def make_split(df: pd.DataFrame, cfg: dict, id_col: str = "CustomerID", target: str = "Churn") -> pd.DataFrame:
    """Ролі рядків: ``test`` (оцінювання) і ``pool`` (L ∪ U). Стратифікація за ціллю."""
    pool_idx, test_idx = train_test_split(df.index, test_size=float(cfg["split"]["test_share"]),
                                          stratify=df[target], random_state=int(cfg["split"]["seed"]))
    roles = pd.Series("pool", index=df.index)
    roles.loc[test_idx] = "test"
    return pd.DataFrame({id_col: df[id_col].to_numpy(), "role": roles.to_numpy()})


def sample_labeled(y_pool: pd.Series, fraction: float, seed: int) -> np.ndarray:
    """Маска розмічених рядків pool: стратифікована вибірка частки ``fraction``."""
    idx = np.arange(len(y_pool))
    lab, _ = train_test_split(idx, train_size=fraction, stratify=y_pool.to_numpy(), random_state=seed)
    mask = np.zeros(len(y_pool), dtype=bool)
    mask[lab] = True
    return mask


# --------------------------------------------------------------------------- #
# Моделі
# --------------------------------------------------------------------------- #
def build_prep(config_path, filter_k: int) -> Pipeline:
    """Препроцесор ЛР1–3 (розгорнутий) + SelectKBest. Навчається лише на розмічених рядках."""
    from src.evaluation import flatten_steps
    from src.pipeline import ExtendedPipelineBuilder

    steps = flatten_steps(ExtendedPipelineBuilder(str(PROJECT_ROOT / config_path)).build())
    return Pipeline(steps + [("filter", SelectKBest(f_classif, k=int(filter_k)))])


def make_rf(model_cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=int(model_cfg["n_estimators"]), max_depth=int(model_cfg["max_depth"]),
                                  class_weight={int(k): float(v) for k, v in model_cfg["class_weight"].items()},
                                  random_state=int(model_cfg["random_state"]), n_jobs=-1)


class _PrepOnLabeled(BaseEstimator, ClassifierMixin):
    """Спільна логіка: y містить -1 для нерозмічених; препроцесор вчиться на y != -1."""

    def _fit_prep(self, X, y):
        y = np.asarray(y)
        lab = y != UNLABELED
        if lab.sum() == 0 or len(np.unique(y[lab])) < 2:
            raise ValueError("Потрібні розмічені приклади обох класів")
        self.prep_ = build_prep(self.config_path, self.filter_k).fit(X[lab], y[lab])
        self.classes_ = np.array([0, 1])
        return y, lab

    def transform_features(self, X):
        return self.prep_.transform(X)


class SupervisedOnLabeled(_PrepOnLabeled):
    """Бейзлайн: лише розмічені рядки, нерозмічені відкидаються."""

    def __init__(self, config_path="config/generated/config_rf_hermetic.yaml", filter_k=30, model_cfg=None):
        self.config_path = config_path
        self.filter_k = filter_k
        self.model_cfg = model_cfg

    def fit(self, X, y):
        y, lab = self._fit_prep(X, y)
        self.model_ = make_rf(self.model_cfg).fit(self.prep_.transform(X[lab]), y[lab])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.prep_.transform(X))


class SelfTrainingModel(_PrepOnLabeled):
    """Self-Training: RF на L → псевдомітки для U з упевненістю ≥ threshold → перенавчання → … (max_iter)."""

    def __init__(self, config_path="config/generated/config_rf_hermetic.yaml", filter_k=30, model_cfg=None,
                 threshold=0.9, max_iter=10):
        self.config_path = config_path
        self.filter_k = filter_k
        self.model_cfg = model_cfg
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X, y):
        from sklearn.semi_supervised import SelfTrainingClassifier

        y, _ = self._fit_prep(X, y)
        Z = self.prep_.transform(X)
        self.model_ = SelfTrainingClassifier(make_rf(self.model_cfg), threshold=float(self.threshold),
                                             criterion="threshold", max_iter=int(self.max_iter)).fit(Z, y)
        self.transduction_ = self.model_.transduction_      # мітки після навчання (-1 — так і не отримав)
        self.labeled_iter_ = self.model_.labeled_iter_      # 0 — початкова мітка, k — псевдомітка на ітерації k
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.prep_.transform(X))


class LabelSpreadingModel(_PrepOnLabeled):
    """Label Spreading (Zhou et al., 2004): мітки «розтікаються» kNN-графом по L ∪ U; α — частка «забування» початкових міток."""

    def __init__(self, config_path="config/generated/config_lr_hermetic.yaml", filter_k=30,
                 kernel="knn", n_neighbors=10, alpha=0.2, max_iter=100):
        self.config_path = config_path
        self.filter_k = filter_k
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        from sklearn.semi_supervised import LabelSpreading

        y, _ = self._fit_prep(X, y)
        Z = self.prep_.transform(X)
        self.scaler_ = StandardScaler().fit(Z)   # без міток: масштаб для відстаней графа (L ∪ U — так задумано в SSL)
        self.model_ = LabelSpreading(kernel=self.kernel, n_neighbors=int(self.n_neighbors), alpha=float(self.alpha),
                                     max_iter=int(self.max_iter)).fit(self.scaler_.transform(Z), y)
        self.transduction_ = self.model_.transduction_
        lab = y != UNLABELED
        self.prior_ = np.array([1 - y[lab].mean(), y[lab].mean()])
        return self

    def predict_proba(self, X):
        """Вузли, до яких мітки не «дотекли» графом (немає розмічених у компоненті зв'язності) → апріорні частки L."""
        p = self.model_.predict_proba(self.scaler_.transform(self.prep_.transform(X)))
        bad = ~np.isfinite(p).all(axis=1)
        self.last_unreached_share_ = float(bad.mean())
        p[bad] = self.prior_
        return p


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def score_test(y_true, proba, threshold: float, cost_matrix: dict) -> dict:
    from src.evaluation import total_cost

    yhat = (np.asarray(proba) >= threshold).astype(int)
    cost = total_cost(y_true, yhat, cost_matrix)
    return {"roc_auc": roc_auc_score(y_true, proba), "f1": f1_score(y_true, yhat),
            "cost_per_customer": cost / len(y_true)}


def pseudo_label_quality(model: SelfTrainingModel, lab_mask: np.ndarray, y_oracle: np.ndarray) -> dict:
    """Якість псевдоміток Self-Training відносно прихованих міток («оракула»)."""
    u = ~lab_mask
    got = u & (model.labeled_iter_ > 0)
    pseudo, truth = model.transduction_[got], y_oracle[got]
    pos = pseudo == 1
    return {"n_unlabeled": int(u.sum()), "n_pseudo": int(got.sum()), "share_pseudo": float(got.sum() / max(u.sum(), 1)),
            "pseudo_accuracy": float((pseudo == truth).mean()) if got.any() else np.nan,
            "pseudo_pos_share": float(pos.mean()) if got.any() else np.nan,
            "pseudo_pos_precision": float(truth[pos].mean()) if pos.any() else np.nan,
            "churn_recall_in_pseudo": float(((pseudo == 1) & (truth == 1)).sum() / max((truth == 1).sum(), 1)),
            "true_pos_share_in_U": float(y_oracle[u].mean())}


# --------------------------------------------------------------------------- #
# Невизначеність і стратегії запитів (active learning → міра uncertainty у ЛР5)
# --------------------------------------------------------------------------- #
def entropy(p1) -> np.ndarray:
    """Ентропія прогнозного розподілу H = −Σ p·log₂p (бінарний випадок: max = 1 при p = 0.5)."""
    p = np.clip(np.asarray(p1, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def margin(p1) -> np.ndarray:
    """Margin uncertainty: 1 − |p(1) − p(0)| (більше — невизначеніше)."""
    p = np.asarray(p1, dtype=float)
    return 1 - np.abs(2 * p - 1)


def least_confidence(p1) -> np.ndarray:
    p = np.asarray(p1, dtype=float)
    return 1 - np.maximum(p, 1 - p)


def select_queries(strategy: str, proba_pool, Z_pool, batch: int, rng: np.random.Generator,
                   n_candidates: int = 100) -> np.ndarray:
    """Індекси (у межах нерозміченого пулу) для наступного запиту до експерта."""
    n = len(proba_pool)
    batch = min(batch, n)
    if strategy == "random":
        return rng.choice(n, size=batch, replace=False)
    unc = entropy(proba_pool)
    if strategy == "entropy":
        return np.argsort(-unc, kind="stable")[:batch]
    if strategy == "entropy_diversity":
        from sklearn.cluster import KMeans
        cand = np.argsort(-unc, kind="stable")[:max(n_candidates, batch)]
        Zc = np.asarray(Z_pool)[cand]
        Zc = (Zc - Zc.mean(0)) / (Zc.std(0) + 1e-9)
        km = KMeans(n_clusters=batch, n_init=3, random_state=int(rng.integers(1_000_000))).fit(Zc)
        chosen = []
        for c in range(batch):
            members = np.flatnonzero(km.labels_ == c)
            d = np.linalg.norm(Zc[members] - km.cluster_centers_[c], axis=1)
            chosen.append(cand[members[np.argmin(d)]])
        return np.array(chosen)
    raise ValueError(f"Невідома стратегія {strategy!r}")


# --------------------------------------------------------------------------- #
# Експерименти
# --------------------------------------------------------------------------- #
def run_ssl_experiments(X_pool, y_pool, X_test, y_test, cfg: dict, costs: dict, log=print):
    """Частки міток × повтори × методи (+ «стеля» з усіма мітками pool). Повертає (метрики, якість псевдоміток, прогнози)."""
    ss, mc = cfg["semi_supervised"], cfg["model"]
    feat, thr = cfg["features"], float(mc["threshold"])
    y_pool = pd.Series(np.asarray(y_pool), index=X_pool.index)
    rows, pseudo_rows, preds = [], [], []

    def record(method, fraction, seed, model, n_lab):
        p = model.predict_proba(X_test)[:, 1]
        rows.append({"method": method, "fraction": fraction, "seed": seed, "n_labeled": n_lab,
                     **score_test(y_test, p, thr, costs["matrix"])})
        if fraction == ss["self_training"]["sensitivity_fraction"] or method == "full_supervised":
            preds.append(pd.DataFrame({"method": method, "fraction": fraction, "seed": seed,
                                       "row": X_test.index, "y": np.asarray(y_test), "p": p}))

    for seed in ss["seeds"]:
        full = SupervisedOnLabeled(feat["config"], feat["filter_k"], mc).fit(X_pool, y_pool.to_numpy())
        record("full_supervised", 1.0, seed, full, len(y_pool))
        for fraction in ss["label_fractions"]:
            mask = sample_labeled(y_pool, fraction, seed)
            y_semi = np.where(mask, y_pool.to_numpy(), UNLABELED)
            n_lab = int(mask.sum())
            record("supervised_L", fraction, seed, SupervisedOnLabeled(feat["config"], feat["filter_k"], mc)
                   .fit(X_pool, y_semi), n_lab)
            st = SelfTrainingModel(feat["config"], feat["filter_k"], mc, threshold=ss["self_training"]["threshold"],
                                   max_iter=ss["self_training"]["max_iter"]).fit(X_pool, y_semi)
            record("self_training", fraction, seed, st, n_lab)
            pseudo_rows.append({"fraction": fraction, "seed": seed, "threshold": ss["self_training"]["threshold"],
                                **pseudo_label_quality(st, mask, y_pool.to_numpy())})
            ls_cfg = ss["label_spreading"]
            record("label_spreading", fraction, seed,
                   LabelSpreadingModel(feat["config_graph"], feat["filter_k"], ls_cfg["kernel"], ls_cfg["n_neighbors"],
                                       ls_cfg["alpha"], ls_cfg["max_iter"]).fit(X_pool, y_semi), n_lab)
            log(f"seed {seed}, fraction {fraction}: {[(r['method'], round(r['roc_auc'], 4)) for r in rows[-3:]]}")
        # чутливість до порогу упевненості (одна частка; не для вибору порогу, а для розуміння)
        fraction = ss["self_training"]["sensitivity_fraction"]
        mask = sample_labeled(y_pool, fraction, seed)
        y_semi = np.where(mask, y_pool.to_numpy(), UNLABELED)
        for t in ss["self_training"]["sensitivity_thresholds"]:
            st = SelfTrainingModel(feat["config"], feat["filter_k"], mc, threshold=t,
                                   max_iter=ss["self_training"]["max_iter"]).fit(X_pool, y_semi)
            p = st.predict_proba(X_test)[:, 1]
            pseudo_rows.append({"fraction": fraction, "seed": seed, "threshold": t, "sensitivity": True,
                                **score_test(y_test, p, thr, costs["matrix"]),
                                **pseudo_label_quality(st, mask, y_pool.to_numpy())})
    return pd.DataFrame(rows), pd.DataFrame(pseudo_rows), pd.concat(preds, ignore_index=True)


def run_active_learning(X_pool, y_pool, X_test, y_test, cfg: dict, costs: dict, log=print):
    """Цикли дорозмітки з оракулом: на кожному раунді — навчання на L, оцінка на test, запит batch міток із U."""
    al, mc, feat = cfg["active_learning"], cfg["model"], cfg["features"]
    thr = float(mc["threshold"])
    y_pool = np.asarray(y_pool)
    curves, queried = [], []
    for seed in al["seeds"]:
        init = sample_labeled(pd.Series(y_pool), al["initial_fraction"], seed)
        for strategy in al["strategies"]:
            rng = np.random.default_rng(seed)
            lab = init.copy()
            for rnd in range(int(al["rounds"]) + 1):
                model = SupervisedOnLabeled(feat["config"], feat["filter_k"], mc).fit(X_pool, np.where(lab, y_pool, UNLABELED))
                curves.append({"strategy": strategy, "seed": seed, "round": rnd, "n_labeled": int(lab.sum()),
                               **score_test(y_test, model.predict_proba(X_test)[:, 1], thr, costs["matrix"])})
                if rnd == int(al["rounds"]):
                    break
                pool_idx = np.flatnonzero(~lab)
                Xu = X_pool.iloc[pool_idx]
                pu = model.predict_proba(Xu)[:, 1]
                pick = pool_idx[select_queries(strategy, pu, model.transform_features(Xu), int(al["batch_size"]), rng,
                                               int(al["diversity_candidates"]))]
                queried.append(pd.DataFrame({"strategy": strategy, "seed": seed, "round": rnd,
                                             "row": X_pool.index[pick], "y_oracle": y_pool[pick],
                                             "entropy": entropy(model.predict_proba(X_pool.iloc[pick])[:, 1])}))
                lab[pick] = True                       # «експерт» (оракул) повідомив справжні мітки
            log(f"AL seed {seed} {strategy}: AUC {curves[-int(al['rounds']) - 1]['roc_auc']:.4f} → {curves[-1]['roc_auc']:.4f}")
    return pd.DataFrame(curves), pd.concat(queried, ignore_index=True)
