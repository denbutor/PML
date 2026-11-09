"""ЛР2, Крок 1 — ручний дизайн табличних ознак.

Три обов'язкові категорії (п. 4.1 ТЗ):

* **А. Часові ознаки** — ``IntervalBinner`` (етап життєвого циклу клієнта з ``Tenure``)
  і ``CyclicalEncoder`` (sin/cos для циклічних величин — година/день тижня відгуку).
* **Б. Взаємодії та трансформації** — ``SafeRatio`` (захищене відношення з ε, REFORMS 4b);
  стабілізація дисперсії Yeo-Johnson — крок ``yeo_johnson`` у ``transform_policy``.
* **В. Кодування високої кардинальності** — ``CategoryCrosser`` (схрещення категорій →
  сегмент високої кардинальності) + ``OOFWoEEncoder`` (out-of-fold WoE).

Усі класи sklearn-сумісні. Назви колонок, межі бінів і параметри задаються в YAML
(секція ``feature_engineering``), а не в коді.

Конструктори (``SafeRatio``, ``IntervalBinner``, ``CategoryCrosser``) — *доповнюючі*:
приймають усю таблицю й повертають її ж плюс нові колонки. Вони безстанові (нічого не
вчать з даних), тому не можуть бути джерелом витоку. ``OOFWoEEncoder`` — станово залежний
і кероване ціллю: навчається тільки на train / у фолді.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from src.utils import _NamedColumnsMixin, _as_frame


# --------------------------------------------------------------------------- #
# Доповнюючі конструктори (безстанові)
# --------------------------------------------------------------------------- #
class _AugmentingTransformer(BaseEstimator, TransformerMixin):
    """База: transform(X) = X + нові колонки. Вхідні колонки перевіряються у fit."""

    def _required(self) -> List[str]:  # pragma: no cover - перевизначається
        raise NotImplementedError

    def _new_columns(self) -> List[str]:  # pragma: no cover - перевизначається
        raise NotImplementedError

    def fit(self, X, y=None):
        X = _as_frame(X)
        missing = [c for c in self._required() if c not in X.columns]
        if missing:
            raise ValueError(f"{type(self).__name__}: у даних немає колонок {missing}")
        clash = [c for c in self._new_columns() if c in X.columns]
        if clash:
            raise ValueError(f"{type(self).__name__}: нові колонки вже існують: {clash}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(list(self.feature_names_in_) + self._new_columns(), dtype=object)


class SafeRatio(_AugmentingTransformer):
    """Взаємодія-відношення ``numerator / (denominator + ε)``.

    * ε (``epsilon``) захищає від ділення на нуль (довідник, п. 1.1; REFORMS 4b).
    * Пропуск у будь-якому операнді → пропуск у результаті (його заповнить AdaptiveImputer
      далі в конвеєрі; індикатор пропуску за потреби додасть політика).
    * Нескінченності (ділення на 0 при ε = 0) примусово замінюються на NaN.
    * ``denominator_scale`` переводить одиниці (напр. місяці → дні).
    """

    def __init__(self, name: str, numerator: str, denominator: str,
                 epsilon: float = 1e-5, denominator_scale: float = 1.0):
        self.name = name
        self.numerator = numerator
        self.denominator = denominator
        self.epsilon = epsilon
        self.denominator_scale = denominator_scale

    def _required(self):
        return [self.numerator, self.denominator]

    def _new_columns(self):
        return [self.name]

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = _as_frame(X, self.feature_names_in_).copy()
        num = X[self.numerator].astype(float)
        den = X[self.denominator].astype(float) * float(self.denominator_scale) + float(self.epsilon)
        ratio = num / den
        X[self.name] = ratio.replace([np.inf, -np.inf], np.nan)
        return X


class IntervalBinner(_AugmentingTransformer):
    """Дискретизація за фіксованими межами з конфігурації (часова ознака «етап»).

    Межі задаються експертно (не з даних), тому трансформер безстановий.
    Результат — порядковий код 0..k-1 (float, NaN зберігається), інтервали ``(a, b]``.
    """

    def __init__(self, name: str, source: str, edges: Sequence[float], labels: Sequence[str] | None = None):
        self.name = name
        self.source = source
        self.edges = edges
        self.labels = labels

    def _required(self):
        return [self.source]

    def _new_columns(self):
        return [self.name]

    def fit(self, X, y=None):
        edges = [float(e) for e in self.edges]
        if edges != sorted(edges) or len(set(edges)) != len(edges):
            raise ValueError(f"IntervalBinner({self.name}): межі мають строго зростати: {self.edges}")
        if self.labels is not None and len(self.labels) != len(edges) - 1:
            raise ValueError(f"IntervalBinner({self.name}): кількість labels має бути len(edges) - 1")
        return super().fit(X, y)

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = _as_frame(X, self.feature_names_in_).copy()
        codes = pd.cut(X[self.source].astype(float), bins=[float(e) for e in self.edges],
                       labels=False, right=True, include_lowest=True)
        X[self.name] = codes.astype(float)
        return X


class CategoryCrosser(_AugmentingTransformer):
    """Схрещення кількох категорій в одну ознаку («сегмент») — створює категорію високої кардинальності.

    Пропуск у будь-якій складовій кодується як окремий токен ``missing_token``,
    щоб сегмент не губив рядок.
    """

    def __init__(self, name: str, sources: Sequence[str], sep: str = " | ", missing_token: str = "NA"):
        self.name = name
        self.sources = sources
        self.sep = sep
        self.missing_token = missing_token

    def _required(self):
        return list(self.sources)

    def _new_columns(self):
        return [self.name]

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = _as_frame(X, self.feature_names_in_).copy()
        parts = [X[c].map(lambda v: self.missing_token if pd.isna(v) else str(v)) for c in self.sources]
        out = parts[0]
        for p in parts[1:]:
            out = out + self.sep + p
        X[self.name] = out.astype(object)
        return X


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Гармонічне кодування циклічних величин: x → (sin 2πx/T, cos 2πx/T).

    ``periods`` — словник «колонка → період» (напр. година → 24, день тижня → 7).
    Безстановий. Повертає лише нові колонки ``<col>_sin``, ``<col>_cos``.
    """

    def __init__(self, periods: Dict[str, float]):
        self.periods = periods

    def fit(self, X, y=None):
        X = _as_frame(X)
        missing = [c for c in self.periods if c not in X.columns]
        if missing:
            raise ValueError(f"CyclicalEncoder: у даних немає колонок {missing}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = _as_frame(X, self.feature_names_in_)
        out = {}
        for col, period in self.periods.items():
            angle = 2 * np.pi * X[col].astype(float) / float(period)
            out[f"{col}_sin"] = np.sin(angle)
            out[f"{col}_cos"] = np.cos(angle)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_{s}" for c in self.periods for s in ("sin", "cos")], dtype=object)


CONSTRUCTORS = {"ratio": SafeRatio, "bins": IntervalBinner, "cross": CategoryCrosser}


def constructor_sources(spec: List[Dict[str, Any]]) -> List[str]:
    """Усі вхідні колонки, які використовують конструктори з секції feature_engineering."""
    cols: List[str] = []
    for item in spec:
        params = {k: v for k, v in item.items() if k != "type"}
        for key in ("numerator", "denominator", "source"):
            if key in params:
                cols.append(params[key])
        cols.extend(params.get("sources", []))
    return cols


def build_constructors(spec: List[Dict[str, Any]]):
    """Список (назва, конструктор) за секцією ``feature_engineering`` конфігурації."""
    steps = []
    for item in spec:
        kind = item.get("type")
        if kind not in CONSTRUCTORS:
            raise ValueError(f"feature_engineering: невідомий тип {kind!r}; доступні {sorted(CONSTRUCTORS)}")
        params = {k: v for k, v in item.items() if k != "type"}
        steps.append((params["name"], CONSTRUCTORS[kind](**params)))
    return steps


# --------------------------------------------------------------------------- #
# Кероване ціллю кодування: WoE (з ЛР1) і його регуляризована OOF-версія
# --------------------------------------------------------------------------- #
class WoEEncoder(_NamedColumnsMixin, BaseEstimator, TransformerMixin):
    """Weight of Evidence для бінарної цілі: WoE = ln(%подій / %не-подій) у категорії.

    Базова версія з ЛР1 (перенесена сюди без змін поведінки).
    * ``smoothing`` — адитивне згладжування Лапласа (немає ±∞ для категорій без подій).
    * Невідома категорія → 0 (нейтральний внесок).
    * **Вада:** ваги обчислюються на всій тренувальній підвибірці, тож при кодуванні train
      кожен рядок «бачить» власну мітку; рідкісні категорії отримують перенавчені значення.
    """

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing

    @staticmethod
    def _check_y(y):
        if y is None:
            raise ValueError("WoE-кодування потребує y у fit(): воно кероване ціллю")
        y = np.asarray(y).astype(int)
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("WoE-кодування підтримує лише бінарну ціль {0, 1}")
        return y

    def _mapping(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict[Any, float]]:
        a = float(self.smoothing)
        total_pos, total_neg = y.sum(), (1 - y).sum()
        mapping = {}
        for col in X.columns:
            stats = pd.DataFrame({"x": X[col].to_numpy(), "y": y}).groupby("x")["y"].agg(["sum", "count"])
            k = len(stats)
            pos = (stats["sum"] + a) / (total_pos + a * k)
            neg = (stats["count"] - stats["sum"] + a) / (total_neg + a * k)
            mapping[col] = np.log(pos / neg).to_dict()
        return mapping

    @staticmethod
    def _encode(X: pd.DataFrame, mapping) -> pd.DataFrame:
        out = {col: np.nan_to_num(X[col].map(mapping[col]).to_numpy(dtype=float), nan=0.0) for col in X.columns}
        return pd.DataFrame(out, index=X.index)

    def fit(self, X, y=None):
        X = self._remember_columns(X)
        self.mapping_ = self._mapping(X, self._check_y(y))
        return self

    def transform(self, X):
        check_is_fitted(self, "mapping_")
        return self._encode(_as_frame(X, self.feature_names_in_), self.mapping_)


class OOFWoEEncoder(WoEEncoder):
    """Out-of-fold WoE — регуляризація керованого кодування (дефолт ТЗ, п. 4.1 В).

    * ``fit_transform(X_train, y_train)``: train ділиться на ``n_splits`` стратифікованих фолдів;
      кожен фолд кодується вагами, порахованими **на решті фолдів** — рядок ніколи не бачить
      власної мітки, рідкісна категорія не «запам'ятовує» свій шум.
    * ``transform(X_new)``: ваги, пораховані на всьому train (як у звичайного WoE).
    * Тобто ``fit_transform(X) != fit(X).transform(X)`` — так само, як у sklearn ``TargetEncoder``.
      sklearn ``Pipeline``/``ColumnTransformer`` під час навчання викликають саме ``fit_transform``.
    """

    def __init__(self, smoothing: float = 0.5, n_splits: int = 5, random_state: int = 42):
        super().__init__(smoothing=smoothing)
        self.n_splits = n_splits
        self.random_state = random_state

    def fit_transform(self, X, y=None, **fit_params):
        X = self._remember_columns(X)
        y = self._check_y(y)
        self.mapping_ = self._mapping(X, y)          # для майбутніх transform()
        oof = pd.DataFrame(0.0, index=X.index, columns=X.columns)
        folds = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for fit_idx, enc_idx in folds.split(X, y):
            fold_map = self._mapping(X.iloc[fit_idx], y[fit_idx])
            oof.iloc[enc_idx] = self._encode(X.iloc[enc_idx], fold_map).to_numpy()
        self.oof_ = True
        return oof
