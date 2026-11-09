"""ЛР2, Кроки 2–3 — мультимодальна екстракція та AutoFE (Featuretools DFS).

Модальності для домену e-commerce (ТЗ, розділ 3.3): текст і зображення з Amazon Reviews 2023.
Корпус не має спільного ключа з таблицею клієнтів, тому відгуки **прив'язуються імітаційно**
(``link_reviews_to_customers``): за категорією покупок, задоволеністю та кількістю замовлень
клієнта. Ціль (Churn) у прив'язці не використовується.

Екстрактори:
* ``TextExtractor`` — TF-IDF(500) → TruncatedSVD(5). **Залежний від даних** (вчить словник і SVD).
* ``ImageExtractor`` — текстурні ознаки GLCM (contrast, correlation, energy, homogeneity).
  Безстановий: кожне зображення обробляється незалежно.

Порожні входи (відгук без тексту, відгук без фото, файл не завантажився або битий)
не ламають екстрактор: відповідні ознаки → NaN, а лічильники пишуться в ``stats_``
(REFORMS 3f, 4b).
"""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from src.data import PROJECT_ROOT, open_data, read_bytes, read_csv
from src.features import CyclicalEncoder

ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_multimodal_config(path: str | Path = "config/multimodal.yaml") -> dict:
    with open(_abs(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Завантаження корпусу (одноразово, з кешем у data/raw/amazon)
# --------------------------------------------------------------------------- #
def _keep_fields(record: dict) -> dict:
    images = record.get("images") or []
    first = images[0] if images else {}
    return {
        "rating": record.get("rating"),
        "title": record.get("title") or "",
        "text": record.get("text") or "",
        "timestamp": record.get("timestamp"),
        "parent_asin": record.get("parent_asin"),
        "image_url": first.get("small_image_url"),
    }


def download_reviews(cfg: dict, verbose: bool = True) -> Dict[str, Path]:
    """Потоково читає перші N рядків кожної категорії й зберігає потрібні поля (jsonl)."""
    import requests

    corpus = cfg["corpus"]
    out_dir = _abs(corpus["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for category, n_lines in corpus["lines_per_category"].items():
        path = out_dir / f"reviews_{category}.jsonl"
        paths[category] = path
        if path.exists() and sum(1 for _ in open(path, encoding="utf-8")) >= n_lines:
            continue
        url = f"{corpus['base_url']}/{category}.jsonl"
        with requests.get(url, stream=True, timeout=120) as r, open(path, "w", encoding="utf-8") as out:
            r.raise_for_status()
            for i, line in enumerate(r.iter_lines()):
                if i >= n_lines:
                    break
                out.write(json.dumps(_keep_fields(json.loads(line)), ensure_ascii=False) + "\n")
        if verbose:
            print(f"{category}: збережено {n_lines} відгуків → {path.name}")
    return paths


def load_corpus(cfg: dict) -> pd.DataFrame:
    frames = []
    for category in cfg["corpus"]["lines_per_category"]:
        path = _abs(cfg["corpus"]["out_dir"]) / f"reviews_{category}.jsonl"
        with open_data(path) as f:                       # через DVC (data/raw/amazon.dvc)
            df = pd.read_json(f, lines=True, convert_dates=False)   # timestamp лишається в мс
        df.insert(0, "source_category", category)
        frames.append(df)
    corpus = pd.concat(frames, ignore_index=True)
    corpus["rating"] = corpus["rating"].round().astype(int)
    return corpus


# --------------------------------------------------------------------------- #
# Імітаційна прив'язка відгуків до клієнтів
# --------------------------------------------------------------------------- #
def link_reviews_to_customers(customers: pd.DataFrame, corpus: pd.DataFrame, link_cfg: dict) -> pd.DataFrame:
    """Кожен клієнт отримує k відгуків (без повторів між клієнтами).

    * k = min(OrderCount, max_reviews), пропуск → 1;
    * категорія — за ``category_map`` від категорії покупок клієнта; пропуск → змішаний пул;
    * оцінка відгуку — найближча доступна до ``SatisfactionScore`` клієнта;
    * кожен відгук корпусу використовується не більше одного разу (REFORMS 6b).
    Ціль у прив'язці не використовується; усе детерміновано (``seed``).
    """
    rng = np.random.default_rng(int(link_cfg["seed"]))
    id_col, cat_col = link_cfg["id_column"], link_cfg["category_column"]
    rating_col, count_col = link_cfg["rating_column"], link_cfg["count_column"]
    cmap, kmax = link_cfg["category_map"], int(link_cfg["max_reviews"])

    pools: Dict[str, Dict[int, List[int]]] = {}
    for (cat, rating), idx in corpus.groupby(["source_category", "rating"]).groups.items():
        pools.setdefault(cat, {})[int(rating)] = list(rng.permutation(np.asarray(idx)))
    categories = sorted(pools)

    def take(cat: str, target: int) -> Optional[int]:
        for r in sorted(range(1, 6), key=lambda v: (abs(v - target), -v)):
            bucket = pools.get(cat, {}).get(r)
            if bucket:
                return bucket.pop()
        return None

    rows = []
    for pos in rng.permutation(len(customers)):
        cust = customers.iloc[pos]
        k = cust[count_col]
        k = 1 if pd.isna(k) else int(min(max(k, 1), kmax))
        target = 3 if pd.isna(cust[rating_col]) else int(cust[rating_col])
        mapped = cmap.get(cust[cat_col]) if not pd.isna(cust[cat_col]) else None
        for _ in range(k):
            cat = mapped or categories[rng.integers(len(categories))]
            ridx = take(cat, target)
            if ridx is None:  # категорію вичерпано → будь-яка інша
                for alt in categories:
                    ridx = take(alt, target)
                    if ridx is not None:
                        break
            if ridx is None:
                raise RuntimeError("Корпус вичерпано: збільште lines_per_category")
            rows.append({id_col: cust[id_col], "corpus_index": int(ridx)})

    links = pd.DataFrame(rows)
    reviews = links.join(corpus, on="corpus_index")
    reviews.insert(0, "review_id", np.arange(len(reviews)))
    return reviews


# --------------------------------------------------------------------------- #
# Завантаження зображень (лише для прив'язаних відгуків, що мають фото)
# --------------------------------------------------------------------------- #
def download_images(reviews: pd.DataFrame, img_cfg: dict) -> pd.Series:
    """Повертає шлях до локального файлу або None (немає фото / не вдалося завантажити)."""
    import requests

    out_dir = _abs(img_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch(url) -> Optional[str]:
        if not isinstance(url, str) or not url:
            return None
        path = out_dir / (hashlib.md5(url.encode()).hexdigest() + ".jpg")
        if path.exists() and path.stat().st_size > 0:
            return str(path.relative_to(PROJECT_ROOT))
        try:
            r = requests.get(url, timeout=img_cfg.get("timeout_s", 20))
            r.raise_for_status()
            path.write_bytes(r.content)
            return str(path.relative_to(PROJECT_ROOT))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=int(img_cfg.get("max_workers", 8))) as pool:
        paths = list(pool.map(fetch, reviews["image_url"].tolist()))
    return pd.Series(paths, index=reviews.index, name="image_path")


def prepare_linked_reviews(customers: pd.DataFrame, cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Корпус → прив'язка → фото. Результат кешується в ``linkage.cache``."""
    cache = _abs(cfg["linkage"]["cache"])
    if cache.exists() and not refresh:
        return read_csv(cache, keep_default_na=True)
    download_reviews(cfg)
    reviews = link_reviews_to_customers(customers, load_corpus(cfg), cfg["linkage"])
    reviews["image_path"] = download_images(reviews, cfg["corpus"]["images"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(cache, index=False)
    return reviews


# --------------------------------------------------------------------------- #
# Екстрактори
# --------------------------------------------------------------------------- #
def _to_series(X) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("Очікується одна колонка")
        return X.iloc[:, 0]
    return pd.Series(np.asarray(X).ravel())


class TextExtractor(BaseEstimator, TransformerMixin):
    """TF-IDF → TruncatedSVD (латентний семантичний аналіз).

    Залежний від даних: словник TF-IDF, IDF-ваги й компоненти SVD вчаться у ``fit``.
    Порожній/відсутній текст → усі компоненти NaN; кількість таких входів — у ``stats_``.
    """

    def __init__(self, max_features: int = 500, n_components: int = 5, random_state: int = 42):
        self.max_features = max_features
        self.n_components = n_components
        self.random_state = random_state

    @staticmethod
    def _clean(X):
        texts = _to_series(X).map(lambda t: t.strip() if isinstance(t, str) else "")
        return texts, texts.str.len() > 0

    def fit(self, X, y=None):
        texts, ok = self._clean(X)
        if ok.sum() < self.n_components + 1:
            raise ValueError("TextExtractor: замало непорожніх текстів для навчання")
        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features, stop_words="english", sublinear_tf=True)
        tfidf = self.vectorizer_.fit_transform(texts[ok])
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=self.random_state).fit(tfidf)
        self.stats_ = {"fit_rows": int(len(texts)), "fit_empty": int((~ok).sum())}
        return self

    def transform(self, X):
        check_is_fitted(self, "svd_")
        texts, ok = self._clean(X)
        out = np.full((len(texts), self.n_components), np.nan)
        if ok.any():
            out[ok.to_numpy()] = self.svd_.transform(self.vectorizer_.transform(texts[ok]))
        self.stats_ = {**self.stats_, "transform_rows": int(len(texts)), "transform_empty": int((~ok).sum())}
        return pd.DataFrame(out, columns=self.get_feature_names_out(), index=texts.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"text_svd_{i}" for i in range(self.n_components)], dtype=object)


class ImageExtractor(BaseEstimator, TransformerMixin):
    """Текстурні ознаки GLCM (матриця суміжності рівнів сірого), усереднені за 4 напрямками.

    Безстановий. Вхід — шляхи до файлів. Відсутній шлях / неіснуючий / битий файл → NaN.
    """

    def __init__(self, levels: int = 32, distances=(1,), props=("contrast", "correlation", "energy", "homogeneity"),
                 reader: str = "dvc"):
        self.levels = levels
        self.distances = distances
        self.props = props
        self.reader = reader      # "dvc" — дані репозиторію через DVC API; "local" — лише для тестів на тимчасових файлах

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def _one(self, path) -> Optional[np.ndarray]:
        from skimage import io
        from skimage.color import rgb2gray
        from skimage.feature import graycomatrix, graycoprops
        from skimage.util import img_as_ubyte

        raw = Path(path).read_bytes() if self.reader == "local" else read_bytes(path)   # DVC: data/raw/amazon.dvc
        img = io.imread(BytesIO(raw))
        if img.ndim == 3:
            img = rgb2gray(img[..., :3])
        img = img_as_ubyte(img)
        if min(img.shape[:2]) < 2:
            raise ValueError("зображення замале")
        q = (img.astype(np.uint16) * self.levels // 256).astype(np.uint8)
        glcm = graycomatrix(q, distances=list(self.distances), angles=ANGLES,
                            levels=self.levels, symmetric=True, normed=True)
        return np.array([graycoprops(glcm, p).mean() for p in self.props])

    def transform(self, X):
        paths = _to_series(X)
        out = np.full((len(paths), len(self.props)), np.nan)
        stats = {"rows": int(len(paths)), "no_image": 0, "failed": 0, "ok": 0}
        for i, path in enumerate(paths):
            if not isinstance(path, str) or not path:
                stats["no_image"] += 1
                continue
            try:
                out[i] = self._one(path)
                stats["ok"] += 1
            except FileNotFoundError:
                stats["no_image"] += 1
            except Exception:
                stats["failed"] += 1
        self.stats_ = stats
        return pd.DataFrame(out, columns=self.get_feature_names_out(), index=paths.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"img_{p}" for p in self.props], dtype=object)


def review_features(reviews: pd.DataFrame, text_extractor: Optional[TextExtractor],
                    image_extractor: ImageExtractor, cfg: dict) -> pd.DataFrame:
    """Таблиця ознак відгуків (один рядок — один відгук) для EntitySet.

    ``text_extractor`` має бути ВЖЕ навченим (на відгуках train-клієнтів — у чесних експериментах).
    """
    out = reviews[["review_id", cfg["linkage"]["id_column"], "rating"]].copy()
    texts = review_texts(reviews)
    out["text_len"] = texts.str.split().str.len().fillna(0).astype(float)
    if text_extractor is not None:   # ЛР3: None — SVD рахується пізніше, всередині фолдів (ReviewTextSVD)
        out = out.join(text_extractor.transform(texts.to_frame("text")).set_axis(out.index))
    out = out.join(image_extractor.transform(reviews[["image_path"]]).set_axis(out.index))
    out["has_image"] = reviews["image_path"].notna().astype(float).to_numpy()
    raw_ts = reviews["timestamp"]
    numeric_ts = pd.to_numeric(raw_ts, errors="coerce")
    ts = (pd.to_datetime(numeric_ts, unit="ms", utc=True) if numeric_ts.notna().all()
          else pd.to_datetime(raw_ts, utc=True, errors="coerce"))   # кеш CSV зберігає дату рядком
    time = pd.DataFrame({"review_hour": ts.dt.hour, "review_dow": ts.dt.dayofweek}, index=out.index)
    out = out.join(CyclicalEncoder(periods=cfg["extraction"]["time"]["periods"]).fit_transform(time))
    return out


def review_texts(reviews: pd.DataFrame) -> pd.Series:
    return (reviews["title"].fillna("").astype(str) + ". " + reviews["text"].fillna("").astype(str)).str.strip(". ")


# --------------------------------------------------------------------------- #
# Крок 3 — Featuretools DFS
# --------------------------------------------------------------------------- #
def run_dfs(clients: pd.DataFrame, reviews_feat: pd.DataFrame, cfg: dict):
    """EntitySet(clients 1 — N reviews) → DFS. Повертає (матриця нових ознак, визначення ознак).

    DFS безстановий: кожна ознака рахується з рядків одного клієнта (агрегати по його відгуках,
    логарифми й відношення його ж значень), тож запуск на всіх клієнтах не переносить
    інформацію між train і test — за умови, що вхідні ознаки відгуків (TF-IDF/SVD) навчено лише на train.
    Нескінченності (ділення на 0) → NaN (REFORMS 4b). Базові колонки клієнта з виходу прибираються
    (вони вже є в конвеєрі ЛР1).
    """
    import warnings

    import featuretools as ft

    dfs_cfg = cfg["dfs"]
    id_col = cfg["linkage"]["id_column"]
    client_cols = list(dfs_cfg["client_columns"])
    es = ft.EntitySet(id="ecommerce")
    es = es.add_dataframe(dataframe_name="clients", dataframe=clients[[id_col] + client_cols].copy(), index=id_col)
    es = es.add_dataframe(dataframe_name="reviews",
                          dataframe=reviews_feat[["review_id", id_col] + list(dfs_cfg["review_columns"])].copy(),
                          index="review_id")
    es = es.add_relationship("clients", id_col, "reviews", id_col)
    keep = dfs_cfg.get("keep") or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = ft.dfs(
            entityset=es, target_dataframe_name="clients", max_depth=int(dfs_cfg["max_depth"]),
            agg_primitives=list(dfs_cfg["agg_primitives"]), trans_primitives=list(dfs_cfg["trans_primitives"]),
            features_only=True,
        )

        def keep_feature(f) -> bool:
            depth, prim = f.get_depth(), type(f.primitive).__name__
            if depth <= int(keep.get("max_depth_default", dfs_cfg["max_depth"])):
                return True
            if prim in keep.get("depth2_primitives", []):
                return not any(s in f.get_name() for s in keep.get("log_exclude_contains", []))
            return False

        defs = [f for f in candidates if keep_feature(f)]
        fm = ft.calculate_feature_matrix(defs, entityset=es)
    fm = fm.drop(columns=[c for c in client_cols if c in fm.columns])
    fm = fm.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    fm.attrs["n_candidates"] = len(candidates)        # повний «вибух» до обмеження
    return fm, defs


# --------------------------------------------------------------------------- #
# ЛР3 — текстовий блок як ланка конвеєра (герметизація TF-IDF/SVD)
# --------------------------------------------------------------------------- #
REVIEW_SEP = " <|> "


def customer_review_text(reviews: pd.DataFrame, id_column: str) -> pd.Series:
    """Усі відгуки клієнта одним рядком (роздільник ``REVIEW_SEP``) — колонка таблиці клієнтів.

    Безстанова операція: лише групування текстів за ключем клієнта. Завдяки цьому TF-IDF/SVD
    можна навчати всередині фолду як звичайну ланку ColumnTransformer.
    """
    texts = review_texts(reviews).str.replace(REVIEW_SEP.strip(), " ", regex=False)
    return texts.groupby(reviews[id_column]).agg(REVIEW_SEP.join).rename("ReviewsText")


class ReviewTextSVD(BaseEstimator, TransformerMixin):
    """TF-IDF → TruncatedSVD на окремих відгуках + агрегація по клієнту (mean, max).

    Вхід — колонка з об'єднаними відгуками клієнта (див. ``customer_review_text``).
    ``fit`` бачить лише відгуки клієнтів поточного (тренувального) фолду — це і є усунення
    ВИТОКУ №1 з ЛР2 (словник/IDF/SVD на всіх текстах). Агрегати відтворюють DFS-ознаки
    ``MEAN/MAX(reviews.text_svd_i)``. Клієнт без жодного непорожнього відгуку → 0 (нейтрально).
    """

    def __init__(self, max_features: int = 500, n_components: int = 5, random_state: int = 42,
                 aggregations=("mean", "max"), sep: str = REVIEW_SEP):
        self.max_features = max_features
        self.n_components = n_components
        self.random_state = random_state
        self.aggregations = aggregations
        self.sep = sep

    def _explode(self, X) -> pd.DataFrame:
        col = _to_series(X).map(lambda t: t if isinstance(t, str) else "")
        rows = col.reset_index(drop=True).str.split(self.sep, regex=False).explode()
        return pd.DataFrame({"row": rows.index.to_numpy(), "text": rows.fillna("").str.strip().to_numpy()})

    def fit(self, X, y=None):
        exploded = self._explode(X)
        self.extractor_ = TextExtractor(max_features=self.max_features, n_components=self.n_components,
                                        random_state=self.random_state).fit(exploded[["text"]])
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        check_is_fitted(self, "extractor_")
        n_rows = len(_to_series(X))
        exploded = self._explode(X)
        comps = self.extractor_.transform(exploded[["text"]]).set_axis(exploded.index)
        comps["row"] = exploded["row"].to_numpy()
        agg = comps.groupby("row").agg(list(self.aggregations))
        agg.columns = [f"{c}_{a}" for c, a in agg.columns]
        agg = agg.reindex(range(n_rows))
        out = agg[list(self.get_feature_names_out())].to_numpy(dtype=float)
        return pd.DataFrame(np.nan_to_num(out, nan=0.0), columns=self.get_feature_names_out(),
                            index=X.index if hasattr(X, "index") else None)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"text_svd_{i}_{a}" for i in range(self.n_components) for a in self.aggregations],
                          dtype=object)
