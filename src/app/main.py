"""ChurnGuard — FastAPI-сервіс підтримки рішень для менеджера з утримання клієнтів (ЛР5).

Запуск:  uvicorn src.app.main:app --port 8000      (Swagger: http://localhost:8000/docs)

Шари: HTTP (цей файл, схеми) → сервіс (src/serving/predictor.py: препроцесор + ONNX + OOD + правило) → артефакти
(MLflow @champion або models/). Модель завантажується один раз у lifespan; поки її немає або контракт порушено —
/health = 503. Ендпоінти з інференсом — звичайні ``def``: FastAPI виконує їх у threadpool, event loop не блокується.

Змінні середовища: MODEL_SOURCE = mlflow | local (за замовчуванням mlflow, якщо є mlflow.db, інакше local);
MODEL_DIR (local); FEEDBACK_DB (шлях до SQLite); MLFLOW_TRACKING_URI.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.app import db
from src.app.schemas import (BATCH_LIMIT, BatchRequest, BatchResponse, ExplainResponse, FeedbackRequest,
                             FeedbackResponse, Meta, PredictionRequest, PredictionResponse)
from src.data import PROJECT_ROOT

NAME = "ChurnGuard"
router = APIRouter()


# --------------------------------------------------------------------------- #
# Перевірка контракту при старті
# --------------------------------------------------------------------------- #
def check_contract(meta, sess, request_model, preprocessor=None, run_id=None) -> list[str]:
    errs = []
    if set(request_model.model_fields) != set(meta["request_fields"]):
        errs.append("поля PredictionRequest не збігаються з request_fields")
    inputs = sess.get_inputs()
    if [i.name for i in inputs] != [i["name"] for i in meta["onnx_inputs"]]:
        errs.append("входи ONNX не збігаються з onnx_inputs")
    if len(inputs) == 1 and isinstance(inputs[0].shape[-1], int) and inputs[0].shape[-1] != len(meta["features"]):
        errs.append("ширина входу ONNX не збігається з кількістю features")
    if preprocessor is not None and list(preprocessor.get_feature_names_out()) != list(meta["features"]):
        errs.append("вихід препроцесора не збігається з features (порядок або склад)")
    if sorted(meta.get("classes", [])) != [0, 1] or meta.get("positive_class") != 1:
        errs.append("classes / positive_class некоректні")
    if set(meta.get("protected_attributes", [])) & set(meta["request_fields"]):
        errs.append("захищені ознаки потрапили в request_fields")
    if run_id is not None and meta.get("run_id") not in (None, run_id):
        errs.append("run_id артефактів не збігається з версією реєстру")
    return errs


def _load_from_mlflow():
    import mlflow
    import mlflow.onnx
    from mlflow import MlflowClient

    from src.serving.predictor import Predictor

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"))
    mv = MlflowClient().get_model_version_by_alias(NAME, "champion")      # аліас резолвиться ОДИН раз
    local = Path(tempfile.mkdtemp(prefix="churnguard_"))
    mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="serving", dst_path=str(local))
    onx = mlflow.onnx.load_model(f"models:/{NAME}/{mv.version}")
    return Predictor.from_dir(local / "serving", onnx_bytes=onx.SerializeToString(), version=str(mv.version),
                              run_id=mv.run_id)


def _load_local():
    from src.serving.predictor import Predictor

    d = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))
    p = Predictor.from_dir(d)
    p.version = f"local-{p.meta.get('tag', 'v?')}"
    return p


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor, app.state.explainer, app.state.contract_errors = None, None, []
    db.init_db()
    source = os.environ.get("MODEL_SOURCE", "mlflow" if (PROJECT_ROOT / "mlflow.db").exists() else "local")
    try:
        pred = _load_from_mlflow() if source == "mlflow" else _load_local()
        errs = check_contract(pred.meta, pred.session, PredictionRequest, pred.preprocessor)
        app.state.contract_errors = errs
        if not errs:
            from src.xai.explain import LocalExplainer
            app.state.explainer = LocalExplainer(pred.export_pipe, pred.background, pred.meta["features"])
            app.state.predictor = pred
    except Exception as exc:                                   # сервіс живий, але не готовий
        app.state.contract_errors = [f"не вдалося завантажити модель ({source}): {exc!r}"]
    yield




# --------------------------------------------------------------------------- #
# Єдиний формат помилок
# --------------------------------------------------------------------------- #
def _error(status: int, code: str, message: str, details=None):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details}})


async def _validation(request: Request, exc: RequestValidationError):
    details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return _error(422, "validation_error", "Запит не відповідає контракту PredictionRequest", details)


async def _http(request: Request, exc: HTTPException):
    code = {404: "not_found", 503: "not_ready"}.get(exc.status_code, "http_error")
    return _error(exc.status_code, code, str(exc.detail))


async def _unhandled(request: Request, exc: Exception):
    return _error(500, "internal_error", "Внутрішня помилка сервісу")


def _predictor(request: Request):
    p = request.app.state.predictor
    if p is None:
        raise HTTPException(503, "Модель не завантажена: " + "; ".join(request.app.state.contract_errors or ["старт"]))
    return p


def _respond(p, item: PredictionRequest, res: dict, log: bool = True) -> PredictionResponse:
    rid = str(uuid.uuid4())
    if log:                                   # без журналу id тимчасовий: фідбек на нього дасть 404, у чергу він не потрапить
        db.log_prediction(rid, item.model_dump(), res, p.version)
    return PredictionResponse(request_id=rid, score=res["score"], label=res["label"], uncertainty=res["uncertainty"],
                              meta=Meta(model_version=p.version, fallback_applied=res["fallback_applied"],
                                        ood_log_density=res["ood_log_density"]))


# --------------------------------------------------------------------------- #
# Ендпоінти
# --------------------------------------------------------------------------- #
@router.get("/health")
def health(request: Request):
    p = request.app.state.predictor
    if p is None:
        return _error(503, "not_ready", "Модель не завантажена", request.app.state.contract_errors)
    return {"status": "ok", "model_version": p.version, "run_id": p.run_id, "ood": p.ood is not None}


@router.post("/v1/predict", response_model=PredictionResponse)
def predict(item: PredictionRequest, request: Request):
    p = _predictor(request)
    res = p.predict([item.model_dump()])[0]
    return _respond(p, item, res)


@router.post("/v1/predict_batch", response_model=BatchResponse, description=f"До {BATCH_LIMIT} запитів")
def predict_batch(batch: BatchRequest, request: Request):
    p = _predictor(request)
    results = p.predict([it.model_dump() for it in batch.items])
    return BatchResponse(items=[_respond(p, it, r) for it, r in zip(batch.items, results)])


@router.post("/v1/feedback", response_model=FeedbackResponse)
def feedback(fb: FeedbackRequest):
    if not db.store_feedback(str(fb.request_id), fb.y_true):
        raise HTTPException(404, f"Невідомий request_id {fb.request_id}")
    return FeedbackResponse(request_id=fb.request_id, stored=True)


@router.post("/v1/explain", response_model=ExplainResponse)
def explain(item: PredictionRequest, request: Request, log: bool = True):
    """Прогноз + SHAP + LIME. ``log=false`` — пояснення вже залогованого запиту (черга розмітки):
    без нового запису в журналі, інакше той самий клієнт з'явився б у черзі ще раз."""
    p = _predictor(request)
    rec = item.model_dump()
    res = p.predict([rec])[0]
    pr = _respond(p, item, res, log=log)
    ex = request.app.state.explainer.explain(p.features([rec])[0])
    return ExplainResponse(prediction=pr, **ex)


@router.get("/v1/model")
def model_info(request: Request):
    p = _predictor(request)
    keys = ["model_name", "tag", "features", "threshold", "model_family", "uncertainty", "non_ml_rule", "ood"]
    return {"model_version": p.version, "run_id": p.run_id, **{k: p.meta.get(k) for k in keys}}


@router.get("/v1/label_queue")
def label_queue(request: Request, size: int = 20):
    """Трек 4A: черга розмітки з журналу (без фідбеку, без OOD), за спаданням невизначеності."""
    from src.observability.label_queue import build_queue

    return build_queue(db.db_path(), size=size, predictor=request.app.state.predictor)


def create_app() -> FastAPI:
    """Фабрика: окремий екземпляр зі своїм станом (тести зі зіпсованим контрактом не чіпають основний)."""
    application = FastAPI(title="ChurnGuard API", version="1.0",
                          description="Ризик відтоку клієнта e-commerce, рекомендація утримання, невизначеність і "
                                      "пояснення. Остаточне рішення ухвалює менеджер.", lifespan=lifespan)
    application.add_exception_handler(RequestValidationError, _validation)
    application.add_exception_handler(HTTPException, _http)
    application.add_exception_handler(Exception, _unhandled)
    application.include_router(router)
    return application


app = create_app()
