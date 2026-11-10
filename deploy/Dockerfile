FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONIOENCODING=utf-8 \
    MLFLOW_DISABLE_AGENT_HINT=1 PYTHONPATH=/app
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt deploy/requirements.txt
RUN grep -v -E "^(dvc|PyALE)==" deploy/requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY src/ src/
COPY config/ config/
COPY models/pipeline.onnx models/pipeline.skops models/preprocessor.skops models/shap_background.npy models/model_meta.json models/
COPY models/ood/ models/ood/
COPY reports/ reports/
COPY deploy/seed_registry.py deploy/seed_registry.py

EXPOSE 8000 8501 5000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]