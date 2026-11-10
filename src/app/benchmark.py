"""p50 / p95 часу відповіді API (ЛР5, етап 4): /v1/predict, /v1/predict_batch (256 рядків), /v1/explain — по 100 запитів.

    python -m src.app.benchmark            → reports/latency.json

Сервіс піднімається як окремий процес uvicorn (реальний HTTP, модель @champion з MLflow), журнал — тимчасова БД,
щоб не засмічувати feedback.db. Запити — реальні клієнти test (holdout @ lab3-v1.0), перший прогрів не враховується.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.serving.data import load_lab5
from src.serving.features import REQUEST_FIELDS

N = 100
BATCH = 256
PORT = 8765


def machine() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(), "cpu": platform.processor(),
            "logical_cores": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            info["cpu"] = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                         capture_output=True, text=True, timeout=30).stdout.strip() or info["cpu"]
        except Exception:
            pass
    return info


def to_payloads(df) -> list[dict]:
    from src.observability.label_round import _normalize
    return [_normalize({k: (None if pd.isna(r[k]) else r[k]) for k in REQUEST_FIELDS}) for _, r in df.iterrows()]


def timed(client, path, payloads) -> dict:
    client.post(path, json=payloads[0])                     # прогрів
    ms = []
    for p in payloads:
        t0 = time.perf_counter()
        r = client.post(path, json=p)
        ms.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
    return {"n": len(ms), "p50_ms": round(float(np.percentile(ms, 50)), 1),
            "p95_ms": round(float(np.percentile(ms, 95)), 1), "mean_ms": round(float(np.mean(ms)), 1)}


def wait_ready(url, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("сервіс не піднявся")


def main():
    test = load_lab5()["test"]
    payloads = to_payloads(test)
    from src.app.schemas import PredictionRequest
    payloads = [p for p in payloads if _valid(PredictionRequest, p)]
    rng = np.random.default_rng(42)
    single = [payloads[i] for i in rng.choice(len(payloads), N, replace=False)]
    batches = [{"items": [payloads[j] for j in rng.choice(len(payloads), BATCH, replace=False)]} for _ in range(N)]

    with socket.socket() as s:
        busy = s.connect_ex(("127.0.0.1", PORT)) == 0
    if busy:
        raise SystemExit(f"порт {PORT} зайнятий")
    tmp = tempfile.mkdtemp()
    env = {**os.environ, "FEEDBACK_DB": os.path.join(tmp, "bench.db")}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.app.main:app", "--port", str(PORT), "--log-level", "warning"],
                            cwd=PROJECT_ROOT, env=env)
    url = f"http://127.0.0.1:{PORT}"
    try:
        health = wait_ready(url)
        with httpx.Client(base_url=url, timeout=120) as c:
            res = {"/v1/predict": timed(c, "/v1/predict", single),
                   f"/v1/predict_batch ({BATCH} рядків)": timed(c, "/v1/predict_batch", batches),
                   "/v1/explain": timed(c, "/v1/explain", single)}
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    out = {"model_version": health["model_version"], "run_id": health["run_id"], "source": os.environ.get("MODEL_SOURCE", "mlflow"),
           "client": "httpx, послідовні запити, той самий хост", "machine": machine(), "results": res}
    (PROJECT_ROOT / "reports" / "latency.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _valid(model, p) -> bool:
    try:
        model.model_validate(p)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
