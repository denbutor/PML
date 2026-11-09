---

### Файл обкладинки проєкту `README.md`

================================================  
FILE: `README.md`  
================================================
```markdown
# Churn-Insight-ML — Прогнозування відтоку клієнтів E-Commerce

Наскрізний навчальний проєкт з прикладного машинного навчання за методологіями **CRISP-ML(Q)** та **REFORMS**.

- **Домен:** E-commerce customer churn
- **Датасет:** Kaggle Ecommerce Customer Churn Analysis (5074 клієнти після дедуплікації)
- **Сенситивні ознаки:** `CityTier` та `Gender`

## Структура репозиторію

- `config/` — YAML-конфігурації обробки та моделей.
- `src/` — Модульний вихідний код пайплайнів, кодувальників та екстракторів.
- `reports/` — Згенеровані звіти та зведені таблиці експериментів.
- `tests/` — Модульні автотести PyTest.

## Швидкий запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Перевірка цілісності пайплайнів
python verify_pipelines.py

# Запуск автотестів
pytest -q