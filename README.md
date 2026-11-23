# Heart Disease Prediction — Task 1

Production-ориентированная версия учебного ML-проекта: предсказание риска сердечно-сосудистого
заболевания по 11 клиническим признакам.

> Учебный проект. Не медицинское изделие и не замена врачу.

## Что было исправлено

| Проблема в старой версии | Исправление |
|---|---|
| Удаление выбросов и `LabelEncoder` fit на всём датасете **до** train/test split | Сначала `split_data`, все преобразования — внутри `Pipeline`, fit только на train |
| Сохранялся только классификатор | `joblib` сохраняет весь Pipeline (clean + preprocessing + model) |
| Жёсткие пути и дублирование | Путь в `configs/config.yaml`, один pipeline, модули `src/` |
| Не было тестов, pre-commit, Poetry | 6 тестов, `.pre-commit-config.yaml`, `pyproject.toml` + `poetry.toml`, ruff/black |

## Данные

- 918 пациентов, 11 признаков, цель `HeartDisease` (0/1).
- Файл: `data/raw/heart.csv`; источник и схема — `data/README.md`.
- `Cholesterol = 0` и `RestingBP = 0` трактуются как пропуски.

## Структура

```text
configs/config.yaml       параметры
data/raw/heart.csv        данные
src/data.py               загрузка, валидация, split, fingerprint
src/preprocessing.py      ZeroToNaN + ColumnTransformer + VotingClassifier
src/evaluate.py           accuracy / precision / recall / F1 / ROC-AUC
src/train.py              CV, обучение, сохранение артефактов
src/predict.py            inference по одному пациенту
tests/                    6 тестов
docs/refactor_before_after.md  что показывать на защите
models/model.joblib       создаётся после обучения
reports/metrics.json      создаётся после обучения
```

## Быстрый старт

```bash
# Poetry (рекомендуется; .venv создаётся внутри проекта)
poetry install
poetry run python -m src.train --config configs/config.yaml
poetry run pytest -q

# Или pip
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --config configs/config.yaml
pytest -q
```

## Результаты

5-fold CV ROC-AUC: **0.9309 ± 0.0266**. Held-out test (184 пациента):

| accuracy | precision | recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.8804 | 0.9082 | 0.8725 | 0.8900 | 0.9301 |

## Что сдавать (Task 1)

1. Скриншоты четырёх тестов hh.ru: ML, Python, SQL, Docker/Git/Linux/OOP.
2. Ссылка на этот GitHub-репозиторий.
3. В репозитории видно: `src/` после рефакторинга, `pyproject.toml`, `poetry.toml`,
   `.pre-commit-config.yaml`, `tests/`, `README.md`.

Проверка:

```bash
poetry run pre-commit run --all-files
poetry run ruff check src tests
poetry run black --check src tests
poetry run pytest -q
```

## Что показывать на защите

Открыть `src/train.py` и показать порядок:

```python
X_train, X_test, y_train, y_test = split_data(df)   # split first
model = build_pipeline()
model.fit(X_train, y_train)                         # preprocessing inside, train only
joblib.dump(model, "models/model.joblib")
```

Фраза:

> «В старой версии preprocessing обучался до split — это утечка. В новой split первый, а весь
> preprocessing находится внутри Pipeline и обучается только на train.»

Подробнее — `docs/refactor_before_after.md`.
