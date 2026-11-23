# Refactor: before / after

## 1. Old `train.py` — leakage

```python
df = load_data(...)
df = remove_outliers(df, numerical_columns)          # fitted on all data
df, encoders = encode_categorical_features(df)       # fitted on all data
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test, scaler = scale_features(X_train, X_test)
model.fit(X_train, y_train)
pickle.dump(model, "model.pkl")                      # only classifier
```

Problems:
- test distribution leaks into outlier removal and encoding;
- inference cannot reproduce encoders/scaler;
- hard-coded paths and duplicated logic.

## 2. New `src/train.py` — split first

```python
df = load_data(config["data"]["path"])
X_train, X_test, y_train, y_test = split_data(df)    # split first
model = build_pipeline()
model.fit(X_train, y_train)                           # preprocessing only on train
joblib.dump(model, "models/model.joblib")             # whole pipeline
```

## 3. New `src/preprocessing.py` — pipeline

```python
Pipeline([
    ("clean", ZeroToNaN()),
    ("preprocess", ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])),
    ("model", VotingClassifier(...)),
])
```

## What to show on the screen

1. `src/train.py` — the order: `split_data` first, then `fit`.
2. `src/preprocessing.py` — all preprocessing inside the pipeline.
3. `tests/test_data.py` — disjoint indices.
4. `pyproject.toml` + `.pre-commit-config.yaml` — reproducible environment and checks.

One sentence:

> «Раньше preprocessing обучался до split — это утечка. Теперь split первый, а весь preprocessing
> находится внутри Pipeline и обучается только на train.»
