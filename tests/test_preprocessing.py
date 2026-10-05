"""Тесты для модуля preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import (
    encode_categorical_features,
    remove_outliers,
    scale_features,
    split_features_target,
    split_train_test,
)


def test_remove_outliers() -> None:
    """Проверяет, что выбросы удаляются."""
    df = pd.DataFrame({
        "Age": [30, 35, 40, 45, 50, 200],
        "Cholesterol": [180, 200, 210, 220, 230, 240],
    })
    result = remove_outliers(df, ["Age"], threshold=0)
    assert 200 not in result["Age"].values
    assert len(result) < len(df)


def test_encode_categorical_features() -> None:
    """Проверяет, что категориальные признаки кодируются в числа."""
    df = pd.DataFrame({
        "Sex": ["M", "F", "M", "F"],
        "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    })
    df_encoded, encoders = encode_categorical_features(
        df, ["Sex", "ChestPainType"]
    )
    assert df_encoded["Sex"].dtype in (np.int32, np.int64, int)
    assert df_encoded["ChestPainType"].dtype in (np.int32, np.int64, int)
    assert set(encoders.keys()) == {"Sex", "ChestPainType"}


def test_split_features_target() -> None:
    """Проверяет разделение на X и y."""
    df = pd.DataFrame({
        "Age": [30, 40, 50],
        "HeartDisease": [0, 1, 0],
    })
    X, y = split_features_target(df, "HeartDisease")
    assert "HeartDisease" not in X.columns
    assert list(y.values) == [0, 1, 0]


def test_split_train_test() -> None:
    """Проверяет корректность разделения на train/test."""
    X = pd.DataFrame({"a": range(100), "b": range(100)})
    y = pd.Series([0, 1] * 50)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=42
    )
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_scale_features() -> None:
    """Проверяет, что после масштабирования среднее ≈ 0."""
    X_train = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    X_test = pd.DataFrame({"a": [6], "b": [60]})
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert abs(X_train_scaled["a"].mean()) < 1e-9
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape