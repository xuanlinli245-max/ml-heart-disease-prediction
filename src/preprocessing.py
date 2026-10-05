"""Предобработка данных: выбросы, кодирование, разделение, скейлинг."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import setup_logger

logger = setup_logger(__name__)


def find_outliers_iqr(
    df: pd.DataFrame,
    features: list[str],
    threshold: int = 1,
) -> list[int]:
    """Находит индексы строк, содержащих более `threshold` выбросов (метод IQR)."""
    outlier_counter: Counter = Counter()

    for column in features:
        q1 = np.percentile(df[column], 25)
        q3 = np.percentile(df[column], 75)
        iqr = q3 - q1
        step = 1.5 * iqr

        outlier_indices = df[
            (df[column] < q1 - step) | (df[column] > q3 + step)
        ].index
        outlier_counter.update(outlier_indices)

    multiple_outliers = [
        idx for idx, count in outlier_counter.items() if count > threshold
    ]

    logger.info("Found %d outlier rows (threshold=%d)", len(multiple_outliers), threshold)
    return multiple_outliers


def remove_outliers(
    df: pd.DataFrame,
    features: list[str],
    threshold: int = 1,
) -> pd.DataFrame:
    """Удаляет выбросы из DataFrame."""
    outliers = find_outliers_iqr(df, features, threshold)
    df_clean = df.drop(outliers, axis=0).reset_index(drop=True)
    logger.info("Data shape after outlier removal: %s", df_clean.shape)
    return df_clean


def encode_categorical_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Кодирует категориальные признаки с помощью LabelEncoder."""
    df_encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in columns:
        if col not in df_encoded.columns:
            logger.warning("Column '%s' not found, skipping.", col)
            continue

        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col])
        encoders[col] = encoder
        logger.info("Encoded column '%s' -> %s", col, list(encoder.classes_))

    return df_encoded, encoders


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Разделяет DataFrame на признаки (X) и целевую переменную (y)."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    logger.info("Features shape: %s, target shape: %s", X.shape, y.shape)
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделяет данные на обучающую и тестовую выборки."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Train/Test split: X_train=%s, X_test=%s", X_train.shape, X_test.shape
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Масштабирует признаки с помощью StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    logger.info("Features scaled with StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler