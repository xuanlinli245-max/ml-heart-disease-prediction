"""Load, validate and split the heart-disease dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "HeartDisease"
NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = NUMERIC + CATEGORICAL

ALLOWED = {
    "Sex": {"M", "F"},
    "ChestPainType": {"ATA", "NAP", "ASY", "TA"},
    "FastingBS": {0, 1},
    "RestingECG": {"Normal", "ST", "LVH"},
    "ExerciseAngina": {"N", "Y"},
    "ST_Slope": {"Up", "Flat", "Down"},
}


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary 0/1 without gaps")
    for col, values in ALLOWED.items():
        unknown = set(df[col].dropna().unique()) - values
        if unknown:
            raise ValueError(f"Unexpected values in {col}: {sorted(unknown)}")
    if (df[["Age", "RestingBP", "Cholesterol", "MaxHR"]] < 0).any().any():
        raise ValueError("Age/BP/Cholesterol/MaxHR must be non-negative")


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split first; all preprocessing is fitted later on the training part only."""
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]
