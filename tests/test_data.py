"""Data contract tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import FEATURES, TARGET, load_data, split_data, validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "heart.csv"


def test_load_and_schema():
    df = load_data(DATA)
    assert len(df) == 918
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert df[TARGET].isin([0, 1]).all()


def test_split_is_disjoint():
    df = load_data(DATA)
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_validation_rejects_bad_target():
    df = pd.read_csv(DATA)
    df.loc[0, TARGET] = 2
    with pytest.raises(ValueError):
        validate(df)
