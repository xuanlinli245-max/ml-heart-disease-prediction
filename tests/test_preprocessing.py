"""Preprocessing tests: zeros are missing, pipeline outputs probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import FEATURES
from src.preprocessing import ZeroToNaN, build_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "heart.csv"


def test_zero_to_nan():
    df = pd.DataFrame({"Cholesterol": [0, 200], "RestingBP": [0, 120], "Age": [50, 60]})
    out = ZeroToNaN().fit_transform(df)
    assert np.isnan(out.loc[0, "Cholesterol"])
    assert np.isnan(out.loc[0, "RestingBP"])
    assert out.loc[1, "Cholesterol"] == 200


def test_pipeline_predicts_probabilities():
    df = pd.read_csv(DATA).sample(200, random_state=1)
    model = build_pipeline().fit(df[FEATURES], df["HeartDisease"])
    proba = model.predict_proba(df[FEATURES].head(5))[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()
