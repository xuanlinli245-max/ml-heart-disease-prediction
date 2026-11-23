"""Single-record inference from the saved pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURES


class Predictor:
    def __init__(self, model_path: str | Path = "models/model.joblib"):
        self.model = joblib.load(model_path)

    def predict(self, record: dict) -> dict:
        missing = [f for f in FEATURES if f not in record]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X = pd.DataFrame([{f: record[f] for f in FEATURES}])
        proba = float(self.model.predict_proba(X)[0, 1])
        return {"probability": round(proba, 4), "prediction": int(proba >= 0.5)}
