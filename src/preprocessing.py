"""Preprocessing and the single train/serve pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL, NUMERIC


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Cholesterol/RestingBP = 0 is a missing-value code, not a real measurement."""

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in ("Cholesterol", "RestingBP"):
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return X


def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ]
    )
    classifier = VotingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=random_state
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    min_samples_leaf=15,
                    l2_regularization=0.1,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ],
        voting="soft",
        weights=[0.3, 0.4, 0.3],
        n_jobs=1,
    )
    return Pipeline([("clean", ZeroToNaN()), ("preprocess", preprocessor), ("model", classifier)])
