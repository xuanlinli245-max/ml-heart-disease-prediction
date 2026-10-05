"""Определение моделей (LR, KNN, SVC, VotingClassifier)."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.utils import setup_logger

logger = setup_logger(__name__)


def create_logistic_regression(params: dict[str, Any]) -> LogisticRegression:
    """Создаёт модель LogisticRegression."""
    model = LogisticRegression(**params)
    logger.info("Created LogisticRegression with params: %s", params)
    return model


def create_knn(params: dict[str, Any]) -> KNeighborsClassifier:
    """Создаёт модель KNeighborsClassifier."""
    model = KNeighborsClassifier(**params)
    logger.info("Created KNeighborsClassifier with params: %s", params)
    return model


def create_svc(params: dict[str, Any]) -> SVC:
    """Создаёт модель SVC."""
    model = SVC(**params)
    logger.info("Created SVC with params: %s", params)
    return model


def create_voting_classifier(
    estimators: list[tuple[str, Any]],
    voting: str = "soft",
    weights: list[float] | None = None,
) -> VotingClassifier:
    """Создаёт ансамбль VotingClassifier.

    Args:
        estimators: Список кортежей (имя, модель).
        voting: 'hard' или 'soft'.
        weights: Веса моделей.

    Returns:
        Обученный VotingClassifier.
    """
    model = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
    )
    logger.info(
        "Created VotingClassifier (voting=%s, weights=%s)", voting, weights
    )
    return model