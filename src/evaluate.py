"""Метрики оценки моделей."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import setup_logger

logger = setup_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Вычисляет основные метрики классификации.

    Args:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.
        y_proba: Предсказанные вероятности (для ROC-AUC).

    Returns:
        Словарь с метриками: accuracy, precision, recall, f1, roc_auc (если есть proba).
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        # y_proba может быть либо 1D (вероятность класса 1), либо 2D (N, 2)
        if y_proba.ndim == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_pos))

    logger.info("Metrics: %s", metrics)
    return metrics


def print_metrics(metrics: dict[str, float]) -> None:
    """Красиво печатает метрики в лог."""
    logger.info("=" * 40)
    logger.info("Model Evaluation Results:")
    for name, value in metrics.items():
        logger.info("  %-10s: %.4f", name, value)
    logger.info("=" * 40)