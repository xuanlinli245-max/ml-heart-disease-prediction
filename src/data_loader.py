"""Загрузка данных из CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)


def load_data(path: str | Path) -> pd.DataFrame:
    """Загружает CSV-файл с данными.

    Args:
        path: Путь к CSV-файлу.

    Returns:
        DataFrame с данными.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если DataFrame пустой.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Data file is empty: {path}")

    logger.info("Loaded data shape: %s", df.shape)
    return df