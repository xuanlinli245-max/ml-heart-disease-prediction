"""Общие утилиты: логирование, работа с путями и конфигами."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml


def setup_logger(
    name: str = "heart_disease",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Настраивает и возвращает логгер.

    Args:
        name: Имя логгера.
        log_file: Путь к файлу лога (опционально).
        level: Уровень логирования.

    Returns:
        Настроенный объект Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Избегаем дублирования хендлеров при повторном вызове
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Хендлер для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Хендлер для файла
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Загружает YAML-конфиг.

    Args:
        config_path: Путь к YAML-файлу.

    Returns:
        Словарь с параметрами конфигурации.

    Raises:
        FileNotFoundError: Если файл конфига не найден.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config