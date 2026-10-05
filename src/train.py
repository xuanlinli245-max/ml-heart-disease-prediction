"""Главный скрипт обучения."""
from __future__ import annotations

import pickle
from pathlib import Path

from src.data_loader import load_data
from src.evaluate import compute_classification_metrics, print_metrics
from src.models import (
    create_knn,
    create_logistic_regression,
    create_svc,
    create_voting_classifier,
)
from src.preprocessing import (
    encode_categorical_features,
    remove_outliers,
    scale_features,
    split_features_target,
    split_train_test,
)
from src.utils import load_config, setup_logger

logger = setup_logger("heart_disease.train", log_file="logs/train.log")


def main(config_path: str = "configs/config.yaml") -> None:
    """Основной пайплайн обучения."""
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # 1. Загрузка конфига
    config = load_config(config_path)

    # 2. Загрузка данных
    df = load_data(config["data"]["raw_path"])

    # 3. Удаление выбросов
    df = remove_outliers(
        df,
        features=config["preprocessing"]["numerical_columns"],
        threshold=config["preprocessing"]["outlier_threshold"],
    )

    # 4. Кодирование категориальных признаков
    df, _ = encode_categorical_features(
        df, columns=config["preprocessing"]["categorical_columns"]
    )

    # 5. Разделение на X и y
    X, y = split_features_target(df, target_column=config["data"]["target_column"])

    # 6. Train/test split
    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # 7. Масштабирование
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 8. Создание базовых моделей
    lr = create_logistic_regression(config["models"]["logistic_regression"])
    knn = create_knn(config["models"]["knn"])
    svc = create_svc(config["models"]["svc"])

    # 9. VotingClassifier
    voting = create_voting_classifier(
        estimators=[("lr", lr), ("knn", knn), ("svc", svc)],
        voting=config["voting"]["voting_type"],
        weights=config["voting"]["weights"],
    )

    # 10. Обучение
    logger.info("Training VotingClassifier...")
    voting.fit(X_train_scaled, y_train)

    # 11. Предсказание
    y_pred = voting.predict(X_test_scaled)
    y_proba = voting.predict_proba(X_test_scaled)

    # 12. Оценка
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics)

    # 13. Сохранение модели
    model_path = Path(config["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(voting, f)
    logger.info("Model saved to %s", model_path)

    logger.info("Training pipeline finished successfully.")


if __name__ == "__main__":
    main()