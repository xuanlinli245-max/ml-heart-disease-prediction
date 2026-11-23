"""Train the model and write reproducible metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import fingerprint, load_data, split_data
from src.evaluate import metrics, positive_scores
from src.preprocessing import build_pipeline

ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("task1")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main(config_path: str = "configs/config.yaml") -> None:
    config = yaml.safe_load(resolve(config_path).read_text(encoding="utf-8"))
    df = load_data(resolve(config["data"]["path"]))
    X_train, X_test, y_train, y_test = split_data(
        df, config["data"]["test_size"], config["data"]["random_state"]
    )
    log.info("train=%s test=%s", X_train.shape, X_test.shape)

    model = build_pipeline(config["data"]["random_state"])
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=True,
        random_state=config["data"]["random_state"],
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    log.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    model.fit(X_train, y_train)
    y_score = positive_scores(model, X_test)
    y_pred = (y_score >= 0.5).astype(int)
    result = {
        "dataset_fingerprint": fingerprint(df),
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "test": metrics(y_test, y_pred, y_score),
    }

    model_path = resolve(config["output"]["model_path"])
    metrics_path = resolve(config["output"]["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== TEST METRICS @0.5 ===")
    for key, value in result["test"].items():
        print(f"{key:>10}: {value:.4f}")
    print(f"model saved: {model_path}")
    print(f"metrics saved: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
