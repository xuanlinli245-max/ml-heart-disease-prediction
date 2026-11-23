"""Training contract: metrics file is created and sane."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.train import main

ROOT = Path(__file__).resolve().parents[1]


def test_training_writes_metrics():
    old = Path.cwd()
    os.chdir(ROOT)
    try:
        main("configs/config.yaml")
    finally:
        os.chdir(old)
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert metrics["test"]["roc_auc"] > 0.75
    assert metrics["test"]["recall"] > 0.70
    assert (ROOT / "models" / "model.joblib").exists()
