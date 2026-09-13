"""Tests for XGBoost training and evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.evaluate_xgboost import evaluate_xgboost
from src.models.xgboost_model import build_xgboost_classifier
from src.preprocessing.prepare_dataset import (
    MODEL_FEATURES,
    SplitRatios,
    prepare_dataset,
)
from src.training.train_xgboost import run_xgboost_training


def make_sample_csv(path: Path, rows: list[str]) -> None:
    import pandas as pd
    columns = list(MODEL_FEATURES) + ["label"]
    records = []
    for index, label in enumerate(rows):
        record = {
            feature: float(index * 10.0 + feat_idx + 1.0)
            for feat_idx, feature in enumerate(MODEL_FEATURES)
        }
        record["label"] = label
        records.append(record)
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    labels = [
        "BenignTraffic",
        "DictionaryBruteForce",
        "DDoS-TCP_Flood",
        "DoS-UDP_Flood",
        "Mirai-greeth_flood",
        "BrowserHijacking",
        "DNS_Spoofing",
        "XSS",
    ] * 5
    make_sample_csv(raw_dir / "sample.csv", labels)
    processed_dir = tmp_path / "processed"
    scaler_path = tmp_path / "scaler.pkl"

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        scaler_path=scaler_path,
        chunksize=10,
        shard_size=20,
        ratios=SplitRatios(0.6, 0.2, 0.2),
        seed=42,
    )
    return processed_dir


def test_xgboost_training_and_evaluation_pipeline(sample_dataset: Path, tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "experiments"
    results_dir = tmp_path / "results"

    # Train
    train_results = run_xgboost_training(
        data_dir=sample_dataset,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        n_estimators=10,
        learning_rate=0.1,
        max_depth=4,
        seed=42,
    )

    assert Path(train_results["checkpoint_model_json"]).exists()
    assert Path(train_results["metadata_json"]).exists()

    # Record test file modification time
    test_file = next((sample_dataset / "test").glob("part-*.npz"))
    mtime_before = test_file.stat().st_mtime

    # Evaluate
    eval_results = evaluate_xgboost(
        checkpoint_model_path=Path(train_results["checkpoint_model_json"]),
        test_dir=sample_dataset / "test",
        output_dir=results_dir,
    )

    # Confirm test data was untouched
    mtime_after = test_file.stat().st_mtime
    assert mtime_before == mtime_after

    metrics = eval_results["test_metrics"]
    assert np.isfinite(metrics["test_accuracy"])
    assert np.isfinite(metrics["test_log_loss"])
    assert np.isfinite(metrics["f1_macro"])
    assert metrics["num_classes"] == 7

    # Verify generated output files
    assert (results_dir / "test_metrics.json").exists()
    assert (results_dir / "classification_report.csv").exists()
    assert (results_dir / "confusion_matrix.csv").exists()
    assert (results_dir / "confusion_matrix.png").exists()
    assert (results_dir / "feature_importance.csv").exists()
    assert (results_dir / "feature_importance.png").exists()
    assert (results_dir / "test_predictions.csv").exists()
    assert (results_dir / "evaluation_summary.json").exists()
    assert (results_dir / "test_evaluation_report.md").exists()
