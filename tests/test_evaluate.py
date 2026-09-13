"""Tests for the model evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.evaluate import evaluate_model
from src.models.cnn_transformer import build_model
from src.preprocessing.prepare_dataset import (
    MODEL_FEATURES,
    SplitRatios,
    prepare_dataset,
)


def make_sample_csv(path: Path, rows: list[str]) -> None:
    import pandas as pd
    columns = list(MODEL_FEATURES) + ["label"]
    records = []
    for index, label in enumerate(rows):
        record = {
            feature: float(index * 5.0 + feat_idx + 1.0)
            for feat_idx, feature in enumerate(MODEL_FEATURES)
        }
        record["label"] = label
        records.append(record)
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)


@pytest.fixture
def dummy_dataset_and_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
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

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / "best_model.keras"

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.save(str(ckpt_path))

    return processed_dir / "test", ckpt_path


def test_evaluate_model_pipeline(dummy_dataset_and_checkpoint: tuple[Path, Path], tmp_path: Path):
    test_dir, ckpt_path = dummy_dataset_and_checkpoint
    output_dir = tmp_path / "results"

    # Record test file modification time before evaluation
    test_file = next(test_dir.glob("part-*.npz"))
    mtime_before = test_file.stat().st_mtime

    results = evaluate_model(
        checkpoint_path=ckpt_path,
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=4,
    )

    # 1. Verification of output metrics structure
    metrics = results["test_metrics"]
    assert np.isfinite(metrics["test_accuracy"])
    assert np.isfinite(metrics["test_loss"])
    assert np.isfinite(metrics["f1_macro"])
    assert np.isfinite(metrics["f1_weighted"])
    assert metrics["num_classes"] == 7
    assert results["num_samples"] == 8

    # 2. Verification of output files creation
    assert (output_dir / "test_metrics.json").exists()
    assert (output_dir / "classification_report.csv").exists()
    assert (output_dir / "confusion_matrix.csv").exists()
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "test_predictions.csv").exists()
    assert (output_dir / "evaluation_summary.json").exists()
    assert (output_dir / "test_evaluation_report.md").exists()

    # 3. Confirm test dataset file was untouched
    mtime_after = test_file.stat().st_mtime
    assert mtime_before == mtime_after
