"""Tests for the MLP training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.prepare_dataset import (
    MODEL_FEATURES,
    SplitRatios,
    prepare_dataset,
)
from src.training.train_mlp import run_mlp_training


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


def test_mlp_training_pipeline_smoke_run(sample_dataset: Path, tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "experiments"

    results = run_mlp_training(
        data_dir=sample_dataset,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=8,
        learning_rate=0.00005,
        seed=42,
    )

    assert np.isfinite(results["final_train_loss"])
    assert np.isfinite(results["final_train_acc"])
    assert np.isfinite(results["final_val_loss"])
    assert np.isfinite(results["final_val_acc"])

    assert Path(results["best_checkpoint"]).exists()
    assert Path(results["final_checkpoint"]).exists()
    assert Path(results["history_json"]).exists()
    assert Path(results["metadata_json"]).exists()
    assert Path(results["run_config_json"]).exists()
    assert (output_dir / "training_log.csv").exists()

    metadata = json.loads(Path(results["metadata_json"]).read_text(encoding="utf-8"))
    assert metadata["model_name"] == "MLP-Baseline"
    assert metadata["optimizer"] == "Adam"
    assert metadata["learning_rate"] == 0.00005
    assert metadata["batch_size"] == 8
    assert metadata["epochs_requested"] == 1
