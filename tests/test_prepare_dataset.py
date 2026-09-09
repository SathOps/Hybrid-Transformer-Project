import pickle

import numpy as np
import pandas as pd

from src.preprocessing.class_mapping import map_label
from src.preprocessing.prepare_dataset import (
    MODEL_FEATURES,
    SplitRatios,
    load_dataset_settings,
    prepare_dataset,
)


def make_csv(path, rows):
    columns = list(MODEL_FEATURES) + ["label"]
    records = []
    for index, label in enumerate(rows):
        record = {feature: float(index + offset + 1) for offset, feature in enumerate(MODEL_FEATURES)}
        record["label"] = label
        records.append(record)
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)


def test_feature_count():
    assert len(MODEL_FEATURES) == 46
    assert len(set(MODEL_FEATURES)) == 46


def test_label_mapping():
    assert map_label("DDoS-TCP_Flood") == "DDoS"
    assert map_label("Backdoor_Malware") is None
    assert map_label("VulnerabilityScan") is None


def test_scaler_fitting_and_transformed_range(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    labels = ["BenignTraffic", "DDoS-TCP_Flood"] * 10
    make_csv(raw_dir / "sample.csv", labels)
    output_dir = tmp_path / "processed"
    scaler_path = tmp_path / "scaler.pkl"

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        scaler_path=scaler_path,
        chunksize=4,
        shard_size=8,
        ratios=SplitRatios(0.6, 0.2, 0.2),
    )

    with scaler_path.open("rb") as file_handle:
        scaler = pickle.load(file_handle)
    assert scaler.n_features_in_ == 46
    train_shards = sorted((output_dir / "train").glob("part-*.npz"))
    with np.load(train_shards[0]) as shard:
        assert np.all(shard["features"] >= 0)
        assert np.all(shard["features"] <= 1)


def test_train_test_separation(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_csv(raw_dir / "sample.csv", ["BenignTraffic", "DDoS-TCP_Flood"] * 15)
    output_dir = tmp_path / "processed"

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        scaler_path=tmp_path / "scaler.pkl",
        chunksize=5,
        shard_size=100,
        ratios=SplitRatios(0.6, 0.2, 0.2),
    )

    with np.load(next((output_dir / "train").glob("part-*.npz"))) as train:
        train_rows = {tuple(row) for row in train["features"]}
    with np.load(next((output_dir / "test").glob("part-*.npz"))) as test:
        test_rows = {tuple(row) for row in test["features"]}
    assert train_rows.isdisjoint(test_rows)


def test_development_mode_limits_records_per_class(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_csv(raw_dir / "sample.csv", ["BenignTraffic", "DDoS-TCP_Flood"] * 10)
    output_dir = tmp_path / "processed"

    manifest = prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        scaler_path=tmp_path / "scaler.pkl",
        chunksize=3,
        shard_size=100,
        ratios=SplitRatios(0.6, 0.2, 0.2),
        mode="development",
        max_records_per_class=2,
    )

    assert manifest["dataset_mode"] == "development"
    assert sum(manifest["split_rows"].values()) == 4


def test_configured_full_mode_has_no_development_limits(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset:\n  mode: full\n  development:\n    max_records_per_class: 2\n",
        encoding="utf-8",
    )

    assert load_dataset_settings(config_path) == ("full", None, None)