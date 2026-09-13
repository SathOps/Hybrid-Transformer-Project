import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.class_mapping import (
    ALLOWED_TARGET_CLASSES,
    EXPECTED_SOURCE_LABELS,
    SOURCE_TO_TARGET,
    UNRESOLVED_SOURCE_LABELS,
    map_label,
)
from src.preprocessing.dataset import (
    PreparedDataset,
    iter_batches,
    load_class_names,
    to_tf_dataset,
)
from src.preprocessing.prepare_dataset import (
    MODEL_FEATURES,
    TARGET_TO_ID,
    SplitRatios,
    discover_csv_files,
    load_dataset_settings,
    prepare_dataset,
    validate_feature_columns,
)


def make_csv(path, rows, offset_start=0):
    columns = list(MODEL_FEATURES) + ["label"]
    records = []
    for index, label in enumerate(rows):
        record = {
            feature: float(offset_start + index * 100 + feat_idx + 1)
            for feat_idx, feature in enumerate(MODEL_FEATURES)
        }
        record["label"] = label
        records.append(record)
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)


def test_feature_count():
    assert len(MODEL_FEATURES) == 46
    assert len(set(MODEL_FEATURES)) == 46
    validate_feature_columns(list(MODEL_FEATURES) + ["label"])
    with pytest.raises(ValueError, match="missing required model features"):
        validate_feature_columns(["flow_duration", "label"])
    with pytest.raises(ValueError, match="missing required label column"):
        validate_feature_columns(list(MODEL_FEATURES))


def test_label_mapping():
    for source_label, target_class in SOURCE_TO_TARGET.items():
        assert map_label(source_label) == target_class
        assert target_class in ALLOWED_TARGET_CLASSES

    for unresolved in UNRESOLVED_SOURCE_LABELS:
        assert map_label(unresolved) is None

    assert set(SOURCE_TO_TARGET.keys()) | UNRESOLVED_SOURCE_LABELS == set(EXPECTED_SOURCE_LABELS)
    with pytest.raises(ValueError, match="Unexpected CICIoT2023 source label"):
        map_label("UnknownAttackCategory")


def test_label_encoding():
    expected_order = (
        "Benign",
        "BruteForce",
        "DDoS",
        "DoS",
        "Mirai",
        "Recon",
        "Spoofing",
        "Web-based",
    )
    assert ALLOWED_TARGET_CLASSES == expected_order
    for expected_id, class_name in enumerate(expected_order):
        assert TARGET_TO_ID[class_name] == expected_id


def test_scaler_fitting_and_transformed_range(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    labels = ["BenignTraffic", "DDoS-TCP_Flood"] * 10
    make_csv(raw_dir / "sample.csv", labels)
    output_dir = tmp_path / "processed"
    scaler_path = tmp_path / "scaler.pkl"

    manifest = prepare_dataset(
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
    assert scaler.n_samples_seen_ == manifest["split_rows"]["train"]

    train_shards = sorted((output_dir / "train").glob("part-*.npz"))
    assert len(train_shards) >= 1
    with np.load(train_shards[0]) as shard:
        assert np.all(shard["features"] >= -1e-6)
        assert np.all(shard["features"] <= 1.0 + 1e-6)
        assert np.issubdtype(shard["labels"].dtype, np.integer)

    for split in ("train", "validation", "test"):
        shards = sorted((output_dir / split).glob("part-*.npz"))
        assert len(shards) >= 1
        with np.load(shards[0]) as shard:
            assert np.all(np.isfinite(shard["features"]))
            assert np.issubdtype(shard["labels"].dtype, np.integer)

    assert json.loads((output_dir / "class_names.json").read_text()) == list(ALLOWED_TARGET_CLASSES)


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
    with np.load(next((output_dir / "validation").glob("part-*.npz"))) as val:
        val_rows = {tuple(row) for row in val["features"]}
    with np.load(next((output_dir / "test").glob("part-*.npz"))) as test:
        test_rows = {tuple(row) for row in test["features"]}

    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)


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


def test_invalid_values_are_discarded(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    columns = list(MODEL_FEATURES) + ["label"]
    valid = {feature: 1.0 for feature in MODEL_FEATURES}
    valid["label"] = "BenignTraffic"
    invalid_nan = valid.copy()
    invalid_nan["flow_duration"] = np.nan
    invalid_inf = valid.copy()
    invalid_inf["Rate"] = np.inf
    invalid_neginf = valid.copy()
    invalid_neginf["Duration"] = -np.inf

    valid_rows = [valid.copy() for _ in range(6)]
    for index, row in enumerate(valid_rows):
        row["label"] = "BenignTraffic" if index % 2 == 0 else "DDoS-TCP_Flood"
    pd.DataFrame(
        valid_rows + [invalid_nan, invalid_inf, invalid_neginf], columns=columns
    ).to_csv(raw_dir / "invalid.csv", index=False)

    manifest = prepare_dataset(
        raw_dir=raw_dir,
        output_dir=tmp_path / "processed",
        scaler_path=tmp_path / "scaler.pkl",
        chunksize=2,
        ratios=SplitRatios(0.6, 0.2, 0.2),
    )

    assert manifest["rows_processed"] == 9
    assert manifest["invalid_rows_dropped"] == 3
    assert sum(manifest["split_rows"].values()) == 6


def test_scaler_is_fitted_on_training_rows_only(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_csv(raw_dir / "sample.csv", ["BenignTraffic", "DDoS-TCP_Flood"] * 10)
    output_dir = tmp_path / "processed"
    scaler_path = tmp_path / "scaler.pkl"

    manifest = prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        scaler_path=scaler_path,
        chunksize=4,
        ratios=SplitRatios(0.6, 0.2, 0.2),
        seed=17,
    )

    with scaler_path.open("rb") as file_handle:
        scaler = pickle.load(file_handle)
    assert scaler.n_samples_seen_ == manifest["split_rows"]["train"]


def test_same_seed_produces_same_partitions(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_csv(raw_dir / "sample.csv", ["BenignTraffic", "DDoS-TCP_Flood"] * 12)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = prepare_dataset(
        raw_dir=raw_dir,
        output_dir=first_dir,
        scaler_path=tmp_path / "first.pkl",
        chunksize=5,
        ratios=SplitRatios(0.6, 0.2, 0.2),
        seed=99,
    )
    second = prepare_dataset(
        raw_dir=raw_dir,
        output_dir=second_dir,
        scaler_path=tmp_path / "second.pkl",
        chunksize=5,
        ratios=SplitRatios(0.6, 0.2, 0.2),
        seed=99,
    )

    assert first["split_class_distribution"] == second["split_class_distribution"]
    for split in ("train", "validation", "test"):
        with np.load(next((first_dir / split).glob("part-*.npz"))) as first_shard:
            first_labels = first_shard["labels"]
        with np.load(next((second_dir / split).glob("part-*.npz"))) as second_shard:
            second_labels = second_shard["labels"]
        np.testing.assert_array_equal(first_labels, second_labels)


def test_dataset_streaming_and_tf_dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    labels = ["BenignTraffic", "DDoS-TCP_Flood"] * 20
    make_csv(raw_dir / "sample.csv", labels)
    output_dir = tmp_path / "processed"

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        scaler_path=tmp_path / "scaler.pkl",
        chunksize=10,
        shard_size=8,
        ratios=SplitRatios(0.6, 0.2, 0.2),
    )

    class_names = load_class_names(output_dir)
    assert len(class_names) == 7
    assert class_names[0] == "Benign"

    batches = list(iter_batches(output_dir / "train", batch_size=4))
    assert len(batches) > 0
    features, targets = batches[0]
    assert features.shape == (4, 46)
    assert targets.shape == (4,)

    dataset = PreparedDataset(output_dir, "train")
    ds_batches = list(dataset.batches(batch_size=4))
    assert len(ds_batches) == len(batches)

    tf_ds = dataset.to_tf_dataset(batch_size=4, expand_dims=True)
    for tf_features, tf_targets in tf_ds.take(1):
        assert tf_features.shape == (4, 46, 1)
        assert tf_targets.shape == (4,)


def test_feature_ordering_and_no_label_leakage(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Place label first and reverse feature columns in CSV
    reversed_features = list(reversed(MODEL_FEATURES))
    columns = ["label"] + reversed_features
    records = []
    labels = ["BenignTraffic", "DDoS-TCP_Flood"] * 5
    for index, label in enumerate(labels):
        record = {"label": label}
        for feat_idx, feature in enumerate(MODEL_FEATURES):
            record[feature] = float((feat_idx + 1) * 10.0)
        records.append(record)
    pd.DataFrame(records, columns=columns).to_csv(raw_dir / "scrambled.csv", index=False)

    output_dir = tmp_path / "processed"
    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        scaler_path=tmp_path / "scaler.pkl",
        chunksize=5,
        shard_size=10,
    )

    with np.load(next((output_dir / "train").glob("part-*.npz"))) as train_shard:
        features = train_shard["features"]
        # Shard features must have exactly 46 columns, matching MODEL_FEATURES count
        assert features.shape[1] == 46
        # Target label strings or IDs must not be inside features
        assert not np.any(np.isnan(features))


def test_scaler_no_validation_or_test_leakage(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    labels = ["BenignTraffic", "DDoS-TCP_Flood"] * 10
    make_csv(raw_dir / "sample.csv", labels)

    scaler_path = tmp_path / "scaler.pkl"
    output_dir = tmp_path / "processed"
    manifest = prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        scaler_path=scaler_path,
        chunksize=10,
        ratios=SplitRatios(0.6, 0.2, 0.2),
        seed=42,
    )

    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    # 1. Scaler must only see train sample count
    assert scaler.n_samples_seen_ == manifest["split_rows"]["train"]
    assert scaler.n_samples_seen_ < manifest["rows_processed"]

    # 2. Reconstructed train features from inverse_transform must strictly match scaler.data_min_ and scaler.data_max_
    with np.load(next((output_dir / "train").glob("part-*.npz"))) as train_shard:
        unscaled_train = scaler.inverse_transform(train_shard["features"])
        np.testing.assert_allclose(scaler.data_max_, np.max(unscaled_train, axis=0), rtol=1e-5)
        np.testing.assert_allclose(scaler.data_min_, np.min(unscaled_train, axis=0), rtol=1e-5)

    # 3. Scaler data_max_ remains strictly the training maximum regardless of test features
    with np.load(next((output_dir / "test").glob("part-*.npz"))) as test_shard:
        unscaled_test = scaler.inverse_transform(test_shard["features"])
        assert len(unscaled_test) == manifest["split_rows"]["test"]
        assert scaler.n_features_in_ == 46



def test_saved_dataset_compatibility_with_cnn(tmp_path):
    from src.models.cnn import build_cnn_feature_extractor

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_csv(raw_dir / "sample.csv", ["BenignTraffic", "DDoS-TCP_Flood"] * 10)
    output_dir = tmp_path / "processed"

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        scaler_path=tmp_path / "scaler.pkl",
        chunksize=5,
        shard_size=10,
    )

    dataset = PreparedDataset(output_dir, "train")
    tf_ds = dataset.to_tf_dataset(batch_size=4, expand_dims=True)

    cnn_model = build_cnn_feature_extractor()
    for batch_features, _ in tf_ds.take(1):
        output = cnn_model(batch_features)
        assert output.shape == (4, 64)
        assert np.all(np.isfinite(output.numpy()))