"""Prepare CICIoT2023 data in streaming passes for later model training."""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import psutil
import yaml
from sklearn.preprocessing import MinMaxScaler

try:
    from .class_mapping import (
        ALLOWED_TARGET_CLASSES,
        EXPECTED_SOURCE_LABELS,
        SOURCE_TO_TARGET,
        UNRESOLVED_SOURCE_LABELS,
    )
except ImportError:
    from class_mapping import (  # type: ignore[no-redef]
        ALLOWED_TARGET_CLASSES,
        EXPECTED_SOURCE_LABELS,
        SOURCE_TO_TARGET,
        UNRESOLVED_SOURCE_LABELS,
    )


LOGGER = logging.getLogger(__name__)
LABEL_COLUMN = "label"
MODEL_FEATURES = (
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Magnitue",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
)

RAW_39_FEATURES = (
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
)

SPLIT_NAMES = ("train", "validation", "test")
TARGET_TO_ID = {name: index for index, name in enumerate(ALLOWED_TARGET_CLASSES)}
MAJORITY_CLASS_IDS = {TARGET_TO_ID["DDoS"], TARGET_TO_ID["DoS"], TARGET_TO_ID["Mirai"]}


class Log1pMinMaxScaler:
    """Scaler applying log1p(max(0, x)) transformation followed by MinMaxScaler."""

    def __init__(self) -> None:
        self.scaler = MinMaxScaler()

    @property
    def n_features_in_(self) -> int:
        return getattr(self.scaler, "n_features_in_", 46)

    @property
    def n_samples_seen_(self) -> int:
        return getattr(self.scaler, "n_samples_seen_", 0)

    @property
    def data_min_(self) -> np.ndarray:
        raw_min = getattr(self.scaler, "data_min_", np.zeros(46))
        return np.expm1(raw_min)

    @property
    def data_max_(self) -> np.ndarray:
        raw_max = getattr(self.scaler, "data_max_", np.ones(46))
        return np.expm1(raw_max)

    def partial_fit(self, X: np.ndarray, y: None = None) -> Log1pMinMaxScaler:
        if len(X) == 0:
            return self
        X_log = np.log1p(np.maximum(0.0, X, dtype=np.float32))
        self.scaler.partial_fit(X_log)
        return self

    def fit(self, X: np.ndarray, y: None = None) -> Log1pMinMaxScaler:
        return self.partial_fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, 46), dtype=np.float32)
        X_log = np.log1p(np.maximum(0.0, X, dtype=np.float32))
        return self.scaler.transform(X_log)

    def fit_transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        return self.partial_fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, 46), dtype=np.float32)
        X_log = self.scaler.inverse_transform(X)
        return np.expm1(X_log)


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0 for value in values) or not math.isclose(sum(values), 1.0):
            raise ValueError("train, validation, and test ratios must be positive and sum to 1")


@dataclass
class InspectionStats:
    files_processed: set[str]
    raw_rows: int = 0
    valid_rows: int = 0
    unresolved_rows: int = 0
    invalid_rows: int = 0
    limited_rows: int = 0

    def merge(self, other: InspectionStats) -> None:
        self.files_processed.update(other.files_processed)
        self.raw_rows += other.raw_rows
        self.valid_rows += other.valid_rows
        self.unresolved_rows += other.unresolved_rows
        self.invalid_rows += other.invalid_rows
        self.limited_rows += other.limited_rows


class DevelopmentLimiter:
    def __init__(
        self,
        max_records_per_class: int | None,
        max_records_per_file: int | None,
    ) -> None:
        for value in (max_records_per_class, max_records_per_file):
            if value is not None and value < 1:
                raise ValueError("development record limits must be positive")
        self.max_records_per_class = max_records_per_class
        self.max_records_per_file = max_records_per_file
        self.class_counts: Counter[int] = Counter()
        self.file_counts: Counter[str] = Counter()

    @property
    def is_complete(self) -> bool:
        if self.max_records_per_class is not None:
            if len(self.class_counts) >= len(ALLOWED_TARGET_CLASSES) and all(
                self.class_counts[target_id] >= self.max_records_per_class
                for target_id in range(len(ALLOWED_TARGET_CLASSES))
            ):
                return True
        return False

    def is_file_complete(self, csv_file: Path) -> bool:
        if self.is_complete:
            return True
        if self.max_records_per_file is not None:
            return self.file_counts[str(csv_file)] >= self.max_records_per_file
        return False

    def select(self, features: np.ndarray, labels: np.ndarray, csv_file: Path) -> tuple[np.ndarray, np.ndarray]:
        if self.max_records_per_class is None and self.max_records_per_file is None:
            return features, labels

        file_key = str(csv_file)
        keep = np.zeros(len(labels), dtype=bool)
        file_remaining = self.max_records_per_file
        if file_remaining is not None:
            file_remaining -= self.file_counts[file_key]
        for index, label in enumerate(labels):
            target_id = int(label)
            if file_remaining is not None and file_remaining <= 0:
                break
            if (
                self.max_records_per_class is not None
                and self.class_counts[target_id] >= self.max_records_per_class
            ):
                continue
            keep[index] = True
            self.class_counts[target_id] += 1
            if file_remaining is not None:
                file_remaining -= 1
                self.file_counts[file_key] += 1
        return features[keep], labels[keep]


def discover_csv_files(raw_dir: Path) -> list[Path]:
    target_dir = raw_dir / "ciciot2023" if (raw_dir / "ciciot2023").is_dir() else raw_dir
    files = sorted(path for path in target_dir.rglob("*.csv") if path.is_file())
    if (raw_dir / "ciciot2023").is_dir():
        files = [p for p in files if "sample_" not in p.name]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    return files


def validate_feature_columns(columns: list[str]) -> None:
    has_model_feats = all(col in columns for col in MODEL_FEATURES)
    has_raw_feats = all(col in columns for col in RAW_39_FEATURES)
    if not (has_model_feats or has_raw_feats):
        missing_model = [col for col in MODEL_FEATURES if col not in columns]
        missing_raw = [col for col in RAW_39_FEATURES if col not in columns]
        raise ValueError(
            f"Dataset is missing required model features. "
            f"Missing MODEL_FEATURES: {missing_model}; Missing RAW_39_FEATURES: {missing_raw}"
        )
    if has_model_feats and LABEL_COLUMN not in columns:
        raise ValueError(f"Dataset is missing required label column: {LABEL_COLUMN}")


def clean_chunk(chunk: pd.DataFrame, csv_file: Path) -> tuple[np.ndarray, np.ndarray, InspectionStats]:
    """Map labels and return only finite, complete rows from one CSV chunk."""
    if LABEL_COLUMN in chunk.columns:
        source_labels = chunk[LABEL_COLUMN].astype("string")
    else:
        folder_name = csv_file.parent.name
        source_labels = pd.Series([folder_name] * len(chunk), index=chunk.index, dtype="string")

    observed = set(source_labels.dropna().unique().tolist())
    unexpected = observed - set(EXPECTED_SOURCE_LABELS)
    if unexpected:
        raise ValueError(f"Unexpected source labels found: {sorted(unexpected)}")

    mapped_labels = source_labels.map(SOURCE_TO_TARGET)
    unresolved_mask = source_labels.isin(UNRESOLVED_SOURCE_LABELS)

    if all(feat in chunk.columns for feat in MODEL_FEATURES):
        numeric_df = chunk.loc[:, list(MODEL_FEATURES)].apply(pd.to_numeric, errors="coerce")
    elif all(feat in chunk.columns for feat in RAW_39_FEATURES):
        raw_df = chunk.loc[:, list(RAW_39_FEATURES)].apply(pd.to_numeric, errors="coerce")
        numeric_df = pd.DataFrame(index=chunk.index)
        for col in RAW_39_FEATURES:
            numeric_df[col] = raw_df[col]
        numeric_df["flow_duration"] = raw_df["IAT"] * raw_df["Number"]
        numeric_df["Duration"] = raw_df["IAT"] * raw_df["Number"]
        numeric_df["Srate"] = raw_df["Rate"]
        numeric_df["Drate"] = raw_df["Rate"] / 2.0
        numeric_df["urg_count"] = 0.0
        numeric_df["Magnitue"] = np.sqrt(raw_df["Min"]**2 + raw_df["Max"]**2 + raw_df["AVG"]**2)
        numeric_df["Radius"] = np.sqrt(raw_df["Std"]**2 + raw_df["Variance"])
        numeric_df["Covariance"] = raw_df["Std"] * raw_df["AVG"]
        numeric_df["Weight"] = raw_df["Number"]
        numeric_df = numeric_df.loc[:, list(MODEL_FEATURES)]
    else:
        raise ValueError(f"CSV file {csv_file} does not contain valid feature columns.")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    valid_mask = mapped_labels.notna() & numeric_df.notna().all(axis=1)

    stats = InspectionStats(
        files_processed=set(),
        raw_rows=len(chunk),
        valid_rows=int(valid_mask.sum()),
        unresolved_rows=int(unresolved_mask.sum()),
        invalid_rows=int((~valid_mask & ~unresolved_mask).sum()),
    )
    features = numeric_df.loc[valid_mask].to_numpy(dtype=np.float32)
    labels = mapped_labels.loc[valid_mask].map(TARGET_TO_ID).to_numpy(dtype=np.int64)
    return features, labels, stats


def iter_clean_chunks(
    csv_files: list[Path],
    chunksize: int,
    max_records_per_class: int | None = None,
    max_records_per_file: int | None = None,
) -> Iterator[tuple[Path, np.ndarray, np.ndarray, InspectionStats]]:
    limiter = DevelopmentLimiter(max_records_per_class, max_records_per_file)
    for csv_file in csv_files:
        if limiter.is_complete:
            break
        header = pd.read_csv(csv_file, nrows=0).columns.tolist()
        validate_feature_columns(header)
        for chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False):
            if limiter.is_file_complete(csv_file):
                break
            features, labels, stats = clean_chunk(chunk, csv_file)
            stats.files_processed.add(str(csv_file))
            valid_before_limit = len(labels)
            features, labels = limiter.select(features, labels, csv_file)
            stats.valid_rows = len(labels)
            stats.limited_rows = valid_before_limit - len(labels)
            yield csv_file, features, labels, stats
            if limiter.is_file_complete(csv_file):
                break


def count_valid_rows(
    csv_files: list[Path],
    chunksize: int,
    max_records_per_class: int | None = None,
    max_records_per_file: int | None = None,
) -> tuple[Counter[int], InspectionStats]:
    class_counts: Counter[int] = Counter()
    stats = InspectionStats(files_processed=set())
    last_logged = 0
    for _, _, labels, chunk_stats in iter_clean_chunks(
        csv_files, chunksize, max_records_per_class, max_records_per_file
    ):
        class_counts.update(labels.tolist())
        stats.merge(chunk_stats)
        if stats.raw_rows - last_logged >= 5_000_000:
            LOGGER.info("Pass 1 (counting): scanned %d rows...", stats.raw_rows)
            last_logged = stats.raw_rows
    return class_counts, stats


def split_and_subsample_masks(
    labels: np.ndarray,
    class_counts: Counter[int],
    class_seen: defaultdict[int, int],
    ratios: SplitRatios,
    seed: int = 42,
    class_orders: dict[int, np.ndarray] | None = None,
    max_majority_train_rows: int = 500_000,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Assign rows to splits and compute subsampling mask for training majority classes."""
    assignments = np.empty(labels.shape[0], dtype="U10")
    subsampled_train_keep = np.zeros(labels.shape[0], dtype=bool)
    if class_orders is None:
        class_orders = {}

    for target_class in np.unique(labels):
        positions = np.flatnonzero(labels == target_class)
        start = class_seen[int(target_class)]
        total = class_counts[int(target_class)]
        if int(target_class) not in class_orders:
            p = np.random.default_rng(seed + int(target_class)).permutation(total)
            inv = np.empty(total, dtype=np.int64)
            inv[p] = np.arange(total)
            class_orders[int(target_class)] = inv

        class_positions = class_orders[int(target_class)][
            np.arange(start, start + len(positions))
        ]
        train_end = int(math.floor(total * ratios.train))
        validation_end = int(math.floor(total * (ratios.train + ratios.validation)))

        train_mask_cls = class_positions < train_end
        validation_mask_cls = (class_positions >= train_end) & (class_positions < validation_end)
        test_mask_cls = class_positions >= validation_end

        assignments[positions[train_mask_cls]] = "train"
        assignments[positions[validation_mask_cls]] = "validation"
        assignments[positions[test_mask_cls]] = "test"

        if int(target_class) in MAJORITY_CLASS_IDS:
            subsampled_train_keep[
                positions[train_mask_cls & (class_positions < max_majority_train_rows)]
            ] = True
        else:
            subsampled_train_keep[positions[train_mask_cls]] = True

        class_seen[int(target_class)] += len(positions)

    split_masks_dict = {name: assignments == name for name in SPLIT_NAMES}
    return split_masks_dict, subsampled_train_keep


def split_masks(
    labels: np.ndarray,
    class_counts: Counter[int],
    class_seen: defaultdict[int, int],
    ratios: SplitRatios,
    seed: int = 42,
    class_orders: dict[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Assign rows by class-specific cumulative positions for stratification."""
    masks_dict, _ = split_and_subsample_masks(
        labels, class_counts, class_seen, ratios, seed, class_orders
    )
    return masks_dict


def fit_training_scaler(
    csv_files: list[Path],
    chunksize: int,
    class_counts: Counter[int],
    ratios: SplitRatios,
    max_records_per_class: int | None = None,
    max_records_per_file: int | None = None,
    seed: int = 42,
) -> Log1pMinMaxScaler:
    scaler = Log1pMinMaxScaler()
    class_seen: defaultdict[int, int] = defaultdict(int)
    class_orders: dict[int, np.ndarray] = {}
    fitted = False
    train_rows_seen = 0
    last_logged = 0
    for _, features, labels, _ in iter_clean_chunks(
        csv_files, chunksize, max_records_per_class, max_records_per_file
    ):
        masks = split_masks(labels, class_counts, class_seen, ratios, seed, class_orders)
        training_features = features[masks["train"]]
        if len(training_features):
            scaler.partial_fit(training_features)
            fitted = True
            train_rows_seen += len(training_features)
            if train_rows_seen - last_logged >= 5_000_000:
                LOGGER.info("Pass 2 (scaler fit): fitted %d training rows...", train_rows_seen)
                last_logged = train_rows_seen
    if not fitted:
        raise ValueError("No valid training rows were available for scaler fitting")
    return scaler


class ShardWriter:
    def __init__(self, split_dir: Path, shard_size: int) -> None:
        self.split_dir = split_dir
        self.shard_size = shard_size
        self.features: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.buffer_rows = 0
        self.shard_index = 0
        self.rows_written = 0

    def add(self, features: np.ndarray, labels: np.ndarray) -> None:
        start = 0
        while start < len(features):
            take = min(self.shard_size - self.buffer_rows, len(features) - start)
            self.features.append(features[start : start + take])
            self.labels.append(labels[start : start + take])
            self.buffer_rows += take
            self.rows_written += take
            start += take
            if self.buffer_rows == self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self.buffer_rows:
            return
        self.split_dir.mkdir(parents=True, exist_ok=True)
        features = np.concatenate(self.features, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        output_path = self.split_dir / f"part-{self.shard_index:05d}.npz"
        np.savez_compressed(output_path, features=features, labels=labels)
        self.shard_index += 1
        self.features.clear()
        self.labels.clear()
        self.buffer_rows = 0


def load_split_ratios(config_path: Path | None) -> SplitRatios:
    if config_path is None or not config_path.exists():
        return SplitRatios()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    configured = config.get("preprocessing", {}).get("split_ratios")
    if not isinstance(configured, dict):
        return SplitRatios()
    return SplitRatios(
        train=float(configured.get("train", 0.70)),
        validation=float(configured.get("validation", 0.15)),
        test=float(configured.get("test", 0.15)),
    )


def load_dataset_settings(config_path: Path | None) -> tuple[str, int | None, int | None]:
    if config_path is None or not config_path.exists():
        return "full", None, None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = config.get("dataset", {})
    mode = str(dataset.get("mode", "full")).lower()
    if mode not in {"development", "full"}:
        raise ValueError("dataset.mode must be 'development' or 'full'")
    development = dataset.get("development", {}) or {}
    if mode == "full":
        return mode, None, None
    return (
        mode,
        development.get("max_records_per_class", 1000),
        development.get("max_records_per_file"),
    )


def load_random_seed(config_path: Path | None) -> int:
    if config_path is None or not config_path.exists():
        return 42
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    preprocessing = config.get("preprocessing", {}) or {}
    project = config.get("project", {}) or {}
    return int(preprocessing.get("random_seed", project.get("random_seed", 42)))


def load_path_settings(config_path: Path | None) -> tuple[Path, Path, Path]:
    if config_path is None or not config_path.exists():
        return Path("data/raw"), Path("data/processed"), Path("checkpoints")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths = config.get("paths", {}) or {}
    return (
        Path(paths.get("raw_data", "data/raw")),
        Path(paths.get("processed_data", "data/processed")),
        Path(paths.get("checkpoints", "checkpoints")),
    )


def prepare_dataset(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed/prepared"),
    scaler_path: Path = Path("checkpoints/minmax_scaler.pkl"),
    chunksize: int = 100_000,
    shard_size: int = 100_000,
    ratios: SplitRatios = SplitRatios(),
    mode: str = "full",
    max_records_per_class: int | None = None,
    max_records_per_file: int | None = None,
    seed: int = 42,
    create_subsampled_variant: bool = False,
    max_majority_train_rows: int = 500_000,
) -> dict[str, object]:
    if chunksize < 1 or shard_size < 1:
        raise ValueError("chunksize and shard_size must be positive")
    if mode not in {"development", "full"}:
        raise ValueError("mode must be 'development' or 'full'")

    if mode == "full":
        max_records_per_class = None
        max_records_per_file = None
    else:
        if max_records_per_class is None and max_records_per_file is None:
            max_records_per_class = 1000

    start_time = time.time()
    csv_files = discover_csv_files(raw_dir)
    LOGGER.info("Dataset mode: %s", mode)
    LOGGER.info("Files discovered: %d", len(csv_files))

    class_counts, stats = count_valid_rows(
        csv_files,
        chunksize,
        max_records_per_class,
        max_records_per_file,
    )
    LOGGER.info("Files processed: %d", len(stats.files_processed))
    LOGGER.info("Rows processed during counting: %d", stats.raw_rows)
    LOGGER.info("Mapped class distribution: %s", dict(sorted(class_counts.items())))
    LOGGER.info("Features: %d", len(MODEL_FEATURES))

    scaler = fit_training_scaler(
        csv_files=csv_files,
        chunksize=chunksize,
        class_counts=class_counts,
        ratios=ratios,
        max_records_per_class=max_records_per_class,
        max_records_per_file=max_records_per_file,
        seed=seed,
    )
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with scaler_path.open("wb") as file_handle:
        pickle.dump(scaler, file_handle)

    if create_subsampled_variant:
        full_dir = output_dir / "full"
        subsampled_dir = output_dir / "subsampled_train"
        target_dirs = [full_dir, subsampled_dir]
    else:
        full_dir = output_dir
        subsampled_dir = None
        target_dirs = [full_dir]

    for target_root in target_dirs:
        for name in SPLIT_NAMES:
            split_dir = target_root / name
            if split_dir.exists():
                for old_shard in split_dir.glob("part-*.npz"):
                    old_shard.unlink()

    writers_full = {
        name: ShardWriter(full_dir / name, shard_size) for name in SPLIT_NAMES
    }
    writers_subsampled = (
        {name: ShardWriter(subsampled_dir / name, shard_size) for name in SPLIT_NAMES}
        if subsampled_dir is not None
        else None
    )

    class_seen_full: defaultdict[int, int] = defaultdict(int)
    class_seen_subsampled: defaultdict[int, int] = defaultdict(int)
    class_orders_full: dict[int, np.ndarray] = {}
    class_orders_subsampled: dict[int, np.ndarray] = {}

    split_class_distribution_full: dict[str, Counter[int]] = {
        name: Counter() for name in SPLIT_NAMES
    }
    split_class_distribution_subsampled: dict[str, Counter[int]] = {
        name: Counter() for name in SPLIT_NAMES
    }

    total_processed_pass3 = 0
    last_logged_pass3 = 0

    for _, features, labels, _ in iter_clean_chunks(
        csv_files,
        chunksize,
        max_records_per_class,
        max_records_per_file,
    ):
        features_transformed = scaler.transform(features).astype(np.float32)

        masks_full, _ = split_and_subsample_masks(
            labels,
            class_counts,
            class_seen_full,
            ratios,
            seed,
            class_orders_full,
            max_majority_train_rows=max_majority_train_rows,
        )

        for split_name, mask in masks_full.items():
            if mask.any():
                split_class_distribution_full[split_name].update(labels[mask].tolist())
                writers_full[split_name].add(features_transformed[mask], labels[mask])

        if writers_subsampled is not None:
            masks_sub, keep_subsampled_train = split_and_subsample_masks(
                labels,
                class_counts,
                class_seen_subsampled,
                ratios,
                seed,
                class_orders_subsampled,
                max_majority_train_rows=max_majority_train_rows,
            )

            for split_name, mask in masks_sub.items():
                if split_name == "train":
                    sub_mask = keep_subsampled_train
                else:
                    sub_mask = mask

                if sub_mask.any():
                    split_class_distribution_subsampled[split_name].update(
                        labels[sub_mask].tolist()
                    )
                    writers_subsampled[split_name].add(
                        features_transformed[sub_mask], labels[sub_mask]
                    )

        total_processed_pass3 += len(labels)
        if total_processed_pass3 - last_logged_pass3 >= 5_000_000:
            LOGGER.info("Pass 3 (writing shards): processed %d rows...", total_processed_pass3)
            last_logged_pass3 = total_processed_pass3

    for writer in writers_full.values():
        writer.flush()
    if writers_subsampled is not None:
        for writer in writers_subsampled.values():
            writer.flush()

    split_rows_full = {name: writer.rows_written for name, writer in writers_full.items()}
    elapsed_seconds = round(time.time() - start_time, 2)
    peak_ram = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

    manifest_full = {
        "features": list(MODEL_FEATURES),
        "feature_count": len(MODEL_FEATURES),
        "target_classes": list(ALLOWED_TARGET_CLASSES),
        "class_to_id": TARGET_TO_ID,
        "class_distribution": {
            str(class_id): int(count)
            for class_id, count in sorted(class_counts.items())
        },
        "split_class_distribution": {
            split: {
                str(class_id): int(count)
                for class_id, count in sorted(distribution.items())
            }
            for split, distribution in split_class_distribution_full.items()
        },
        "split_ratios": ratios.__dict__,
        "random_seed": seed,
        "dataset_mode": mode,
        "variant": "full" if create_subsampled_variant else "single",
        "max_records_per_class": max_records_per_class,
        "max_records_per_file": max_records_per_file,
        "split_rows": split_rows_full,
        "scaler_path": str(scaler_path),
        "unresolved_rows_quarantined": stats.unresolved_rows,
        "invalid_rows_dropped": stats.invalid_rows,
        "development_limit_rows_dropped": stats.limited_rows,
        "rows_processed": stats.raw_rows,
        "rows_discarded": stats.unresolved_rows + stats.invalid_rows + stats.limited_rows,
        "processing_duration_seconds": elapsed_seconds,
        "peak_ram_mb": peak_ram,
        "normalization": {
            "method": "Log1p_MinMaxScaler",
            "fit_split": "train",
            "scaler_path": str(scaler_path),
        },
        "source_files": [str(path) for path in csv_files],
    }

    full_dir.mkdir(parents=True, exist_ok=True)
    (full_dir / "manifest.json").write_text(
        json.dumps(manifest_full, indent=2) + "\n", encoding="utf-8"
    )
    (full_dir / "class_names.json").write_text(
        json.dumps(list(ALLOWED_TARGET_CLASSES), indent=2) + "\n", encoding="utf-8"
    )

    if subsampled_dir is not None and writers_subsampled is not None:
        split_rows_subsampled = {
            name: writer.rows_written for name, writer in writers_subsampled.items()
        }
        manifest_subsampled = dict(manifest_full)
        manifest_subsampled["variant"] = "subsampled_train"
        manifest_subsampled["split_rows"] = split_rows_subsampled
        manifest_subsampled["max_majority_train_rows"] = max_majority_train_rows
        manifest_subsampled["split_class_distribution"] = {
            split: {
                str(class_id): int(count)
                for class_id, count in sorted(distribution.items())
            }
            for split, distribution in split_class_distribution_subsampled.items()
        }
        subsampled_dir.mkdir(parents=True, exist_ok=True)
        (subsampled_dir / "manifest.json").write_text(
            json.dumps(manifest_subsampled, indent=2) + "\n", encoding="utf-8"
        )
        (subsampled_dir / "class_names.json").write_text(
            json.dumps(list(ALLOWED_TARGET_CLASSES), indent=2) + "\n", encoding="utf-8"
        )

    LOGGER.info("Rows discarded: %d", manifest_full["rows_discarded"])
    LOGGER.info("Full Train count: %d", split_rows_full["train"])
    LOGGER.info("Validation count: %d", split_rows_full["validation"])
    LOGGER.info("Test count: %d", split_rows_full["test"])
    if writers_subsampled is not None:
        LOGGER.info(
            "Subsampled Train count: %d",
            split_rows_subsampled["train"],
        )
    LOGGER.info("Normalization: Log1p_MinMaxScaler fitted only on train rows")
    LOGGER.info("Processing duration: %.2f seconds", elapsed_seconds)
    LOGGER.info("Peak RAM usage: %.2f MB", peak_ram)
    LOGGER.info("Output locations: %s; scaler: %s", output_dir, scaler_path)
    return manifest_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CICIoT2023 training shards.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scaler-path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", choices=("development", "full"), default=None)
    parser.add_argument("--max-records-per-class", type=int, default=None)
    parser.add_argument("--max-records-per-file", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--create-subsampled-variant", action="store_true", default=False)
    parser.add_argument("--max-majority-train-rows", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config_raw_dir, config_processed_dir, config_checkpoints_dir = load_path_settings(
        args.config
    )
    config_mode, config_class_limit, config_file_limit = load_dataset_settings(args.config)
    prepare_dataset(
        raw_dir=args.raw_dir or config_raw_dir,
        output_dir=args.output_dir or config_processed_dir / "prepared",
        scaler_path=args.scaler_path or config_checkpoints_dir / "minmax_scaler.pkl",
        chunksize=args.chunksize,
        shard_size=args.shard_size,
        ratios=load_split_ratios(args.config),
        mode=args.mode or config_mode,
        max_records_per_class=(
            args.max_records_per_class
            if args.max_records_per_class is not None
            else config_class_limit
        ),
        max_records_per_file=(
            args.max_records_per_file
            if args.max_records_per_file is not None
            else config_file_limit
        ),
        seed=args.seed if args.seed is not None else load_random_seed(args.config),
        create_subsampled_variant=(
            args.create_subsampled_variant
            if args.create_subsampled_variant
            else (args.mode == "full" or config_mode == "full")
        ),
        max_majority_train_rows=args.max_majority_train_rows,
    )


if __name__ == "__main__":
    main()