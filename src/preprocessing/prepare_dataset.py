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
SPLIT_NAMES = ("train", "validation", "test")
TARGET_TO_ID = {name: index for index, name in enumerate(ALLOWED_TARGET_CLASSES)}


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

    def merge(self, other: "InspectionStats") -> None:
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
    files = sorted(path for path in raw_dir.rglob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    return files


def validate_feature_columns(columns: list[str]) -> None:
    missing = [column for column in MODEL_FEATURES if column not in columns]
    if missing:
        raise ValueError(f"Dataset is missing required model features: {missing}")
    if LABEL_COLUMN not in columns:
        raise ValueError(f"Dataset is missing required label column: {LABEL_COLUMN}")


def clean_chunk(chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, InspectionStats]:
    """Map labels and return only finite, complete rows from one CSV chunk."""
    source_labels = chunk[LABEL_COLUMN].astype("string")
    observed = set(source_labels.dropna().unique().tolist())
    unexpected = observed - set(EXPECTED_SOURCE_LABELS)
    if unexpected:
        raise ValueError(f"Unexpected source labels found: {sorted(unexpected)}")

    mapped_labels = source_labels.map(SOURCE_TO_TARGET)
    unresolved_mask = source_labels.isin(UNRESOLVED_SOURCE_LABELS)
    numeric_features = chunk.loc[:, MODEL_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
    valid_mask = mapped_labels.notna() & numeric_features.notna().all(axis=1)

    stats = InspectionStats(
        files_processed=set(),
        raw_rows=len(chunk),
        valid_rows=int(valid_mask.sum()),
        unresolved_rows=int(unresolved_mask.sum()),
        invalid_rows=int((~valid_mask & ~unresolved_mask).sum()),
    )
    features = numeric_features.loc[valid_mask].to_numpy(dtype=np.float32)
    labels = mapped_labels.loc[valid_mask].map(TARGET_TO_ID).to_numpy(dtype=np.int64)
    return features, labels, stats


def iter_clean_chunks(
    csv_files: list[Path],
    chunksize: int,
    max_records_per_class: int | None = None,
    max_records_per_file: int | None = None,
) -> Iterator[tuple[Path, np.ndarray, np.ndarray, InspectionStats]]:
    usecols = list(MODEL_FEATURES) + [LABEL_COLUMN]
    limiter = DevelopmentLimiter(max_records_per_class, max_records_per_file)
    for csv_file in csv_files:
        if limiter.is_complete:
            break
        header = pd.read_csv(csv_file, nrows=0).columns.tolist()
        validate_feature_columns(header)
        for chunk in pd.read_csv(
            csv_file, usecols=usecols, chunksize=chunksize, low_memory=False
        ):
            if limiter.is_file_complete(csv_file):
                break
            features, labels, stats = clean_chunk(chunk)
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
) -> tuple[Counter[str], InspectionStats]:
    class_counts: Counter[str] = Counter()
    stats = InspectionStats(files_processed=set())
    last_logged = 0
    for _, _, labels, chunk_stats in iter_clean_chunks(
        csv_files, chunksize, max_records_per_class, max_records_per_file
    ):
        class_counts.update(labels.tolist())
        stats.merge(chunk_stats)
        if stats.raw_rows - last_logged >= 1_000_000:
            LOGGER.info("Pass 1 (counting): scanned %d rows...", stats.raw_rows)
            last_logged = stats.raw_rows
    return class_counts, stats


def split_masks(
    labels: np.ndarray,
    class_counts: Counter[str],
    class_seen: defaultdict[str, int],
    ratios: SplitRatios,
    seed: int = 42,
    class_orders: dict[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Assign rows by class-specific cumulative positions for stratification."""
    assignments = np.empty(labels.shape[0], dtype="U10")
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
        assignments[positions[class_positions < train_end]] = "train"
        validation_mask = (class_positions >= train_end) & (class_positions < validation_end)
        assignments[positions[validation_mask]] = "validation"
        assignments[positions[class_positions >= validation_end]] = "test"
        class_seen[int(target_class)] += len(positions)
    return {name: assignments == name for name in SPLIT_NAMES}


def fit_training_scaler(
    csv_files: list[Path],
    chunksize: int,
    class_counts: Counter[str],
    ratios: SplitRatios,
    max_records_per_class: int | None = None,
    max_records_per_file: int | None = None,
    seed: int = 42,
) -> MinMaxScaler:
    scaler = MinMaxScaler()
    class_seen: defaultdict[str, int] = defaultdict(int)
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
            if train_rows_seen - last_logged >= 1_000_000:
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
) -> dict[str, object]:
    if chunksize < 1 or shard_size < 1:
        raise ValueError("chunksize and shard_size must be positive")
    if mode not in {"development", "full"}:
        raise ValueError("mode must be 'development' or 'full'")
    if mode == "full":
        max_records_per_class = None
        max_records_per_file = None
    elif max_records_per_class is None and max_records_per_file is None:
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

    for name in SPLIT_NAMES:
        split_dir = output_dir / name
        if split_dir.exists():
            for old_shard in split_dir.glob("part-*.npz"):
                old_shard.unlink()

    writers = {
        name: ShardWriter(output_dir / name, shard_size) for name in SPLIT_NAMES
    }
    class_seen: defaultdict[str, int] = defaultdict(int)
    class_orders: dict[int, np.ndarray] = {}
    split_class_distribution: dict[str, Counter[int]] = {
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
        masks = split_masks(labels, class_counts, class_seen, ratios, seed, class_orders)
        for split_name, mask in masks.items():
            if mask.any():
                split_class_distribution[split_name].update(labels[mask].tolist())
                writers[split_name].add(
                    scaler.transform(features[mask]).astype(np.float32), labels[mask]
                )
        total_processed_pass3 += len(labels)
        if total_processed_pass3 - last_logged_pass3 >= 1_000_000:
            LOGGER.info("Pass 3 (writing shards): processed %d rows...", total_processed_pass3)
            last_logged_pass3 = total_processed_pass3
    for writer in writers.values():
        writer.flush()

    split_rows = {name: writer.rows_written for name, writer in writers.items()}
    elapsed_seconds = round(time.time() - start_time, 2)
    peak_ram = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    manifest = {
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
            for split, distribution in split_class_distribution.items()
        },
        "split_ratios": ratios.__dict__,
        "random_seed": seed,
        "dataset_mode": mode,
        "max_records_per_class": max_records_per_class,
        "max_records_per_file": max_records_per_file,
        "split_rows": split_rows,
        "scaler_path": str(scaler_path),
        "unresolved_rows_quarantined": stats.unresolved_rows,
        "invalid_rows_dropped": stats.invalid_rows,
        "development_limit_rows_dropped": stats.limited_rows,
        "rows_processed": stats.raw_rows,
        "rows_discarded": stats.unresolved_rows + stats.invalid_rows + stats.limited_rows,
        "processing_duration_seconds": elapsed_seconds,
        "peak_ram_mb": peak_ram,
        "normalization": {
            "method": "MinMaxScaler",
            "fit_split": "train",
            "scaler_path": str(scaler_path),
        },
        "source_files": [str(path) for path in csv_files],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "class_names.json").write_text(
        json.dumps(list(ALLOWED_TARGET_CLASSES), indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Rows discarded: %d", manifest["rows_discarded"])
    LOGGER.info("Train count: %d", split_rows["train"])
    LOGGER.info("Validation count: %d", split_rows["validation"])
    LOGGER.info("Test count: %d", split_rows["test"])
    LOGGER.info("Normalization: MinMaxScaler fitted only on train rows")
    LOGGER.info("Processing duration: %.2f seconds", elapsed_seconds)
    LOGGER.info("Peak RAM usage: %.2f MB", peak_ram)
    LOGGER.info("Output locations: %s; scaler: %s", output_dir, scaler_path)
    return manifest


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
    )


if __name__ == "__main__":
    main()