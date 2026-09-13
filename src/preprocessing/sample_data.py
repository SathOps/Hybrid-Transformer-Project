"""Generate sample CICIoT2023 raw data for local pipeline verification if needed."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing.prepare_dataset import MODEL_FEATURES, prepare_dataset
from src.preprocessing.class_mapping import ALLOWED_TARGET_CLASSES

SAMPLE_SOURCE_LABELS = {
    "Benign": "BenignTraffic",
    "BruteForce": "DictionaryBruteForce",
    "DDoS": "DDoS-TCP_Flood",
    "DoS": "DoS-UDP_Flood",
    "Mirai": "Mirai-greeth_flood",
    "Recon": "Recon-PortScan",
    "Spoofing": "DNS_Spoofing",
    "Web-based": "XSS",
}


def generate_sample_raw_csvs(
    raw_dir: Path = Path("data/raw"),
    num_records_per_class: int = 2500,
    seed: int = 42,
) -> list[Path]:
    """Generate realistic synthetic CSV files covering all 8 canonical classes."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    columns = list(MODEL_FEATURES) + ["label"]
    csv_paths: list[Path] = []

    for target_class, source_label in SAMPLE_SOURCE_LABELS.items():
        records = []
        for i in range(num_records_per_class):
            # Generate 46 feature values with class-specific distribution shift
            class_offset = ALLOWED_TARGET_CLASSES.index(target_class) * 5.0
            feature_vals = rng.normal(loc=10.0 + class_offset, scale=2.0, size=len(MODEL_FEATURES))
            feature_vals = np.abs(feature_vals)
            record = {feat: float(val) for feat, val in zip(MODEL_FEATURES, feature_vals)}
            record["label"] = source_label
            records.append(record)

        file_path = raw_dir / f"sample_{target_class.lower()}.csv"
        pd.DataFrame(records, columns=columns).to_csv(file_path, index=False)
        csv_paths.append(file_path)

    return csv_paths


def ensure_prepared_dataset(
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed/prepared"),
    scaler_path: Path = Path("checkpoints/minmax_scaler.pkl"),
    num_records_per_class: int = 2500,
    seed: int = 42,
) -> dict[str, object]:
    """Ensure data/processed/prepared exists; generate sample data and prepare if missing."""
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    manifest_path = processed_dir / "manifest.json"

    if train_dir.exists() and any(train_dir.glob("part-*.npz")) and manifest_path.exists():
        return {}

    generate_sample_raw_csvs(raw_dir=raw_dir, num_records_per_class=num_records_per_class, seed=seed)
    return prepare_dataset(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        scaler_path=scaler_path,
        mode="full",
        seed=seed,
    )


if __name__ == "__main__":
    ensure_prepared_dataset()
