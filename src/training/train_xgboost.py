"""Training pipeline for the XGBoost baseline classifier."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import psutil
import xgboost as xgb

from src.models.xgboost_model import build_xgboost_classifier
from src.preprocessing.dataset import iter_batches
from src.preprocessing.prepare_dataset import MODEL_FEATURES

LOGGER = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_system_info() -> dict[str, object]:
    """Inspect memory and CPU status."""
    process = psutil.Process()
    ram_info = process.memory_info()
    return {
        "ram_rss_mb": round(ram_info.rss / (1024 * 1024), 2),
        "cpu_count": psutil.cpu_count(logical=True),
    }


def load_split_data(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all feature and label arrays from a processed split directory."""
    feature_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    for features, labels in iter_batches(split_dir, batch_size=10000, shuffle=False):
        feature_list.append(features)
        label_list.append(labels)
    if not feature_list:
        raise ValueError(f"No data shards found under {split_dir}")
    return np.concatenate(feature_list, axis=0), np.concatenate(label_list, axis=0)


def run_xgboost_training(
    data_dir: Path = Path("data/processed/prepared"),
    train_dir: Path | None = None,
    val_dir: Path | None = None,
    checkpoint_dir: Path = Path("checkpoints/xgboost/real_ciciot2023/baseline"),
    output_dir: Path = Path("results/xgboost/real_ciciot2023/baseline"),
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 10,
    seed: int = 42,
) -> dict[str, object]:
    """Train XGBoost multi-class baseline classifier with validation early stopping."""
    set_random_seeds(seed)
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    checkpoint_dir = checkpoint_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_output_log_path = output_dir / "task_output.log"

    def log_and_print(msg: str) -> None:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
        with open(task_output_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists() and (data_dir.parent / "manifest.json").exists():
        manifest_path = data_dir.parent / "manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    # Determine training directory
    if train_dir is None:
        if (data_dir / "subsampled_train" / "train").exists():
            train_path = data_dir / "subsampled_train" / "train"
        elif (data_dir / "train").exists():
            train_path = data_dir / "train"
        else:
            train_path = data_dir
    else:
        train_path = train_dir

    # Determine validation directory
    if val_dir is None:
        if (data_dir / "subsampled_train" / "validation").exists():
            val_path = data_dir / "subsampled_train" / "validation"
        elif (data_dir / "validation").exists():
            val_path = data_dir / "validation"
        else:
            val_path = data_dir
    else:
        val_path = val_dir

    log_and_print("==================================================")
    log_and_print("STARTING FRESH REAL CICIoT2023 XGBOOST BASELINE")
    log_and_print(f"Data Dir: {data_dir}")
    log_and_print(f"Train Path: {train_path}")
    log_and_print(f"Val Path: {val_path}")
    log_and_print(f"Checkpoint Dir: {checkpoint_dir}")
    log_and_print(f"Output Dir: {output_dir}")
    log_and_print("==================================================")

    LOGGER.info("Loading training data for XGBoost from %s...", train_path)
    X_train, y_train = load_split_data(train_path)
    LOGGER.info("Loaded X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)

    LOGGER.info("Loading validation data for XGBoost from %s...", val_path)
    X_val, y_val = load_split_data(val_path)
    LOGGER.info("Loaded X_val shape: %s, y_val shape: %s", X_val.shape, y_val.shape)

    clf = build_xgboost_classifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
    )

    LOGGER.info("Fitting XGBoost classifier (tree_method='hist')...")
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10,
    )

    training_duration_seconds = round(time.time() - start_time, 2)
    sys_info = get_system_info()
    peak_ram_mb = sys_info["ram_rss_mb"]

    # Evaluate validation metrics at best iteration
    val_probs = clf.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_acc = float(np.mean(val_preds == y_val))

    cce = float(-np.mean(np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-15)))

    best_iteration = getattr(clf, "best_iteration", n_estimators - 1)

    # Save XGBoost model to JSON
    model_json_path = checkpoint_dir / "model.json"
    clf.save_model(str(model_json_path))
    out_model_json_path = output_dir / "model.json"
    clf.save_model(str(out_model_json_path))
    LOGGER.info("Saved XGBoost model to %s and %s", model_json_path, out_model_json_path)

    dataset_note = (
        "The paper reports approximately 46.18M records, whereas our available local dataset "
        "contains 37,422,631 raw rows and 37,421,798 retained rows across 7 active classes (Recon absent)."
    )

    metadata = {
        "timestamp": start_timestamp,
        "model_name": "XGBoost-Baseline",
        "xgboost_version": xgb.__version__,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": "hist",
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "early_stopping_rounds": early_stopping_rounds,
            "random_seed": seed,
        },
        "dataset_note": dataset_note,
        "data_summary": {
            "num_features": int(X_train.shape[1]),
            "num_classes": int(len(np.unique(y_train))),
            "train_samples": int(X_train.shape[0]),
            "validation_samples": int(X_val.shape[0]),
        },
        "training_results": {
            "best_iteration": int(best_iteration) if best_iteration is not None else n_estimators,
            "validation_accuracy": val_acc,
            "validation_log_loss": cce,
            "training_duration_seconds": training_duration_seconds,
            "peak_ram_mb": peak_ram_mb,
        },
        "paths": {
            "checkpoint_model_json": str(model_json_path),
            "output_model_json": str(out_model_json_path),
            "metadata_json": str(output_dir / "metadata.json"),
            "task_output_log": str(task_output_log_path),
            "results_dir": str(output_dir),
        },
        "system_info": sys_info,
        "dataset_manifest": dataset_manifest,
    }

    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved metadata to %s and %s", checkpoint_dir, output_dir)

    summary_msg = (
        "\n==================================================\n"
        "XGBOOST BASELINE TRAINING COMPLETED\n"
        f"Best Iteration: {best_iteration}\n"
        f"Validation Accuracy: {val_acc * 100:.2f}% ({val_acc:.6f})\n"
        f"Validation Log Loss: {cce:.6f}\n"
        f"Training Duration: {training_duration_seconds}s\n"
        f"Peak RAM: {peak_ram_mb} MB\n"
        f"Model Saved: {model_json_path.as_posix()}\n"
        "==================================================\n"
    )
    log_and_print(summary_msg)

    return {
        "model": clf,
        "best_iteration": int(best_iteration) if best_iteration is not None else n_estimators,
        "val_acc": val_acc,
        "val_loss": cce,
        "duration_seconds": training_duration_seconds,
        "peak_ram_mb": peak_ram_mb,
        "checkpoint_model_json": str(model_json_path),
        "output_model_json": str(out_model_json_path),
        "metadata_json": str(output_dir / "metadata.json"),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "num_features": int(X_train.shape[1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost baseline classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/prepared"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/xgboost/real_ciciot2023/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/xgboost/real_ciciot2023/baseline"))
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    results = run_xgboost_training(
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )

    print("\n--- XGBoost Baseline Training Summary ---")
    print(f"XGBoost Version: {xgb.__version__}")
    print(f"Best Iteration: {results['best_iteration']}")
    print(f"Validation Accuracy: {results['val_acc'] * 100:.2f}%")
    print(f"Validation Log Loss: {results['val_loss']:.6f}")
    print(f"Training Duration: {results['duration_seconds']}s")
    print(f"Peak RAM: {results['peak_ram_mb']} MB")
    print("Model Saved:", str(results["checkpoint_model_json"]).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
