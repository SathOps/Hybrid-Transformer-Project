"""Training pipeline for the CICIoT2023 CNN-Transformer model."""

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
import tensorflow as tf
from tensorflow import keras
import yaml

from src.models.cnn_transformer import build_model
from src.preprocessing.dataset import to_tf_dataset

LOGGER = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: Path = Path("config.yaml")) -> dict[str, object]:
    """Load configuration from config.yaml if available."""
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def get_system_info() -> dict[str, object]:
    """Inspect GPU availability, memory, and CPU/RAM status."""
    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0
    gpu_names = [gpu.name for gpu in gpus]
    process = psutil.Process()
    ram_info = process.memory_info()

    return {
        "gpu_available": gpu_available,
        "gpu_count": len(gpus),
        "gpu_devices": gpu_names,
        "ram_rss_mb": round(ram_info.rss / (1024 * 1024), 2),
        "cpu_count": psutil.cpu_count(logical=True),
    }


def count_tf_dataset_steps(tf_dataset: tf.data.Dataset) -> int:
    """Safely count the number of batches in a tf.data.Dataset generator."""
    count = 0
    for _ in tf_dataset:
        count += 1
    return count


def run_training(
    data_dir: Path = Path("data/processed/prepared"),
    train_dir: Path | None = None,
    val_dir: Path | None = None,
    checkpoint_dir: Path = Path("checkpoints/cnn_transformer/real_ciciot2023/fresh_25_epochs"),
    output_dir: Path = Path("results/cnn_transformer/real_ciciot2023/fresh_25_epochs"),
    epochs: int = 25,
    batch_size: int = 1024,
    learning_rate: float = 0.00005,
    seed: int = 42,
    early_stopping: bool = True,
    patience: int = 5,
    fresh: bool = False,
    class_weight: dict[int, float] | None = None,
    feature_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Compile and fit the CNN-Transformer model on prepared datasets."""
    set_random_seeds(seed)
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    checkpoint_dir = checkpoint_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_epochs = 0
    initial_cumulative_time = 0.0
    initial_best_epoch = 0
    initial_best_val_loss = float("inf")

    latest_checkpoint_path = checkpoint_dir / "latest.keras"
    best_checkpoint_path = checkpoint_dir / "best_model.keras"
    final_checkpoint_path = checkpoint_dir / "final_model.keras"
    training_log_path = output_dir / "training_log.csv"
    task_output_log_path = output_dir / "task_output.log"
    tb_log_dir = output_dir / "tensorboard"
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    if not fresh and latest_checkpoint_path.exists() and training_log_path.exists():
        try:
            import pandas as pd
            if training_log_path.stat().st_size > 0:
                log_df = pd.read_csv(training_log_path)
                if not log_df.empty and "epoch" in log_df.columns:
                    completed_epochs = int(log_df["epoch"].max())
                    if "cumulative_training_time_seconds" in log_df.columns:
                        initial_cumulative_time = float(log_df["cumulative_training_time_seconds"].iloc[-1])
                    if "val_loss" in log_df.columns:
                        initial_best_val_loss = float(log_df["val_loss"].min())
                        best_rows = log_df[log_df["val_loss"] == initial_best_val_loss]
                        if not best_rows.empty:
                            initial_best_epoch = int(best_rows["epoch"].iloc[0])
        except Exception as exc:
            LOGGER.warning("Could not read existing training log for resume state: %s", exc)

    # Determine training directory
    if train_dir is None:
        if (data_dir / "subsampled_train" / "train").exists():
            train_path = data_dir / "subsampled_train" / "train"
        elif (data_dir / "subsampled_train").exists() and list((data_dir / "subsampled_train").glob("part-*.npz")):
            train_path = data_dir / "subsampled_train"
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

    LOGGER.info("Loading training streaming dataset from %s", train_path)
    LOGGER.info("Loading validation streaming dataset from %s", val_path)

    train_ds = to_tf_dataset(
        train_path,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        expand_dims=True,
        feature_indices=feature_indices,
    )
    val_ds = to_tf_dataset(
        val_path,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        expand_dims=True,
        feature_indices=feature_indices,
    )

    if not fresh and completed_epochs > 0 and latest_checkpoint_path.exists():
        LOGGER.info("Resuming model from %s (completed epoch %d)...", latest_checkpoint_path, completed_epochs)
        model = keras.models.load_model(str(latest_checkpoint_path))
    else:
        LOGGER.info("Building FRESH CNN-Transformer model initialization...")
        num_feats = len(feature_indices) if feature_indices is not None else 46
        model = build_model(
            input_shape=(num_feats, 1),
            num_classes=7,
            aggregation="global_average",
            name="cnn_transformer_classifier",
        )
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"],
        )

    class ResumableTrainingCallback(keras.callbacks.Callback):
        def __init__(
            self,
            ckpt_dir: Path,
            out_dir: Path,
            total_epochs: int,
            cum_time: float,
            b_epoch: int,
            b_val_loss: float,
        ):
            super().__init__()
            self.ckpt_dir = ckpt_dir
            self.out_dir = out_dir
            self.total_epochs = total_epochs
            self.cumulative_time = cum_time
            self.best_epoch = b_epoch
            self.best_val_loss = b_val_loss
            self.epoch_start_time = None
            self.csv_path = out_dir / "training_log.csv"
            self.log_file_path = out_dir / "task_output.log"

            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                with open(self.csv_path, "w", encoding="utf-8") as f:
                    f.write(
                        "epoch,train_loss,train_accuracy,val_loss,val_accuracy,best_epoch,best_val_loss,epoch_time_seconds,cumulative_training_time_seconds\n"
                    )

        def on_epoch_begin(self, epoch: int, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch: int, logs=None):
            logs = logs or {}
            epoch_num = epoch + 1
            epoch_time = time.time() - (self.epoch_start_time or time.time())
            self.cumulative_time += epoch_time

            train_loss = float(logs.get("loss", 0.0))
            train_acc = float(logs.get("accuracy", 0.0))
            val_loss = float(logs.get("val_loss", 0.0))
            val_acc = float(logs.get("val_accuracy", 0.0))

            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch_num
                is_best = True

            # 1. Save epoch-specific checkpoint
            ep_ckpt = (self.ckpt_dir / f"epoch_{epoch_num:03d}.keras").resolve()
            self.model.save(ep_ckpt.as_posix())

            # 2. Save latest checkpoint
            lat_ckpt = (self.ckpt_dir / "latest.keras").resolve()
            self.model.save(lat_ckpt.as_posix())

            # 3. Save best checkpoint if best
            if is_best:
                b_ckpt = (self.ckpt_dir / "best_model.keras").resolve()
                self.model.save(b_ckpt.as_posix())
                b_out_ckpt = (self.out_dir / "best_model.keras").resolve()
                self.model.save(b_out_ckpt.as_posix())

            # 4. Save/update training_log.csv
            with open(self.csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch_num},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},"
                    f"{self.best_epoch},{self.best_val_loss:.6f},{epoch_time:.2f},{self.cumulative_time:.2f}\n"
                )

            # 5. Output exact required terminal block
            msg = (
                "\n==================================================\n"
                f"EPOCH {epoch_num}/{self.total_epochs} COMPLETED\n"
                f"Train Loss: {train_loss:.6f}\n"
                f"Train Accuracy: {train_acc:.6f}\n"
                f"Validation Loss: {val_loss:.6f}\n"
                f"Validation Accuracy: {val_acc:.6f}\n"
                f"Best Epoch: {self.best_epoch}\n"
                f"Best Validation Loss: {self.best_val_loss:.6f}\n"
                f"Epoch Time: {epoch_time:.2f}s\n"
                f"Cumulative Time: {self.cumulative_time:.2f}s\n"
                f"Checkpoint Saved: {ep_ckpt.as_posix()}\n"
                "==================================================\n"
            )
            try:
                print(msg, flush=True)
            except UnicodeEncodeError:
                print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
            with open(self.log_file_path, "a", encoding="utf-8") as log_f:
                log_f.write(msg)
                log_f.flush()

    resumable_cb = ResumableTrainingCallback(
        ckpt_dir=checkpoint_dir,
        out_dir=output_dir,
        total_epochs=epochs,
        cum_time=initial_cumulative_time,
        b_epoch=initial_best_epoch,
        b_val_loss=initial_best_val_loss,
    )

    callbacks: list[keras.callbacks.Callback] = [resumable_cb]

    try:
        tb_path_str = tb_log_dir.as_posix()
        tb_path_str.encode("ascii")
        callbacks.append(keras.callbacks.TensorBoard(log_dir=tb_path_str))
    except (UnicodeEncodeError, Exception) as exc:
        LOGGER.warning(
            "TensorBoard callback skipped (path contains non-ASCII characters or TF summary writer unsupported): %s",
            exc,
        )

    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

    if completed_epochs == 0:
        start_msg = (
            "\n==================================================\n"
            "CNN-TRANSFORMER REAL CICIoT2023 BASELINE\n"
            "FRESH RUN (25 EPOCHS)\n"
            f"TOTAL EPOCHS: {epochs}\n"
            "STARTING EPOCH: 1\n"
            "==================================================\n"
        )
    else:
        start_msg = (
            "\n==================================================\n"
            "CNN-TRANSFORMER REAL CICIoT2023 BASELINE\n"
            "RESUMING RUN\n"
            f"TOTAL EPOCHS: {epochs}\n"
            f"STARTING EPOCH: {completed_epochs + 1}\n"
            "==================================================\n"
        )
    print(start_msg, flush=True)
    with open(task_output_log_path, "a", encoding="utf-8") as log_f:
        log_f.write(start_msg)
        log_f.flush()

    if completed_epochs < epochs:
        LOGGER.info("Starting training run (%d to %d epoch(s), batch size %d)...", completed_epochs + 1, epochs, batch_size)
        if class_weight:
            LOGGER.info("Applying class weights: %s", class_weight)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            initial_epoch=completed_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
    else:
        LOGGER.info("Training already complete (%d epochs).", completed_epochs)

    # Save final model state
    model.save(final_checkpoint_path.as_posix())
    out_final_path = output_dir / "final_model.keras"
    model.save(out_final_path.as_posix())
    LOGGER.info("Saved final checkpoint to %s and %s", final_checkpoint_path, out_final_path)

    training_duration_seconds = round(time.time() - start_time, 2)
    sys_info = get_system_info()
    peak_ram_mb = sys_info["ram_rss_mb"]

    # Process history dict into Python standard types
    history_dict = {
        key: [float(val) for val in values]
        for key, values in history.history.items()
    }
    history_json_path = output_dir / "history.json"
    history_json_path.write_text(json.dumps(history_dict, indent=2) + "\n", encoding="utf-8")

    # Extract final step metrics
    final_train_loss = history_dict["loss"][-1]
    final_train_acc = history_dict["accuracy"][-1]
    final_val_loss = history_dict["val_loss"][-1]
    final_val_acc = history_dict["val_accuracy"][-1]

    # Calculate best validation epoch (1-indexed)
    best_val_idx = int(np.argmin(history_dict["val_loss"]))
    best_epoch = best_val_idx + 1
    best_val_loss = history_dict["val_loss"][best_val_idx]
    best_val_acc = history_dict["val_accuracy"][best_val_idx]

    # Calculate step counts
    train_steps = len(history.history["loss"])
    val_steps = count_tf_dataset_steps(val_ds)

    dataset_note = (
        "The paper reports approximately 46.18M records, whereas our available local dataset "
        "contains 37,422,631 raw rows and 37,421,798 retained rows across 7 active classes (Recon absent)."
    )

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists() and (data_dir.parent / "manifest.json").exists():
        manifest_path = data_dir.parent / "manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    metadata = {
        "timestamp": start_timestamp,
        "dataset_type": "REAL CICIoT2023",
        "active_classes_count": 7,
        "training_partition": "SUBSAMPLED TRAINING PARTITION (2,626,223 rows)",
        "validation_partition": "NATURAL VALIDATION SET (5,613,269 rows)",
        "test_partition": "NATURAL HOLD-OUT TEST SET (5,613,274 rows)",
        "synthetic_data_used": False,
        "model_name": "CNN-Transformer",
        "architecture_summary": (
            "CNN -> Reshape(8, 8) -> Transformer Encoder -> GlobalAveragePooling1D -> Dense(7, Softmax)"
        ),
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": train_steps,
        "random_seed": seed,
        "early_stopping_enabled": early_stopping,
        "class_weight": class_weight,
        "dataset_note": dataset_note,
        "paths": {
            "data_dir": str(data_dir),
            "train_dir": str(train_path),
            "val_dir": str(val_path),
            "best_checkpoint": str(best_checkpoint_path),
            "final_checkpoint": str(final_checkpoint_path),
            "training_log": str(training_log_path),
            "history_json": str(history_json_path),
            "metadata_json": str(output_dir / "metadata.json"),
            "task_output_log": str(task_output_log_path),
            "tensorboard_dir": str(tb_log_dir),
        },
        "system_info": sys_info,
        "dataset_manifest": dataset_manifest,
        "metrics_summary": {
            "best_validation_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "best_validation_accuracy": best_val_acc,
            "final_training_loss": final_train_loss,
            "final_training_accuracy": final_train_acc,
            "final_validation_loss": final_val_loss,
            "final_validation_accuracy": final_val_acc,
            "training_duration_seconds": training_duration_seconds,
            "average_epoch_duration_seconds": round(training_duration_seconds / train_steps, 2) if train_steps else 0,
            "peak_ram_mb": peak_ram_mb,
        },
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved metadata and run configuration to %s", output_dir)

    return {
        "model": model,
        "history": history_dict,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "duration_seconds": training_duration_seconds,
        "peak_ram_mb": peak_ram_mb,
        "gpu_info": sys_info,
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
        "history_json": str(history_json_path),
        "metadata_json": str(metadata_path),
        "run_config_json": str(run_config_path),
        "val_steps": val_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN-Transformer cyber attack detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/prepared"))
    parser.add_argument("--train-dir", type=Path, default=None, help="Explicit path to train dataset directory.")
    parser.add_argument("--val-dir", type=Path, default=None, help="Explicit path to validation dataset directory.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/cnn_transformer/real_ciciot2023/fresh_25_epochs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cnn_transformer/real_ciciot2023/fresh_25_epochs"),
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default 25).")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default 1024).")
    parser.add_argument("--learning-rate", type=float, default=0.00005, help="Learning rate for Adam optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--early-stopping", action="store_true", default=True, help="Enable early stopping callback.")
    parser.add_argument("--no-early-stopping", action="store_false", dest="early_stopping", help="Disable early stopping.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping if enabled.")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh run, ignoring existing checkpoints.")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    results = run_training(
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        early_stopping=args.early_stopping,
        patience=args.patience,
        fresh=args.fresh,
    )

    print("\n--- Training Results Summary ---")
    print(f"Epochs Completed: {len(results['history']['loss'])}")
    print(f"Final Training Loss: {results['final_train_loss']:.6f}")
    print(f"Final Training Accuracy: {results['final_train_acc']:.6f}")
    print(f"Final Validation Loss: {results['final_val_loss']:.6f}")
    print(f"Final Validation Accuracy: {results['final_val_acc']:.6f}")
    print(f"Duration: {results['duration_seconds']}s")
    print(f"Peak RAM: {results['peak_ram_mb']} MB")
    print("Best Checkpoint:", str(results["best_checkpoint"]).encode("ascii", "replace").decode("ascii"))
    print("Final Checkpoint:", str(results["final_checkpoint"]).encode("ascii", "replace").decode("ascii"))
    print("Run Config Saved:", str(results["run_config_json"]).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()

