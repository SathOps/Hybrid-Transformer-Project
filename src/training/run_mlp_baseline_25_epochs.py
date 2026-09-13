"""Orchestration script for Fresh 25-Epoch MLP Baseline on Real CICIoT2023."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from src.evaluation.evaluate import evaluate_model
from src.training.train_mlp import run_mlp_training

LOGGER = logging.getLogger(__name__)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    data_dir = Path("data/processed/prepared").resolve()
    subsampled_dir = data_dir / "subsampled_train"
    train_dir = subsampled_dir / "train"
    val_dir = subsampled_dir / "validation"
    test_dir = subsampled_dir / "test"

    checkpoint_dir = Path("checkpoints/mlp/real_ciciot2023/baseline_25_epochs").resolve()
    output_dir = Path("results/mlp/real_ciciot2023/baseline_25_epochs").resolve()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_log_path = output_dir / "task_output.log"

    def log_and_print(msg: str) -> None:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
        with open(task_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    log_and_print("==================================================")
    log_and_print("STARTING FRESH 25-EPOCH MLP BASELINE EXPERIMENT")
    log_and_print(f"Data Dir: {data_dir}")
    log_and_print(f"Train Dir: {train_dir}")
    log_and_print(f"Val Dir: {val_dir}")
    log_and_print(f"Test Dir: {test_dir}")
    log_and_print(f"Checkpoint Dir: {checkpoint_dir}")
    log_and_print(f"Output Dir: {output_dir}")
    log_and_print("==================================================")

    # 1. Run MLP Training
    train_results = run_mlp_training(
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        epochs=25,
        batch_size=1024,
        learning_rate=0.00005,
        seed=42,
        early_stopping=True,
        patience=5,
        fresh=True,
    )

    epochs_completed = len(train_results["history"]["loss"])
    log_and_print("\n==================================================")
    log_and_print("MLP TRAINING COMPLETED / EARLY STOPPED")
    log_and_print(f"Epochs Completed: {epochs_completed}/25")
    log_and_print(f"Best Validation Epoch: {train_results['best_epoch']}")
    log_and_print(f"Best Validation Loss: {train_results['best_val_loss']:.6f}")
    log_and_print(f"Best Validation Accuracy: {train_results['best_val_acc']:.6f}")
    log_and_print(f"Total Training Duration: {train_results['duration_seconds']}s")
    log_and_print("==================================================")

    # 2. Run Test Evaluation
    log_and_print("\nSTARTING UNTOUCHED NATURAL TEST SET EVALUATION...")
    eval_results = evaluate_model(
        checkpoint_path=Path(train_results["best_checkpoint"]),
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=1024,
    )

    test_metrics = eval_results["test_metrics"]
    log_and_print("\n==================================================")
    log_and_print("FINAL 25-EPOCH MLP BASELINE EXPERIMENT SUMMARY")
    log_and_print("==================================================")
    log_and_print(f"Epochs Completed: {epochs_completed}/25")
    log_and_print(f"Best Epoch: {train_results['best_epoch']}")
    log_and_print(f"Best Validation Loss: {train_results['best_val_loss']:.6f}")
    log_and_print(f"Best Validation Accuracy: {train_results['best_val_acc']:.6f}")
    log_and_print(f"Test Accuracy: {test_metrics['test_accuracy'] * 100:.2f}% ({test_metrics['test_accuracy']:.6f})")
    log_and_print(f"Test Loss: {test_metrics['test_loss']:.6f}")
    log_and_print(f"Macro F1: {test_metrics['f1_macro']:.6f}")
    log_and_print(f"Weighted F1: {test_metrics['f1_weighted']:.6f}")
    log_and_print(f"Macro Precision: {test_metrics['precision_macro']:.6f}")
    log_and_print(f"Weighted Precision: {test_metrics['precision_weighted']:.6f}")
    log_and_print(f"Macro Recall: {test_metrics['recall_macro']:.6f}")
    log_and_print(f"Weighted Recall: {test_metrics['recall_weighted']:.6f}")
    log_and_print(f"Macro ROC-AUC: {test_metrics['roc_auc_macro']:.6f}")
    log_and_print(f"Macro PR-AUC: {test_metrics['pr_auc_macro']:.6f}")
    log_and_print(f"Total Training Time: {train_results['duration_seconds']}s")
    log_and_print(f"Checkpoint Locations: {checkpoint_dir}")
    log_and_print(f"Report Locations: {output_dir}")
    log_and_print("==================================================")


if __name__ == "__main__":
    main()
