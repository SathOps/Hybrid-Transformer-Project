"""Orchestration script for Fresh Real CICIoT2023 XGBoost Baseline."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from src.evaluation.evaluate_xgboost import evaluate_xgboost
from src.training.train_xgboost import run_xgboost_training

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

    checkpoint_dir = Path("checkpoints/xgboost/real_ciciot2023/baseline").resolve()
    output_dir = Path("results/xgboost/real_ciciot2023/baseline").resolve()

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
    log_and_print("STARTING FRESH REAL CICIoT2023 XGBOOST BASELINE EXPERIMENT")
    log_and_print(f"Data Dir: {data_dir}")
    log_and_print(f"Train Dir: {train_dir}")
    log_and_print(f"Val Dir: {val_dir}")
    log_and_print(f"Test Dir: {test_dir}")
    log_and_print(f"Checkpoint Dir: {checkpoint_dir}")
    log_and_print(f"Output Dir: {output_dir}")
    log_and_print("==================================================")

    # 1. Run XGBoost Training
    train_results = run_xgboost_training(
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        seed=42,
    )

    log_and_print("\n==================================================")
    log_and_print("XGBOOST TRAINING COMPLETED")
    log_and_print(f"Train Samples: {train_results['train_samples']}")
    log_and_print(f"Validation Samples: {train_results['val_samples']}")
    log_and_print(f"Features: {train_results['num_features']}")
    log_and_print(f"Best Iteration: {train_results['best_iteration']}")
    log_and_print(f"Validation Accuracy: {train_results['val_acc'] * 100:.2f}% ({train_results['val_acc']:.6f})")
    log_and_print(f"Validation Log Loss: {train_results['val_loss']:.6f}")
    log_and_print(f"Total Training Duration: {train_results['duration_seconds']}s")
    log_and_print("==================================================")

    # 2. Run Test Evaluation
    log_and_print("\nSTARTING UNTOUCHED NATURAL TEST SET EVALUATION...")
    eval_results = evaluate_xgboost(
        checkpoint_model_path=Path(train_results["checkpoint_model_json"]),
        test_dir=test_dir,
        output_dir=output_dir,
    )

    test_metrics = eval_results["test_metrics"]
    log_and_print("\n==================================================")
    log_and_print("FINAL REAL CICIoT2023 XGBOOST BASELINE SUMMARY")
    log_and_print("==================================================")
    log_and_print(f"Train Samples: {train_results['train_samples']}")
    log_and_print(f"Validation Samples: {train_results['val_samples']}")
    log_and_print(f"Test Samples: {eval_results['num_samples']}")
    log_and_print(f"Features: {train_results['num_features']}")
    log_and_print(f"Active Classes: 7")
    log_and_print(f"Best Iteration: {train_results['best_iteration']}")
    log_and_print(f"Validation Accuracy: {train_results['val_acc'] * 100:.2f}% ({train_results['val_acc']:.6f})")
    log_and_print(f"Test Accuracy: {test_metrics['test_accuracy'] * 100:.2f}% ({test_metrics['test_accuracy']:.6f})")
    log_and_print(f"Test Log Loss: {test_metrics['test_log_loss']:.6f}")
    log_and_print(f"Macro F1: {test_metrics['f1_macro']:.6f}")
    log_and_print(f"Weighted F1: {test_metrics['f1_weighted']:.6f}")
    log_and_print(f"Macro Precision: {test_metrics['precision_macro']:.6f}")
    log_and_print(f"Weighted Precision: {test_metrics['precision_weighted']:.6f}")
    log_and_print(f"Macro Recall: {test_metrics['recall_macro']:.6f}")
    log_and_print(f"Weighted Recall: {test_metrics['recall_weighted']:.6f}")
    log_and_print(f"Macro ROC-AUC: {test_metrics['roc_auc_macro']:.6f}")
    log_and_print(f"Macro PR-AUC: {test_metrics['pr_auc_macro']:.6f}")
    log_and_print(f"Training Time: {train_results['duration_seconds']}s")
    log_and_print(f"Test Evaluation Time: {test_metrics['inference_duration_seconds']}s")
    log_and_print(f"Checkpoint Locations: {checkpoint_dir}")
    log_and_print(f"Report Locations: {output_dir}")
    log_and_print("==================================================")


if __name__ == "__main__":
    main()
