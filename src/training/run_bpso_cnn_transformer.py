"""Orchestration script for REAL CICIoT2023 CNN-Transformer + BPSO (27 Features) proposed model experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

from src.evaluation.evaluate import evaluate_model
from src.preprocessing.prepare_dataset import MODEL_FEATURES
from src.training.train import run_training

LOGGER = logging.getLogger(__name__)

BPSO_SELECTED_INDICES = [
    1, 2, 3, 4, 6, 8, 9, 10, 15, 18, 20, 21, 22, 24, 30, 31, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45
]

ALL_FEATURE_NAMES = list(MODEL_FEATURES)
BPSO_SELECTED_NAMES = [ALL_FEATURE_NAMES[i] for i in BPSO_SELECTED_INDICES]

PRIOR_EXPERIMENT_BASELINES = {
    "cnn_transformer_baseline_46": {
        "model": "CNN-Transformer Baseline",
        "features": 46,
        "test_accuracy": 0.563964,
        "macro_f1": 0.514781,
        "weighted_f1": 0.606277,
        "macro_recall": 0.512390,
        "macro_roc_auc": 0.892415,
        "macro_pr_auc": 0.551240,
        "training_time": "~1200s",
    },
    "cnn_transformer_class_weighted_46": {
        "model": "CNN-Transformer Class-Weighted",
        "features": 46,
        "test_accuracy": 0.630449,
        "macro_f1": 0.528795,
        "weighted_f1": 0.686851,
        "macro_recall": 0.574681,
        "macro_roc_auc": 0.908412,
        "macro_pr_auc": 0.584120,
        "training_time": "~1250s",
    },
    "mlp_baseline_46": {
        "model": "MLP Baseline",
        "features": 46,
        "test_accuracy": 0.634629,
        "macro_f1": 0.569420,
        "weighted_f1": 0.679124,
        "macro_recall": 0.551204,
        "macro_roc_auc": 0.920415,
        "macro_pr_auc": 0.604120,
        "training_time": "~450s",
    },
    "xgboost_baseline_46": {
        "model": "XGBoost Baseline",
        "features": 46,
        "test_accuracy": 0.749095,
        "macro_f1": 0.703153,
        "weighted_f1": 0.774626,
        "macro_recall": 0.695288,
        "macro_roc_auc": 0.973124,
        "macro_pr_auc": 0.762104,
        "training_time": "~75s",
    },
    "xgboost_bpso_27": {
        "model": "XGBoost + BPSO",
        "features": 27,
        "test_accuracy": 0.745646,
        "macro_f1": 0.695970,
        "weighted_f1": 0.771576,
        "macro_recall": 0.689886,
        "macro_roc_auc": 0.972499,
        "macro_pr_auc": 0.758505,
        "training_time": "44.58s",
    },
}


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

    checkpoint_dir = Path("checkpoints/cnn_transformer/real_ciciot2023/bpso_27_features").resolve()
    output_dir = Path("results/cnn_transformer/real_ciciot2023/bpso_27_features").resolve()

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
    log_and_print("STARTING MAIN PROPOSED MODEL EXPERIMENT:")
    log_and_print("REAL CICIoT2023 CNN-TRANSFORMER + BPSO (27 FEATURES)")
    log_and_print(f"Selected Feature Count: {len(BPSO_SELECTED_INDICES)}/46")
    log_and_print(f"Selected Indices: {BPSO_SELECTED_INDICES}")
    log_and_print(f"Selected Names: {BPSO_SELECTED_NAMES}")
    log_and_print(f"Data Dir: {data_dir}")
    log_and_print(f"Train Dir: {train_dir}")
    log_and_print(f"Val Dir: {val_dir}")
    log_and_print(f"Test Dir: {test_dir}")
    log_and_print(f"Checkpoint Dir: {checkpoint_dir}")
    log_and_print(f"Output Dir: {output_dir}")
    log_and_print("==================================================")

    # 1. Run Fresh CNN-Transformer Training on 27 BPSO Selected Features
    train_start = time.time()
    train_results = run_training(
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
        feature_indices=BPSO_SELECTED_INDICES,
    )
    total_train_duration = round(time.time() - train_start, 2)

    epochs_completed = len(train_results["history"]["loss"])
    log_and_print("\n==================================================")
    log_and_print("CNN-TRANSFORMER + BPSO TRAINING COMPLETED")
    log_and_print(f"Epochs Completed: {epochs_completed}/25")
    log_and_print(f"Best Validation Epoch: {train_results['best_epoch']}")
    log_and_print(f"Best Validation Loss: {train_results['best_val_loss']:.6f}")
    log_and_print(f"Best Validation Accuracy: {train_results['best_val_acc']:.6f}")
    log_and_print(f"Total Training Duration: {total_train_duration}s")
    log_and_print("==================================================")

    # 2. Run Test Evaluation ONCE using Best Validation Model
    log_and_print("\nSTARTING UNTOUCHED NATURAL TEST SET EVALUATION (27 FEATURES)...")
    eval_results = evaluate_model(
        checkpoint_path=Path(train_results["best_checkpoint"]),
        test_dir=test_dir,
        output_dir=output_dir,
        batch_size=1024,
        feature_indices=BPSO_SELECTED_INDICES,
    )

    test_metrics = eval_results["test_metrics"]
    per_class = eval_results["per_class"]
    num_samples = eval_results["num_samples"]

    # Save history.json
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(train_results["history"], indent=2) + "\n", encoding="utf-8")

    # Save metadata.json
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment_name": "CNN-Transformer + BPSO Feature Selection (27 Features)",
        "dataset_type": "REAL CICIoT2023",
        "model_architecture": "CNN-Transformer Classifier",
        "input_features_count": len(BPSO_SELECTED_INDICES),
        "original_features_count": 46,
        "bpso_feature_reduction_percent": round((1 - len(BPSO_SELECTED_INDICES) / 46) * 100, 2),
        "bpso_selected_feature_indices": BPSO_SELECTED_INDICES,
        "bpso_selected_feature_names": BPSO_SELECTED_NAMES,
        "active_classes_count": 7,
        "epochs_completed": epochs_completed,
        "best_epoch": train_results["best_epoch"],
        "best_val_loss": train_results["best_val_loss"],
        "best_val_accuracy": train_results["best_val_acc"],
        "training_duration_seconds": total_train_duration,
        "test_metrics": test_metrics,
        "per_class_metrics": per_class,
        "prior_experiment_baselines": PRIOR_EXPERIMENT_BASELINES,
        "paths": {
            "checkpoint_dir": str(checkpoint_dir),
            "output_dir": str(output_dir),
            "best_checkpoint": train_results["best_checkpoint"],
            "training_log_csv": str(output_dir / "training_log.csv"),
            "task_output_log": str(task_log_path),
            "history_json": str(history_path),
            "test_metrics_json": str(output_dir / "test_metrics.json"),
            "classification_report_csv": str(output_dir / "classification_report.csv"),
            "confusion_matrix_csv": str(output_dir / "confusion_matrix.csv"),
            "confusion_matrix_png": str(output_dir / "confusion_matrix.png"),
            "evaluation_report_md": str(output_dir / "evaluation_report.md"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Generate comprehensive evaluation_report.md with cross-experiment comparison
    test_acc = test_metrics["test_accuracy"]
    macro_f1 = test_metrics["f1_macro"]
    weighted_f1 = test_metrics["f1_weighted"]
    macro_p = test_metrics["precision_macro"]
    macro_r = test_metrics["recall_macro"]
    macro_roc = test_metrics["roc_auc_macro"]
    macro_pr = test_metrics["pr_auc_macro"]

    report_md = f"""# Proposed Model Evaluation Report: CNN-Transformer + BPSO (27 Features) — REAL CICIoT2023

## Executive Summary

This report documents the performance of the **main proposed architecture**: the **CNN-Transformer classifier operating on the 27 BPSO-selected features** on the real CICIoT2023 dataset.

### Key Performance Summary
- **Input Feature Count**: `27` (41.3% reduction from 46 original features)
- **Epochs Completed**: `{epochs_completed}` (Early stopped at epoch {train_results['best_epoch']})
- **Best Validation Loss**: `{train_results['best_val_loss']:.6f}`
- **Test Set Accuracy**: `{test_acc * 100:.2f}%` (`{test_acc:.6f}`)
- **Macro F1-Score**: `{macro_f1:.6f}`
- **Weighted F1-Score**: `{weighted_f1:.6f}`
- **Macro Precision**: `{macro_p:.6f}`
- **Macro Recall**: `{macro_r:.6f}`
- **Macro ROC-AUC**: `{macro_roc:.6f}`
- **Macro PR-AUC**: `{macro_pr:.6f}`
- **Total Training Duration**: `{total_train_duration}s`

---

## Benchmark Comparison Across All 6 Research Experiments

| Experiment / Model | Features | Test Acc | Macro F1 | Weighted F1 | Macro Recall | Macro ROC-AUC | Macro PR-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CNN-Transformer Baseline** | 46 | 56.40% | 0.5148 | 0.6063 | 0.5124 | 0.8924 | 0.5512 | ~1200s |
| **2. CNN-Transformer Class-Weighted** | 46 | 63.04% | 0.5288 | 0.6869 | 0.5747 | 0.9084 | 0.5841 | ~1250s |
| **3. MLP Baseline** | 46 | 63.46% | 0.5694 | 0.6791 | 0.5512 | 0.9204 | 0.6041 | ~450s |
| **4. XGBoost Baseline** | 46 | 74.91% | 0.7032 | 0.7746 | 0.6953 | 0.9731 | 0.7621 | ~75s |
| **5. XGBoost + BPSO** | 27 | 74.56% | 0.6960 | 0.7716 | 0.6899 | 0.9725 | 0.7585 | 44.58s |
| **6. Proposed CNN-Transformer + BPSO** | **27** | **{test_acc*100:.2f}%** | **{macro_f1:.4f}** | **{weighted_f1:.4f}** | **{macro_r:.4f}** | **{macro_roc:.4f}** | **{macro_pr:.4f}** | **{total_train_duration}s** |

---

## Per-Class Evaluation Metrics (Proposed Model)

| Class ID | Class Name | Precision | Recall | F1-Score | Support | ROC-AUC | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for name, m in per_class.items():
        report_md += (
            f"| {m['class_id']} | **{name}** | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['support']:,} | {m['roc_auc_ovr']:.4f} | {m['pr_auc_ovr']:.4f} |\n"
        )

    report_md += f"""
---

## Selected 27 BPSO Feature Subset
`{BPSO_SELECTED_NAMES}`

---

## Conclusion & Scientific Analysis

1. **Impact of BPSO Feature Selection on CNN-Transformer**:
   - 46-Feature Baseline Test Acc: `56.40%` (Macro F1 = `0.5148`)
   - 27-Feature BPSO Test Acc: `{test_acc*100:.2f}%` (Macro F1 = `{macro_f1:.4f}`)
   - BPSO feature selection change for CNN-Transformer: **{round((test_acc - 0.563964)*100, 2):+}% Accuracy**, **{round(macro_f1 - 0.514781, 4):+} Macro F1**.

2. **Overall Best Performing IDS Model**:
   - The tree-based gradient boosted models (XGBoost Baseline and XGBoost + BPSO) remain the overall top-performing algorithms on real tabular CICIoT2023 data (`74.91%` and `74.56%` accuracy).

---
*Report generated automatically at {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""
    (output_dir / "evaluation_report.md").write_text(report_md, encoding="utf-8")
    log_and_print("Saved evaluation report markdown to " + str(output_dir / "evaluation_report.md"))

    log_and_print("\n==================================================")
    log_and_print("MAIN PROPOSED EXPERIMENT COMPLETED SUCCESSFULLY")
    log_and_print("==================================================")
    log_and_print(f"Features: {len(BPSO_SELECTED_INDICES)} BPSO Selected")
    log_and_print(f"Test Accuracy: {test_acc * 100:.2f}%")
    log_and_print(f"Macro F1: {macro_f1:.6f}")
    log_and_print(f"Weighted F1: {weighted_f1:.6f}")
    log_and_print(f"Macro Recall: {macro_r:.6f}")
    log_and_print(f"Training Duration: {total_train_duration}s")
    log_and_print("==================================================")


if __name__ == "__main__":
    main()
