"""Orchestration script for Binary Particle Swarm Optimization (BPSO) Feature Selection on Real CICIoT2023."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb

from src.evaluation.evaluate import (
    compute_confusion_matrix,
    compute_ovr_pr_auc,
    compute_ovr_roc_auc,
)
from src.features.bpso import BinaryPSO
from src.models.cnn_transformer import CLASS_NAMES
from src.preprocessing.prepare_dataset import MODEL_FEATURES
from src.training.train_xgboost import load_split_data

LOGGER = logging.getLogger(__name__)

# Standard 46-feature XGBoost baseline benchmark values
BASELINE_METRICS = {
    "num_features": 46,
    "test_accuracy": 0.749095,
    "macro_f1": 0.703153,
    "weighted_f1": 0.774626,
    "macro_precision": 0.865403,
    "weighted_precision": 0.872807,
    "macro_recall": 0.694793,
    "weighted_recall": 0.749095,
    "macro_roc_auc": 0.972964,
    "macro_pr_auc": 0.763626,
    "training_duration_seconds": 50.68,
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

    checkpoint_dir = Path("checkpoints/bpso/real_ciciot2023/feature_selection").resolve()
    output_dir = Path("results/bpso/real_ciciot2023/feature_selection").resolve()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_log_path = output_dir / "task_output.log"
    bpso_csv_path = output_dir / "bpso_log.csv"

    def log_and_print(msg: str) -> None:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
        with open(task_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

    # Save feature index mapping JSON
    feature_names = list(MODEL_FEATURES)
    feature_map = {idx: name for idx, name in enumerate(feature_names)}
    (output_dir / "feature_index_mapping.json").write_text(
        json.dumps(feature_map, indent=2) + "\n", encoding="utf-8"
    )

    log_and_print("==================================================")
    log_and_print("STARTING REAL CICIoT2023 BPSO FEATURE SELECTION EXPERIMENT")
    log_and_print(f"Total Original Features: {len(feature_names)}")
    log_and_print(f"Data Dir: {data_dir}")
    log_and_print(f"Train Path: {train_path if 'train_path' in locals() else train_dir}")
    log_and_print(f"Val Path: {val_dir}")
    log_and_print(f"Test Path: {test_dir}")
    log_and_print(f"Checkpoint Dir: {checkpoint_dir}")
    log_and_print(f"Output Dir: {output_dir}")
    log_and_print("==================================================")

    # 1. Load Data for BPSO Fitness Evaluation (Fixed, reproducible subset 200,000 rows)
    LOGGER.info("Loading training partition for BPSO fitness evaluation...")
    X_train_full, y_train_full = load_split_data(train_dir)
    LOGGER.info("Loaded X_train_full shape: %s", X_train_full.shape)

    LOGGER.info("Loading validation partition for BPSO fitness evaluation...")
    X_val_full, y_val_full = load_split_data(val_dir)
    LOGGER.info("Loaded X_val_full shape: %s", X_val_full.shape)

    # Create reproducible fitness subset (200,000 samples)
    subset_rng = np.random.default_rng(42)
    subset_size_tr = min(200000, len(X_train_full))
    subset_size_va = min(200000, len(X_val_full))

    tr_idx = subset_rng.choice(len(X_train_full), size=subset_size_tr, replace=False)
    va_idx = subset_rng.choice(len(X_val_full), size=subset_size_va, replace=False)

    X_tr_fit = X_train_full[tr_idx]
    y_tr_fit = y_train_full[tr_idx]
    X_va_fit = X_val_full[va_idx]
    y_va_fit = y_val_full[va_idx]

    log_and_print(f"BPSO Fitness Subset Created: Train={len(X_tr_fit):,}, Val={len(X_va_fit):,} (Seed=42)")

    # 2. BPSO Initialization
    bpso = BinaryPSO(
        num_features=46,
        num_particles=20,
        max_iterations=20,
        w=0.7,
        c1=1.5,
        c2=1.5,
        v_min=-6.0,
        v_max=6.0,
        seed=42,
    )

    bpso_config = {
        "num_features": 46,
        "num_particles": 20,
        "max_iterations": 20,
        "inertia_weight_w": 0.7,
        "cognitive_c1": 1.5,
        "social_c2": 1.5,
        "v_bounds": [-6.0, 6.0],
        "random_seed": 42,
        "fitness_subset_train_size": len(X_tr_fit),
        "fitness_subset_val_size": len(X_va_fit),
        "fitness_model": "XGBoost (hist, n_estimators=30, max_depth=5)",
        "fitness_primary_metric": "Validation Macro F1",
    }
    (output_dir / "bpso_config.json").write_text(
        json.dumps(bpso_config, indent=2) + "\n", encoding="utf-8"
    )

    # Prepare BPSO log CSV
    with open(bpso_csv_path, "w", encoding="utf-8") as f:
        f.write(
            "iteration,gbest_val_macro_f1,gbest_val_accuracy,gbest_val_weighted_f1,"
            "gbest_val_macro_precision,gbest_val_macro_recall,num_selected_features,"
            "selected_feature_indices,iteration_duration_seconds,cumulative_duration_seconds\n"
        )

    bpso_start_time = time.time()
    cum_bpso_duration = 0.0

    def fitness_function(feature_mask: np.ndarray) -> tuple[float, dict[str, float]]:
        selected_idx = np.where(feature_mask == 1)[0]
        if len(selected_idx) == 0:
            return 0.0, {}

        # Train lightweight XGBoost model on selected features
        clf = xgb.XGBClassifier(
            n_estimators=30,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="multi:softprob",
            num_class=7,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr_fit[:, selected_idx], y_tr_fit)
        preds = clf.predict(X_va_fit[:, selected_idx])

        macro_f1 = float(f1_score(y_va_fit, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(y_va_fit, preds))
        weighted_f1 = float(f1_score(y_va_fit, preds, average="weighted", zero_division=0))
        macro_p = float(precision_score(y_va_fit, preds, average="macro", zero_division=0))
        macro_r = float(recall_score(y_va_fit, preds, average="macro", zero_division=0))

        metrics = {
            "val_macro_f1": macro_f1,
            "val_accuracy": acc,
            "val_weighted_f1": weighted_f1,
            "val_macro_precision": macro_p,
            "val_macro_recall": macro_r,
        }
        return macro_f1, metrics

    log_and_print("\n==================================================")
    log_and_print("RUNNING BPSO FEATURE SELECTION (20 PARTICLES x 20 ITERATIONS)...")
    log_and_print("==================================================")

    for iter_num in range(1, 21):
        iter_start = time.time()
        summary = bpso.step(fitness_function, iter_num)
        iter_duration = time.time() - iter_start
        cum_bpso_duration += iter_duration

        sel_indices = summary["selected_feature_indices"]
        sel_names = [feature_names[i] for i in sel_indices]
        sel_names_str = "; ".join(sel_names)

        # Write to bpso_log.csv
        with open(bpso_csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{iter_num},{summary['gbest_val_macro_f1']:.6f},{summary['gbest_val_accuracy']:.6f},"
                f"{summary['gbest_val_weighted_f1']:.6f},{summary['gbest_val_macro_precision']:.6f},"
                f"{summary['gbest_val_macro_recall']:.6f},{summary['num_selected_features']},"
                f"\"{sel_indices}\",{iter_duration:.2f},{cum_bpso_duration:.2f}\n"
            )

        # Save checkpoint state for resume/inspection
        state_data = {
            "iteration": iter_num,
            "gbest_score": bpso.gbest_score,
            "gbest_position": bpso.gbest_position.tolist(),
            "gbest_metrics": bpso.gbest_metrics,
            "history": bpso.history,
        }
        (checkpoint_dir / "latest_bpso_state.json").write_text(
            json.dumps(state_data, indent=2) + "\n", encoding="utf-8"
        )

        iter_msg = (
            f"\n==================================================\n"
            f"BPSO ITERATION {iter_num}/20 COMPLETED\n"
            f"Global Best Validation Macro F1: {summary['gbest_val_macro_f1']:.6f}\n"
            f"Global Best Validation Accuracy: {summary['gbest_val_accuracy']:.6f}\n"
            f"Global Best Validation Weighted F1: {summary['gbest_val_weighted_f1']:.6f}\n"
            f"Selected Features Count: {summary['num_selected_features']}/46 ({(summary['num_selected_features']/46)*100:.1f}%)\n"
            f"Selected Feature Indices: {sel_indices}\n"
            f"Iteration Time: {iter_duration:.2f}s\n"
            f"Cumulative Time: {cum_bpso_duration:.2f}s\n"
            "=================================================="
        )
        log_and_print(iter_msg)

    # 3. Save BPSO Results & Feature Selection Artifacts
    best_mask = bpso.gbest_position.tolist()
    best_indices = np.where(bpso.gbest_position == 1)[0].tolist()
    best_names = [feature_names[i] for i in best_indices]

    (output_dir / "best_feature_mask.json").write_text(
        json.dumps(best_mask, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "best_feature_indices.json").write_text(
        json.dumps(best_indices, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "best_feature_names.json").write_text(
        json.dumps(best_names, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "fitness_history.json").write_text(
        json.dumps(bpso.history, indent=2) + "\n", encoding="utf-8"
    )

    log_and_print("\n==================================================")
    log_and_print("BPSO FEATURE SELECTION OPTIMIZATION FINISHED")
    log_and_print(f"Original Feature Count: 46")
    log_and_print(f"Selected Feature Count: {len(best_indices)} ({len(best_indices)/46*100:.1f}%)")
    log_and_print(f"Best Iteration: {bpso.gbest_iteration}")
    log_and_print(f"Best Validation Macro F1: {bpso.gbest_score:.6f}")
    log_and_print(f"Selected Feature Indices: {best_indices}")
    log_and_print(f"Selected Feature Names: {best_names}")
    log_and_print(f"Total BPSO Search Time: {cum_bpso_duration:.2f}s")
    log_and_print("==================================================")

    # 4. Final Model Training on Selected Features
    log_and_print("\n==================================================")
    log_and_print("TRAINING FINAL XGBOOST MODEL ON SELECTED FEATURE SUBSET...")
    log_and_print(f"Training Features Subset Shape: {len(X_train_full):,} rows x {len(best_indices)} columns")
    log_and_print("==================================================")

    final_train_start = time.time()
    final_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="multi:softprob",
        num_class=7,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=10,
        n_jobs=-1,
    )

    X_train_sel = X_train_full[:, best_indices]
    X_val_sel = X_val_full[:, best_indices]

    final_clf.fit(
        X_train_sel,
        y_train_full,
        eval_set=[(X_train_sel, y_train_full), (X_val_sel, y_val_full)],
        verbose=10,
    )
    final_train_duration = round(time.time() - final_train_start, 2)

    # Save final model
    final_model_json_path = checkpoint_dir / "final_xgboost_model.json"
    final_clf.save_model(str(final_model_json_path))
    out_final_model_json = output_dir / "final_xgboost_model.json"
    final_clf.save_model(str(out_final_model_json))

    # 5. Final Test Evaluation ONCE on Untouched Test Set
    log_and_print("\nSTARTING UNTOUCHED NATURAL TEST SET EVALUATION (SELECTED FEATURES ONLY)...")
    LOGGER.info("Loading test partition for final evaluation...")
    X_test_full, y_test_full = load_split_data(test_dir)
    X_test_sel = X_test_full[:, best_indices]
    num_test_samples = len(y_test_full)

    eval_start_time = time.time()
    test_probs = final_clf.predict_proba(X_test_sel)
    eval_duration = round(time.time() - eval_start_time, 4)

    test_preds = np.argmax(test_probs, axis=1)

    # Compute metrics
    test_acc = float(np.mean(y_test_full == test_preds))
    eps = 1e-15
    y_prob_clipped = np.clip(test_probs, eps, 1 - eps)
    test_log_loss = float(-np.mean(np.log(y_prob_clipped[np.arange(num_test_samples), y_test_full])))

    cm = compute_confusion_matrix(y_test_full, test_preds, num_classes=len(CLASS_NAMES))

    tp = np.diag(cm).astype(float)
    fp = (cm.sum(axis=0) - tp).astype(float)
    fn = (cm.sum(axis=1) - tp).astype(float)
    support = cm.sum(axis=1).astype(int)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    macro_precision = float(np.mean(precision))
    weighted_precision = float(np.average(precision, weights=support))
    macro_recall = float(np.mean(recall))
    weighted_recall = float(np.average(recall, weights=support))
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support))

    macro_roc_auc, weighted_roc_auc, per_class_roc = compute_ovr_roc_auc(
        y_test_full, test_probs, num_classes=len(CLASS_NAMES)
    )
    macro_pr_auc, weighted_pr_auc, per_class_pr = compute_ovr_pr_auc(
        y_test_full, test_probs, num_classes=len(CLASS_NAMES)
    )

    per_class_dict = {}
    for idx, name in enumerate(CLASS_NAMES):
        per_class_dict[name] = {
            "class_id": idx,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1[idx]),
            "support": int(support[idx]),
            "roc_auc_ovr": float(per_class_roc[idx]),
            "pr_auc_ovr": float(per_class_pr[idx]),
        }

    # Save test_metrics.json
    test_metrics = {
        "test_log_loss": test_log_loss,
        "test_accuracy": test_acc,
        "precision_macro": macro_precision,
        "precision_weighted": weighted_precision,
        "recall_macro": macro_recall,
        "recall_weighted": weighted_recall,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "roc_auc_macro": macro_roc_auc,
        "roc_auc_weighted": weighted_roc_auc,
        "pr_auc_macro": macro_pr_auc,
        "pr_auc_weighted": weighted_pr_auc,
        "total_test_samples": num_test_samples,
        "selected_features_count": len(best_indices),
        "original_features_count": 46,
        "feature_reduction_percent": round((1 - len(best_indices) / 46) * 100, 2),
        "evaluation_duration_seconds": eval_duration,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8"
    )

    # Save classification_report.csv
    report_rows = []
    for idx, name in enumerate(CLASS_NAMES):
        report_rows.append(
            {
                "class_name": name,
                "precision": precision[idx],
                "recall": recall[idx],
                "f1_score": f1[idx],
                "support": support[idx],
            }
        )
    report_rows.append(
        {
            "class_name": "macro avg",
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "support": num_test_samples,
        }
    )
    report_rows.append(
        {
            "class_name": "weighted avg",
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1,
            "support": num_test_samples,
        }
    )
    pd.DataFrame(report_rows).to_csv(output_dir / "classification_report.csv", index=False)

    # Save confusion_matrix.csv & png
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv", index=True
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=f"BPSO XGBoost ({len(best_indices)} Features) — Test Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Save metadata.json
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "BPSO Feature Selection + XGBoost Baseline",
        "dataset_type": "REAL CICIoT2023",
        "active_classes_count": 7,
        "bpso_config": bpso_config,
        "feature_selection_summary": {
            "original_features": 46,
            "selected_features": len(best_indices),
            "feature_reduction_percent": round((1 - len(best_indices) / 46) * 100, 2),
            "best_bpso_iteration": bpso.gbest_iteration,
            "best_validation_macro_f1": bpso.gbest_score,
            "selected_feature_indices": best_indices,
            "selected_feature_names": best_names,
        },
        "final_xgboost_metrics": test_metrics,
        "comparison_against_46_feature_baseline": {
            "baseline_46_features_accuracy": BASELINE_METRICS["test_accuracy"],
            "baseline_46_features_macro_f1": BASELINE_METRICS["macro_f1"],
            "baseline_46_features_weighted_f1": BASELINE_METRICS["weighted_f1"],
            "accuracy_change_pct": round((test_acc - BASELINE_METRICS["test_accuracy"]) * 100, 2),
            "macro_f1_change": round(macro_f1 - BASELINE_METRICS["macro_f1"], 6),
            "weighted_f1_change": round(weighted_f1 - BASELINE_METRICS["weighted_f1"], 6),
        },
        "paths": {
            "bpso_config": str(output_dir / "bpso_config.json"),
            "bpso_log_csv": str(bpso_csv_path),
            "best_feature_mask_json": str(output_dir / "best_feature_mask.json"),
            "best_feature_indices_json": str(output_dir / "best_feature_indices.json"),
            "best_feature_names_json": str(output_dir / "best_feature_names.json"),
            "final_model_json": str(out_final_model_json),
            "test_metrics_json": str(output_dir / "test_metrics.json"),
            "metadata_json": str(output_dir / "metadata.json"),
            "task_output_log": str(task_log_path),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Generate evaluation_report.md
    report_md = f"""# BPSO Feature Selection & XGBoost Evaluation Report — REAL CICIoT2023

## Executive Summary

Binary Particle Swarm Optimization (BPSO) was applied to select an optimal feature subset from the **46** original preprocessed CICIoT2023 features.
The BPSO fitness objective optimized **Validation Macro F1-Score** using a fixed, reproducible subset ($N=200,000$) sampled strictly from training and validation partitions with `seed=42`. Test data was kept **100% untouched** during feature selection.

### BPSO Feature Reduction Summary
- **Original Feature Count**: `46`
- **BPSO Selected Feature Count**: `{len(best_indices)}` (`{len(best_indices)/46*100:.1f}%` retained, `{round((1 - len(best_indices)/46)*100, 2)}%` reduction)
- **Best BPSO Iteration**: Epoch/Iteration `{bpso.gbest_iteration}` / `20`
- **Best BPSO Validation Macro F1**: `{bpso.gbest_score:.6f}`

---

## Benchmark Comparison: BPSO Selected Subset vs. Full 46-Feature Baseline

| Metric | Full 46-Feature XGBoost Baseline | **BPSO Selected ({len(best_indices)} Features) XGBoost** | Absolute Change |
| :--- | :---: | :---: | :---: |
| **Feature Count** | 46 | **{len(best_indices)}** | **-{46 - len(best_indices)} features (-{round((1 - len(best_indices)/46)*100, 1)}%)** |
| **Test Accuracy** | 74.91% (`0.749095`) | **{test_acc * 100:.2f}% (`{test_acc:.6f}`)** | **{test_acc - BASELINE_METRICS['test_accuracy']:+.4f} ({((test_acc - BASELINE_METRICS['test_accuracy'])*100):+.2f}%)** |
| **Test Log Loss** | 0.420332 | **{test_log_loss:.6f}** | **{test_log_loss - 0.420332:+.6f}** |
| **Macro F1-Score** | 0.703153 | **{macro_f1:.6f}** | **{macro_f1 - BASELINE_METRICS['macro_f1']:+.6f}** |
| **Weighted F1-Score** | 0.774626 | **{weighted_f1:.6f}** | **{weighted_f1 - BASELINE_METRICS['weighted_f1']:+.6f}** |
| **Macro Precision** | 0.865403 | **{macro_precision:.6f}** | **{macro_precision - BASELINE_METRICS['macro_precision']:+.6f}** |
| **Weighted Precision** | 0.872807 | **{weighted_precision:.6f}** | **{weighted_precision - BASELINE_METRICS['weighted_precision']:+.6f}** |
| **Macro Recall** | 0.694793 | **{macro_recall:.6f}** | **{macro_recall - BASELINE_METRICS['macro_recall']:+.6f}** |
| **Weighted Recall** | 0.749095 | **{weighted_recall:.6f}** | **{weighted_recall - BASELINE_METRICS['weighted_recall']:+.6f}** |
| **Macro ROC-AUC** | 0.972964 | **{macro_roc_auc:.6f}** | **{macro_roc_auc - BASELINE_METRICS['macro_roc_auc']:+.6f}** |
| **Macro PR-AUC** | 0.763626 | **{macro_pr_auc:.6f}** | **{macro_pr_auc - BASELINE_METRICS['macro_pr_auc']:+.6f}** |
| **Training Time** | 50.68 s | **{final_train_duration:.2f} s** | **{final_train_duration - 50.68:+.2f} s** |

---

## Selected Features List ({len(best_indices)} Features)

| Selected Index | Feature Name | Original Index |
| :---: | :--- | :---: |
"""
    for idx in best_indices:
        report_md += f"| **{idx}** | `{feature_names[idx]}` | {idx} |\n"

    report_md += f"""
---

## Per-Class Test Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
    for name in CLASS_NAMES:
        c_p = per_class_dict[name]["precision"]
        c_r = per_class_dict[name]["recall"]
        c_f1 = per_class_dict[name]["f1_score"]
        c_supp = per_class_dict[name]["support"]
        report_md += f"| **{name}** | {c_p:.4f} | {c_r:.4f} | {c_f1:.4f} | {c_supp:,} |\n"

    (output_dir / "evaluation_report.md").write_text(report_md, encoding="utf-8")

    final_summary_msg = (
        "\n==================================================\n"
        "FINAL BPSO FEATURE SELECTION & TEST EVALUATION SUMMARY\n"
        "==================================================\n"
        f"Original Features: 46\n"
        f"Selected Features: {len(best_indices)} ({len(best_indices)/46*100:.1f}%)\n"
        f"Feature Reduction: {round((1 - len(best_indices)/46)*100, 2)}%\n"
        f"BPSO Best Iteration: {bpso.gbest_iteration}\n"
        f"BPSO Best Validation Macro F1: {bpso.gbest_score:.6f}\n"
        f"Selected Feature Indices: {best_indices}\n"
        f"Selected Feature Names: {best_names}\n"
        f"Test Accuracy: {test_acc * 100:.2f}% ({test_acc:.6f})\n"
        f"Test Log Loss: {test_log_loss:.6f}\n"
        f"Macro F1: {macro_f1:.6f}\n"
        f"Weighted F1: {weighted_f1:.6f}\n"
        f"Macro Precision: {macro_precision:.6f}\n"
        f"Weighted Precision: {weighted_precision:.6f}\n"
        f"Macro Recall: {macro_recall:.6f}\n"
        f"Weighted Recall: {weighted_recall:.6f}\n"
        f"Macro ROC-AUC: {macro_roc_auc:.6f}\n"
        f"Macro PR-AUC: {macro_pr_auc:.6f}\n"
        f"BPSO Search Time: {cum_bpso_duration:.2f}s\n"
        f"Final Model Training Time: {final_train_duration:.2f}s\n"
        f"Test Evaluation Time: {eval_duration:.4f}s\n"
        f"Checkpoint Locations: {checkpoint_dir}\n"
        f"Report Locations: {output_dir}\n"
        "=================================================="
    )
    log_and_print(final_summary_msg)


if __name__ == "__main__":
    main()
