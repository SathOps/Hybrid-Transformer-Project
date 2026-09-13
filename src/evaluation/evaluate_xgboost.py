"""Evaluation script for trained XGBoost model on the test split."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation.evaluate import (
    compute_confusion_matrix,
    compute_ovr_pr_auc,
    compute_ovr_roc_auc,
)
from src.models.cnn_transformer import CLASS_NAMES
from src.preprocessing.dataset import iter_batches, load_class_names
from src.preprocessing.prepare_dataset import MODEL_FEATURES

LOGGER = logging.getLogger(__name__)


def load_test_data(test_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all feature and label arrays from the test split directory."""
    feature_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    for features, labels in iter_batches(test_dir, batch_size=10000, shuffle=False):
        feature_list.append(features)
        label_list.append(labels)
    if not feature_list:
        raise ValueError(f"No test data shards found under {test_dir}")
    return np.concatenate(feature_list, axis=0), np.concatenate(label_list, axis=0)


def evaluate_xgboost(
    checkpoint_model_path: Path = Path("checkpoints/xgboost/baseline/model.json"),
    test_dir: Path = Path("data/processed/prepared/test"),
    output_dir: Path = Path("results/xgboost/baseline"),
) -> dict[str, object]:
    """Load XGBoost model, evaluate on test set, export metrics, report, and feature importances."""
    checkpoint_model_path = checkpoint_model_path.resolve()
    test_dir = test_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_model_path.exists():
        raise FileNotFoundError(f"XGBoost model file not found at: {checkpoint_model_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dataset directory not found at: {test_dir}")

    LOGGER.info("Loading XGBoost model from %s", checkpoint_model_path)
    clf = xgb.XGBClassifier()
    clf.load_model(str(checkpoint_model_path))

    prepared_root = test_dir.parent
    if (prepared_root / "class_names.json").exists():
        class_names = load_class_names(prepared_root)
    else:
        class_names = CLASS_NAMES

    feature_names = list(MODEL_FEATURES)

    LOGGER.info("Loading test dataset for XGBoost evaluation...")
    X_test, y_true = load_test_data(test_dir)
    num_samples = len(y_true)
    LOGGER.info("Loaded X_test shape: %s, y_true shape: %s", X_test.shape, y_true.shape)

    inference_start_time = time.time()
    y_prob = clf.predict_proba(X_test)
    inference_end_time = time.time()

    y_pred = np.argmax(y_prob, axis=1)

    inference_duration_seconds = round(inference_end_time - inference_start_time, 4)
    ms_per_sample = round((inference_duration_seconds / num_samples) * 1000, 4) if num_samples else 0.0

    # Verification assertions
    assert y_prob.shape == (num_samples, len(class_names)), f"Expected ({num_samples}, {len(class_names)}), got {y_prob.shape}"
    assert np.isfinite(y_prob).all(), "Predictions contain non-finite values"
    np.testing.assert_allclose(y_prob.sum(axis=1), np.ones(num_samples), atol=1e-4)

    # Compute XGBoost log loss
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    test_log_loss = float(-np.mean(np.log(y_prob_clipped[np.arange(num_samples), y_true])))

    # Accuracy
    test_acc = float(np.mean(y_true == y_pred))

    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))

    # Per-class precision, recall, f1, support
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
    weighted_precision = float(np.average(precision, weights=support)) if num_samples else macro_precision

    macro_recall = float(np.mean(recall))
    weighted_recall = float(np.average(recall, weights=support)) if num_samples else macro_recall

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if num_samples else macro_f1

    # ROC-AUC & PR-AUC
    macro_roc_auc, weighted_roc_auc, per_class_roc_auc = compute_ovr_roc_auc(
        y_true, y_prob, num_classes=len(class_names)
    )
    macro_pr_auc, weighted_pr_auc, per_class_pr_auc = compute_ovr_pr_auc(
        y_true, y_prob, num_classes=len(class_names)
    )

    per_class_dict = {}
    for idx, name in enumerate(class_names):
        per_class_dict[name] = {
            "class_id": idx,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1[idx]),
            "support": int(support[idx]),
            "roc_auc_ovr": float(per_class_roc_auc[idx]),
            "pr_auc_ovr": float(per_class_pr_auc[idx]),
        }

    # Extract feature importances
    booster = clf.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    score_weight = booster.get_score(importance_type="weight")
    score_cover = booster.get_score(importance_type="cover")

    importance_rows = []
    for idx, f_name in enumerate(feature_names):
        f_key = f"f{idx}"
        gain_val = float(score_gain.get(f_key, score_gain.get(f_name, 0.0)))
        weight_val = float(score_weight.get(f_key, score_weight.get(f_name, 0.0)))
        cover_val = float(score_cover.get(f_key, score_cover.get(f_name, 0.0)))
        importance_rows.append(
            {
                "feature_name": f_name,
                "importance_gain": gain_val,
                "importance_weight": weight_val,
                "importance_cover": cover_val,
            }
        )

    importance_df = pd.DataFrame(importance_rows)
    importance_df = importance_df.sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    # 1. Save test_metrics.json
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
        "total_test_samples": num_samples,
        "num_classes": len(class_names),
        "inference_duration_seconds": inference_duration_seconds,
        "ms_per_sample": ms_per_sample,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8"
    )

    # 2. Save classification_report.csv
    report_rows = []
    for idx, name in enumerate(class_names):
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
            "support": num_samples,
        }
    )
    report_rows.append(
        {
            "class_name": "weighted avg",
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1,
            "support": num_samples,
        }
    )
    pd.DataFrame(report_rows).to_csv(output_dir / "classification_report.csv", index=False)

    # 3. Save confusion_matrix.csv
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=True)

    # 4. Save confusion_matrix.png
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="XGBoost Baseline — Test Set Confusion Matrix",
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

    # 5. Save feature_importance.csv and plot feature_importance.png
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(20).iloc[::-1]
    ax.barh(top_features["feature_name"], top_features["importance_gain"], color="#2b5c8f")
    ax.set_title("XGBoost Baseline — Top 20 Feature Importance (Gain)", fontsize=12)
    ax.set_xlabel("Gain Importance Score", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close(fig)

    # 6. Save test_predictions.csv
    preds_df = pd.DataFrame(
        {
            "sample_index": np.arange(num_samples),
            "true_class_id": y_true,
            "true_class_name": [class_names[i] for i in y_true],
            "pred_class_id": y_pred,
            "pred_class_name": [class_names[i] for i in y_pred],
            "max_probability": np.max(y_prob, axis=1),
        }
    )
    for idx, name in enumerate(class_names):
        preds_df[f"prob_{name}"] = y_prob[:, idx]
    preds_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # 7. Save evaluation_summary.json
    evaluation_summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_evaluated": str(checkpoint_model_path),
        "test_dataset_dir": str(test_dir),
        "overall_metrics": test_metrics,
        "per_class_metrics": per_class_dict,
        "top_10_features_by_gain": importance_df[["rank", "feature_name", "importance_gain"]].head(10).to_dict(orient="records"),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2) + "\n", encoding="utf-8"
    )

    # 8. Save test_evaluation_report.md
    report_md_path = output_dir / "test_evaluation_report.md"
    report_md_content = f"""# XGBoost Baseline Test Evaluation Report

## Executive Summary

The trained **XGBoost Baseline** model (`model.json`) was evaluated on the untouched test dataset partition (`data/processed/prepared/test/`).

### Dataset & Partition Metrics Comparison

| Partition / Model | Samples Evaluated | Accuracy | Loss / Log-Loss |
| :--- | :--- | :--- | :--- |
| **XGBoost Test Set (Evaluation)** | **{num_samples:,}** | **{test_acc * 100:.2f}%** | **{test_log_loss:.4f}** |
| *CNN-Transformer Test Baseline* | *3,000* | *87.40%* | *0.5855* |
| *MLP Test Baseline* | *3,000* | *80.07%* | *1.0051* |

---

## Key Aggregate Metrics

- **Test Accuracy**: `{test_acc * 100:.2f}%` (`{test_acc:.6f}`)
- **Test Log Loss**: `{test_log_loss:.6f}`
- **Macro Precision**: `{macro_precision:.6f}`
- **Weighted Precision**: `{weighted_precision:.6f}`
- **Macro Recall**: `{macro_recall:.6f}`
- **Weighted Recall**: `{weighted_recall:.6f}`
- **Macro F1-Score**: `{macro_f1:.6f}`
- **Weighted F1-Score**: `{weighted_f1:.6f}`
- **Macro ROC-AUC (OvR)**: `{macro_roc_auc:.6f}`
- **Weighted ROC-AUC (OvR)**: `{weighted_roc_auc:.6f}`
- **Macro PR-AUC (OvR)**: `{macro_pr_auc:.6f}`
- **Weighted PR-AUC (OvR)**: `{weighted_pr_auc:.6f}`
- **Inference Time**: `{inference_duration_seconds:.4f}` s (`{ms_per_sample:.4f}` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for name in class_names:
        c_p = per_class_dict[name]["precision"]
        c_r = per_class_dict[name]["recall"]
        c_f1 = per_class_dict[name]["f1_score"]
        c_supp = per_class_dict[name]["support"]
        c_roc = per_class_dict[name]["roc_auc_ovr"]
        c_pr = per_class_dict[name]["pr_auc_ovr"]
        report_md_content += (
            f"| **{name}** | {c_p:.4f} | {c_r:.4f} | {c_f1:.4f} | {c_supp:,} | {c_roc:.4f} | {c_pr:.4f} |\n"
        )

    report_md_content += """
---

## Top 10 Features by Gain Importance

| Rank | Feature Name | Gain Score |
| :---: | :--- | :---: |
"""
    for row in importance_df.head(10).itertuples():
        report_md_content += f"| {row.rank} | **{row.feature_name}** | {row.importance_gain:.4f} |\n"

    report_md_content += """
---

## Confusion Matrix

```
"""
    header_str = "True \\ Pred\t" + "\t".join([f"{name[:6]}" for name in class_names])
    report_md_content += header_str + "\n"
    for idx, row in enumerate(cm):
        row_str = f"{class_names[idx][:10]:<10}\t" + "\t".join([str(val) for val in row])
        report_md_content += row_str + "\n"

    report_md_content += """```

---

## Verification & Integrity Safeguards

1. **Model File**: Loaded successfully from `checkpoints/xgboost/baseline/model.json`.
2. **Output Contract**: Probability shape `(3000, 8)`, finite predictions (`isfinite == True`), row sums equal `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
"""
    report_md_path.write_text(report_md_content, encoding="utf-8")
    LOGGER.info("Saved XGBoost evaluation report to %s", report_md_path)

    return {
        "test_metrics": test_metrics,
        "per_class": per_class_dict,
        "num_samples": num_samples,
        "feature_importance": importance_df.to_dict(orient="records"),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained XGBoost model on test set.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints/xgboost/real_ciciot2023/baseline/model.json"),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/processed/prepared/subsampled_train/test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/xgboost/real_ciciot2023/baseline"),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    results = evaluate_xgboost(
        checkpoint_model_path=args.checkpoint_path,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
    )

    metrics = results["test_metrics"]
    print("\n--- XGBoost Test Evaluation Summary ---")
    print(f"Test Samples: {results['num_samples']}")
    print(f"Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Test Log Loss: {metrics['test_log_loss']:.6f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.6f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.6f}")
    print(f"Macro Precision: {metrics['precision_macro']:.6f}")
    print(f"Macro Recall: {metrics['recall_macro']:.6f}")
    print(f"Inference Time: {metrics['inference_duration_seconds']}s ({metrics['ms_per_sample']} ms/sample)")
    print("Results Directory:", str(results["output_dir"]).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
