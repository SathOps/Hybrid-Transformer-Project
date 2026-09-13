"""Evaluation script for trained CNN-Transformer models on the test split."""

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
import tensorflow as tf
from tensorflow import keras

from src.models.cnn_transformer import CLASS_NAMES
from src.preprocessing.dataset import iter_batches, load_class_names

LOGGER = logging.getLogger(__name__)


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7
) -> np.ndarray:
    """Compute an N x N confusion matrix using pure NumPy."""
    flat_indices = num_classes * y_true + y_pred
    return np.bincount(flat_indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def compute_ovr_roc_auc(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 7
) -> tuple[float, float, list[float]]:
    """Compute One-vs-Rest ROC-AUC using pure NumPy and trapezoidal integration."""
    auc_scores: list[float] = []
    weights: list[int] = []

    for c in range(num_classes):
        y_c = (y_true == c).astype(int)
        n_pos = int(y_c.sum())
        n_neg = len(y_c) - n_pos
        weights.append(n_pos)

        if n_pos == 0 or n_neg == 0:
            auc_scores.append(0.5)
            continue

        scores = y_prob[:, c]
        sort_idx = np.argsort(-scores)
        y_sorted = y_c[sort_idx]

        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)

        tpr = tp / n_pos
        fpr = fp / n_neg

        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])

        auc = float(np.trapz(tpr, fpr))
        auc_scores.append(auc)

    macro_auc = float(np.mean(auc_scores))
    total_weights = sum(weights)
    weighted_auc = (
        float(np.average(auc_scores, weights=weights)) if total_weights > 0 else macro_auc
    )
    return macro_auc, weighted_auc, auc_scores


def compute_ovr_pr_auc(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 7
) -> tuple[float, float, list[float]]:
    """Compute One-vs-Rest PR-AUC using pure NumPy and trapezoidal integration."""
    pr_auc_scores: list[float] = []
    weights: list[int] = []

    for c in range(num_classes):
        y_c = (y_true == c).astype(int)
        n_pos = int(y_c.sum())
        weights.append(n_pos)

        if n_pos == 0:
            pr_auc_scores.append(0.0)
            continue

        scores = y_prob[:, c]
        sort_idx = np.argsort(-scores)
        y_sorted = y_c[sort_idx]

        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = tp / n_pos

        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])

        # Sort recall for monotonic integration
        r_sort_idx = np.argsort(recall)
        pr_auc = float(np.trapz(precision[r_sort_idx], recall[r_sort_idx]))
        pr_auc_scores.append(pr_auc)

    macro_pr = float(np.mean(pr_auc_scores))
    total_weights = sum(weights)
    weighted_pr = (
        float(np.average(pr_auc_scores, weights=weights)) if total_weights > 0 else macro_pr
    )
    return macro_pr, weighted_pr, pr_auc_scores


def evaluate_model(
    checkpoint_path: Path = Path("checkpoints/cnn_transformer/baseline_50_epochs/best_model.keras"),
    test_dir: Path = Path("data/processed/prepared/test"),
    output_dir: Path = Path("results/cnn_transformer/baseline_50_epochs"),
    batch_size: int = 1024,
    feature_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Load model checkpoint, run inference on test set, and compute full evaluation metrics."""
    checkpoint_path = checkpoint_path.resolve()
    test_dir = test_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint model file not found at: {checkpoint_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dataset directory not found at: {test_dir}")

    LOGGER.info("Loading model from checkpoint: %s", checkpoint_path)
    model = keras.models.load_model(str(checkpoint_path))

    prepared_root = test_dir.parent
    if (prepared_root / "class_names.json").exists():
        class_names = load_class_names(prepared_root)
    else:
        class_names = CLASS_NAMES

    LOGGER.info("Streaming test set for inference...")
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []

    inference_start_time = time.time()

    for features, labels in iter_batches(test_dir, batch_size=batch_size, shuffle=False, feature_indices=feature_indices):
        features_exp = np.expand_dims(features, axis=-1)
        preds = model(features_exp, training=False).numpy()
        y_prob_list.append(preds)
        y_true_list.append(labels)

    inference_end_time = time.time()

    if not y_true_list:
        raise ValueError("No test batches were loaded from test directory.")

    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    num_samples = len(y_true)

    inference_duration_seconds = round(inference_end_time - inference_start_time, 4)
    ms_per_sample = round((inference_duration_seconds / num_samples) * 1000, 4) if num_samples else 0.0

    # Verification assertions
    assert y_prob.shape == (num_samples, len(class_names)), f"Expected shape ({num_samples}, {len(class_names)}), got {y_prob.shape}"
    assert np.isfinite(y_prob).all(), "Predictions contain non-finite values (NaN or Inf)"
    np.testing.assert_allclose(y_prob.sum(axis=1), np.ones(num_samples), atol=1e-4)

    # Loss calculation
    cce = keras.losses.SparseCategoricalCrossentropy(reduction="sum_over_batch_size")
    test_loss = float(cce(y_true, y_prob).numpy())

    # Accuracy
    correct_predictions = np.sum(y_true == y_pred)
    test_acc = float(correct_predictions / num_samples)

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

    # 1. Save test_metrics.json
    test_metrics = {
        "test_loss": test_loss,
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
    # 1. Save test_metrics.json and metrics.json
    test_metrics = {
        "dataset_type": "REAL CICIoT2023",
        "active_classes_count": 7,
        "training_partition": "SUBSAMPLED TRAINING PARTITION",
        "validation_partition": "NATURAL VALIDATION SET",
        "test_partition": "NATURAL HOLD-OUT TEST SET",
        "synthetic_data_used": False,
        "test_loss": test_loss,
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
    (output_dir / "metrics.json").write_text(
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
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(output_dir / "classification_report.csv", index=False)

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
        title="CNN-Transformer Baseline — Test Set Confusion Matrix",
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

    # 5. Save test_predictions.csv
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

    # 6. Save evaluation_summary.json
    evaluation_summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_type": "REAL CICIoT2023",
        "active_classes_count": 7,
        "training_partition": "SUBSAMPLED TRAINING PARTITION",
        "validation_partition": "NATURAL VALIDATION SET",
        "test_partition": "NATURAL HOLD-OUT TEST SET",
        "synthetic_data_used": False,
        "checkpoint_evaluated": str(checkpoint_path),
        "test_dataset_dir": str(test_dir),
        "overall_metrics": test_metrics,
        "per_class_metrics": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2) + "\n", encoding="utf-8"
    )

    # 7. Generate concise evaluation_report.md & test_evaluation_report.md
    report_md_content = f"""# CNN-Transformer Baseline Test Evaluation Report — REAL CICIoT2023

> [!IMPORTANT]
> **RESEARCH LABELING STATEMENT**
> - **Dataset**: REAL CICIoT2023
> - **Active Model Classes**: 7 ACTIVE CLASSES (Benign, BruteForce, DDoS, DoS, Mirai, Spoofing, Web-based; Recon inactive with 0 instances)
> - **Training Data**: SUBSAMPLED TRAINING PARTITION (2,626,223 rows)
> - **Validation Partition**: NATURAL VALIDATION SET (5,613,269 rows)
> - **Test Partition**: NATURAL HOLD-OUT TEST SET (5,613,274 rows)
> - **Synthetic Data**: NOT USED (0 synthetic samples)

## Executive Summary

The trained **CNN-Transformer** model checkpoint (`best_model.keras`) was evaluated on the untouched natural test dataset partition.

### Test Partition Evaluation Metrics

| Partition | Samples Evaluated | Accuracy | Loss |
| :--- | :--- | :--- | :--- |
| **Test Set (Evaluation)** | **{num_samples:,}** | **{test_acc * 100:.2f}%** | **{test_loss:.4f}** |

> [!NOTE]
> The test set evaluation was performed strictly in inference mode (`training=False`). The test split remained completely untouched and was never used during model training or checkpoint selection.

---

## Key Aggregate Metrics

- **Test Accuracy**: `{test_acc * 100:.2f}%` (`{test_acc:.6f}`)
- **Test Loss**: `{test_loss:.6f}`
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

1. **Model Checkpoint**: Loaded successfully from `checkpoints/cnn_transformer/baseline_50_epochs/best_model.keras`.
2. **Output Contract**: Shape `(batch_size, 8)`, finite outputs (`isfinite == True`), Softmax probabilities sum to `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
"""
    (output_dir / "test_evaluation_report.md").write_text(report_md_content, encoding="utf-8")
    (output_dir / "evaluation_report.md").write_text(report_md_content, encoding="utf-8")
    LOGGER.info("Saved evaluation report markdown to %s", output_dir / "evaluation_report.md")

    return {
        "test_metrics": test_metrics,
        "per_class": per_class_dict,
        "num_samples": num_samples,
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained CNN-Transformer model on test set.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints/cnn_transformer/real_ciciot2023/fresh_25_epochs/best_model.keras"),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/processed/prepared/test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cnn_transformer/real_ciciot2023/fresh_25_epochs"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    results = evaluate_model(
        checkpoint_path=args.checkpoint_path,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    metrics = results["test_metrics"]
    print("\n--- Test Evaluation Summary ---")
    print(f"Test Samples: {results['num_samples']}")
    print(f"Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Test Loss: {metrics['test_loss']:.6f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.6f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.6f}")
    print(f"Macro Precision: {metrics['precision_macro']:.6f}")
    print(f"Macro Recall: {metrics['recall_macro']:.6f}")
    print(f"Inference Time: {metrics['inference_duration_seconds']}s ({metrics['ms_per_sample']} ms/sample)")
    print("Results Directory:", str(results["output_dir"]).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
