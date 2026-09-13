"""Adaptive Ensemble Weight Optimization Module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LOGGER = logging.getLogger(__name__)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute comprehensive validation evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = Path("results/optimized_hybrid/real_ciciot2023").resolve()
    val_probs_path = out_dir / "validation_probabilities.npz"

    if not val_probs_path.exists():
        raise FileNotFoundError(f"Validation probabilities not found at {val_probs_path}")

    LOGGER.info("Loading cached validation probability matrices...")
    data = np.load(val_probs_path)
    P_xgb = data["val_probs_xgb"]
    P_mlp = data["val_probs_mlp"]
    P_cnn = data["val_probs_cnn"]
    y_val = data["y_val"]

    num_samples = len(y_val)
    LOGGER.info("Validation set size: %d samples across 7 active classes.", num_samples)

    canonical_setups = [
        ("CNN Only", 1.0, 0.0, 0.0),
        ("XGBoost Only", 0.0, 1.0, 0.0),
        ("MLP Only", 0.0, 0.0, 1.0),
        ("CNN + XGBoost", 0.5, 0.5, 0.0),
        ("CNN + MLP", 0.5, 0.0, 0.5),
        ("XGBoost + MLP", 0.0, 0.5, 0.5),
        ("Equal Weights (1/3, 1/3, 1/3)", 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ]

    search_records = []
    best_score = -1.0
    best_weights = (0.0, 1.0, 0.0)

    # 1. Evaluate Canonical Setups
    for name, w_cnn, w_xgb, w_mlp in canonical_setups:
        P_combo = w_cnn * P_cnn + w_xgb * P_xgb + w_mlp * P_mlp
        preds = np.argmax(P_combo, axis=1)
        m = evaluate_metrics(y_val, preds)

        search_records.append({
            "setup_name": name,
            "w_cnn": round(w_cnn, 4),
            "w_xgb": round(w_xgb, 4),
            "w_mlp": round(w_mlp, 4),
            "val_macro_f1": round(m["macro_f1"], 6),
            "val_accuracy": round(m["accuracy"], 6),
            "val_weighted_f1": round(m["weighted_f1"], 6),
            "val_macro_precision": round(m["macro_precision"], 6),
            "val_macro_recall": round(m["macro_recall"], 6),
        })

        if m["macro_f1"] > best_score:
            best_score = m["macro_f1"]
            best_weights = (w_cnn, w_xgb, w_mlp)

    # 2. Grid Search over Weights (Step Size 0.02 for speed)
    LOGGER.info("Searching constrained ensemble weight space (w_cnn + w_xgb + w_mlp = 1)...")
    steps = np.linspace(0.0, 1.0, 51)  # 0.00, 0.02, 0.04, ..., 1.00

    for w_cnn in steps:
        for w_xgb in steps:
            w_mlp = 1.0 - w_cnn - w_xgb
            if w_mlp < -1e-5 or w_mlp > 1.0 + 1e-5:
                continue
            w_mlp = max(0.0, min(1.0, w_mlp))

            P_combo = w_cnn * P_cnn + w_xgb * P_xgb + w_mlp * P_mlp
            preds = np.argmax(P_combo, axis=1)
            macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

            if macro_f1 > best_score:
                best_score = macro_f1
                best_weights = (float(w_cnn), float(w_xgb), float(w_mlp))

    w_cnn_opt, w_xgb_opt, w_mlp_opt = best_weights
    LOGGER.info("Optimal Ensemble Weights Found: w_cnn=%.4f, w_xgb=%.4f, w_mlp=%.4f | Val Macro F1=%.6f",
                w_cnn_opt, w_xgb_opt, w_mlp_opt, best_score)

    P_opt = w_cnn_opt * P_cnn + w_xgb_opt * P_xgb + w_mlp_opt * P_mlp
    preds_opt = np.argmax(P_opt, axis=1)
    opt_metrics = evaluate_metrics(y_val, preds_opt)

    search_records.append({
        "setup_name": "Optimal Adaptive Ensemble",
        "w_cnn": round(w_cnn_opt, 4),
        "w_xgb": round(w_xgb_opt, 4),
        "w_mlp": round(w_mlp_opt, 4),
        "val_macro_f1": round(opt_metrics["macro_f1"], 6),
        "val_accuracy": round(opt_metrics["accuracy"], 6),
        "val_weighted_f1": round(opt_metrics["weighted_f1"], 6),
        "val_macro_precision": round(opt_metrics["macro_precision"], 6),
        "val_macro_recall": round(opt_metrics["macro_recall"], 6),
    })

    search_df = pd.DataFrame(search_records)
    search_df.to_csv(out_dir / "ensemble_weight_search.csv", index=False)

    best_ensemble_config = {
        "val_macro_f1": best_score,
        "weights": {
            "w_cnn": round(w_cnn_opt, 4),
            "w_xgb": round(w_xgb_opt, 4),
            "w_mlp": round(w_mlp_opt, 4),
        },
        "validation_metrics": opt_metrics,
        "setups_evaluated": search_records,
    }

    with open(out_dir / "best_ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(best_ensemble_config, f, indent=2)

    LOGGER.info("Ensemble weight optimization finished successfully.")


if __name__ == "__main__":
    main()
