"""Hyperparameter Optimization Module for Base Models (XGBoost, MLP, CNN-Transformer)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from src.features.bpso import BinaryPSO
from src.preprocessing.dataset import iter_batches
from src.training.train_xgboost import load_split_data

LOGGER = logging.getLogger(__name__)

BPSO_SELECTED_INDICES = [
    1, 2, 3, 4, 6, 8, 9, 10, 15, 18, 20, 21, 22, 24, 30, 31, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45
]

FITNESS_SEED = 42
FITNESS_SUBSET_SIZE = 200000


def evaluate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> float:
    """Compute Macro F1-score safely across all active target classes."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, out_dir: Path) -> dict:
    """Perform controlled hyperparameter search for XGBoost on the BPSO 27-feature subset."""
    import xgboost as xgb

    LOGGER.info("Starting XGBoost Hyperparameter Search (Target: Validation Macro F1)...")

    search_space = [
        {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 100, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 2.0},
        {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    ]

    trials = []
    best_score = -1.0
    best_config = search_space[0]

    for idx, params in enumerate(search_space, 1):
        t0 = time.time()
        clf = xgb.XGBClassifier(
            **params,
            tree_method="hist",
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=FITNESS_SEED,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = clf.predict(X_val)
        
        macro_f1 = evaluate_macro_f1(y_val, preds)
        acc = float((preds == y_val).mean())
        dur = round(time.time() - t0, 2)

        trial_record = {
            "trial_id": idx,
            "val_macro_f1": round(macro_f1, 6),
            "val_accuracy": round(acc, 6),
            "duration_seconds": dur,
            **params,
        }
        trials.append(trial_record)
        LOGGER.info("XGBoost Trial %d/%d: Val Macro F1=%.6f, Acc=%.4f (%.2fs)", idx, len(search_space), macro_f1, acc, dur)

        if macro_f1 > best_score:
            best_score = macro_f1
            best_config = params

    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(out_dir / "xgboost_trials.csv", index=False)
    with open(out_dir / "best_xgboost_config.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_macro_f1": best_score, "hyperparameters": best_config}, f, indent=2)

    return {"best_config": best_config, "best_val_macro_f1": best_score}


def tune_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, out_dir: Path) -> dict:
    """Perform controlled hyperparameter search for MLP on the BPSO 27-feature subset."""
    import tensorflow as tf
    from tensorflow import keras

    LOGGER.info("Starting MLP Hyperparameter Search (Target: Validation Macro F1)...")

    search_space = [
        {"layers": [256, 128, 64], "learning_rate": 0.00005, "dropout_rate": 0.05, "batch_size": 1024, "epochs": 10},
        {"layers": [512, 256, 128], "learning_rate": 0.0001, "dropout_rate": 0.05, "batch_size": 1024, "epochs": 10},
        {"layers": [256, 128], "learning_rate": 0.0001, "dropout_rate": 0.0, "batch_size": 512, "epochs": 10},
        {"layers": [512, 256, 128, 64], "learning_rate": 0.00005, "dropout_rate": 0.1, "batch_size": 1024, "epochs": 10},
    ]

    trials = []
    best_score = -1.0
    best_config = search_space[0]

    for idx, params in enumerate(search_space, 1):
        t0 = time.time()
        tf.random.set_seed(FITNESS_SEED)
        
        inputs = keras.Input(shape=(27,), name="input_features")
        x = inputs
        for l_idx, units in enumerate(params["layers"]):
            x = keras.layers.Dense(units, activation="relu", name=f"dense_{units}_{l_idx}")(x)
            if params["dropout_rate"] > 0:
                x = keras.layers.Dropout(params["dropout_rate"], name=f"dropout_{l_idx}")(x)
        logits = keras.layers.Dense(7, name="logits")(x)
        outputs = keras.layers.Activation("softmax", name="softmax")(logits)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            verbose=0,
        )

        probs = model.predict(X_val, batch_size=params["batch_size"], verbose=0)
        preds = np.argmax(probs, axis=1)

        macro_f1 = evaluate_macro_f1(y_val, preds)
        acc = float((preds == y_val).mean())
        dur = round(time.time() - t0, 2)

        trial_record = {
            "trial_id": idx,
            "val_macro_f1": round(macro_f1, 6),
            "val_accuracy": round(acc, 6),
            "duration_seconds": dur,
            "architecture": str(params["layers"]),
            "learning_rate": params["learning_rate"],
            "dropout_rate": params["dropout_rate"],
            "batch_size": params["batch_size"],
        }
        trials.append(trial_record)
        LOGGER.info("MLP Trial %d/%d: Val Macro F1=%.6f, Acc=%.4f (%.2fs)", idx, len(search_space), macro_f1, acc, dur)

        if macro_f1 > best_score:
            best_score = macro_f1
            best_config = params

    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(out_dir / "mlp_trials.csv", index=False)
    with open(out_dir / "best_mlp_config.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_macro_f1": best_score, "hyperparameters": best_config}, f, indent=2)

    return {"best_config": best_config, "best_val_macro_f1": best_score}


def tune_cnn_transformer(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, out_dir: Path) -> dict:
    """Perform controlled hyperparameter search for CNN-Transformer on the BPSO 27-feature subset."""
    import tensorflow as tf
    from tensorflow import keras
    from src.models.cnn_transformer import build_model

    LOGGER.info("Starting CNN-Transformer Hyperparameter Search (Target: Validation Macro F1)...")

    search_space = [
        {"learning_rate": 0.00005, "batch_size": 1024, "epochs": 5},
        {"learning_rate": 0.0001, "batch_size": 1024, "epochs": 5},
        {"learning_rate": 0.00002, "batch_size": 512, "epochs": 5},
    ]

    trials = []
    best_score = -1.0
    best_config = search_space[0]

    X_tr_exp = np.expand_dims(X_train, axis=-1)
    X_va_exp = np.expand_dims(X_val, axis=-1)

    for idx, params in enumerate(search_space, 1):
        t0 = time.time()
        tf.random.set_seed(FITNESS_SEED)

        model = build_model(input_shape=(27, 1), num_classes=7)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        model.fit(
            X_tr_exp,
            y_train,
            validation_data=(X_va_exp, y_val),
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            verbose=0,
        )

        probs = model.predict(X_va_exp, batch_size=params["batch_size"], verbose=0)
        preds = np.argmax(probs, axis=1)

        macro_f1 = evaluate_macro_f1(y_val, preds)
        acc = float((preds == y_val).mean())
        dur = round(time.time() - t0, 2)

        trial_record = {
            "trial_id": idx,
            "val_macro_f1": round(macro_f1, 6),
            "val_accuracy": round(acc, 6),
            "duration_seconds": dur,
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
        }
        trials.append(trial_record)
        LOGGER.info("CNN-Transformer Trial %d/%d: Val Macro F1=%.6f, Acc=%.4f (%.2fs)", idx, len(search_space), macro_f1, acc, dur)

        if macro_f1 > best_score:
            best_score = macro_f1
            best_config = params

    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(out_dir / "cnn_transformer_trials.csv", index=False)
    with open(out_dir / "best_cnn_config.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_macro_f1": best_score, "hyperparameters": best_config}, f, indent=2)

    return {"best_config": best_config, "best_val_macro_f1": best_score}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_dir = Path("data/processed/prepared").resolve()
    subsampled_dir = data_dir / "subsampled_train"
    train_dir = subsampled_dir / "train"
    val_dir = subsampled_dir / "validation"

    out_dir = Path("results/optimized_hybrid/real_ciciot2023").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading training partition for hyperparameter tuning...")
    X_tr_full, y_tr_full = load_split_data(train_dir)
    LOGGER.info("Loading validation partition for hyperparameter tuning...")
    X_va_full, y_va_full = load_split_data(val_dir)

    rng = np.random.default_rng(FITNESS_SEED)
    tr_idx = rng.choice(len(X_tr_full), size=min(FITNESS_SUBSET_SIZE, len(X_tr_full)), replace=False)
    va_idx = rng.choice(len(X_va_full), size=min(FITNESS_SUBSET_SIZE, len(X_va_full)), replace=False)

    X_tr_sub = X_tr_full[tr_idx][:, BPSO_SELECTED_INDICES]
    y_tr_sub = y_tr_full[tr_idx]
    X_va_sub = X_va_full[va_idx][:, BPSO_SELECTED_INDICES]
    y_va_sub = y_va_full[va_idx]

    LOGGER.info("Fitness Subset Prepared: Train=%d, Val=%d, Features=27 (Seed=%d)", len(X_tr_sub), len(X_va_sub), FITNESS_SEED)

    tune_xgboost(X_tr_sub, y_tr_sub, X_va_sub, y_va_sub, out_dir)
    tune_mlp(X_tr_sub, y_tr_sub, X_va_sub, y_va_sub, out_dir)
    tune_cnn_transformer(X_tr_sub, y_tr_sub, X_va_sub, y_va_sub, out_dir)

    LOGGER.info("Hyperparameter optimization for base models finished successfully.")


if __name__ == "__main__":
    main()
