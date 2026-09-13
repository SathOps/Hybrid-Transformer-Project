"""Trains the three optimized base models on full training data using 27 BPSO features and caches validation probabilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb

from src.models.cnn_transformer import build_model
from src.preprocessing.dataset import iter_batches
from src.training.train_xgboost import load_split_data

LOGGER = logging.getLogger(__name__)

BPSO_SELECTED_INDICES = [
    1, 2, 3, 4, 6, 8, 9, 10, 15, 18, 20, 21, 22, 24, 30, 31, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45
]

SEED = 42


def train_optimized_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ckpt_dir: Path,
    out_dir: Path,
) -> tuple[xgb.XGBClassifier, np.ndarray, float]:
    """Fit optimized XGBoost model on 27 features and generate validation probabilities."""
    LOGGER.info("Training Optimized XGBoost on 2,626,223 training samples x 27 features...")
    
    best_config_path = out_dir / "best_xgboost_config.json"
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    if best_config_path.exists():
        with open(best_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            params = cfg.get("hyperparameters", params)

    t0 = time.time()
    model = xgb.XGBClassifier(
        **params,
        tree_method="hist",
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)
    dur = round(time.time() - t0, 2)

    LOGGER.info("Generating XGBoost validation probability predictions...")
    val_probs = model.predict_proba(X_val)

    # Save checkpoint model json
    model.save_model(str(ckpt_dir / "model.json"))
    with open(out_dir / "xgboost_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"params": params, "training_duration_seconds": dur}, f, indent=2)

    return model, val_probs, dur


def train_optimized_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    ckpt_dir: Path,
    out_dir: Path,
) -> tuple[keras.Model, np.ndarray, float]:
    """Fit optimized MLP model on 27 features and generate validation probabilities."""
    LOGGER.info("Training Optimized MLP on 2,626,223 training samples x 27 features...")

    tf.random.set_seed(SEED)
    best_config_path = out_dir / "best_mlp_config.json"
    layers = [256, 128, 64]
    lr = 0.00005
    dropout = 0.05
    batch_size = 1024

    if best_config_path.exists():
        with open(best_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            hp = cfg.get("hyperparameters", {})
            layers = hp.get("layers", layers)
            lr = hp.get("learning_rate", lr)
            dropout = hp.get("dropout_rate", dropout)
            batch_size = hp.get("batch_size", batch_size)

    inputs = keras.Input(shape=(27,), name="input_features")
    x = inputs
    for l_idx, units in enumerate(layers):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{units}_{l_idx}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"dropout_{l_idx}")(x)
    logits = keras.layers.Dense(7, name="classifier_dense")(x)
    outputs = keras.layers.Activation("softmax", name="classifier_softmax")(logits)
    model = keras.Model(inputs=inputs, outputs=outputs, name="optimized_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cb = [
        keras.callbacks.ModelCheckpoint(str(ckpt_dir / "best_model.keras"), save_best_only=True, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=25,
        callbacks=cb,
        verbose=1,
    )
    dur = round(time.time() - t0, 2)

    model.save(str(ckpt_dir / "final_model.keras"))
    LOGGER.info("Generating MLP validation probability predictions...")
    val_probs = model.predict(X_val, batch_size=batch_size, verbose=1)

    return model, val_probs, dur


def train_optimized_cnn_transformer(
    train_dir: Path,
    val_dir: Path,
    ckpt_dir: Path,
    out_dir: Path,
) -> tuple[keras.Model, np.ndarray, float]:
    """Fit optimized CNN-Transformer model on 27 features and generate validation probabilities."""
    LOGGER.info("Training Optimized CNN-Transformer on prepared streaming datasets...")

    from src.preprocessing.dataset import to_tf_dataset

    tf.random.set_seed(SEED)
    best_config_path = out_dir / "best_cnn_config.json"
    lr = 0.00005
    batch_size = 1024

    if best_config_path.exists():
        with open(best_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            hp = cfg.get("hyperparameters", {})
            lr = hp.get("learning_rate", lr)
            batch_size = hp.get("batch_size", batch_size)

    train_ds = to_tf_dataset(train_dir, batch_size=batch_size, shuffle=True, seed=SEED, feature_indices=BPSO_SELECTED_INDICES)
    val_ds = to_tf_dataset(val_dir, batch_size=batch_size, shuffle=False, seed=SEED, feature_indices=BPSO_SELECTED_INDICES)

    model = build_model(input_shape=(27, 1), num_classes=7, name="optimized_cnn_transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cb = [
        keras.callbacks.ModelCheckpoint(str(ckpt_dir / "best_model.keras"), save_best_only=True, monitor="val_loss"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=cb,
        verbose=1,
    )
    dur = round(time.time() - t0, 2)

    model.save(str(ckpt_dir / "final_model.keras"))
    LOGGER.info("Generating CNN-Transformer validation probability predictions...")
    val_probs = model.predict(val_ds, verbose=1)

    return model, val_probs, dur


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data_dir = Path("data/processed/prepared").resolve()
    subsampled_dir = data_dir / "subsampled_train"
    train_dir = subsampled_dir / "train"
    val_dir = subsampled_dir / "validation"

    out_dir = Path("results/optimized_hybrid/real_ciciot2023").resolve()
    ckpt_root = Path("checkpoints/optimized_hybrid/real_ciciot2023").resolve()

    ckpt_xgb = ckpt_root / "xgboost"
    ckpt_mlp = ckpt_root / "mlp"
    ckpt_cnn = ckpt_root / "cnn_transformer"

    for p in [out_dir, ckpt_xgb, ckpt_mlp, ckpt_cnn]:
        p.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading full training partition (2,626,223 rows)...")
    X_tr, y_tr = load_split_data(train_dir)
    LOGGER.info("Loading full validation partition (5,613,269 rows)...")
    X_va, y_va = load_split_data(val_dir)

    X_tr_27 = X_tr[:, BPSO_SELECTED_INDICES]
    X_va_27 = X_va[:, BPSO_SELECTED_INDICES]

    # 1. Train Optimized XGBoost
    xgb_model, val_probs_xgb, dur_xgb = train_optimized_xgboost(X_tr_27, y_tr, X_va_27, y_va, ckpt_xgb, out_dir)

    # 2. Train Optimized MLP
    mlp_model, val_probs_mlp, dur_mlp = train_optimized_mlp(X_tr_27, y_tr, X_va_27, y_va, ckpt_mlp, out_dir)

    # 3. Train Optimized CNN-Transformer
    cnn_model, val_probs_cnn, dur_cnn = train_optimized_cnn_transformer(train_dir, val_dir, ckpt_cnn, out_dir)

    # Save cached validation probabilities
    np.savez_compressed(
        out_dir / "validation_probabilities.npz",
        val_probs_xgb=val_probs_xgb,
        val_probs_mlp=val_probs_mlp,
        val_probs_cnn=val_probs_cnn,
        y_val=y_va,
    )
    LOGGER.info("Saved cached validation probabilities to %s", out_dir / "validation_probabilities.npz")


if __name__ == "__main__":
    main()
