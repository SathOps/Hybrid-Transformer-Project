"""XGBoost baseline classifier model for the CICIoT2023 project."""

from __future__ import annotations

from typing import Any
import xgboost as xgb


from src.preprocessing.class_mapping import NUM_ACTIVE_CLASSES


def build_xgboost_classifier(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    tree_method: str = "hist",
    seed: int = 42,
    **kwargs: Any,
) -> xgb.XGBClassifier:
    """Build an XGBoost multi-class classifier instance for 7 active classes NIDS task."""
    params = {
        "objective": "multi:softprob",
        "num_class": NUM_ACTIVE_CLASSES,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "tree_method": tree_method,
        "random_state": seed,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
    }
    params.update(kwargs)
    return xgb.XGBClassifier(**params)
