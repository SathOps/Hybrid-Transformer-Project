"""Tests for the XGBoost model configuration."""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb

from src.models.xgboost_model import build_xgboost_classifier


def test_xgboost_classifier_instantiation():
    clf = build_xgboost_classifier(n_estimators=10, seed=42)
    assert isinstance(clf, xgb.XGBClassifier)
    assert clf.n_estimators == 10
    assert clf.random_state == 42


def test_xgboost_fit_and_predict_contract():
    clf = build_xgboost_classifier(n_estimators=5, seed=42)
    X_train = np.random.normal(size=(20, 46)).astype(np.float32)
    y_train = np.random.randint(0, 7, size=(20,))

    clf.fit(X_train, y_train)

    X_test = np.random.normal(size=(5, 46)).astype(np.float32)
    probs = clf.predict_proba(X_test)

    assert probs.shape == (5, 7)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-4)
