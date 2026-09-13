"""Tests for the MLP baseline model architecture."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.models.mlp import build_mlp_model


def test_mlp_model_instantiation():
    model = build_mlp_model()
    assert model is not None
    assert model.name == "mlp_classifier"
    assert model.output_shape == (None, 7)


def test_mlp_model_forward_pass():
    model = build_mlp_model()
    dummy_input = tf.random.normal((10, 46), seed=42)
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (10, 7)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output.sum(axis=1), np.ones(10), atol=1e-5)


def test_mlp_model_handles_expand_dims_input():
    model = build_mlp_model(input_shape=(46, 1))
    dummy_input = tf.random.normal((5, 46, 1), seed=7)
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (5, 7)
    assert np.isfinite(output).all()
