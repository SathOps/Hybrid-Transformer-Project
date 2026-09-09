import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.cnn_transformer import build_model


def test_complete_model_can_be_instantiated():
    assert build_model() is not None


def test_model_shapes_and_component_contracts():
    model = build_model()
    cnn = model.get_layer("cnn_feature_extractor")
    bridge = model.get_layer("cnn_to_transformer_bridge")
    transformer = model.get_layer("transformer_encoder")

    assert model.input_shape == (None, 46, 1)
    assert cnn.output_shape == (None, 64)
    assert tuple(bridge.output.shape) == (None, 8, 8)
    assert transformer.input_shape == (None, 8, 8)
    assert transformer.output_shape == (None, 8, 8)
    assert model.output_shape == (None, 8)


def test_softmax_probabilities_sum_to_one():
    model = build_model()
    output = model(tf.random.normal((3, 46, 1), seed=42), training=False).numpy()

    np.testing.assert_allclose(output.sum(axis=1), np.ones(3), atol=1e-6)


def test_random_forward_pass_is_finite_and_batch_dynamic():
    model = build_model()
    output = model(tf.random.normal((5, 46, 1), seed=7), training=False).numpy()

    assert output.shape == (5, 8)
    assert np.isfinite(output).all()


def test_global_average_aggregation_is_used():
    model = build_model()

    aggregation = model.get_layer("transformer_global_average_pool")
    assert isinstance(aggregation, keras.layers.GlobalAveragePooling1D)


def test_model_compiles_and_executes_one_training_step():
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    features = tf.random.normal((4, 46, 1), seed=11)
    labels = tf.constant([0, 1, 2, 3], dtype=tf.int32)

    result = model.train_on_batch(features, labels)

    assert np.isfinite(float(result))