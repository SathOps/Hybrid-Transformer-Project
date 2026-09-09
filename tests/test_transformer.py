import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.transformer import build_transformer_encoder


def test_transformer_can_be_instantiated():
    assert build_transformer_encoder() is not None


def test_transformer_input_and_output_shapes():
    model = build_transformer_encoder()

    assert model.input_shape == (None, 8, 8)
    assert model.output_shape == (None, 8, 8)


def test_attention_configuration():
    model = build_transformer_encoder()
    attention = model.get_layer("multi_head_self_attention")

    assert attention.num_heads == 3
    assert attention.key_dim == 8
    assert attention.dropout == 0.05


def test_layer_normalization_layers_exist():
    model = build_transformer_encoder()

    layer_norms = [
        layer for layer in model.layers if isinstance(layer, keras.layers.LayerNormalization)
    ]
    assert len(layer_norms) == 2


def test_residual_connections_are_present():
    model = build_transformer_encoder()

    residual_layers = [
        layer for layer in model.layers if isinstance(layer, keras.layers.Add)
    ]
    assert {layer.name for layer in residual_layers} == {
        "attention_residual",
        "feed_forward_residual",
    }


def test_random_tensor_passes_without_non_finite_values():
    model = build_transformer_encoder()
    random_input = tf.random.normal((4, 8, 8), seed=42)
    output = model(random_input, training=False).numpy()

    assert output.shape == (4, 8, 8)
    assert np.isfinite(output).all()