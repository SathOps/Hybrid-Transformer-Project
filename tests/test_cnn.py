import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from src.models.cnn import DEFAULT_INPUT_SHAPE, build_cnn_feature_extractor


def test_default_cnn_output_shape_and_tensor_execution():
    model = build_cnn_feature_extractor()

    assert model.input_shape == (None, 46, 1)
    assert model.output_shape == (None, 64)
    output = model(tf.zeros((2, *DEFAULT_INPUT_SHAPE)))
    assert tuple(output.shape) == (2, 64)


def test_cnn_contains_only_specified_layer_sequence():
    model = build_cnn_feature_extractor()
    layer_types = [type(layer) for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]

    assert layer_types == [
        keras.layers.BatchNormalization,
        keras.layers.Conv1D,
        keras.layers.MaxPooling1D,
        keras.layers.BatchNormalization,
        keras.layers.Conv1D,
        keras.layers.MaxPooling1D,
        keras.layers.BatchNormalization,
        keras.layers.Conv1D,
        keras.layers.MaxPooling1D,
        keras.layers.Flatten,
        keras.layers.Dense,
        keras.layers.Dropout,
        keras.layers.Dense,
        keras.layers.Dropout,
        keras.layers.Dense,
        keras.layers.Dropout,
    ]


def test_cnn_layer_parameters_match_specification():
    model = build_cnn_feature_extractor()
    conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv1D)]
    dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dropout)]

    assert [(layer.filters, layer.kernel_size) for layer in conv_layers] == [
        (64, (3,)),
        (128, (3,)),
        (256, (3,)),
    ]
    assert [layer.units for layer in dense_layers] == [256, 128, 64]
    assert all(layer.activation == keras.activations.selu for layer in conv_layers + dense_layers)
    assert all(layer.kernel_regularizer.l2 == 0.001 for layer in dense_layers)
    assert [layer.rate for layer in dropout_layers] == [0.05, 0.05, 0.05]


def test_cnn_accepts_configurable_input_shape():
    model = build_cnn_feature_extractor(input_shape=(50, 2))

    assert model.input_shape == (None, 50, 2)
    assert model.output_shape == (None, 64)


@pytest.mark.parametrize("input_shape", [(46,), (4, 1), (0, 1), (46, 0)])
def test_cnn_rejects_invalid_input_shapes(input_shape):
    with pytest.raises(ValueError):
        build_cnn_feature_extractor(input_shape=input_shape)