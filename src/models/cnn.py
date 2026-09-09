"""CNN feature extractor specified for the CICIoT2023 project."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf
from tensorflow import keras


DEFAULT_INPUT_SHAPE = (46, 1)
L2_REGULARIZATION = 0.001
DROPOUT_RATE = 0.05


def _validate_input_shape(
    input_shape: Sequence[int], third_pool_size: int
) -> tuple[int, int]:
    shape = tuple(input_shape)
    if len(shape) != 2:
        raise ValueError("input_shape must be (features, channels) for Conv1D")
    if any(not isinstance(dimension, int) or dimension < 1 for dimension in shape):
        raise ValueError("input_shape dimensions must be positive integers")

    sequence_length, _ = shape
    for pool_size in (2, 2, third_pool_size):
        sequence_length = sequence_length - 3 + 1
        sequence_length = (sequence_length - pool_size) // pool_size + 1
        if sequence_length < 1:
            raise ValueError("input_shape is too short for the three CNN blocks")
    return shape


def build_cnn_feature_extractor(
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    third_pool_size: int = 2,
    name: str = "cnn_feature_extractor",
) -> keras.Model:
    """Build the CNN block and return its 64-dimensional feature output.

    With the default ``input_shape=(46, 1)`` and ``third_pool_size=2``, the
    tensors are:

    ``(None, 46, 1)`` -> ``(None, 46, 1)`` BatchNorm
    -> ``(None, 44, 64)`` Conv1D -> ``(None, 22, 64)`` MaxPool
    -> ``(None, 22, 64)`` BatchNorm -> ``(None, 20, 128)`` Conv1D
    -> ``(None, 10, 128)`` MaxPool -> ``(None, 10, 128)`` BatchNorm
    -> ``(None, 8, 256)`` Conv1D -> ``(None, 4, 256)`` MaxPool
    -> ``(None, 1024)`` Flatten -> ``(None, 256)`` -> ``(None, 128)``
    -> ``(None, 64)``. Dropout preserves each preceding tensor shape.

    The paper does not specify the third pool size. The default ``2`` matches
    Keras's MaxPooling1D default and remains configurable rather than hidden.
    """
    if not isinstance(third_pool_size, int) or third_pool_size < 1:
        raise ValueError("third_pool_size must be a positive integer")
    shape = _validate_input_shape(input_shape, third_pool_size)

    inputs = keras.Input(shape=shape, name="input_features")

    # (None, features, channels) -> (None, features, channels)
    x = keras.layers.BatchNormalization(name="batch_norm_1")(inputs)
    # (None, 46, 1) -> (None, 44, 64) for the default input shape.
    x = keras.layers.Conv1D(64, 3, activation="selu", name="conv1d_64")(x)
    # (None, 44, 64) -> (None, 22, 64) for the default input shape.
    x = keras.layers.MaxPooling1D(2, name="max_pool_1")(x)

    # (None, 22, 64) -> (None, 22, 64) for the default input shape.
    x = keras.layers.BatchNormalization(name="batch_norm_2")(x)
    # (None, 22, 64) -> (None, 20, 128) for the default input shape.
    x = keras.layers.Conv1D(128, 3, activation="selu", name="conv1d_128")(x)
    # (None, 20, 128) -> (None, 10, 128) for the default input shape.
    x = keras.layers.MaxPooling1D(2, name="max_pool_2")(x)

    # (None, 10, 128) -> (None, 10, 128) for the default input shape.
    x = keras.layers.BatchNormalization(name="batch_norm_3")(x)
    # (None, 10, 128) -> (None, 8, 256) for the default input shape.
    x = keras.layers.Conv1D(256, 3, activation="selu", name="conv1d_256")(x)
    # (None, 8, 256) -> (None, 4, 256) for the default input shape.
    x = keras.layers.MaxPooling1D(third_pool_size, name="max_pool_3")(x)
    # (None, 4, 256) -> (None, 1024) for the default input shape.
    x = keras.layers.Flatten(name="flatten")(x)

    # Dense tensor shapes: (None, 1024) -> (None, 256) -> (None, 128) -> (None, 64).
    x = keras.layers.Dense(
        256,
        activation="selu",
        kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
        name="dense_256",
    )(x)
    x = keras.layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = keras.layers.Dense(
        128,
        activation="selu",
        kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
        name="dense_128",
    )(x)
    x = keras.layers.Dropout(DROPOUT_RATE, name="dropout_2")(x)
    x = keras.layers.Dense(
        64,
        activation="selu",
        kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
        name="dense_64",
    )(x)
    outputs = keras.layers.Dropout(DROPOUT_RATE, name="dropout_3")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)