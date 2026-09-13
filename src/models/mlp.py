"""MLP (Multi-Layer Perceptron) baseline model for the CICIoT2023 project."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf
from tensorflow import keras

from src.preprocessing.class_mapping import NUM_ACTIVE_CLASSES

DEFAULT_INPUT_SHAPE = (46,)
DEFAULT_NUM_CLASSES = NUM_ACTIVE_CLASSES
DROPOUT_RATE = 0.05


def build_mlp_model(
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    name: str = "mlp_classifier",
) -> keras.Model:
    """Build a standard MLP baseline model for 7 active classes network intrusion classification.

    Tensor shapes:
    ``(None, 46)`` -> ``Dense(256, ReLU)`` -> ``Dropout(0.05)``
    -> ``Dense(128, ReLU)`` -> ``Dropout(0.05)``
    -> ``Dense(64, ReLU)`` -> ``Dropout(0.05)``
    -> ``Dense(7)`` -> ``Softmax``.
    """
    shape = tuple(input_shape)
    inputs = keras.Input(shape=shape, name="input_features")

    x = inputs
    if len(shape) > 1:
        x = keras.layers.Flatten(name="flatten")(inputs)

    x = keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_2")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_3")(x)

    logits = keras.layers.Dense(num_classes, name="classifier_dense")(x)
    outputs = keras.layers.Activation("softmax", name="classifier_softmax")(logits)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)
