"""Reusable CNN-Transformer classifier assembled from existing components."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf
from tensorflow import keras

from .cnn import DEFAULT_INPUT_SHAPE, build_cnn_feature_extractor
from .transformer import DEFAULT_INPUT_SHAPE as DEFAULT_TRANSFORMER_INPUT_SHAPE
from .transformer import build_transformer_encoder


from src.preprocessing.class_mapping import ACTIVE_TARGET_CLASSES, NUM_ACTIVE_CLASSES

CLASS_NAMES = ACTIVE_TARGET_CLASSES
DEFAULT_NUM_CLASSES = NUM_ACTIVE_CLASSES
DEFAULT_AGGREGATION = "global_average"


def build_model(
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    aggregation: str = DEFAULT_AGGREGATION,
    third_pool_size: int = 2,
    name: str = "cnn_transformer_classifier",
) -> keras.Model:
    """Build the complete classifier without compiling or training it.

    Tensor shapes for the default configuration are:

    ``(None, 46, 1)`` -> CNN -> ``(None, 64)``
    -> reshape bridge -> ``(None, 8, 8)``
    -> Transformer encoder -> ``(None, 8, 8)``
    -> GlobalAveragePooling1D -> ``(None, 8)``
    -> Dense(7) -> Softmax -> ``(None, 7)``.
    """
    if num_classes != DEFAULT_NUM_CLASSES:
        raise ValueError(
            f"This project classifier requires {DEFAULT_NUM_CLASSES} active target classes"
        )
    if aggregation != DEFAULT_AGGREGATION:
        raise ValueError(
            "The documented classifier aggregation is 'global_average'"
        )
    if len(input_shape) != 2 or input_shape[1] != 1:
        raise ValueError("The CNN-Transformer integration requires input_shape=(num_features, 1)")

    inputs = keras.Input(shape=tuple(input_shape), name="traffic_features")
    cnn = build_cnn_feature_extractor(
        input_shape=input_shape,
        third_pool_size=third_pool_size,
        name="cnn_feature_extractor",
    )
    cnn_output = cnn(inputs)

    # The documented project bridge preserves the 64 CNN values as 8 tokens
    # with width 8: (batch, 64) -> (batch, 8, 8).
    bridged_output = keras.layers.Reshape(
        DEFAULT_TRANSFORMER_INPUT_SHAPE,
        name="cnn_to_transformer_bridge",
    )(cnn_output)

    transformer = build_transformer_encoder(name="transformer_encoder")
    transformer_output = transformer(bridged_output)
    aggregated_output = keras.layers.GlobalAveragePooling1D(
        name="transformer_global_average_pool"
    )(transformer_output)
    logits = keras.layers.Dense(num_classes, name="classifier_dense")(aggregated_output)
    outputs = keras.layers.Activation("softmax", name="classifier_softmax")(logits)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)