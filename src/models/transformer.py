"""Reusable Transformer encoder for the resolved CNN interface."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf
from tensorflow import keras


DEFAULT_INPUT_SHAPE = (8, 8)
NUM_HEADS = 3
KEY_DIM = 8
ATTENTION_DROPOUT = 0.05
FEED_FORWARD_UNITS = 8


def _validate_input_shape(input_shape: Sequence[int]) -> tuple[int, int]:
    shape = tuple(input_shape)
    if len(shape) != 2:
        raise ValueError("input_shape must be (tokens, embedding_width)")
    if any(not isinstance(dimension, int) or dimension < 1 for dimension in shape):
        raise ValueError("input_shape dimensions must be positive integers")
    if shape[1] != KEY_DIM:
        raise ValueError("input_shape embedding width must be 8")
    return shape


def build_transformer_encoder(
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    name: str = "transformer_encoder",
) -> keras.Model:
    """Build one self-attention encoder block for ``(batch, 8, 8)`` input.

    The resolved bridge supplies 8 tokens with width 8. Keras uses
    ``num_heads=3`` and ``key_dim=8``; the attention projections therefore use
    an internal attention width of 24 before the attention output projection
    returns to width 8. The block preserves the input shape:

    ``(None, 8, 8)`` -> attention -> residual -> LayerNorm
    -> Dense(8, SeLU) -> Dense(8) -> residual -> LayerNorm
    -> ``(None, 8, 8)``.
    """
    shape = _validate_input_shape(input_shape)
    inputs = keras.Input(shape=shape, name="transformer_input")

    attention = keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=KEY_DIM,
        dropout=ATTENTION_DROPOUT,
        name="multi_head_self_attention",
    )
    attention_output = attention(inputs, inputs)
    attention_residual = keras.layers.Add(name="attention_residual")(
        [inputs, attention_output]
    )
    normalized_attention = keras.layers.LayerNormalization(
        name="attention_layer_norm"
    )(attention_residual)

    feed_forward_hidden = keras.layers.Dense(
        FEED_FORWARD_UNITS,
        activation="selu",
        name="feed_forward_selu",
    )(normalized_attention)
    feed_forward_output = keras.layers.Dense(
        KEY_DIM,
        name="feed_forward_projection",
    )(feed_forward_hidden)
    feed_forward_residual = keras.layers.Add(name="feed_forward_residual")(
        [normalized_attention, feed_forward_output]
    )
    outputs = keras.layers.LayerNormalization(name="feed_forward_layer_norm")(
        feed_forward_residual
    )

    return keras.Model(inputs=inputs, outputs=outputs, name=name)