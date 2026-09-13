"""Streaming access to prepared CICIoT2023 NumPy shards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
import json

import numpy as np


from src.preprocessing.class_mapping import (
    ACTIVE_TARGET_CLASSES,
    canonical_to_active_labels,
)


def load_class_names(root_dir: Path) -> tuple[str, ...]:
    """Load the active target class ordering saved beside processed shards or default to active classes."""
    active_path = root_dir / "active_class_names.json"
    if active_path.exists():
        return tuple(json.loads(active_path.read_text(encoding="utf-8")))
    class_names_path = root_dir / "class_names.json"
    if class_names_path.exists():
        names = json.loads(class_names_path.read_text(encoding="utf-8"))
        if len(names) == 7:
            return tuple(names)
    return ACTIVE_TARGET_CLASSES


def iter_batches(
    split_dir: Path,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
    map_to_active: bool = True,
    feature_indices: Sequence[int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield batches while loading one compressed shard at a time."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    shard_paths = sorted(split_dir.glob("part-*.npz"))
    random = np.random.default_rng(seed)
    if shuffle:
        random.shuffle(shard_paths)
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            features = shard["features"]
            labels = shard["labels"]
            if feature_indices is not None:
                features = features[:, feature_indices]
            if map_to_active:
                labels = canonical_to_active_labels(labels)
            indices = np.arange(len(features))
            if shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                yield features[batch_indices], labels[batch_indices]


class PreparedDataset:
    """Small wrapper exposing a repeatable, shard-streaming batch iterator."""

    def __init__(self, root_dir: Path, split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split_dir = root_dir / split

    def batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
        map_to_active: bool = True,
        feature_indices: Sequence[int] | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return iter_batches(
            self.split_dir,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            map_to_active=map_to_active,
            feature_indices=feature_indices,
        )

    def to_tf_dataset(
        self,
        batch_size: int = 1024,
        shuffle: bool = False,
        seed: int = 42,
        expand_dims: bool = True,
        map_to_active: bool = True,
        feature_indices: Sequence[int] | None = None,
    ):
        """Create a tf.data.Dataset for this partition."""
        return to_tf_dataset(
            self.split_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            expand_dims=expand_dims,
            map_to_active=map_to_active,
            feature_indices=feature_indices,
        )


def to_tf_dataset(
    split_dir: Path,
    batch_size: int = 1024,
    shuffle: bool = False,
    seed: int = 42,
    expand_dims: bool = True,
    map_to_active: bool = True,
    feature_indices: Sequence[int] | None = None,
):
    """Create a tf.data.Dataset from processed shards for TensorFlow training."""
    import tensorflow as tf

    def generator():
        for features, labels in iter_batches(
            split_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            map_to_active=map_to_active,
            feature_indices=feature_indices,
        ):
            if expand_dims:
                yield np.expand_dims(features, axis=-1), labels
            else:
                yield features, labels

    num_feats = len(feature_indices) if feature_indices is not None else 46
    feature_shape = (None, num_feats, 1) if expand_dims else (None, num_feats)
    output_signature = (
        tf.TensorSpec(shape=feature_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)