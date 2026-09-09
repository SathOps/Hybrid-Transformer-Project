"""Streaming access to prepared CICIoT2023 NumPy shards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def iter_batches(
    split_dir: Path, batch_size: int, shuffle: bool = False, seed: int = 42
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
        self, batch_size: int, shuffle: bool = False, seed: int = 42
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return iter_batches(self.split_dir, batch_size, shuffle, seed)