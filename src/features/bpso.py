"""Binary Particle Swarm Optimization (BPSO) engine for feature selection."""

from __future__ import annotations

from typing import Callable, Sequence
import json
import logging
from pathlib import Path
import numpy as np

LOGGER = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute element-wise sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


class BinaryPSO:
    """Binary Particle Swarm Optimization algorithm for selecting feature subsets."""

    def __init__(
        self,
        num_features: int = 46,
        num_particles: int = 20,
        max_iterations: int = 20,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_min: float = -6.0,
        v_max: float = 6.0,
        seed: int = 42,
    ) -> None:
        self.num_features = num_features
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_min = v_min
        self.v_max = v_max
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Initialize particle positions (50% random selection probability per bit)
        self.positions = self.rng.choice([0, 1], size=(num_particles, num_features), p=[0.5, 0.5])
        # Ensure every particle selects at least 1 feature
        for i in range(num_particles):
            if self.positions[i].sum() == 0:
                rand_idx = self.rng.integers(0, num_features)
                self.positions[i, rand_idx] = 1

        # Initialize particle velocities uniformly in [v_min/2, v_max/2]
        self.velocities = self.rng.uniform(v_min / 2.0, v_max / 2.0, size=(num_particles, num_features))

        # Personal best positions and scores
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(num_particles, -np.inf)

        # Global best position and score
        self.gbest_position = np.copy(self.positions[0])
        self.gbest_score = -np.inf
        self.gbest_metrics: dict[str, float] = {}
        self.gbest_iteration = 0

        self.history: list[dict[str, object]] = []

    def evaluate_swarm(
        self,
        fitness_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
        iteration: int,
    ) -> dict[str, object]:
        """Evaluate fitness for all particles in the swarm and update pbest/gbest."""
        iteration_best_score = -np.inf
        iteration_best_particle_idx = -1

        for i in range(self.num_particles):
            feature_mask = self.positions[i]
            # Ensure at least 1 feature is selected
            if feature_mask.sum() == 0:
                rand_idx = self.rng.integers(0, self.num_features)
                feature_mask[rand_idx] = 1
                self.positions[i, rand_idx] = 1

            fitness, metrics = fitness_fn(feature_mask)

            # Update personal best
            if fitness > self.pbest_scores[i]:
                self.pbest_scores[i] = fitness
                self.pbest_positions[i] = np.copy(feature_mask)

            # Update global best
            if fitness > self.gbest_score:
                self.gbest_score = fitness
                self.gbest_position = np.copy(feature_mask)
                self.gbest_metrics = dict(metrics)
                self.gbest_iteration = iteration

            if fitness > iteration_best_score:
                iteration_best_score = fitness
                iteration_best_particle_idx = i

        num_selected = int(self.gbest_position.sum())
        selected_indices = np.where(self.gbest_position == 1)[0].tolist()

        iter_summary = {
            "iteration": iteration,
            "gbest_fitness": float(self.gbest_score),
            "gbest_val_macro_f1": float(self.gbest_metrics.get("val_macro_f1", self.gbest_score)),
            "gbest_val_accuracy": float(self.gbest_metrics.get("val_accuracy", 0.0)),
            "gbest_val_weighted_f1": float(self.gbest_metrics.get("val_weighted_f1", 0.0)),
            "gbest_val_macro_precision": float(self.gbest_metrics.get("val_macro_precision", 0.0)),
            "gbest_val_macro_recall": float(self.gbest_metrics.get("val_macro_recall", 0.0)),
            "num_selected_features": num_selected,
            "selected_feature_indices": selected_indices,
            "iteration_best_score": float(iteration_best_score),
            "iteration_best_particle": iteration_best_particle_idx,
        }
        self.history.append(iter_summary)
        return iter_summary

    def update_particles(self) -> None:
        """Update particle velocities and binary positions using standard BPSO equations."""
        r1 = self.rng.random(size=(self.num_particles, self.num_features))
        r2 = self.rng.random(size=(self.num_particles, self.num_features))

        # Velocity update
        cognitive_component = self.c1 * r1 * (self.pbest_positions - self.positions)
        social_component = self.c2 * r2 * (self.gbest_position - self.positions)
        self.velocities = self.w * self.velocities + cognitive_component + social_component

        # Clamp velocities
        self.velocities = np.clip(self.velocities, self.v_min, self.v_max)

        # Sigmoid probability & position update
        probs = sigmoid(self.velocities)
        rand_draws = self.rng.random(size=(self.num_particles, self.num_features))
        self.positions = (rand_draws < probs).astype(int)

        # Ensure at least 1 feature per particle
        for i in range(self.num_particles):
            if self.positions[i].sum() == 0:
                best_feat_idx = int(np.argmax(self.velocities[i]))
                self.positions[i, best_feat_idx] = 1

    def step(
        self,
        fitness_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
        iteration: int,
    ) -> dict[str, object]:
        """Perform one complete BPSO iteration (evaluate then update positions)."""
        summary = self.evaluate_swarm(fitness_fn, iteration)
        self.update_particles()
        return summary
