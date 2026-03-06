"""
src/utils/per_buffer.py
────────────────────────
Prioritized Experience Replay (PER) buffer.

Uses a sum-tree for O(log n) weighted sampling and O(log n) priority updates.
Implements importance sampling (IS) weights to correct for the sampling bias.

Reference: Schaul et al. (2016) — Prioritized Experience Replay
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────
# Sum-tree
# ──────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary tree where each parent = sum of its two children.
    Leaf nodes store priorities; internal nodes store sums.
    Enables O(log n) weighted sampling and priority updates.

    Layout (capacity=4):
              [0]
            /       \\
          [1]        [2]
         /   \\      /   \\
       [3]  [4]  [5]  [6]   ← leaves (indices 3..6)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, priority: float) -> None:
        """Update priority at leaf index `idx` (0-based leaf index)."""
        tree_idx = idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def get(self, value: float) -> Tuple[int, float]:
        """
        Retrieve the leaf whose cumulative sum contains `value`.
        Returns (leaf_index, priority).
        """
        idx = 0
        while idx < self.capacity - 1:
            left  = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, self.tree[idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        leaves = self.tree[self.capacity - 1:]
        return float(leaves.max()) if leaves.max() > 0 else 1.0


# ──────────────────────────────────────────────────────────────
# PER Buffer
# ──────────────────────────────────────────────────────────────

class PERBuffer:
    """
    Prioritized Experience Replay buffer.

    Parameters
    ----------
    capacity : Maximum transitions stored.
    obs_shape: Shape of a single observation.
    alpha    : Priority exponent (0 = uniform, 1 = full prioritisation).
    beta     : IS weight exponent (0 = no correction, 1 = full correction).
    beta_inc : Increment β toward 1 over training.
    eps      : Small constant to prevent zero priorities.
    """

    def __init__(
        self,
        capacity:  int,
        obs_shape: Tuple[int, ...],
        alpha:     float = 0.6,
        beta:      float = 0.4,
        beta_inc:  float = 1e-6,
        eps:       float = 1e-6,
    ):
        self.capacity  = capacity
        self.alpha     = alpha
        self.beta      = beta
        self.beta_inc  = beta_inc
        self.eps       = eps
        self.size      = 0
        self.ptr       = 0

        self.tree        = SumTree(capacity)
        self.states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        priority = self.tree.max_priority ** self.alpha
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)
        self.tree.update(self.ptr, priority)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a prioritised minibatch.

        Returns
        -------
        states, actions, rewards, next_states, dones, weights, indices
        weights : IS correction weights (float32 array)
        indices : leaf indices (for priority update after TD error)
        """
        indices   = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        segment   = self.tree.total / batch_size

        self.beta = min(1.0, self.beta + self.beta_inc)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            val = np.random.uniform(lo, hi)
            idx, p = self.tree.get(val)
            indices[i]    = idx
            priorities[i] = p

        # IS weights
        probs   = priorities / (self.tree.total + self.eps)
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()   # normalise so max weight = 1

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled transitions using new TD errors."""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(indices, priorities):
            self.tree.update(int(idx), float(p))

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size
