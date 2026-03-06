"""
src/utils/replay_buffer.py
───────────────────────────
Uniform experience replay buffer backed by a circular numpy array.
Stores (state, action, reward, next_state, done) transitions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Circular uniform replay buffer.

    Parameters
    ----------
    capacity    : Maximum number of transitions stored.
    obs_shape   : Shape of a single observation, e.g. (4, 84, 84).
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity  = capacity
        self.ptr       = 0          # write pointer
        self.size      = 0          # current fill level

        self.states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    # ── write ──────────────────────────────────────────────────

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ── read ───────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random minibatch. Returns numpy arrays."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size
