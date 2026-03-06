"""
src/networks/dueling_network.py
────────────────────────────────
Dueling CNN Q-network.

Splits the FC head into two streams:
  V(s)       — scalar state value
  A(s, a)    — advantage for each action

Combined via:
  Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]

The mean-subtraction ensures identifiability: Q uniquely determines V and A.

Reference: Wang et al. (2016) — Dueling Network Architectures
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    """
    Dueling DQN convolutional network.

    Parameters
    ----------
    n_actions   : Number of discrete actions.
    in_channels : Number of stacked frames (default 4).
    """

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out = self._conv_output_size(in_channels)

        # Value stream: s → scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage stream: s → A(s, a) for all a
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _conv_output_size(self, in_channels: int) -> int:
        dummy = torch.zeros(1, in_channels, 84, 84)
        return int(self.conv(dummy).view(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 4, 84, 84) float tensor

        Returns
        -------
        q : (batch, n_actions) Q-values
        """
        feat = self.conv(x).view(x.size(0), -1)
        value     = self.value_stream(feat)             # (B, 1)
        advantage = self.advantage_stream(feat)         # (B, A)

        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
