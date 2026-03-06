"""
src/networks/cnn_q_network.py
──────────────────────────────
Standard CNN Q-network matching the DeepMind DQN architecture.

Input  : (batch, 4, 84, 84) — 4-frame stack
Output : (batch, n_actions) — Q-value per action

Architecture
------------
  Conv(32, 8×8, stride 4) → ReLU
  Conv(64, 4×4, stride 2) → ReLU
  Conv(64, 3×3, stride 1) → ReLU
  Flatten
  Linear(512) → ReLU
  Linear(n_actions)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNQNetwork(nn.Module):
    """
    Standard DQN convolutional Q-network.

    Parameters
    ----------
    n_actions   : Number of discrete actions in the environment.
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

        # compute conv output size dynamically
        conv_out = self._conv_output_size(in_channels)

        self.fc = nn.Sequential(
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
        x : (batch, 4, 84, 84) float tensor, values in [0, 1]

        Returns
        -------
        q : (batch, n_actions) Q-values
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
