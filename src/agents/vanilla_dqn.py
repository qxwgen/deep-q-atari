"""
src/agents/vanilla_dqn.py
──────────────────────────
Vanilla DQN — baseline agent.

Loss: MSE( r + γ·max_a' Q_target(s',a') − Q_online(s,a) )

Known issues this baseline suffers from:
  - Q-value overestimation (uses same net for selection + evaluation)
  - Uniform replay (wastes capacity on uninformative transitions)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.networks.cnn_q_network import CNNQNetwork
from src.utils.replay_buffer import ReplayBuffer


class VanillaDQN(BaseAgent):
    """
    Vanilla Deep Q-Network.

    Parameters
    ----------
    env_id      : Atari gym env id.
    n_actions   : Number of discrete actions.
    **kwargs    : Forwarded to BaseAgent.
    """

    def __init__(self, env_id: str, n_actions: int, **kwargs):
        online = CNNQNetwork(n_actions)
        target = CNNQNetwork(n_actions)

        buffer = ReplayBuffer(
            capacity=kwargs.pop("buffer_size", 100_000),
            obs_shape=(4, 84, 84),
        )

        super().__init__(
            env_id=env_id,
            network=online,
            target_network=target,
            buffer=buffer,
            **kwargs,
        )

    def _compute_loss(self, batch) -> Tuple[torch.Tensor, float]:
        states, actions, rewards, next_states, dones = batch

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss   = F.mse_loss(q_values, targets)
        mean_q = float(q_values.mean().item())
        return loss, mean_q
