"""
src/agents/dueling_double_dqn.py
─────────────────────────────────
Dueling Double DQN — the strongest single agent in this project.

Combines:
  ✓ Dueling architecture (better V/A decomposition)
  ✓ Double Q-learning (no overestimation)
  ✓ Optional Prioritized Experience Replay (learn from mistakes faster)

This is the configuration closest to the Rainbow paper's component analysis,
without n-step returns or distributional RL.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.networks.dueling_network import DuelingNetwork
from src.utils.replay_buffer import ReplayBuffer
from src.utils.per_buffer import PERBuffer


class DuelingDoubleDQN(BaseAgent):
    """
    Dueling Double DQN with optional Prioritized Experience Replay.

    Parameters
    ----------
    env_id    : Atari gym env id.
    n_actions : Number of discrete actions.
    use_per   : Enable Prioritized Experience Replay.
    **kwargs  : Forwarded to BaseAgent.
    """

    def __init__(self, env_id: str, n_actions: int, use_per: bool = True, **kwargs):
        online = DuelingNetwork(n_actions)
        target = DuelingNetwork(n_actions)

        buffer_size = kwargs.pop("buffer_size", 100_000)

        if use_per:
            buffer = PERBuffer(capacity=buffer_size, obs_shape=(4, 84, 84))
        else:
            buffer = ReplayBuffer(capacity=buffer_size, obs_shape=(4, 84, 84))

        self.use_per = use_per

        super().__init__(
            env_id=env_id,
            network=online,
            target_network=target,
            buffer=buffer,
            **kwargs,
        )

    def _compute_loss(self, batch) -> Tuple[torch.Tensor, float]:
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = batch
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target with dueling network
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            targets      = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (q_values - targets).detach().cpu().numpy()

        # PER: weighted loss
        elementwise_loss = F.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (weights * elementwise_loss).mean()

        # Update priorities in PER buffer
        if self.use_per and indices is not None:
            self.buffer.update_priorities(indices, td_errors)

        mean_q = float(q_values.mean().item())
        return loss, mean_q
