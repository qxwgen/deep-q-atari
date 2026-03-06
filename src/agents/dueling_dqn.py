"""
src/agents/dueling_dqn.py
──────────────────────────
Dueling DQN — better value estimation in states where action choice
doesn't matter much.

Key change vs vanilla:
  Network architecture splits into two heads:
    V(s)    — how good is this state regardless of action?
    A(s,a)  — how much better is action a vs the average?
  Combined: Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]

Uses vanilla (non-double) targets — swap for DuelingDoubleDQN if desired.

Reference: Wang et al. (2016)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.networks.dueling_network import DuelingNetwork
from src.utils.replay_buffer import ReplayBuffer


class DuelingDQN(BaseAgent):
    """
    Dueling Deep Q-Network.

    Parameters
    ----------
    env_id    : Atari gym env id.
    n_actions : Number of discrete actions.
    **kwargs  : Forwarded to BaseAgent.
    """

    def __init__(self, env_id: str, n_actions: int, **kwargs):
        online = DuelingNetwork(n_actions)
        target = DuelingNetwork(n_actions)

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

        q_values = self.online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss   = F.mse_loss(q_values, targets)
        mean_q = float(q_values.mean().item())
        return loss, mean_q
