"""
src/agents/base_agent.py
─────────────────────────
Abstract base class for all DQN variants.
Provides the shared training loop, epsilon schedule,
target network updates, CSV logging, and checkpointing.
All subclasses override only `_compute_loss()`.
"""

from __future__ import annotations

import abc
import csv
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.env_wrappers import make_atari_env


class BaseAgent(abc.ABC):
    """
    Shared DQN training infrastructure.

    Parameters
    ----------
    env_id          : Atari gym env id.
    network         : Online Q-network (nn.Module).
    target_network  : Target Q-network (nn.Module).
    buffer          : Replay buffer (uniform or PER).
    lr              : Adam learning rate.
    gamma           : Discount factor.
    batch_size      : Minibatch size.
    min_replay_size : Transitions to collect before training starts.
    target_update   : Hard target update frequency (steps). Set 0 for soft updates.
    tau             : Soft update coefficient (only used if target_update=0).
    eps_start       : Initial epsilon for ε-greedy.
    eps_end         : Final epsilon.
    eps_decay_steps : Steps over which epsilon decays linearly.
    save_dir        : Directory for checkpoints and logs.
    device          : 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        env_id:           str,
        network:          nn.Module,
        target_network:   nn.Module,
        buffer,
        lr:               float = 1e-4,
        gamma:            float = 0.99,
        batch_size:       int   = 32,
        min_replay_size:  int   = 10_000,
        target_update:    int   = 1_000,
        tau:              float = 1e-3,
        eps_start:        float = 1.0,
        eps_end:          float = 0.01,
        eps_decay_steps:  int   = 500_000,
        save_dir:         str   = "results",
        device:           str   = "cpu",
    ):
        self.env_id           = env_id
        self.gamma            = gamma
        self.batch_size       = batch_size
        self.min_replay_size  = min_replay_size
        self.target_update    = target_update
        self.tau              = tau
        self.eps_start        = eps_start
        self.eps_end          = eps_end
        self.eps_decay_steps  = eps_decay_steps
        self.device           = torch.device(device)

        self.online_net = network.to(self.device)
        self.target_net = target_network.to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.buffer    = buffer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # logging
        agent_name = self.__class__.__name__
        self.save_dir = Path(save_dir) / env_id / agent_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.save_dir / "training_log.csv"
        self._init_log()

        # counters
        self.steps        = 0
        self.episodes     = 0
        self.best_reward  = -float("inf")

    # ── logging ───────────────────────────────────────────────

    def _init_log(self) -> None:
        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "steps", "reward", "loss", "epsilon", "mean_q"])

    def _log(self, reward: float, loss: float, epsilon: float, mean_q: float) -> None:
        with open(self._log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.episodes, self.steps, reward, loss, epsilon, mean_q])

    # ── epsilon schedule ──────────────────────────────────────

    def _epsilon(self) -> float:
        fraction = min(self.steps / self.eps_decay_steps, 1.0)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - fraction)

    # ── action selection ──────────────────────────────────────

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online_net(state_t).argmax(dim=1).item())

    # ── target network update ─────────────────────────────────

    def _update_target(self, hard: bool = False) -> None:
        if hard:
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

    # ── abstract ──────────────────────────────────────────────

    @abc.abstractmethod
    def _compute_loss(self, batch) -> tuple[torch.Tensor, float]:
        """
        Compute the loss and mean Q for a batch of transitions.

        Parameters
        ----------
        batch : tuple of arrays from replay buffer sample()

        Returns
        -------
        (loss_tensor, mean_q_value)
        """
        ...

    # ── training loop ─────────────────────────────────────────

    def train(self, total_steps: int = 1_000_000, log_every: int = 10) -> None:
        """
        Main training loop.

        Parameters
        ----------
        total_steps : Total environment steps to train for.
        log_every   : Log to console every N episodes.
        """
        self.env = make_atari_env(self.env_id, clip_rewards=True, episode_life=True)
        obs, _ = self.env.reset()

        ep_reward  = 0.0
        ep_loss    = []
        ep_q       = []
        start_time = time.time()

        print(f"\n🎮 Training {self.__class__.__name__} on {self.env_id}")
        print(f"   Device: {self.device}  |  Total steps: {total_steps:,}\n")

        while self.steps < total_steps:
            epsilon = self._epsilon()
            action  = self.select_action(obs, epsilon)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.push(obs, action, reward, next_obs, done)
            obs        = next_obs
            ep_reward += reward
            self.steps += 1

            # train when buffer is ready
            if self.buffer.is_ready(self.min_replay_size):
                batch = self.buffer.sample(self.batch_size)
                loss, mean_q = self._compute_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
                self.optimizer.step()

                ep_loss.append(loss.item())
                ep_q.append(mean_q)

                # target update
                if self.target_update > 0 and self.steps % self.target_update == 0:
                    self._update_target(hard=True)
                elif self.target_update == 0:
                    self._update_target(hard=False)

            if done:
                self.episodes += 1
                mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
                mean_q    = float(np.mean(ep_q))    if ep_q    else 0.0

                self._log(ep_reward, mean_loss, epsilon, mean_q)

                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    self._save_checkpoint("best.pt")

                if self.episodes % log_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"  ep={self.episodes:5d}  steps={self.steps:7,}  "
                        f"reward={ep_reward:6.1f}  best={self.best_reward:6.1f}  "
                        f"loss={mean_loss:.4f}  ε={epsilon:.3f}  "
                        f"t={elapsed:.0f}s"
                    )

                obs, _     = self.env.reset()
                ep_reward  = 0.0
                ep_loss    = []
                ep_q       = []

        self._save_checkpoint("final.pt")
        self.env.close()
        print(f"\n✅ Training done. Best reward: {self.best_reward:.1f}")

    # ── checkpointing ─────────────────────────────────────────

    def _save_checkpoint(self, name: str) -> None:
        path = self.save_dir / name
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "steps":       self.steps,
            "episodes":    self.episodes,
            "best_reward": self.best_reward,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps       = ckpt["steps"]
        self.episodes    = ckpt["episodes"]
        self.best_reward = ckpt["best_reward"]
        print(f"✅ Loaded checkpoint: {path}  (step={self.steps:,})")
