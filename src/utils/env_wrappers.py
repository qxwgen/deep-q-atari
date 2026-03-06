"""
src/utils/env_wrappers.py
──────────────────────────
Standard Atari preprocessing pipeline matching the DeepMind DQN paper:
  1. NoopReset       — random no-ops at episode start for stochasticity
  2. MaxAndSkip      — max-pool over 2 frames, repeat action for 4 frames
  3. EpisodicLife    — treat loss of life as episode end during training
  4. FireReset       — press FIRE to start games that require it
  5. WarpFrame       — grayscale + resize to 84×84
  6. ScaledFloat     — uint8 [0,255] → float32 [0,1]
  7. FrameStack      — stack 4 consecutive frames → (4, 84, 84) observation
  8. ClipReward      — clip rewards to {-1, 0, +1} for stability
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────────────────────
# Individual wrappers
# ──────────────────────────────────────────────────────────────

class NoopResetEnv(gym.Wrapper):
    """Execute random no-ops on reset for stochastic starts."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return max-pooled obs over last 2 frames; repeat action for `skip` steps."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Treat loss of a life as episode end (training only)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE to start games that require it."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """Grayscale + resize to 84×84."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width  = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]  # (H, W, 1)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Scale uint8 [0, 255] to float32 [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        low  = env.observation_space.low  / 255.0
        high = env.observation_space.high / 255.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.array(obs, dtype=np.float32) / 255.0


class FrameStack(gym.Wrapper):
    """Stack `n_frames` consecutive frames into a single observation."""

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self._frames  = deque(maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(n_frames, shp[0], shp[1]),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # frames are (H, W, 1); stack → (n_frames, H, W)
        return np.concatenate(list(self._frames), axis=2).transpose(2, 0, 1)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} for training stability."""

    def reward(self, reward: float) -> float:
        return np.sign(reward)


# ──────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────

def make_atari_env(
    env_id: str,
    clip_rewards: bool = True,
    episode_life: bool = True,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Build a fully preprocessed Atari environment.

    Parameters
    ----------
    env_id       : Gymnasium env id, e.g. 'PongNoFrameskip-v4'
    clip_rewards : Clip rewards to {-1,0,+1} during training
    episode_life : Treat life loss as episode end (training only)
    seed         : Optional random seed

    Returns
    -------
    Wrapped gym.Env with observation shape (4, 84, 84) and float32 obs.
    """
    env = gym.make(env_id, render_mode=None)
    if seed is not None:
        env.reset(seed=seed)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if episode_life:
        env = EpisodicLifeEnv(env)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, n_frames=4)

    if clip_rewards:
        env = ClipRewardEnv(env)

    return env
