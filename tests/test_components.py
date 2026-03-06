"""
tests/test_components.py
─────────────────────────
Unit tests for all components — no GPU, no Atari ROM required.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── replay buffers ────────────────────────────────────────────

class TestReplayBuffer:
    def _make(self, cap=100):
        from src.utils.replay_buffer import ReplayBuffer
        return ReplayBuffer(capacity=cap, obs_shape=(4, 84, 84))

    def test_push_and_len(self):
        buf = self._make()
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        buf.push(obs, 0, 1.0, obs, False)
        assert len(buf) == 1

    def test_circular_overwrite(self):
        buf = self._make(cap=5)
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        for _ in range(10):
            buf.push(obs, 0, 1.0, obs, False)
        assert len(buf) == 5

    def test_sample_shapes(self):
        buf = self._make()
        obs = np.random.rand(4, 84, 84).astype(np.float32)
        for _ in range(50):
            buf.push(obs, 1, 0.5, obs, False)
        s, a, r, ns, d = buf.sample(16)
        assert s.shape == (16, 4, 84, 84)
        assert a.shape == (16,)
        assert r.shape == (16,)

    def test_is_ready(self):
        buf = self._make()
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        assert not buf.is_ready(10)
        for _ in range(10):
            buf.push(obs, 0, 0.0, obs, False)
        assert buf.is_ready(10)


class TestPERBuffer:
    def _make(self, cap=200):
        from src.utils.per_buffer import PERBuffer
        return PERBuffer(capacity=cap, obs_shape=(4, 84, 84))

    def test_push_and_len(self):
        buf = self._make()
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        buf.push(obs, 0, 1.0, obs, False)
        assert len(buf) == 1

    def test_sample_returns_weights_and_indices(self):
        buf = self._make()
        obs = np.random.rand(4, 84, 84).astype(np.float32)
        for _ in range(100):
            buf.push(obs, 1, 1.0, obs, False)
        s, a, r, ns, d, w, idx = buf.sample(16)
        assert w.shape == (16,)
        assert idx.shape == (16,)
        assert (w > 0).all()

    def test_priority_update(self):
        buf = self._make()
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        for _ in range(100):
            buf.push(obs, 0, 1.0, obs, False)
        _, _, _, _, _, _, indices = buf.sample(8)
        td_errors = np.random.rand(8).astype(np.float32)
        buf.update_priorities(indices, td_errors)   # should not raise


# ── sum-tree ──────────────────────────────────────────────────

class TestSumTree:
    def test_total_after_updates(self):
        from src.utils.per_buffer import SumTree
        tree = SumTree(capacity=8)
        for i in range(8):
            tree.update(i, float(i + 1))
        assert abs(tree.total - sum(range(1, 9))) < 1e-5

    def test_get_returns_valid_leaf(self):
        from src.utils.per_buffer import SumTree
        tree = SumTree(capacity=4)
        for i in range(4):
            tree.update(i, 1.0)
        idx, p = tree.get(tree.total * 0.5)
        assert 0 <= idx < 4
        assert p > 0


# ── networks ──────────────────────────────────────────────────

class TestNetworks:
    def test_cnn_output_shape(self):
        import torch
        from src.networks.cnn_q_network import CNNQNetwork
        net = CNNQNetwork(n_actions=6)
        x = torch.zeros(2, 4, 84, 84)
        out = net(x)
        assert out.shape == (2, 6)

    def test_dueling_output_shape(self):
        import torch
        from src.networks.dueling_network import DuelingNetwork
        net = DuelingNetwork(n_actions=18)
        x = torch.zeros(4, 4, 84, 84)
        out = net(x)
        assert out.shape == (4, 18)

    def test_dueling_advantage_zero_mean(self):
        """Mean advantage should be ~0 by construction of the dueling formula."""
        import torch
        from src.networks.dueling_network import DuelingNetwork
        net = DuelingNetwork(n_actions=6)
        net.eval()
        x = torch.rand(8, 4, 84, 84)
        with torch.no_grad():
            feat = net.conv(x).view(8, -1)
            adv  = net.advantage_stream(feat)
        # check that mean is subtracted implicitly in forward
        q = net(x)
        v = net.value_stream(feat)
        reconstructed = v + adv - adv.mean(dim=1, keepdim=True)
        assert torch.allclose(q, reconstructed, atol=1e-5)
