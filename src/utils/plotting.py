"""
src/utils/plotting.py
──────────────────────
Training curve visualisation utilities.
Reads CSV logs produced by BaseAgent and generates publication-ready plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────

def _smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_log(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


# ──────────────────────────────────────────────────────────────
# single agent plots
# ──────────────────────────────────────────────────────────────

def plot_training(csv_path: str | Path, save_dir: Optional[str | Path] = None) -> None:
    """
    Generate a 2×2 grid of training curves for one agent.
    Plots: episode reward, loss, epsilon, mean Q-value.
    """
    df = load_log(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training Curves — {Path(csv_path).parent.name}", fontweight="bold")

    panels = [
        ("steps", "reward",  "Episode Reward",  "steelblue"),
        ("steps", "loss",    "Training Loss",   "tomato"),
        ("steps", "epsilon", "Epsilon",         "goldenrod"),
        ("steps", "mean_q",  "Mean Q-Value",    "mediumseagreen"),
    ]

    for ax, (x_col, y_col, title, color) in zip(axes.flat, panels):
        raw = df[y_col].values
        ax.plot(df[x_col].values, raw, alpha=0.25, color=color, linewidth=0.8)
        smoothed = _smooth(raw)
        ax.plot(df[x_col].values[len(raw) - len(smoothed):], smoothed,
                color=color, linewidth=1.8, label="smoothed")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Steps")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    plt.tight_layout()
    if save_dir:
        out = Path(save_dir) / "training_curves.png"
        plt.savefig(out, bbox_inches="tight")
        print(f"📊 Saved: {out}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# ablation comparison plot
# ──────────────────────────────────────────────────────────────

def plot_comparison(
    logs: Dict[str, str | Path],
    save_path: Optional[str | Path] = None,
    smooth_window: int = 30,
) -> None:
    """
    Overlay reward curves for multiple agents on the same axes.

    Parameters
    ----------
    logs      : {agent_name: csv_path}
    save_path : Where to save the figure.
    """
    colors = ["steelblue", "tomato", "mediumseagreen", "darkorchid", "goldenrod"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for (name, path), color in zip(logs.items(), colors):
        df = load_log(path)
        raw = df["reward"].values
        smoothed = _smooth(raw, window=smooth_window)
        steps = df["steps"].values[len(raw) - len(smoothed):]
        ax.plot(steps, smoothed, label=name, color=color, linewidth=2.0)

    ax.set_title("Agent Comparison — Smoothed Episode Reward", fontweight="bold", fontsize=13)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Reward")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved comparison plot: {save_path}")
    plt.show()


def plot_reward_bars(
    logs: Dict[str, str | Path],
    last_n: int = 50,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Bar chart of mean reward over the last N episodes per agent.
    Useful for the ablation table in the README.
    """
    names, means, stds = [], [], []
    for name, path in logs.items():
        df = load_log(path)
        rewards = df["reward"].values[-last_n:]
        names.append(name)
        means.append(rewards.mean())
        stds.append(rewards.std())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "tomato", "mediumseagreen", "darkorchid"]
    ax.bar(names, means, yerr=stds, capsize=5,
           color=colors[:len(names)], alpha=0.85, edgecolor="white")
    ax.set_title(f"Mean Reward (last {last_n} episodes)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Episode Reward")

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.3, f"{m:.1f}", ha="center", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved bar chart: {save_path}")
    plt.show()
