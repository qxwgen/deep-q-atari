"""
scripts/compare_agents.py
──────────────────────────
Ablation study: train all 4 DQN variants on the same game and compare.

Usage
-----
    python scripts/compare_agents.py --env PongNoFrameskip-v4 --steps 500000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env_wrappers import make_atari_env
from src.agents.vanilla_dqn import VanillaDQN
from src.agents.double_dqn import DoubleDQN
from src.agents.dueling_dqn import DuelingDQN
from src.agents.dueling_double_dqn import DuelingDoubleDQN
from src.utils.plotting import plot_comparison, plot_reward_bars


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation study across all DQN variants")
    p.add_argument("--env",     default="PongNoFrameskip-v4")
    p.add_argument("--steps",   type=int, default=500_000)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--save_dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()

    tmp = make_atari_env(args.env, clip_rewards=False, episode_life=False)
    n_actions = tmp.action_space.n
    tmp.close()

    shared = dict(
        env_id=args.env,
        n_actions=n_actions,
        device=args.device,
        save_dir=args.save_dir,
    )

    agents = [
        ("Vanilla DQN",           VanillaDQN(**shared)),
        ("Double DQN",            DoubleDQN(**shared)),
        ("Dueling DQN",           DuelingDQN(**shared)),
        ("Dueling Double DQN+PER", DuelingDoubleDQN(**shared, use_per=True)),
    ]

    log_paths = {}

    for name, agent in agents:
        print(f"\n{'='*55}")
        print(f"  Training: {name}")
        print(f"{'='*55}")
        agent.train(total_steps=args.steps, log_every=20)
        log_paths[name] = agent._log_path

    # plots
    out_dir = Path(args.save_dir) / args.env
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison(
        logs=log_paths,
        save_path=out_dir / "comparison_rewards.png",
    )
    plot_reward_bars(
        logs=log_paths,
        save_path=out_dir / "ablation_bars.png",
    )
    print(f"\n🏁 Ablation study complete. Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
