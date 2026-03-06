"""
scripts/train.py
─────────────────
CLI: train any DQN agent on any Atari game.

Usage
-----
    # best agent with PER
    python scripts/train.py --agent dueling_double --env PongNoFrameskip-v4 --per

    # vanilla baseline
    python scripts/train.py --agent vanilla --env BreakoutNoFrameskip-v4

    # double DQN, 2M steps, on GPU
    python scripts/train.py --agent double --env SpaceInvadersNoFrameskip-v4 \
        --total_steps 2000000 --device cuda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
from src.utils.env_wrappers import make_atari_env
from src.agents.vanilla_dqn import VanillaDQN
from src.agents.double_dqn import DoubleDQN
from src.agents.dueling_dqn import DuelingDQN
from src.agents.dueling_double_dqn import DuelingDoubleDQN


AGENT_MAP = {
    "vanilla":        VanillaDQN,
    "double":         DoubleDQN,
    "dueling":        DuelingDQN,
    "dueling_double": DuelingDoubleDQN,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN agent on Atari")
    p.add_argument("--agent",       required=True, choices=list(AGENT_MAP.keys()))
    p.add_argument("--env",         default="PongNoFrameskip-v4")
    p.add_argument("--total_steps", type=int,   default=1_000_000)
    p.add_argument("--per",         action="store_true", help="Enable PER (dueling_double only)")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--buffer_size", type=int,   default=100_000)
    p.add_argument("--target_update", type=int, default=1_000)
    p.add_argument("--min_replay",  type=int,   default=10_000)
    p.add_argument("--save_dir",    default="results")
    p.add_argument("--log_every",   type=int,   default=10)
    return p.parse_args()


def main():
    args = parse_args()

    # get n_actions from a temp env
    tmp = make_atari_env(args.env, clip_rewards=False, episode_life=False)
    n_actions = tmp.action_space.n
    tmp.close()
    print(f"🕹️  Env: {args.env}  |  Actions: {n_actions}")

    AgentClass = AGENT_MAP[args.agent]

    agent_kwargs = dict(
        env_id=args.env,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update=args.target_update,
        min_replay_size=args.min_replay,
        save_dir=args.save_dir,
        device=args.device,
    )

    if args.agent == "dueling_double":
        agent_kwargs["use_per"] = args.per

    agent = AgentClass(**agent_kwargs)
    agent.train(total_steps=args.total_steps, log_every=args.log_every)


if __name__ == "__main__":
    main()
