"""
scripts/evaluate.py
────────────────────
CLI: load a saved checkpoint and run evaluation episodes.

Usage
-----
    python scripts/evaluate.py \
        --agent dueling_double \
        --env   PongNoFrameskip-v4 \
        --checkpoint results/PongNoFrameskip-v4/DuelingDoubleDQN/best.pt \
        --episodes 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    p = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    p.add_argument("--agent",      required=True, choices=list(AGENT_MAP.keys()))
    p.add_argument("--env",        default="PongNoFrameskip-v4")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes",   type=int, default=20)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--render",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    env = make_atari_env(args.env, clip_rewards=False, episode_life=False)
    n_actions = env.action_space.n

    AgentClass = AGENT_MAP[args.agent]
    agent = AgentClass(
        env_id=args.env,
        n_actions=n_actions,
        device=args.device,
    )
    agent.load_checkpoint(args.checkpoint)
    agent.online_net.eval()

    rewards = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            obs_t = __import__("torch").FloatTensor(obs).unsqueeze(0).to(agent.device)
            with __import__("torch").no_grad():
                action = agent.online_net(obs_t).argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"  Episode {ep:3d}: reward = {ep_reward:6.1f}")

    env.close()
    print(f"\n📊 Results over {args.episodes} episodes:")
    print(f"   Mean   : {np.mean(rewards):.2f}")
    print(f"   Std    : {np.std(rewards):.2f}")
    print(f"   Max    : {np.max(rewards):.2f}")
    print(f"   Min    : {np.min(rewards):.2f}")


if __name__ == "__main__":
    main()
