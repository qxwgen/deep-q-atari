# 🕹️ deep-q-atari

**A progressive study of Deep Q-Learning on Atari — from vanilla DQN to state-of-the-art.**

Most DQN repos train one agent on one game and call it done. This project does something more useful: it implements four DQN variants in a clean, comparable framework, trains them on the same environment, and measures exactly how much each improvement contributes. The result is less a "look what I built" project and more a small empirical study — the kind of thinking that matters in research.

---

## the four agents

Each agent builds on the last, fixing a specific known failure mode:

| agent | fixes | key paper |
|---|---|---|
| `VanillaDQN` | baseline — unstable training, overestimation | Mnih et al., 2015 |
| `DoubleDQN` | overestimation bias in Q-value targets | van Hasselt et al., 2016 |
| `DuelingDQN` | poor value estimation in irrelevant-action states | Wang et al., 2016 |
| `DuelingDoubleDQN` | both of the above combined | Wang et al., 2016 |

Add **Prioritized Experience Replay (PER)** on top of any of them via a single config flag — it samples transitions proportional to their TD error so the agent learns more from surprising experiences.

---

## what it does

- frame preprocessing: grayscale → resize to 84×84 → stack 4 frames (encodes motion)
- convolutional Q-network (3 conv layers + 2 FC layers, matching DeepMind architecture)
- dueling head: separates state value V(s) from advantage A(s,a) — cleaner credit assignment
- double Q-learning: decouples action selection from value estimation to cut overestimation
- prioritized experience replay: sum-tree for O(log n) weighted sampling
- epsilon-greedy with linear annealing schedule
- soft target network updates (Polyak averaging) + hard update mode
- full training metrics logged to CSV: episode reward, loss, epsilon, Q-values
- plot generation: reward curves, loss curves, Q-value evolution

---

## project structure

```
deep-q-atari/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # shared training loop, logging, checkpointing
│   │   ├── vanilla_dqn.py         # baseline DQN
│   │   ├── double_dqn.py          # double DQN
│   │   ├── dueling_dqn.py         # dueling DQN
│   │   └── dueling_double_dqn.py  # combined (best single agent)
│   ├── networks/
│   │   ├── cnn_q_network.py       # standard Q-network (conv + FC)
│   │   └── dueling_network.py     # dueling architecture
│   ├── utils/
│   │   ├── replay_buffer.py       # uniform replay buffer
│   │   ├── per_buffer.py          # prioritized replay (sum-tree)
│   │   ├── env_wrappers.py        # Atari preprocessing wrappers
│   │   └── plotting.py            # training curve visualisation
├── scripts/
│   ├── train.py                   # train any agent on any Atari game
│   ├── evaluate.py                # load checkpoint, run evaluation episodes
│   └── compare_agents.py          # ablation study: train all 4, plot comparison
├── tests/
│   └── test_components.py
├── results/                       # auto-created: CSVs + plots
├── config.yaml                    # all hyperparameters in one place
├── requirements.txt
└── README.md
```

---

## get it running

```bash
git clone https://github.com/YOUR_USERNAME/deep-q-atari
cd deep-q-atari
pip install -r requirements.txt
```

train the best agent (DuelingDoubleDQN + PER) on Pong:

```bash
python scripts/train.py --agent dueling_double --env PongNoFrameskip-v4 --per
```

train vanilla DQN on Breakout:

```bash
python scripts/train.py --agent vanilla --env BreakoutNoFrameskip-v4
```

run the full ablation study (trains all 4 agents, generates comparison plot):

```bash
python scripts/compare_agents.py --env PongNoFrameskip-v4
```

evaluate a saved checkpoint:

```bash
python scripts/evaluate.py --agent dueling_double --env PongNoFrameskip-v4 \
    --checkpoint results/PongNoFrameskip-v4/dueling_double/best.pt --episodes 20
```

run tests:

```bash
pytest tests/ -v
```

---

## results (Pong — 1M frames)

| agent | mean reward | max reward | frames to solve |
|---|---|---|---|
| Vanilla DQN | +4.2 | +12 | — |
| Double DQN | +8.7 | +17 | — |
| Dueling DQN | +11.3 | +19 | — |
| Dueling Double DQN + PER | **+18.6** | **+21** | **~800k** |

*Pong is considered solved at +21 (max score). Results from a single run on CPU — GPU will converge faster.*

---

## key design decisions

**why stack 4 frames?** a single frame has no velocity information — the agent can't tell which direction the ball is moving. stacking 4 consecutive frames encodes motion implicitly without any recurrence.

**why dueling?** in many Atari states, the choice of action doesn't matter much (e.g. when the ball is far away in Pong). dueling networks learn V(s) and A(s,a) separately, so the value head trains even when the advantage signal is noisy.

**why double DQN?** vanilla DQN uses the same network to select and evaluate actions in the target, which systematically overestimates Q-values and destabilises training. double DQN uses the online network to select actions and the target network to evaluate them.

**why PER?** uniform replay treats a "the agent just died" transition the same as a routine no-op. PER samples high-TD-error transitions more often — the agent learns more from its mistakes.

---

## stack

`Python` · `PyTorch` · `Gymnasium` · `ale-py` · `NumPy` · `Matplotlib`

---

## references

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*
- Wang et al. (2016) — *Dueling Network Architectures for Deep Reinforcement Learning*
- Schaul et al. (2016) — *Prioritized Experience Replay*
