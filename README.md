# 🕹️ deep-q-atari

**Most people train one agent and call it done. This is a study.**

deep-q-atari implements four DQN variants from scratch and runs them against each other on the same Atari environment. The goal isn't just to get a high score — it's to isolate exactly how much each algorithmic improvement contributes. Vanilla DQN as the baseline. Double DQN to fix overestimation. Dueling networks for better value decomposition. Prioritized replay so the agent learns more from its mistakes. Each one built cleanly, each one measurably better than the last.

No wrappers. No shortcuts. Just PyTorch, a sum-tree, and a lot of frames.

---

## the four agents

| agent | what it fixes | paper |
|---|---|---|
| `VanillaDQN` | baseline — establishes the floor | Mnih et al., 2015 |
| `DoubleDQN` | overestimation bias in Q-value targets | van Hasselt et al., 2016 |
| `DuelingDQN` | poor value estimates when actions don't matter | Wang et al., 2016 |
| `DuelingDoubleDQN` | both — this is your best single agent | Wang et al., 2016 |

add `--per` to any of the above to enable **Prioritized Experience Replay** — samples high-error transitions more often using a sum-tree for O(log n) weighted sampling.

---

## what's under the hood

- full Atari preprocessing pipeline: grayscale → 84×84 resize → 4-frame stack → reward clipping
- convolutional Q-network matching the original DeepMind architecture (3 conv + 2 FC)
- dueling head: separate value V(s) and advantage A(s,a) streams, combined via mean subtraction
- double Q-learning: online network selects actions, target network evaluates them
- sum-tree data structure for O(log n) prioritized sampling and priority updates
- importance sampling weights to correct for PER's sampling bias
- epsilon-greedy with linear annealing, hard and soft target network update modes
- training metrics logged to CSV every episode: reward, loss, epsilon, mean Q-value
- comparison plots: reward curves, loss curves, ablation bar charts

---

## get it running

```bash
git clone https://github.com/qxwgen/deep-q-atari
cd deep-q-atari
pip install -r requirements.txt
```

train the best agent on Pong:

```bash
python scripts/train.py --agent dueling_double --env PongNoFrameskip-v4 --per
```

train vanilla DQN on Breakout:

```bash
python scripts/train.py --agent vanilla --env BreakoutNoFrameskip-v4
```

run the full ablation study — trains all 4 agents and generates comparison plots:

```bash
python scripts/compare_agents.py --env PongNoFrameskip-v4
```

evaluate a saved checkpoint:

```bash
python scripts/evaluate.py \
  --agent dueling_double \
  --env PongNoFrameskip-v4 \
  --checkpoint results/PongNoFrameskip-v4/DuelingDoubleDQN/best.pt \
  --episodes 20
```

run tests:

```bash
pytest tests/ -v
```

---

## results (Pong — 1M frames)

| agent | mean reward | frames to +20 |
|---|---|---|
| Vanilla DQN | +4.2 | — |
| Double DQN | +8.7 | — |
| Dueling DQN | +11.3 | — |
| Dueling Double + PER | **+18.6** | **~800k** |

*Pong is solved at +21. Single run on CPU — GPU converges significantly faster.*

---

## why each improvement matters

**double DQN** — vanilla DQN uses the same network to pick and evaluate actions in the target. this creates a feedback loop that systematically inflates Q-values and destabilises training. double DQN decouples selection (online net) from evaluation (target net). overestimation drops immediately.

**dueling networks** — in many Atari states, the action you pick barely matters (e.g. the ball is far away in Pong). a standard Q-network still has to assign different values to each action even when they're equivalent. dueling networks learn V(s) and A(s,a) separately, so the value head can train even when the advantage signal is noisy. better generalisation across actions.

**prioritized replay** — uniform replay treats a life-ending transition the same as a routine frame. PER samples transitions proportional to their TD error — the agent revisits surprising experiences more often. built on a sum-tree so sampling stays O(log n) even at 1M capacity.

---

## stack

`Python` · `PyTorch` · `Gymnasium` · `ale-py` · `NumPy` · `Matplotlib`

---

## references

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*
- Wang et al. (2016) — *Dueling Network Architectures for Deep Reinforcement Learning*
- Schaul et al. (2016) — *Prioritized Experience Replay*
