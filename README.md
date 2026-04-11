# RewardForge

**LLM-driven reward shaping for reinforcement learning agents.**

RewardForge watches a reinforcement learning agent train in real time, reads the reward curve, and uses an LLM (qwen3-32b via Groq) to dynamically rewrite the reward function — accelerating learning without human intervention.

Inspired by NVIDIA's [Eureka paper (2023)](https://eureka-research.github.io/), implemented with Stable-Baselines3 + Groq API (qwen/qwen3-32b).

---

## Experiment Results — CartPole-v1

4 conditions × 5 seeds (20 runs total, 20k steps each).

| Condition | Median Best Reward | Steps to 450 |
|---|---|---|
| RewardForge | 500.0 | 12,000 |
| Baseline PPO | 491.3 | 14,000 |
| Ablation (blind) | 500.0 | 16,000 |
| Random shaping | 572.3 | 12,000 |

**Best single run:** RewardForge seed 0 → 629.9 (only run where the LLM actually fired)

**Mann-Whitney U: p = 0.069** — trend real, n=5 too small to claim significance.

### What we learned

CartPole is solved too fast by PPO for the stagnation trigger to fire consistently.
The LLM intervened in only 1 of 5 RewardForge seeds — but that seed won the whole experiment.

Key finding: any reward shaping (even random) beats vanilla PPO on CartPole.
This validates the shaping hypothesis but exposes the real question:
**what happens on an environment where random shaping actively hurts?**

That's LunarLander-v3. Results coming next.

---

## Key CartPole Result

On CartPole-v1 (20k steps, no GPU):

| Step  | Mean Reward | Event |
|-------|-------------|-------|
| 2000  | 9.00        | Barely balancing |
| 4000  | 450.00      | PPO clicked |
| 6000  | 332.20      | LLM triggered (drop detected) |
| 12000 | 525.99      | Near-perfect balance |
| 18000 | **576.16**  | **Best ever (CartPole max = 500)** |

**qwen3-32b independently discovered parabolic reward shaping** — without being told it exists:

```python
def custom_reward(obs, action, reward, terminated, info):
    theta, x = obs[2], obs[0]
    shaping = 0.4 * (1 - (theta / 0.209)**2) + \
              0.1 * (1 - (x / 2.4)**2) - 0.1
    return float(reward + shaping)
```

`0.209 radians ≈ 12°` — CartPole's exact failure threshold. The LLM found this boundary from training history alone.

---

## How It Works

```
Train PPO → Log reward every 500 steps
         → Reward stagnates? → Trigger RewardForge
                             → Send curve + current fn to qwen3-32b (Groq)
                             → LLM returns new reward fn (Python)
                             → exec() safely, inject into env wrapper
                             → Reset value head → Resume training
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/rewardforge
cd rewardforge
pip install -r requirements.txt
```

Add your Groq API key to `.env`:

```
GROQ_API_KEY=gsk_your_key_here
```

Run CartPole:

```bash
python main.py
```

Run LunarLander:

```bash
python lunarlander/main.py
```

Run full experiment (4 conditions × 5 seeds):

```bash
python lunarlander/experiment_runner.py
```

Results are saved under `runs/`.

---

## Stack

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO
- [Gymnasium](https://gymnasium.farama.org/) — CartPole-v1, LunarLander-v3
- [Groq API](https://console.groq.com/) — qwen/qwen3-32b (60 RPM, 1K RPD free tier)
- Python 3.10+

---

## File Structure

```
rewardforge/
├── main.py                   # CartPole entry point
├── env_wrapper.py            # CustomCartPole with injectable reward fn
├── rewardforge_agent.py      # LLM agent logic (Groq / qwen3-32b)
├── experiment_runner.py      # CartPole 4-condition experiment
├── test_groq.py              # API diagnostic script
├── requirements.txt
├── .env                      # GROQ_API_KEY (not committed)
├── runs/                     # CartPole logs
└── lunarlander/
    ├── main.py               # LunarLander entry point
    ├── env_wrapper.py        # CustomLunarLander
    ├── rewardforge_agent.py  # LLM agent (LunarLander prompt)
    └── experiment_runner.py  # LunarLander 4-condition experiment
```

---

## Roadmap

- [x] CartPole-v1 prototype with LLM reward shaping (qwen3-32b)
- [x] Controlled experiments (baseline vs RewardForge vs ablation) — CartPole
- [x] LunarLander-v3 (sparser reward — real test)
- [ ] LunarLander experiment results (4 conditions × 5 seeds)
- [ ] LangGraph multi-agent orchestration (Planner → Diagnoser → RewardWriter → Evaluator)
- [ ] A4000 GPU training via SSH

---

## Citation / Reference

> Ma, Y., et al. "Eureka: Human-Level Reward Design via Coding Large Language Models." arXiv 2023.

---

## Author

M.Tech AI — PDEU, Gandhinagar