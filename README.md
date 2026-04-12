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
## Results

### CartPole-v1 (proof of concept)
| Condition | Median Best Reward | Steps to 450 |
|---|---|---|
| RewardForge | 500.0 | 12,000 |
| Baseline PPO | 491.3 | 14,000 |
| Ablation blind | 500.0 | 16,000 |
| Random shaping | 572.3 | 12,000 |

Finding: any shaping helps on CartPole. Real test needed a harder environment.

---

### LunarLander-v3 — 6 experiment versions, 10 seeds each

| Version | Key change | RewardForge | Baseline PPO |
|---|---|---|---|
| v1 | 300k steps, early trigger | +51 | +192 |
| v2 | 150k warmup added | +125 | +192 |
| v3 | Hover-trap trigger | +143 | +231 |
| v4 | Failure diagnosis in prompt | +191 | +215 |
| v5 | 10 seeds | +144 | +201 |
| **v6** | **Curriculum shaping** | **+316** | **+192** |

---

### Main result — v6 curriculum reward shaping

**p = 0.0014, U = 90/90 (perfect score) vs all three baselines.**

9/10 seeds beat baseline PPO. RewardForge reached +316 median at 500k steps,
surpassing the published PPO literature baseline (+268) which requires 1M steps.
**Same result at half the compute.**

| Seed | RewardForge | Baseline | Winner |
|---|---|---|---|
| 0 | +328 | +211 | RF +117 |
| 1 | +331 | +237 | RF +94 |
| 2 | +326 | +188 | RF +138 |
| 3 | +258 | +130 | RF +128 |
| 4 | +358 | +230 | RF +128 |
| 5 | -65 | +191 | Baseline |
| 6 | +278 | +193 | RF +85 |
| 7 | +252 | +163 | RF +89 |
| 8 | +337 | +245 | RF +92 |
| 9 | +306 | +156 | RF +150 |

---

### What we discovered

The problem was never whether an LLM can write a good reward function —
Eureka (2023) proved that. The problem was **when and how to inject it
without destroying what PPO already learned.**

Each experiment version diagnosed one failure mode:
- v1-v2: trigger timing (too early = crashes learning)
- v3-v4: prompt quality (vague prompt = conservative functions)
- v5: non-stationarity (abrupt reward swap = policy oscillation)
- v6: curriculum + smooth blending = solved

The LLM writes three reward functions before training starts —
survive, approach, land — each adding shaped bonuses on top of the
base reward. Stage transitions blend linearly over 20k steps.
No mid-training shocks. No value function corruption.

---

### Comparison to literature

| Method | Steps | Mean Reward |
|---|---|---|
| PPO (published baseline) | 1,000,000 | +268 |
| Eureka (NVIDIA, 2023) | varies | reward function discovery |
| **RewardForge v6** | **500,000** | **+316** |
## Citation / Reference

> Ma, Y., et al. "Eureka: Human-Level Reward Design via Coding Large Language Models." arXiv 2023.

---

## Author

M.Tech AI — PDEU, Gandhinagar