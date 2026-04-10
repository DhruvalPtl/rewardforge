# RewardForge

**LLM-driven reward shaping for reinforcement learning agents.**

RewardForge watches a reinforcement learning agent train in real time, reads the reward curve, and uses an LLM (Gemini) to dynamically rewrite the reward function — accelerating learning without human intervention.

Inspired by NVIDIA's [Eureka paper (2023)](https://eureka-research.github.io/), implemented with Stable-Baselines3 + Google Gemini API.

---

## Key Result

On CartPole-v1 (20k steps, no GPU):

| Step  | Mean Reward | Event |
|-------|-------------|-------|
| 2000  | 9.00        | Barely balancing |
| 4000  | 450.00      | PPO clicked |
| 6000  | 332.20      | Gemini triggered (drop detected) |
| 12000 | 525.99      | Near-perfect balance |
| 18000 | **576.16**  | **Best ever (CartPole max = 500)** |

**Gemini independently discovered parabolic reward shaping** — without being told it exists:

```python
def custom_reward(obs, action, reward, terminated, info):
    theta, x = obs[2], obs[0]
    shaping = 0.4 * (1 - (theta / 0.209)**2) + \
              0.1 * (1 - (x / 2.4)**2) - 0.1
    return float(reward + shaping)
```

`0.209 radians ≈ 12°` — CartPole's exact failure threshold. Gemini found this boundary from training history alone.

---

## How It Works

```
Train PPO → Log reward every 500 steps
         → Reward stagnates? → Trigger RewardForge
                             → Send curve + current fn to Gemini
                             → Gemini returns new reward fn (Python)
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

Add your Gemini API key to `.env`:

```
GOOGLE_API_KEY=your_key_here
```

Run:

```bash
python main.py
```

Results are saved as CSV in `runs/`.

---

## Stack

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO
- [Gymnasium](https://gymnasium.farama.org/) — CartPole-v1
- [Google Gemini API](https://ai.google.dev/) — gemini-1.5-flash (free tier)
- Python 3.10+

---

## File Structure

```
rewardforge/
├── main.py               # entry point
├── env_wrapper.py        # CustomCartPole with injectable reward fn
├── rewardforge_agent.py  # Gemini agent logic
├── requirements.txt
├── .env                  # GOOGLE_API_KEY (not committed)
└── runs/                 # CSV logs
```

---

## Roadmap

- [x] CartPole-v1 prototype with Gemini reward shaping
- [ ] Controlled experiments (baseline vs RewardForge vs ablation)
- [ ] LunarLander-v2 (sparser reward — real test)
- [ ] LangGraph multi-agent orchestration (Planner → Diagnoser → RewardWriter → Evaluator)
- [ ] A4000 GPU training via SSH

---

## Citation / Reference

> Ma, Y., et al. "Eureka: Human-Level Reward Design via Coding Large Language Models." arXiv 2023.

---

## Author

M.Tech AI — PDEU, Gandhinagar# RewardForge

**LLM-driven reward shaping for reinforcement learning agents.**

RewardForge watches a reinforcement learning agent train in real time, reads the reward curve, and uses an LLM (Gemini) to dynamically rewrite the reward function — accelerating learning without human intervention.

Inspired by NVIDIA's [Eureka paper (2023)](https://eureka-research.github.io/), implemented with Stable-Baselines3 + Google Gemini API.

---

## Key Result

On CartPole-v1 (20k steps, no GPU):

| Step  | Mean Reward | Event |
|-------|-------------|-------|
| 2000  | 9.00        | Barely balancing |
| 4000  | 450.00      | PPO clicked |
| 6000  | 332.20      | Gemini triggered (drop detected) |
| 12000 | 525.99      | Near-perfect balance |
| 18000 | **576.16**  | **Best ever (CartPole max = 500)** |

**Gemini independently discovered parabolic reward shaping** — without being told it exists:

```python
def custom_reward(obs, action, reward, terminated, info):
    theta, x = obs[2], obs[0]
    shaping = 0.4 * (1 - (theta / 0.209)**2) + \
              0.1 * (1 - (x / 2.4)**2) - 0.1
    return float(reward + shaping)
```

`0.209 radians ≈ 12°` — CartPole's exact failure threshold. Gemini found this boundary from training history alone.

---

## How It Works

```
Train PPO → Log reward every 500 steps
         → Reward stagnates? → Trigger RewardForge
                             → Send curve + current fn to Gemini
                             → Gemini returns new reward fn (Python)
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

Add your Gemini API key to `.env`:

```
GOOGLE_API_KEY=your_key_here
```

Run:

```bash
python main.py
```

Results are saved as CSV in `runs/`.

---

## Stack

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO
- [Gymnasium](https://gymnasium.farama.org/) — CartPole-v1
- [Google Gemini API](https://ai.google.dev/) — gemini-1.5-flash (free tier)
- Python 3.10+

---

## File Structure

```
rewardforge/
├── main.py               # entry point
├── env_wrapper.py        # CustomCartPole with injectable reward fn
├── rewardforge_agent.py  # Gemini agent logic
├── requirements.txt
├── .env                  # GOOGLE_API_KEY (not committed)
└── runs/                 # CSV logs
```

---

## Roadmap

- [x] CartPole-v1 prototype with Gemini reward shaping
- [ ] Controlled experiments (baseline vs RewardForge vs ablation)
- [ ] LunarLander-v2 (sparser reward — real test)
- [ ] LangGraph multi-agent orchestration (Planner → Diagnoser → RewardWriter → Evaluator)
- [ ] A4000 GPU training via SSH

---

## Citation / Reference

> Ma, Y., et al. "Eureka: Human-Level Reward Design via Coding Large Language Models." arXiv 2023.

---

## Author

M.Tech AI — PDEU, Gandhinagar