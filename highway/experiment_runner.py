"""
highway/experiment_runner.py — 2-condition experiment on highway-v0.

Conditions
──────────────────────────────────────────────────────────────────────────────
  llm_single    PPO + single LLM reward fn (20k-step warmup from step 0)
  baseline_ppo  PPO only, built-in env reward

Output
──────────────────────────────────────────────────────────────────────────────
  runs/experiments/highway/{timestamp}_{label}/
    ├── experiment_metadata.json
    ├── results_summary.csv
    ├── results_analysis.txt
    └── {condition}/seed_{seed}/
        ├── training_log.csv
        ├── summary.txt
        └── reward_fn.py

Usage
──────────────────────────────────────────────────────────────────────────────
  cd rewardforge/
  python highway/experiment_runner.py
"""

import csv
import json
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from highway.env_wrapper   import CustomHighwayEnv              # noqa: E402
from highway.highway_agent import (                             # noqa: E402
    BLEND_STEPS, SingleFnState, make_single_blend_fn, request_single_fn,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS   = 30_000    # 161 steps/sec → ~3 min/seed, ~60 min for 20 seeds
CHECKPOINT_EVERY  = 3_000     # 10 checkpoints per run
N_EVAL_EPISODES   = 5         # 5 ep × 30 steps = 150 eval steps each checkpoint

SEEDS      = list(range(10))
CONDITIONS = ["llm_single", "baseline_ppo"]

RUN_LABEL  = "v1_highway_llm_single_vs_baseline"

# highway-env config overrides applied to every env (train + eval)
# Fewer vehicles = faster Python physics simulation
_ENV_CONFIG = {
    "vehicles_count": 3,    # default 5 → 3 (fewer vehicles = much faster sim)
    "duration":       30,   # max episode steps (default 40 → 30, same behaviour)
    "lanes_count":     3,   # 3 lanes (default 4)
}

_EXPR_BASE = _ROOT / "runs" / "experiments" / "highway"

Condition = Literal["llm_single", "baseline_ppo"]

# Catastrophic failure: best episode reward below this
CATASTROPHIC_THRESHOLD = 5.0   # a crashed-dominated agent gets < 5 per episode

# Literature: highway-env trained PPO typically reaches 25-35/episode
# No single canonical paper benchmark; using community observed values
LITERATURE_MEAN   = 30.0
LITERATURE_STD    = 5.0
LITERATURE_STEPS  = 1_000_000
LITERATURE_SOURCE = "highway-env community PPO benchmark (approx)"


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helper
# Episodes are short (~40 steps) so 20 episodes ≈ 800 steps total
# ═══════════════════════════════════════════════════════════════════════════════
def _evaluate(model: PPO, eval_env: CustomHighwayEnv,
              n: int = N_EVAL_EPISODES) -> tuple[float, float]:
    rewards = []
    for _ in range(n):
        obs, _ = eval_env.reset()
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = eval_env.step(action)
            total += r
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


# ═══════════════════════════════════════════════════════════════════════════════
# Callback
# ═══════════════════════════════════════════════════════════════════════════════
class HighwayCallback(BaseCallback):

    def __init__(self, train_env: CustomHighwayEnv,
                 condition: Condition, seed: int):
        super().__init__(verbose=0)
        self.train_env = train_env
        self.condition = condition
        self.seed      = seed

        self._eval_env = CustomHighwayEnv(**_ENV_CONFIG)
        self._eval_env.reward_fn      = train_env.reward_fn
        self._eval_env.reward_fn_code = train_env.reward_fn_code

        self.log_rows:   list[dict] = []
        self.best_reward = -float("inf")
        self._single_state: SingleFnState | None = None
        self._single_code:  str | None = None

        if condition == "llm_single":
            self._init_single()

    def _init_single(self) -> None:
        result = request_single_fn()
        if result is None:
            print("  \u26a0\ufe0f  llm_single failed \u2014 using base reward.")
            return
        fn, code = result
        self._single_code  = code
        self._single_state = SingleFnState()
        blended = make_single_blend_fn(fn, self._single_state)
        label   = "llm_single:warmup"
        self.train_env.reward_fn      = blended
        self.train_env.reward_fn_code = label
        self._eval_env.reward_fn      = blended
        self._eval_env.reward_fn_code = label
        print(f"  \U0001f52e  Warmup: alpha 0\u21921 over {BLEND_STEPS:,} steps")

    def _tick_warmup(self) -> None:
        st = self._single_state
        if st is None or not st.blending:
            return
        elapsed  = self.num_timesteps - st.blend_start_step
        st.alpha = min(1.0, elapsed / BLEND_STEPS)
        if elapsed >= BLEND_STEPS:
            st.blending = False
            st.advances = 1
            print(f"\n    \U0001f52e  Full shaping active @ step {self.num_timesteps:,}")

    def _on_step(self) -> bool:
        if self.condition == "llm_single" and self._single_state is not None:
            self._tick_warmup()

        if self.num_timesteps % CHECKPOINT_EVERY != 0:
            return True

        mean_rew, std_rew = _evaluate(self.model, self._eval_env)
        step = self.num_timesteps

        if mean_rew > self.best_reward:
            self.best_reward = mean_rew

        alpha_str = (f"  \u03b1={self._single_state.alpha:.2f}"
                     if self._single_state else "")
        print(f"  step {step:>7,}  mean={mean_rew:>+7.2f}  std={std_rew:>5.2f}{alpha_str}")

        self.log_rows.append({
            "step": step, "mean_reward": mean_rew, "std_reward": std_rew,
        })
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Per-run save
# ═══════════════════════════════════════════════════════════════════════════════
def _save_run(run_dir: Path, cb: HighwayCallback,
              condition: str, seed: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward", "std_reward"])
        for row in cb.log_rows:
            w.writerow([row["step"],
                        f"{row['mean_reward']:.4f}",
                        f"{row['std_reward']:.4f}"])

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("RewardForge Highway-v0 Experiment Summary\n")
        f.write("=" * 44 + "\n")
        f.write(f"Condition  : {condition}\n")
        f.write(f"Seed       : {seed}\n")
        f.write(f"Best reward: {cb.best_reward:.4f}\n")

    if cb._single_code:
        (run_dir / "reward_fn.py").write_text(cb._single_code, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════════
def run_single(condition: Condition, seed: int, run_dir: Path) -> dict:
    _set_seeds(seed)

    train_env = CustomHighwayEnv(**_ENV_CONFIG)
    train_env.reset(seed=seed)

    # PPO tuned for highway-v0 short episodes
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,        # smaller buffer → training starts after fewer env steps
        batch_size=64,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.8,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
    )

    cb = HighwayCallback(train_env, condition, seed)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb,
                reset_num_timesteps=True)

    final_reward = cb.log_rows[-1]["mean_reward"] if cb.log_rows else cb.best_reward
    final_std    = cb.log_rows[-1]["std_reward"]   if cb.log_rows else 0.0

    _save_run(run_dir, cb, condition, seed)
    train_env.close()
    cb._eval_env.close()

    return {
        "condition":    condition,
        "seed":         seed,
        "best_reward":  cb.best_reward,
        "final_reward": final_reward,
        "final_std":    final_std,
        "catastrophic": int(cb.best_reward < CATASTROPHIC_THRESHOLD),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical analysis
# ═══════════════════════════════════════════════════════════════════════════════
def _iqr(arr) -> float:
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def generate_analysis(results: list[dict], out_path: Path) -> str:
    from scipy import stats as sp_stats

    sep = "─" * 56
    lines = [
        "RewardForge Highway-v0 Experiment Analysis",
        "=" * 56,
        f"  {len(CONDITIONS)} conditions \u00d7 {len(SEEDS)} seeds = {len(results)} runs",
        f"  {TOTAL_TIMESTEPS:,} timesteps per run  (episodes \u2264 40 steps)",
        "",
        f"Reference ({LITERATURE_SOURCE})",
        f"  PPO @ {LITERATURE_STEPS:,} steps:  ~{LITERATURE_MEAN:.0f} \u00b1 {LITERATURE_STD:.0f} reward/episode",
        f"  (Our runs use {TOTAL_TIMESTEPS:,} steps)",
        "",
        "Per-condition statistics:",
        sep,
    ]

    for cond in CONDITIONS:
        rows  = [r for r in results if r["condition"] == cond]
        best  = [r["best_reward"]  for r in rows]
        final = [r["final_reward"] for r in rows]
        fstd  = [r["final_std"]    for r in rows]
        fails = sum(r["catastrophic"] for r in rows)
        lines += [
            f"\n  [{cond}]",
            f"    best_reward  : median={np.median(best):+.2f}  IQR={_iqr(best):.2f}"
            f"  range=[{min(best):+.1f}, {max(best):+.1f}]",
            f"    final_reward : median={np.median(final):+.2f}  IQR={_iqr(final):.2f}",
            f"    final_std    : median={np.median(fstd):.2f}",
            f"    catastrophic fails (best < {CATASTROPHIC_THRESHOLD:.0f}): {fails}/{len(rows)}",
        ]

    # ── Mann-Whitney ─────────────────────────────────────────────────────────
    if "llm_single" in CONDITIONS and "baseline_ppo" in CONDITIONS:
        s_best = [r["best_reward"] for r in results if r["condition"] == "llm_single"]
        b_best = [r["best_reward"] for r in results if r["condition"] == "baseline_ppo"]
        u, p   = sp_stats.mannwhitneyu(s_best, b_best, alternative="greater")
        _, p2  = sp_stats.mannwhitneyu(s_best, b_best, alternative="two-sided")
        sig    = "\u2705 p<0.05" if p < 0.05 else "\u274c p\u22650.05"

        s_final = [r["final_reward"] for r in results if r["condition"] == "llm_single"]
        s_fstd  = [r["final_std"]    for r in results if r["condition"] == "llm_single"]
        s_fails = sum(r["catastrophic"] for r in results if r["condition"] == "llm_single")
        b_final = [r["final_reward"] for r in results if r["condition"] == "baseline_ppo"]
        b_fstd  = [r["final_std"]    for r in results if r["condition"] == "baseline_ppo"]
        b_fails = sum(r["catastrophic"] for r in results if r["condition"] == "baseline_ppo")

        lines += [
            "", sep,
            "Statistical Test: Mann-Whitney U  (one-sided: llm_single > baseline)",
            sep,
            f"  llm_single vs baseline_ppo  : U={u:.0f}  p={p:.4f}  {sig}",
            f"  (two-sided p={p2:.4f})",
            "",
            "  Focused comparison: llm_single vs baseline_ppo",
            "  " + "-" * 52,
            f"  {'':26s} {'llm_single':>14}  {'baseline':>10}",
            f"  {'median best_reward':<26} {np.median(s_best):>+14.2f}  {np.median(b_best):>+10.2f}",
            f"  {'median final_reward':<26} {np.median(s_final):>+14.2f}  {np.median(b_final):>+10.2f}",
            f"  {'median final_std':<26} {np.median(s_fstd):>14.2f}  {np.median(b_fstd):>10.2f}",
            f"  {'catastrophic fails (< {:.0f})'.format(CATASTROPHIC_THRESHOLD):<26}"
            f" {s_fails:>14d}  {b_fails:>10d}",
            f"  {'n seeds':<26} {len(s_best):>14d}  {len(b_best):>10d}",
        ]

        lines += ["", sep, "Conclusion:", sep]
        if p < 0.05:
            lines.append(
                "  \u2705 SIGNIFICANT \u2014 llm_single outperformed baseline_ppo (p < 0.05).\n"
                "     Pre-training a single LLM reward function helps on highway-v0."
            )
        else:
            lines.append(
                "  \u274c NOT SIGNIFICANT \u2014 p \u2265 0.05.\n"
                "     Directional trend may be visible but not confirmed at n=10."
            )

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder    = f"{timestamp}_{RUN_LABEL}" if RUN_LABEL else timestamp
    EXPR_ROOT = _EXPR_BASE / folder
    EXPR_ROOT.mkdir(parents=True, exist_ok=True)

    with open(EXPR_ROOT / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_label":       RUN_LABEL,
            "timestamp":       timestamp,
            "environment":     "highway-v0",
            "conditions":      CONDITIONS,
            "seeds":           SEEDS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "n_eval_episodes": N_EVAL_EPISODES,
            "blend_steps":     BLEND_STEPS,
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "model":           "llama-3.3-70b-versatile",
            "ppo_gamma":       0.8,
            "ppo_n_steps":     256,
            "ppo_lr":          5e-4,
            "ppo_ent_coef":    0.01,
        }, f, indent=2)

    plan  = [(cond, seed) for cond in CONDITIONS for seed in SEEDS]
    total = len(plan)

    print("\n\u2554" + "\u2550" * 58 + "\u2557")
    print("\u2551   RewardForge \u2014 highway-v0 Experiment Runner           \u2551")
    print("\u255a" + "\u2550" * 58 + "\u255d")
    print(f"  {len(CONDITIONS)} conditions \u00d7 {len(SEEDS)} seeds = {total} total runs")
    print(f"  {TOTAL_TIMESTEPS:,} steps \u00d7 max 40 steps/episode \u2248 5,000 episodes/seed")
    print(f"  Output \u2192 {EXPR_ROOT}\n")

    all_results: list[dict] = []

    for idx, (condition, seed) in enumerate(plan, start=1):
        run_dir = EXPR_ROOT / condition / f"seed_{seed}"
        print(f"\n[{idx:>2}/{total}] {condition:<16} seed_{seed}", flush=True)

        result = run_single(condition, seed, run_dir)
        all_results.append(result)

        print(
            f"        \u2713  best={result['best_reward']:>+7.2f}"
            f"  final={result['final_reward']:>+7.2f}"
            f"  std={result['final_std']:>5.2f}"
            f"  catastrophic={'YES' if result['catastrophic'] else 'no'}"
        )

    csv_path = EXPR_ROOT / "results_summary.csv"
    fields   = ["condition", "seed", "best_reward", "final_reward", "final_std", "catastrophic"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"\n\U0001f4ca  Results CSV  \u2192 {csv_path}")

    analysis_path = EXPR_ROOT / "results_analysis.txt"
    analysis_text = generate_analysis(all_results, analysis_path)
    print(f"\U0001f4c8  Analysis     \u2192 {analysis_path}")

    print("\n" + "=" * 58)
    print(analysis_text)
    print("=" * 58)
    print("\u2705  All experiments complete.\n")


if __name__ == "__main__":
    main()
