"""
bipedal/experiment_runner.py — 2-condition scientific comparison on BipedalWalker-v3.

Conditions
──────────────────────────────────────────────────────────────────────────────
  llm_single    PPO + single LLM reward fn applied from step 0 (20k warmup)
  baseline_ppo  PPO only, built-in env reward, no shaping

Literature Reference
──────────────────────────────────────────────────────────────────────────────
  SB3 Zoo PPO BipedalWalker-v3 @ 5,000,000 steps: 312 ± 61
  Source: https://github.com/DLR-RM/rl-baselines3-zoo

Output
──────────────────────────────────────────────────────────────────────────────
  runs/experiments/bipedal/{timestamp}_{label}/
    ├── experiment_metadata.json
    ├── results_summary.csv
    ├── results_analysis.txt
    └── {condition}/seed_{seed}/
        ├── training_log.csv
        ├── summary.txt
        └── reward_fn.py          (llm_single only)

Usage
──────────────────────────────────────────────────────────────────────────────
  cd rewardforge/
  python bipedal/experiment_runner.py
"""

import csv
import json
import os
import random
import sys
import time
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

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()   # rewardforge/bipedal/
_ROOT = _HERE.parent                      # rewardforge/
sys.path.insert(0, str(_ROOT))

from bipedal.env_wrapper   import CustomBipedalWalker          # noqa: E402
from bipedal.bipedal_agent import (                            # noqa: E402
    BLEND_STEPS, SingleFnState, make_single_blend_fn, request_single_fn,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment configuration
# ═══════════════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS    = 500_000
CHECKPOINT_EVERY   = 25_000    # eval every 25k steps → 20 checkpoints per run
N_EVAL_EPISODES    = 5         # BipedalWalker episodes can be 1600 steps

SEEDS      = list(range(10))
CONDITIONS = ["llm_single", "baseline_ppo"]

# ── Folder label: change before each run ─────────────────────────────────────
RUN_LABEL  = "v1_llm_single_vs_baseline"

_EXPR_BASE = _ROOT / "runs" / "experiments" / "bipedal"

Condition = Literal["llm_single", "baseline_ppo"]

# ── Catastrophic failure threshold ────────────────────────────────────────────
# Any run whose best_reward stays below this is a catastrophic failure.
CATASTROPHIC_THRESHOLD = 0.0

# ── Literature reference ──────────────────────────────────────────────────────
LITERATURE_MEAN   = 312.0
LITERATURE_STD    =  61.0
LITERATURE_STEPS  = 5_000_000
LITERATURE_SOURCE = "SB3 Zoo PPO BipedalWalker-v3 (DLR-RM, 2021)"


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
# ═══════════════════════════════════════════════════════════════════════════════
def _evaluate(model: PPO, eval_env: CustomBipedalWalker,
              n: int = N_EVAL_EPISODES) -> tuple[float, float]:
    """Roll out n episodes; return (mean_reward, std_reward)."""
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
class BipedalCallback(BaseCallback):
    """
    Per-checkpoint evaluation + llm_single warmup blend management.
    """

    def __init__(self, train_env: CustomBipedalWalker,
                 condition: Condition, seed: int):
        super().__init__(verbose=0)
        self.train_env = train_env
        self.condition = condition
        self.seed      = seed

        self._eval_env = CustomBipedalWalker()
        self._eval_env.reward_fn      = train_env.reward_fn
        self._eval_env.reward_fn_code = train_env.reward_fn_code

        self.log_rows:    list[dict] = []
        self.best_reward  = -float("inf")
        self.failure_log: list[str]  = []
        self._single_state: SingleFnState | None = None
        self._single_code:  str | None = None

        if condition == "llm_single":
            self._init_single()

    # ── llm_single setup ──────────────────────────────────────────────────────
    def _init_single(self) -> None:
        result = request_single_fn()
        if result is None:
            print("  \u26a0\ufe0f  llm_single failed — using base reward.")
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
        print(f"  \U0001f52e  Warmup blend: alpha 0\u21921 over {BLEND_STEPS:,} steps")

    def _tick_warmup(self) -> None:
        """Update alpha every step during warmup."""
        st = self._single_state
        if st is None or not st.blending:
            return
        elapsed  = self.num_timesteps - st.blend_start_step
        st.alpha = min(1.0, elapsed / BLEND_STEPS)
        if elapsed >= BLEND_STEPS:
            st.blending = False
            st.advances = 1
            print(f"\n    \U0001f52e  Full shaping active @ step {self.num_timesteps:,}")

    # ── SB3 callback ─────────────────────────────────────────────────────────
    def _on_step(self) -> bool:
        # Warm-up tick every step (very cheap)
        if self.condition == "llm_single" and self._single_state is not None:
            self._tick_warmup()

        if self.num_timesteps % CHECKPOINT_EVERY != 0:
            return True

        mean_rew, std_rew = _evaluate(self.model, self._eval_env)
        step = self.num_timesteps

        if mean_rew > self.best_reward:
            self.best_reward = mean_rew

        alpha_str = (f"  α={self._single_state.alpha:.2f}"
                     if self._single_state else "")
        print(f"  step {step:>7,}  mean={mean_rew:>+8.1f}  std={std_rew:>6.1f}{alpha_str}")

        self.log_rows.append({
            "step": step, "mean_reward": mean_rew, "std_reward": std_rew,
        })
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Per-run save
# ═══════════════════════════════════════════════════════════════════════════════
def _save_run(run_dir: Path, cb: BipedalCallback,
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
        f.write("RewardForge BipedalWalker Experiment Summary\n")
        f.write("=" * 44 + "\n")
        f.write(f"Condition  : {condition}\n")
        f.write(f"Seed       : {seed}\n")
        f.write(f"Best reward: {cb.best_reward:.2f}\n")

    if cb._single_code:
        (run_dir / "reward_fn.py").write_text(cb._single_code, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════════
def run_single(condition: Condition, seed: int, run_dir: Path) -> dict:
    _set_seeds(seed)

    train_env = CustomBipedalWalker()
    train_env.reset(seed=seed)

    # PPO tuned for BipedalWalker continuous control
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
    )

    cb = BipedalCallback(train_env, condition, seed)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, reset_num_timesteps=True)

    final_reward = cb.log_rows[-1]["mean_reward"] if cb.log_rows else cb.best_reward
    final_std    = cb.log_rows[-1]["std_reward"]   if cb.log_rows else 0.0

    _save_run(run_dir, cb, condition, seed)
    train_env.close()
    if cb._eval_env:
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
        "RewardForge BipedalWalker-v3 Experiment Analysis",
        "=" * 56,
        f"  {len(CONDITIONS)} conditions × {len(SEEDS)} seeds = {len(results)} runs",
        f"  {TOTAL_TIMESTEPS:,} timesteps per run",
        "",
        f"Literature Reference ({LITERATURE_SOURCE})",
        f"  PPO @ {LITERATURE_STEPS:,} steps:  {LITERATURE_MEAN:.1f} ± {LITERATURE_STD:.1f}",
        f"  (Our runs use {TOTAL_TIMESTEPS:,} steps — expect lower absolute values)",
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
            f"    best_reward  : median={np.median(best):+.1f}  IQR={_iqr(best):.1f}"
            f"  range=[{min(best):+.0f}, {max(best):+.0f}]",
            f"    final_reward : median={np.median(final):+.1f}  IQR={_iqr(final):.1f}",
            f"    final_std    : median={np.median(fstd):.1f}",
            f"    catastrophic fails (best < {CATASTROPHIC_THRESHOLD:.0f}): {fails}/{len(rows)}",
        ]

    # ── Mann-Whitney ─────────────────────────────────────────────────────────
    lines += ["", sep, "Statistical Tests: Mann-Whitney U  (one-sided: llm_single > baseline)", sep]

    if "llm_single" in CONDITIONS and "baseline_ppo" in CONDITIONS:
        s_best = [r["best_reward"] for r in results if r["condition"] == "llm_single"]
        b_best = [r["best_reward"] for r in results if r["condition"] == "baseline_ppo"]
        u, p   = sp_stats.mannwhitneyu(s_best, b_best, alternative="greater")
        sig    = "\u2705 p<0.05" if p < 0.05 else "\u274c p\u22650.05"
        lines.append(f"  llm_single vs baseline_ppo  : U={u:.0f}  p={p:.4f}  {sig}")

        # two-sided for paper reporting
        _, p2 = sp_stats.mannwhitneyu(s_best, b_best, alternative="two-sided")
        lines.append(f"  (two-sided p={p2:.4f})")

        # ── Focused comparison table ─────────────────────────────────────────
        s_final = [r["final_reward"] for r in results if r["condition"] == "llm_single"]
        s_fstd  = [r["final_std"]    for r in results if r["condition"] == "llm_single"]
        s_fails = sum(r["catastrophic"] for r in results if r["condition"] == "llm_single")
        b_final = [r["final_reward"] for r in results if r["condition"] == "baseline_ppo"]
        b_fstd  = [r["final_std"]    for r in results if r["condition"] == "baseline_ppo"]
        b_fails = sum(r["catastrophic"] for r in results if r["condition"] == "baseline_ppo")
        lines += [
            "",
            "  Focused comparison: llm_single vs baseline_ppo",
            "  " + "-" * 52,
            f"  {'':26s} {'llm_single':>14}  {'baseline':>10}",
            f"  {'median best_reward':<26} {np.median(s_best):>+14.1f}  {np.median(b_best):>+10.1f}",
            f"  {'median final_reward':<26} {np.median(s_final):>+14.1f}  {np.median(b_final):>+10.1f}",
            f"  {'median final_std':<26} {np.median(s_fstd):>14.1f}  {np.median(b_fstd):>10.1f}",
            f"  {'catastrophic fails (< 0)':<26} {s_fails:>14d}  {b_fails:>10d}",
            f"  {'n seeds':<26} {len(s_best):>14d}  {len(b_best):>10d}",
        ]

    # ── Conclusion ───────────────────────────────────────────────────────────
    lines += ["", sep, "Conclusion:", sep]
    if "llm_single" in CONDITIONS and "baseline_ppo" in CONDITIONS:
        s_best = [r["best_reward"] for r in results if r["condition"] == "llm_single"]
        b_best = [r["best_reward"] for r in results if r["condition"] == "baseline_ppo"]
        _, p   = sp_stats.mannwhitneyu(s_best, b_best, alternative="greater")
        if p < 0.05:
            lines.append(
                "  \u2705 SIGNIFICANT \u2014 llm_single outperformed baseline_ppo (p < 0.05).\n"
                "     Pre-training a single LLM reward function helps on BipedalWalker-v3."
            )
        else:
            lines.append(
                "  \u274c NOT SIGNIFICANT \u2014 p \u2265 0.05.\n"
                "     Directional trend may be visible but not confirmed at n=10."
            )

    s_median = np.median([r["best_reward"] for r in results if r["condition"] == "llm_single"])
    lines += [
        "",
        f"  Gap to literature baseline ({LITERATURE_STEPS//1_000_000}M steps): ",
        f"    llm_single median = {s_median:+.1f}  vs  literature = {LITERATURE_MEAN:.1f}",
        f"    Gap = {LITERATURE_MEAN - s_median:.1f} pts "
        f"(expected given {TOTAL_TIMESTEPS//1000}k vs {LITERATURE_STEPS//1000}k steps)",
    ]

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

    # Self-documenting metadata
    with open(EXPR_ROOT / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_label":       RUN_LABEL,
            "timestamp":       timestamp,
            "environment":     "BipedalWalker-v3",
            "conditions":      CONDITIONS,
            "seeds":           SEEDS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "n_eval_episodes": N_EVAL_EPISODES,
            "blend_steps":     BLEND_STEPS,
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "model":           "llama-3.3-70b-versatile",
            "ppo_n_steps":     2048,
            "ppo_lr":          3e-4,
        }, f, indent=2)

    plan  = [(cond, seed) for cond in CONDITIONS for seed in SEEDS]
    total = len(plan)

    print("\n\u2554" + "\u2550" * 58 + "\u2557")
    print("\u2551   RewardForge \u2014 BipedalWalker-v3 Experiment Runner      \u2551")
    print("\u255a" + "\u2550" * 58 + "\u255d")
    print(f"  {len(CONDITIONS)} conditions \u00d7 {len(SEEDS)} seeds = {total} total runs")
    print(f"  {TOTAL_TIMESTEPS:,} steps per run  |  checkpoint every {CHECKPOINT_EVERY:,}")
    print(f"  Output \u2192 {EXPR_ROOT}\n")

    all_results: list[dict] = []

    for idx, (condition, seed) in enumerate(plan, start=1):
        run_dir = EXPR_ROOT / condition / f"seed_{seed}"
        print(f"\n[{idx:>2}/{total}] {condition:<16} seed_{seed}", flush=True)

        result = run_single(condition, seed, run_dir)
        all_results.append(result)

        print(
            f"        \u2713  best={result['best_reward']:>+8.1f}"
            f"  final={result['final_reward']:>+8.1f}"
            f"  std={result['final_std']:>6.1f}"
            f"  catastrophic={'YES' if result['catastrophic'] else 'no'}"
        )

    # ── results_summary.csv ──────────────────────────────────────────────────
    csv_path = EXPR_ROOT / "results_summary.csv"
    fields   = ["condition", "seed", "best_reward", "final_reward", "final_std", "catastrophic"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"\n\U0001f4ca  Results CSV  \u2192 {csv_path}")

    # ── results_analysis.txt ─────────────────────────────────────────────────
    analysis_path = EXPR_ROOT / "results_analysis.txt"
    analysis_text = generate_analysis(all_results, analysis_path)
    print(f"\U0001f4c8  Analysis     \u2192 {analysis_path}")

    print("\n" + "=" * 58)
    print(analysis_text)
    print("=" * 58)
    print("\u2705  All experiments complete.\n")


if __name__ == "__main__":
    main()
