"""
experiment_runner.py — Comparative experiments for RewardForge.

Runs 20 total experiments:
    rewardforge     — PPO + Gemini with full reward history
    baseline_ppo    — PPO only, no shaping, no Gemini
    ablation_blind  — Gemini triggered at same steps but NO history sent
    ablation_random — reward function replaced with randomly perturbed v0

Usage:
    python experiment_runner.py

Output:
    runs/experiments/{condition}/seed_{seed}/  ← per-run artifacts (same schema as main.py)
    runs/experiments/results_summary.csv       ← all 20 runs combined
    runs/experiments/results_analysis.txt      ← statistical analysis + conclusion
"""

import csv
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Literal

# ── Silence all deprecation/lifecycle noise before any heavy imports ──────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ── Make sure local modules are importable when run from any CWD ──────────────
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from env_wrapper import CustomCartPole          # noqa: E402
from rewardforge_agent import request_new_reward_fn  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration  (must stay aligned with main.py)
# ═══════════════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS   = 20_000
CHECKPOINT_EVERY  = 2_000
IMPROVEMENT_PCT   = 10.0
MAX_REWRITES      = 3
N_EVAL_EPISODES   = 10
GRACE_CHECKPOINTS = 1

SEEDS      = [0, 1, 2, 3, 4]
CONDITIONS = ["rewardforge", "baseline_ppo", "ablation_blind", "ablation_random"]
EXPR_ROOT  = _HERE / "runs" / "experiments"

Condition = Literal["rewardforge", "baseline_ppo", "ablation_blind", "ablation_random"]


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# Value head reset  (reproduced from main.py — not imported to avoid side-effects)
# ═══════════════════════════════════════════════════════════════════════════════
def _reset_value_head(model: PPO) -> None:
    """Re-initialize only the critic head so the actor keeps what it learned."""
    policy = model.policy
    for module in policy.mlp_extractor.value_net.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    if isinstance(policy.value_net, torch.nn.Linear):
        torch.nn.init.orthogonal_(policy.value_net.weight, gain=1.0)
        if policy.value_net.bias is not None:
            torch.nn.init.zeros_(policy.value_net.bias)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helper  (uses isolated eval env — never touches the training env)
# ═══════════════════════════════════════════════════════════════════════════════
def _evaluate(model: PPO, eval_env: CustomCartPole, n: int = N_EVAL_EPISODES) -> float:
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
    return float(np.mean(rewards))


# ═══════════════════════════════════════════════════════════════════════════════
# ablation_random: randomly perturbed shaping coefficients
# ═══════════════════════════════════════════════════════════════════════════════
def _make_random_reward_fn(rng: np.random.RandomState):
    """
    Build a randomly perturbed reward function that:
      - Uses the original +1/step `reward` as base
      - Adds shaping bonuses for pole being near vertical and cart near center
      - Keeps total reward in ~[0.7, 1.3] — safe for the PPO value network
    """
    pole_w = float(np.clip(rng.normal(0.30, 0.08), 0.05, 0.50))
    cart_w = float(np.clip(rng.normal(0.10, 0.04), 0.01, 0.20))
    # Centering offset: subtract half the max shaping bonus so the mean
    # returned reward stays near +1.0/step
    center = (pole_w + cart_w) / 2.0

    def _fn(obs, action, reward, terminated, info):
        if terminated:
            return 0.0
        theta_norm = abs(obs[2]) / 0.209   # 0 = vertical, 1 = at limit
        x_norm     = abs(obs[0]) / 2.4    # 0 = center,   1 = at limit
        shaping = (
            pole_w * (1.0 - theta_norm)
            + cart_w * (1.0 - x_norm)
            - center
        )
        return float(reward + shaping)

    # Human-readable source so it can be saved to reward_functions/v*.py
    code = (
        "def custom_reward(obs, action, reward, terminated, info):\n"
        "    if terminated:\n"
        "        return 0.0\n"
        f"    # ablation_random shaping: pole_w={pole_w:.4f}, cart_w={cart_w:.4f}, center={center:.4f}\n"
        "    theta_norm = abs(obs[2]) / 0.209\n"
        "    x_norm     = abs(obs[0]) / 2.4\n"
        f"    shaping = {pole_w:.4f} * (1.0 - theta_norm) + {cart_w:.4f} * (1.0 - x_norm) - {center:.4f}\n"
        "    return float(reward + shaping)"
    )
    return _fn, code


# ═══════════════════════════════════════════════════════════════════════════════
# Unified SB3 callback  —  all 4 condition modes share one loop
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentCallback(BaseCallback):
    """
    Single callback that drives all 4 experimental conditions.

    The only thing that varies by condition is what happens at trigger time:
        rewardforge     → call Gemini with last-3 reward history
        baseline_ppo    → never trigger
        ablation_blind  → call Gemini with EMPTY history ([]); function only
        ablation_random → replace with randomly perturbed function; no Gemini
    """

    def __init__(self, train_env: CustomCartPole, condition: Condition, seed: int):
        super().__init__(verbose=0)
        self.train_env  = train_env
        self.condition  = condition
        self.seed       = seed

        # Isolated eval env — same reward fn as train env, but owned by us
        self._eval_env = CustomCartPole()
        self._sync_eval_env(train_env.reward_fn, train_env.reward_fn_code)

        # Seeded RNG for ablation_random (independent of global seed so
        # different seeds give different perturbations)
        self._rng = np.random.RandomState(seed * 137 + 31)

        # Tracking
        self.reward_history:   list[tuple[int, float]] = []
        self.reward_fn_history: list[str]  = [train_env.reward_fn_code]
        self.log_rows:          list[dict] = []
        self.failure_log:       list[str]  = []   # Gemini failures, kept for analysis
        self.rewrite_count     = 0
        self.best_reward       = -float("inf")
        self.best_version      = 0
        self._grace_remaining  = 0

    # ── SB3 callback hook ────────────────────────────────────────────────────
    def _on_step(self) -> bool:
        if self.num_timesteps % CHECKPOINT_EVERY != 0:
            return True

        mean_rew = _evaluate(self.model, self._eval_env)
        step     = self.num_timesteps
        self.reward_history.append((step, mean_rew))

        if mean_rew > self.best_reward:
            self.best_reward  = mean_rew
            self.best_version = self.train_env.reward_fn_version

        triggered = self._maybe_trigger(mean_rew)

        self.log_rows.append({
            "step":        step,
            "mean_reward": mean_rew,
            "version":     self.train_env.reward_fn_version,
            "triggered":   "YES" if triggered else "",
        })
        return True

    # ── Trigger logic ────────────────────────────────────────────────────────
    def _maybe_trigger(self, mean_rew: float) -> bool:
        # baseline_ppo: never reshape
        if self.condition == "baseline_ppo":
            return False

        # Grace period after a recent rewrite — let the agent adapt
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            return False

        # Need at least 2 checkpoints, and rewrites budget remaining
        if len(self.reward_history) < 2 or self.rewrite_count >= MAX_REWRITES:
            return False

        # Stagnation check
        prev = self.reward_history[-2][1]
        if prev > 0 and (mean_rew - prev) / abs(prev) * 100 >= IMPROVEMENT_PCT:
            return False  # improving fast enough — no trigger

        # Condition-specific rewrite strategy
        if self.condition == "rewardforge":
            return self._trigger_gemini(include_history=True)
        elif self.condition == "ablation_blind":
            return self._trigger_gemini(include_history=False)
        elif self.condition == "ablation_random":
            return self._trigger_random()
        return False

    def _trigger_gemini(self, include_history: bool) -> bool:
        """Call Gemini.  include_history=False → ablation_blind condition."""
        history = (
            self.reward_history[-3:] if include_history else []
        )
        try:
            result = request_new_reward_fn(
                current_reward_fn_code=self.train_env.reward_fn_code,
                reward_history=history,
            )
        except Exception as exc:
            note = f"[step {self.reward_history[-1][0]}] Gemini API failure: {exc}"
            self.failure_log.append(note)
            print(f"    ⚠️  {note}  → keeping current fn")
            return False

        if result is None:
            note = f"[step {self.reward_history[-1][0]}] Gemini returned unusable code"
            self.failure_log.append(note)
            return False

        fn, code = result
        self._apply_rewrite(fn, code)
        return True

    def _trigger_random(self) -> bool:
        fn, code = _make_random_reward_fn(self._rng)
        self._apply_rewrite(fn, code)
        return True

    def _apply_rewrite(self, fn, code: str) -> None:
        self.train_env.set_reward_fn(fn, code)
        self._sync_eval_env(fn, code)
        self.rewrite_count += 1
        self.reward_fn_history.append(code)
        _reset_value_head(self.model)
        self._grace_remaining = GRACE_CHECKPOINTS
        print(
            f"    🔄 v{self.train_env.reward_fn_version} applied"
            f"  ⏳ grace={GRACE_CHECKPOINTS}ckpt"
        )

    def _sync_eval_env(self, fn, code: str) -> None:
        """Keep eval env's reward fn in lockstep with training env."""
        self._eval_env.reward_fn      = fn
        self._eval_env.reward_fn_code = code


# ═══════════════════════════════════════════════════════════════════════════════
# Artifact saving  (identical schema to main.py)
# ═══════════════════════════════════════════════════════════════════════════════
def _save_run(
    run_dir:           Path,
    cb:                ExperimentCallback,
    condition:         str,
    seed:              int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # training_log.csv
    with open(run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward", "reward_fn_version", "triggered"])
        for row in cb.log_rows:
            w.writerow([row["step"], f"{row['mean_reward']:.4f}",
                        f"v{row['version']}", row["triggered"]])

    # summary.txt
    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("RewardForge Experiment Summary\n")
        f.write("=" * 42 + "\n")
        f.write(f"Condition          : {condition}\n")
        f.write(f"Seed               : {seed}\n")
        f.write(f"Total timesteps    : {TOTAL_TIMESTEPS}\n")
        f.write(f"Checkpoint interval: {CHECKPOINT_EVERY}\n")
        f.write(f"Improvement thresh : {IMPROVEMENT_PCT}%\n")
        f.write(f"Max rewrites       : {MAX_REWRITES}\n")
        f.write(f"Rewrites used      : {cb.rewrite_count}\n")
        f.write(f"Best reward        : {cb.best_reward:.2f}\n")
        f.write(f"Best reward fn ver : v{cb.best_version}\n")
        if cb.failure_log:
            f.write("\nGemini failures:\n")
            for note in cb.failure_log:
                f.write(f"  {note}\n")

    # reward_functions/v*.py
    rf_dir = run_dir / "reward_functions"
    rf_dir.mkdir(exist_ok=True)
    for i, code in enumerate(cb.reward_fn_history):
        (rf_dir / f"v{i}.py").write_text(code, encoding="utf-8")

    # config.json
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "condition":          condition,
            "seed":               seed,
            "total_timesteps":    TOTAL_TIMESTEPS,
            "checkpoint_every":   CHECKPOINT_EVERY,
            "improvement_pct":    IMPROVEMENT_PCT,
            "max_rewrites":       MAX_REWRITES,
            "n_eval_episodes":    N_EVAL_EPISODES,
            "grace_checkpoints":  GRACE_CHECKPOINTS,
        }, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-run orchestrator
# ═══════════════════════════════════════════════════════════════════════════════
def run_single(condition: Condition, seed: int, run_dir: Path) -> dict:
    """Train one (condition, seed) pair and return a metrics dict."""
    set_global_seeds(seed)

    env   = CustomCartPole()
    env.reset(seed=seed)            # seed the gymnasium env
    model = PPO("MlpPolicy", env, seed=seed, verbose=0)
    cb    = ExperimentCallback(env, condition, seed)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)

    # Derived metrics from the log
    rows = cb.log_rows
    steps_to_450 = next((r["step"] for r in rows if r["mean_reward"] >= 450), -1)
    steps_to_500 = next((r["step"] for r in rows if r["mean_reward"] >= 500), -1)
    final_reward = rows[-1]["mean_reward"] if rows else 0.0

    _save_run(run_dir, cb, condition, seed)

    return {
        "condition":      condition,
        "seed":           seed,
        "best_reward":    cb.best_reward,
        "steps_to_450":   steps_to_450,
        "steps_to_500":   steps_to_500,
        "total_triggers": cb.rewrite_count,
        "final_reward":   final_reward,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical analysis
# ═══════════════════════════════════════════════════════════════════════════════
def _iqr(arr) -> float:
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _median_str(vals, missing_sentinel=-1) -> str:
    """Median, ignoring -1 (never-reached) entries."""
    valid = [v for v in vals if v != missing_sentinel]
    if not valid:
        return "never reached"
    return f"{np.median(valid):.0f}  (IQR {_iqr(valid):.0f})"


def generate_analysis(results: list[dict], out_path: Path) -> str:
    from scipy import stats as sp_stats

    sep  = "─" * 52
    lines = [
        "RewardForge Experiment Analysis",
        "=" * 52,
        f"  {len(CONDITIONS)} conditions × {len(SEEDS)} seeds = {len(results)} runs",
        "",
    ]

    # ── Per-condition stats ──────────────────────────────────────────────────
    lines.append("Per-condition statistics:")
    lines.append(sep)
    for cond in CONDITIONS:
        rows  = [r for r in results if r["condition"] == cond]
        best  = [r["best_reward"]  for r in rows]
        final = [r["final_reward"] for r in rows]
        s450  = [r["steps_to_450"] for r in rows]
        s500  = [r["steps_to_500"] for r in rows]
        trigs = [r["total_triggers"] for r in rows]

        lines += [
            f"\n  [{cond}]",
            f"    best_reward  : median={np.median(best):.2f}  IQR={_iqr(best):.2f}"
            f"  range=[{min(best):.1f}, {max(best):.1f}]",
            f"    final_reward : median={np.median(final):.2f}  IQR={_iqr(final):.2f}",
            f"    steps_to_450 : {_median_str(s450)}",
            f"    steps_to_500 : {_median_str(s500)}",
            f"    triggers used: {np.median(trigs):.1f} median  (max={MAX_REWRITES})",
        ]

    # ── Mann-Whitney U: rewardforge vs baseline_ppo ──────────────────────────
    rf_best   = [r["best_reward"] for r in results if r["condition"] == "rewardforge"]
    base_best = [r["best_reward"] for r in results if r["condition"] == "baseline_ppo"]

    u_stat, p_val = sp_stats.mannwhitneyu(rf_best, base_best, alternative="greater")

    lines += [
        "",
        sep,
        "Statistical Test: Mann-Whitney U  (one-sided: rewardforge > baseline_ppo)",
        sep,
        f"  rewardforge  best_rewards : {[round(v,1) for v in rf_best]}",
        f"  baseline_ppo best_rewards : {[round(v,1) for v in base_best]}",
        f"  U = {u_stat:.1f},  p = {p_val:.4f}",
        "",
    ]

    # ── ablation comparisons ─────────────────────────────────────────────────
    blind_best  = [r["best_reward"] for r in results if r["condition"] == "ablation_blind"]
    random_best = [r["best_reward"] for r in results if r["condition"] == "ablation_random"]

    _, p_blind  = sp_stats.mannwhitneyu(rf_best, blind_best,  alternative="two-sided")
    _, p_random = sp_stats.mannwhitneyu(rf_best, random_best, alternative="two-sided")

    lines += [
        "Ablation comparisons (two-sided, rewardforge vs ablation):",
        f"  vs ablation_blind  : p = {p_blind:.4f}"
        + ("  ← history context matters" if p_blind < 0.05 else "  ← history not decisive"),
        f"  vs ablation_random : p = {p_random:.4f}"
        + ("  ← Gemini beats random"     if p_random < 0.05 else "  ← Gemini not sig. better than random"),
        "",
    ]

    # ── Plain-English conclusion ──────────────────────────────────────────────
    lines.append("Conclusion:")
    lines.append(sep)
    if p_val < 0.05:
        verdict = (
            f"✅  SIGNIFICANT — RewardForge outperformed baseline PPO "
            f"(p = {p_val:.4f} < 0.05). Gemini-driven reward shaping produced "
            f"a statistically significant gain in best reward over 5 seeds."
        )
    else:
        verdict = (
            f"❌  NOT SIGNIFICANT — RewardForge did not significantly outperform "
            f"baseline PPO (p = {p_val:.4f} ≥ 0.05). The effect may require more "
            f"seeds, longer training, or a harder environment to detect."
        )

    lines.append(f"  {verdict}")

    if p_blind >= 0.05:
        lines.append(
            "  💡 History context (rewardforge vs ablation_blind) is not decisive — "
            "Gemini seems to design reasonable shapes even without the reward curve."
        )
    else:
        lines.append(
            "  💡 History context matters: rewardforge significantly outperformed "
            "ablation_blind, confirming the reward curve is useful signal for Gemini."
        )

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    EXPR_ROOT.mkdir(parents=True, exist_ok=True)

    # Build ordered run plan: conditions × seeds
    plan: list[tuple[Condition, int]] = [
        (cond, seed) for cond in CONDITIONS for seed in SEEDS
    ]
    total = len(plan)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║        RewardForge — Experiment Runner               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  {len(CONDITIONS)} conditions × {len(SEEDS)} seeds = {total} total runs")
    print(f"  Output root → {EXPR_ROOT}\n")

    all_results: list[dict] = []

    for idx, (condition, seed) in enumerate(plan, start=1):
        run_dir = EXPR_ROOT / condition / f"seed_{seed}"
        print(f"[{idx:>2}/{total}] Running {condition:<20} seed_{seed} …", flush=True)

        result = run_single(condition, seed, run_dir)
        all_results.append(result)

        # Inline summary after each run
        print(
            f"        ✓  best={result['best_reward']:>7.1f}"
            f"  final={result['final_reward']:>7.1f}"
            f"  triggers={result['total_triggers']}"
            f"  →450@{result['steps_to_450']}"
            f"  →500@{result['steps_to_500']}"
        )

    # ── Write results_summary.csv ─────────────────────────────────────────────
    csv_path = EXPR_ROOT / "results_summary.csv"
    fieldnames = ["condition", "seed", "best_reward",
                  "steps_to_450", "steps_to_500", "total_triggers", "final_reward"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)
    print(f"\n📊  Results CSV     → {csv_path}")

    # ── Write results_analysis.txt ────────────────────────────────────────────
    analysis_path = EXPR_ROOT / "results_analysis.txt"
    analysis_text = generate_analysis(all_results, analysis_path)
    print(f"📈  Analysis        → {analysis_path}")

    print("\n" + "═" * 54)
    print(analysis_text)
    print("═" * 54)
    print("✅  All experiments complete.\n")


if __name__ == "__main__":
    main()
