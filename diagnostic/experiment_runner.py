"""
diagnostic/experiment_runner.py — 3-condition comparison on BipedalWalker-v3.

╔══════════════════════════════════════════════════════════════════════════════╗
║  THE CORE SCIENTIFIC QUESTION                                                ║
║  Does giving the LLM actual behavioral data (diagnostic_llm) produce        ║
║  better reward functions than calling it blindly (llm_single)?              ║
║                                                                              ║
║  If diagnostic_llm > llm_single  →  behavioral grounding matters           ║
║  If diagnostic_llm ≈ llm_single  →  the LLM is already good at guessing   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Conditions  (all 500k steps, 10 seeds)
──────────────────────────────────────────────────────────────────────────────
  baseline_ppo    PPO only — no LLM, built-in reward
  llm_single      LLM called ONCE at step 0, blind (env description only)
                  20k step warmup blend  [our best previous method]
  diagnostic_llm  PPO warm-up 50k steps → behavioral audit (20 episodes) →
                  LLM called with audit report → targeted reward fn + 20k blend
                  [novel contribution]

Output
──────────────────────────────────────────────────────────────────────────────
  runs/experiments/diagnostic/{timestamp}_{label}/
    ├── experiment_metadata.json
    ├── results_summary.csv
    ├── results_analysis.txt          (with Mann-Whitney p-values)
    └── {condition}/seed_{seed}/
        ├── training_log.csv
        ├── summary.txt
        ├── behavior_report.txt       (diagnostic_llm only — raw audit output)
        └── reward_fn.py              (llm_single / diagnostic_llm)

Usage
──────────────────────────────────────────────────────────────────────────────
  cd rewardforge/
  python diagnostic/experiment_runner.py
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

_HERE = Path(__file__).parent.resolve()   # rewardforge/diagnostic/
_ROOT = _HERE.parent                      # rewardforge/
sys.path.insert(0, str(_ROOT))

from bipedal.env_wrapper   import CustomBipedalWalker              # noqa: E402
from bipedal.bipedal_agent import request_single_fn as _blind_llm  # noqa: E402
from bipedal.bipedal_agent import (                                 # noqa: E402
    SingleFnState as BlindState, make_single_blend_fn as make_blind_blend,
    BLEND_STEPS as BLIND_BLEND_STEPS,
)
from diagnostic.behavior_audit   import run_audit                   # noqa: E402
from diagnostic.diagnostic_agent import (                           # noqa: E402
    request_diagnostic_fn, SingleFnState as DiagState,
    make_blend_fn as make_diag_blend, BLEND_STEPS as DIAG_BLEND_STEPS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS  = 500_000
CHECKPOINT_EVERY = 25_000    # 20 checkpoints per run
N_EVAL_EPISODES  = 5         # BipedalWalker episodes can be 1600 steps

AUDIT_STEP  = 50_000    # diagnostic_llm runs behavioral probe at this step
AUDIT_N_EPS = 20        # episodes to roll out for the audit

SEEDS      = list(range(10))
CONDITIONS = ["baseline_ppo", "llm_single", "diagnostic_llm"]

RUN_LABEL  = "v1_diagnostic_vs_blind_vs_baseline"
_EXPR_BASE = _ROOT / "runs" / "experiments" / "diagnostic"

Condition = Literal["baseline_ppo", "llm_single", "diagnostic_llm"]

CATASTROPHIC_THRESHOLD = 0.0

# Literature: BipedalWalker-v3 solved at +300.  SB3 Zoo PPO @ 5M steps ≈ 312.
LITERATURE_MEAN  = 312.0
LITERATURE_STEPS = 5_000_000
LITERATURE_SRC   = "SB3 Zoo PPO BipedalWalker-v3 (DLR-RM, 2021)"


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
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def _evaluate(model: PPO, eval_env: CustomBipedalWalker,
              n: int = N_EVAL_EPISODES) -> tuple[float, float]:
    rewards = []
    for _ in range(n):
        obs, _ = eval_env.reset()
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(action)
            total += r
            done = term or trunc
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Callback — all three conditions share one implementation
# ═══════════════════════════════════════════════════════════════════════════════
class DiagnosticCallback(BaseCallback):
    """
    State machine:
      baseline_ppo   → no LLM involvement at all
      llm_single     → LLM called at __init__ with blind prompt; 20k warmup
      diagnostic_llm → silent PPO until step AUDIT_STEP, then:
                        1. run_audit(model)
                        2. request_diagnostic_fn(report)
                        3. install blended reward + 20k warmup from AUDIT_STEP
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

        self.log_rows:   list[dict] = []
        self.best_reward = -float("inf")
        self.final_std   = 0.0

        # Reward-fn warmup state (used by both llm_single and diagnostic_llm)
        self._fn_state:  BlindState | DiagState | None = None
        self._fn_code:   str  | None = None
        self._behavior_report: object | None = None   # BehaviorReport

        # diagnostic_llm: triggered once at AUDIT_STEP
        self._diag_done  = False

        if condition == "llm_single":
            self._init_blind()

    # ── Condition initialisers ─────────────────────────────────────────────────
    def _init_blind(self) -> None:
        """llm_single: blind LLM call at step 0."""
        result = _blind_llm()
        if result is None:
            print("  ??  llm_single LLM failed ? using base reward.")
            return
        fn, code = result
        self._fn_code  = code
        self._fn_state = BlindState()
        blended = make_blind_blend(fn, self._fn_state)
        self._install(blended, "llm_single:warmup")
        print(f"  Warmup blend: ? 0?1 over {BLIND_BLEND_STEPS:,} steps")

    def _init_diagnostic(self) -> None:
        """diagnostic_llm: called at AUDIT_STEP after warm-up training."""
        print(f"\n  ?  Running behavioral audit ({AUDIT_N_EPS} episodes) ?",
              flush=True)
        report = run_audit(self.model, n_episodes=AUDIT_N_EPS)
        self._behavior_report = report
        print(report.diagnosis_str())

        result = request_diagnostic_fn(report, train_steps=self.num_timesteps)
        self._diag_done = True
        if result is None:
            print("  ??  Diagnostic LLM failed ? continuing with base reward.")
            return
        fn, code = result
        self._fn_code  = code
        self._fn_state = DiagState(start_step=self.num_timesteps)
        blended = make_diag_blend(fn, self._fn_state)
        self._install(blended, "diagnostic_llm:active")
        print(f"  ?  Installed @ step {self.num_timesteps:,}",
              f"  Warmup blend 20k steps")

    def _install(self, fn, label: str) -> None:
        """Apply blended fn to both train and eval envs."""
        self.train_env.reward_fn      = fn
        self.train_env.reward_fn_code = label
        self._eval_env.reward_fn      = fn
        self._eval_env.reward_fn_code = label

    # ── Per-step logic ─────────────────────────────────────────────────────────
    def _tick_warmup(self) -> None:
        st = self._fn_state
        if st is None or not st.blending:
            return
        blend_steps = (BLIND_BLEND_STEPS if self.condition == "llm_single"
                       else DIAG_BLEND_STEPS)
        elapsed  = self.num_timesteps - st.blend_start_step
        st.alpha = min(1.0, elapsed / blend_steps)
        if elapsed >= blend_steps:
            st.blending = False
            st.advances = 1
            print(f"\n    Full shaping active @ step {self.num_timesteps:,}")

    def _on_step(self) -> bool:
        # Warmup tick (every step)
        if self._fn_state is not None:
            self._tick_warmup()

        # Diagnostic trigger (diagnostic_llm only, fires once)
        if (self.condition == "diagnostic_llm"
                and not self._diag_done
                and self.num_timesteps >= AUDIT_STEP):
            self._init_diagnostic()

        if self.num_timesteps % CHECKPOINT_EVERY != 0:
            return True

        mean_rew, std_rew = _evaluate(self.model, self._eval_env)
        step = self.num_timesteps

        if mean_rew > self.best_reward:
            self.best_reward = mean_rew
        self.final_std = std_rew

        alpha_str = (f"  α={self._fn_state.alpha:.2f}"
                     if self._fn_state and self._fn_state.blending else "")
        diag_tag  = "  [DIAG active]" \
                    if self.condition == "diagnostic_llm" and self._diag_done else ""
        print(f"  step {step:>7,}  mean={mean_rew:>+8.1f}  "
              f"std={std_rew:>6.1f}{alpha_str}{diag_tag}")

        self.log_rows.append({
            "step": step, "mean_reward": mean_rew, "std_reward": std_rew,
        })
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Per-run save
# ═══════════════════════════════════════════════════════════════════════════════
def _save_run(run_dir: Path, cb: DiagnosticCallback,
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
        f.write(f"Condition : {condition}\nSeed      : {seed}\n"
                f"Best      : {cb.best_reward:.4f}\n")
        if cb._fn_code:
            f.write(f"\nReward fn installed at step "
                    f"{'0' if condition=='llm_single' else str(AUDIT_STEP)}\n")

    if cb._fn_code:
        (run_dir / "reward_fn.py").write_text(cb._fn_code, encoding="utf-8")

    if cb._behavior_report is not None:
        (run_dir / "behavior_report.txt").write_text(
            cb._behavior_report.diagnosis_str(), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Single run
# ═══════════════════════════════════════════════════════════════════════════════
def run_single(condition: Condition, seed: int, run_dir: Path) -> dict:
    _set_seeds(seed)
    train_env = CustomBipedalWalker()
    train_env.reset(seed=seed)

    model = PPO(
        "MlpPolicy", train_env,
        n_steps=2048, batch_size=64, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        seed=seed, verbose=0,
    )

    cb = DiagnosticCallback(train_env, condition, seed)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb,
                reset_num_timesteps=True)

    final_reward = cb.log_rows[-1]["mean_reward"] if cb.log_rows else cb.best_reward
    _save_run(run_dir, cb, condition, seed)
    train_env.close()
    cb._eval_env.close()

    return {
        "condition":    condition,
        "seed":         seed,
        "best_reward":  cb.best_reward,
        "final_reward": final_reward,
        "final_std":    cb.final_std,
        "catastrophic": int(cb.best_reward < CATASTROPHIC_THRESHOLD),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def _iqr(arr) -> float:
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def generate_analysis(results: list[dict], out_path: Path) -> str:
    from scipy import stats as sp_stats

    sep = "─" * 60
    lines = [
        "Diagnostic RewardForge — BipedalWalker-v3 Analysis",
        "=" * 60,
        f"  {len(CONDITIONS)} conditions × {len(SEEDS)} seeds = {len(results)} runs",
        f"  {TOTAL_TIMESTEPS:,} timesteps per run",
        f"  diagnostic_llm: base PPO {AUDIT_STEP:,} steps → audit → targeted reward",
        "",
        f"Literature: {LITERATURE_SRC}",
        f"  PPO @ {LITERATURE_STEPS:,} steps ≈ {LITERATURE_MEAN:.0f} reward",
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
            f"    final_reward : median={np.median(final):+.1f}",
            f"    final_std    : median={np.median(fstd):.1f}",
            f"    catast. fails: {fails}/{len(rows)} (best < {CATASTROPHIC_THRESHOLD:.0f})",
        ]

    # ── Statistical tests ────────────────────────────────────────────────────
    diag_best    = [r["best_reward"] for r in results if r["condition"] == "diagnostic_llm"]
    blind_best   = [r["best_reward"] for r in results if r["condition"] == "llm_single"]
    base_best    = [r["best_reward"] for r in results if r["condition"] == "baseline_ppo"]

    lines += ["", sep, "Mann-Whitney U  (one-sided  >  alternative):", sep]

    def _mw(a, b, label_a, label_b):
        if not a or not b:
            return
        u, p = sp_stats.mannwhitneyu(a, b, alternative="greater")
        sig  = "✅ p<0.05" if p < 0.05 else "❌ p≥0.05"
        lines.append(f"  {label_a:<20} vs {label_b:<20}: U={u:.0f}  p={p:.4f}  {sig}")

    _mw(diag_best,  base_best,  "diagnostic_llm", "baseline_ppo")
    _mw(blind_best, base_best,  "llm_single",     "baseline_ppo")
    _mw(diag_best,  blind_best, "diagnostic_llm", "llm_single")

    # ── Key comparison table ─────────────────────────────────────────────────
    if diag_best and blind_best:
        _, p2 = sp_stats.mannwhitneyu(diag_best, blind_best, alternative="two-sided")
        d_fails = sum(r["catastrophic"] for r in results if r["condition"] == "diagnostic_llm")
        b_fails = sum(r["catastrophic"] for r in results if r["condition"] == "llm_single")

        lines += [
            "",
            "  ┌─── Core comparison: does behavioral grounding help? ──────────┐",
            f"  │  {'':26s} {'diagnostic_llm':>16}  {'llm_single':>12}  │",
            f"  │  {'median best_reward':<26} {np.median(diag_best):>+16.1f}"
            f"  {np.median(blind_best):>+12.1f}  │",
            f"  │  {'median final_std':<26} {np.median([r['final_std'] for r in results if r['condition']=='diagnostic_llm']):>16.1f}"
            f"  {np.median([r['final_std'] for r in results if r['condition']=='llm_single']):>12.1f}  │",
            f"  │  {'catastrophic fails':<26} {d_fails:>16d}  {b_fails:>12d}  │",
            f"  │  {'two-sided p-value':<26} {p2:>16.4f}  {'—':>12s}  │",
            "  └───────────────────────────────────────────────────────────────┘",
        ]

        if p2 < 0.05:
            if np.median(diag_best) > np.median(blind_best):
                verdict = "✅ BEHAVIORAL GROUNDING HELPS — diagnostic_llm > llm_single (p<0.05)"
            else:
                verdict = "⚠️  BLIND IS BETTER — llm_single > diagnostic_llm (p<0.05)"
        else:
            verdict = "❌ NO SIGNIFICANT DIFFERENCE (p≥0.05) — grounding not decisive at n=10"
        lines += ["", f"  {verdict}"]

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder    = f"{timestamp}_{RUN_LABEL}"
    EXPR_ROOT = _EXPR_BASE / folder
    EXPR_ROOT.mkdir(parents=True, exist_ok=True)

    with open(EXPR_ROOT / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_label":        RUN_LABEL,
            "timestamp":        timestamp,
            "environment":      "BipedalWalker-v3",
            "conditions":       CONDITIONS,
            "seeds":            SEEDS,
            "total_timesteps":  TOTAL_TIMESTEPS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "audit_step":       AUDIT_STEP,
            "audit_n_episodes": AUDIT_N_EPS,
            "blend_steps":      DIAG_BLEND_STEPS,
            "llm_model":        "llama-3.3-70b-versatile",
            "hypothesis":       "behavioral-grounded LLM reward > blind LLM reward",
        }, f, indent=2)

    plan  = [(c, s) for c in CONDITIONS for s in SEEDS]
    total = len(plan)

    print("\n?" + "?" * 62 + "?")
    print("?   Diagnostic RewardForge ? BipedalWalker-v3                   ?")
    print("?   Hypothesis: behavioral-grounded reward > blind LLM reward   ?")
    print("?" + "?" * 62 + "?")
    print(f"  {len(CONDITIONS)} conditions ? {len(SEEDS)} seeds = {total} runs")
    print(f"  {TOTAL_TIMESTEPS:,} steps/run  |  audit @ step {AUDIT_STEP:,}")
    print(f"  Output ? {EXPR_ROOT}\n")

    all_results: list[dict] = []

    for idx, (condition, seed) in enumerate(plan, start=1):
        run_dir = EXPR_ROOT / condition / f"seed_{seed}"
        print(f"\n[{idx:>2}/{total}] {condition:<20} seed_{seed}", flush=True)

        result = run_single(condition, seed, run_dir)
        all_results.append(result)

        print(
            f"        ✓  best={result['best_reward']:>+8.1f}"
            f"  final={result['final_reward']:>+8.1f}"
            f"  std={result['final_std']:>6.1f}"
            f"  {'💀 CATASTROPHIC' if result['catastrophic'] else 'ok'}"
        )

    csv_path = EXPR_ROOT / "results_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\n?  Results  ? {csv_path}")

    analysis_path = EXPR_ROOT / "results_analysis.txt"
    text = generate_analysis(all_results, analysis_path)
    print(f"?  Analysis ? {analysis_path}")

    print("\n" + "=" * 62)
    print(text)
    print("=" * 62)
    print("?  Done.\n")


if __name__ == "__main__":
    main()
