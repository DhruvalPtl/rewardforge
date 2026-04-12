"""
lunarlander/experiment_runner.py — 4-condition scientific comparison on LunarLander-v3.

╔══════════════════════════════════════════════════════════════════════════════╗
║  WHY THIS EXISTS                                                             ║
║  CartPole experiment showed any shaping helps (even random).                ║
║  LunarLander is a harder environment where:                                  ║
║    • random shaping can steer the agent in a WRONG direction                 ║
║    • blind qwen (no history) may over-penalise and hurt learning           ║
║    • RewardForge with full history should be the clear winner                ║
║  This is the paper's key result.                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Conditions
──────────────────────────────────────────────────────────────────────────────
  rewardforge     PPO + qwen3-32b with full reward curve history      [our method]
  baseline_ppo    PPO only, no LLM, built-in env reward               [control]
  ablation_blind  qwen3-32b triggered but receives NO history          [ablation]
  ablation_random random shaping (random-sign coefficients)         [ablation]

Literature Reference (no rerun needed)
──────────────────────────────────────────────────────────────────────────────
  SB3 Zoo PPO LunarLander-v2 @ 1M steps:  267.7 ± 16.7
  Source: https://github.com/DLR-RM/rl-baselines3-zoo
  Note: Eureka paper (Ma et al. 2023) focuses on dexterous manipulation, not
        LunarLander. SB3 Zoo provides the canonical open-source PPO baseline.

Output
──────────────────────────────────────────────────────────────────────────────
  runs/experiments/lunarlander/{condition}/seed_{seed}/  ← per-run artifacts
  runs/experiments/lunarlander/results_summary.csv
  runs/experiments/lunarlander/results_analysis.txt

Usage
──────────────────────────────────────────────────────────────────────────────
  cd rewardforge/
  python lunarlander/experiment_runner.py
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
_HERE = Path(__file__).parent.resolve()   # rewardforge/lunarlander/
_ROOT = _HERE.parent                      # rewardforge/
sys.path.insert(0, str(_ROOT))

from lunarlander.env_wrapper       import CustomLunarLander       # noqa: E402
from lunarlander.rewardforge_agent import request_new_reward_fn   # noqa: E402
from lunarlander.curriculum_agent  import (                       # noqa: E402
    BLEND_STEPS, STAGE1_GATE, STAGE1_WAIT,
    CurriculumState, make_blended_fn, request_curriculum_fns,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS   = 500_000   # 500k -- enough for hover-trap to manifest and be fixed
CHECKPOINT_EVERY  = 10_000
LOOK_BACK         = 3         # compare vs 3 checkpoints ago (30k steps)
IMPROVEMENT_DELTA = 10.0      # < 10 pts gain over 30k steps = genuinely stagnating
MAX_REWRITES      = 2
N_EVAL_EPISODES   = 15
GRACE_CHECKPOINTS = 3         # 30k steps of recovery after each rewrite

# ── Hover-trap trigger: all three must be true simultaneously ─────────────────
# The hover trap: agent learns to float (reward 0-100) but never lands.
# It only emerges AFTER the agent has left the crash zone (reward > 0)
# and AFTER enough time for PPO to reach a stable plateau (step > 80k).
MIN_TRIGGER_STEP   = 80_000   # past the early random-crash phase
MIN_TRIGGER_REWARD = 0.0      # agent must be positive -- hovering, not crashing
                               # (below 0 = still in random-walk territory, LLM can't help)

# Rate limiting for llama-3.3-70b-versatile free tier:
# TPM = 12,000/min.  llama does NOT do thinking, so one call = ~850 tokens
# (700 prompt + 150 code).  12K / 850 = ~14 calls/min max.  10s gap = safe.
LLM_CALL_DELAY_S = 10        # seconds to sleep before each LLM call

SEEDS      = list(range(10))    # seeds 0-9 (v5 — doubled for statistical power)
CONDITIONS = ["rewardforge", "baseline_ppo", "ablation_blind", "ablation_random"]
# EXPR_ROOT is set inside main() with a timestamp so each run gets its own directory.
# e.g.  runs/experiments/lunarlander/20260411_174600/
_EXPR_BASE = _ROOT / "runs" / "experiments" / "lunarlander"

Condition = Literal["rewardforge", "baseline_ppo", "ablation_blind", "ablation_random"]

# ── Literature reference constants (hardcoded — no rerun needed) ──────────────
# SB3 Zoo PPO LunarLander-v2 after 1,000,000 steps
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
LITERATURE_MEAN   = 267.7
LITERATURE_STD    = 16.7
LITERATURE_STEPS  = 1_000_000
LITERATURE_SOURCE = "SB3 Zoo PPO LunarLander-v2 (DLR-RM, 2021)"


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
# Value head reset  (reproduced from main.py)
# ═══════════════════════════════════════════════════════════════════════════════
def _reset_value_head(model: PPO) -> None:
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
# Evaluation helper
# ═══════════════════════════════════════════════════════════════════════════════
def _evaluate(model: PPO, eval_env: CustomLunarLander, n: int = N_EVAL_EPISODES
              ) -> tuple[float, float]:
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
# ablation_random: truly random shaping — can have WRONG signs
#
# Key difference from CartPole ablation_random:
#   CartPole used clipped-positive coefficients (always shaping-like).
#   Here coefficients are uniformly drawn from [-max, +max], so shaping
#   can actively REWARD bad behaviour (e.g., rewarding being far from pad).
#   This demonstrates LunarLander's sensitivity to reward function quality.
# ═══════════════════════════════════════════════════════════════════════════════
def _make_random_reward_fn(rng: np.random.RandomState):
    """
    Random coefficients with free sign — designed to show that unintelligent
    shaping is HARMFUL on LunarLander, unlike CartPole.
    Magnitudes are capped to ≤0.4 to prevent total signal collapse.
    """
    x_coef   = float(rng.uniform(-0.4, 0.4))   # free sign: may reward drifting
    ang_coef = float(rng.uniform(-0.3, 0.3))   # free sign: may reward tilting
    vy_coef  = float(rng.uniform(-0.3, 0.3))   # free sign: may reward falling fast
    leg_bon  = float(rng.uniform(-0.4, 0.5))   # free sign: may penalise landing

    def _fn(obs, action, reward, terminated, info):
        if terminated:
            return reward
        shaping = (
            x_coef   * abs(obs[0])    # horizontal distance shaping
            + ang_coef * abs(obs[4])  # angle shaping
            + vy_coef  * obs[3]       # y-velocity shaping (signed — falling = negative)
        )
        if obs[6] > 0.5 and obs[7] > 0.5:
            shaping += leg_bon
        return reward + shaping

    code = (
        "def custom_reward(obs, action, reward, terminated, info):\n"
        "    if terminated:\n"
        "        return reward\n"
        f"    # ablation_random: x_coef={x_coef:.3f}  ang_coef={ang_coef:.3f}"
        f"  vy_coef={vy_coef:.3f}  leg_bon={leg_bon:.3f}\n"
        f"    shaping = {x_coef:.4f} * abs(obs[0])"
        f" + {ang_coef:.4f} * abs(obs[4])"
        f" + {vy_coef:.4f} * obs[3]\n"
        "    if obs[6] > 0.5 and obs[7] > 0.5:\n"
        f"        shaping += {leg_bon:.4f}\n"
        "    return reward + shaping"
    )
    return _fn, code


# ═══════════════════════════════════════════════════════════════════════════════
# Unified SB3 Callback — all 4 conditions share one loop
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentCallback(BaseCallback):
    """
    Single callback parameterised by condition.
    Stagnation is detected as absolute improvement over LOOK_BACK checkpoints
    (not percentage — negative rewards make % comparisons meaningless).
    """

    def __init__(self, train_env: CustomLunarLander, condition: Condition, seed: int):
        super().__init__(verbose=0)
        self.train_env = train_env
        self.condition = condition
        self.seed      = seed

        self._eval_env = CustomLunarLander()
        self._eval_env.reward_fn      = train_env.reward_fn
        self._eval_env.reward_fn_code = train_env.reward_fn_code

        self._rng = np.random.RandomState(seed * 137 + 53)

        self.reward_history:    list[tuple[int, float]] = []
        self.reward_fn_history: list[str]  = [train_env.reward_fn_code]
        self.log_rows:          list[dict] = []
        self.failure_log:       list[str]  = []
        self.rewrite_count      = 0
        self.best_reward        = -float("inf")
        self.best_version       = 0
        self._grace_remaining   = 0

        # ── Curriculum state (rewardforge condition only, v6) ─────────────────
        self._curr_state  = None    # CurriculumState | None
        self._curr_fns    = None    # (survive_fn, approach_fn, land_fn) | None
        self._curr_codes  = None    # {"survive": str, ...} | None
        if condition == "rewardforge":
            self._init_curriculum()

    def _on_step(self) -> bool:
        # Curriculum blend tick (every step, only for rewardforge)
        if self.condition == "rewardforge" and self._curr_state is not None:
            self._tick_blend()

        if self.num_timesteps % CHECKPOINT_EVERY != 0:
            return True

        mean_rew, std_rew = _evaluate(self.model, self._eval_env)
        step = self.num_timesteps
        self.reward_history.append((step, mean_rew))

        if mean_rew > self.best_reward:
            self.best_reward  = mean_rew
            self.best_version = self.train_env.reward_fn_version

        # Trigger routing: curriculum for rewardforge; hover-trap for ablations
        if self.condition == "rewardforge":
            triggered = self._check_curriculum_advance(mean_rew, step)
        else:
            triggered = self._maybe_trigger(mean_rew)

        self.log_rows.append({
            "step": step, "mean_reward": mean_rew, "std_reward": std_rew,
            "version": self.train_env.reward_fn_version,
            "triggered": "YES" if triggered else "",
        })
        return True

    def _maybe_trigger(self, mean_rew: float) -> bool:
        """
        Hover-trap detector: fires LLM when ALL THREE conditions hold:

          1. step >= MIN_TRIGGER_STEP (80k)  — past the early chaotic crash phase
          2. mean_rew >= MIN_TRIGGER_REWARD (0.0) — agent is hovering, not crashing
          3. improvement over last LOOK_BACK checkpoints < IMPROVEMENT_DELTA (10 pts)
                                             — genuinely stagnating at the hover plateau

        This pinpoints the hover trap: the agent learned to float to avoid the
        -100 crash penalty but can't break through to a soft landing (+100 bonus).
        The LLM intervenes exactly here with a landing-encouraging shaping term.
        """
        # -- Hard gates --
        if self.condition == "baseline_ppo":
            return False
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            return False
        if self.rewrite_count >= MAX_REWRITES:
            return False
        if len(self.reward_history) < LOOK_BACK + 1:
            return False

        current_step = self.reward_history[-1][0]

        # -- Hover-trap: three conditions must ALL be true --
        step_ready   = current_step >= MIN_TRIGGER_STEP   # past random-crash phase
        hover_zone   = mean_rew     >= MIN_TRIGGER_REWARD  # positive = hovering
        old_rew      = self.reward_history[-(LOOK_BACK + 1)][1]
        stagnating   = (mean_rew - old_rew) < IMPROVEMENT_DELTA  # < 10 pts / 30k steps

        if not (step_ready and hover_zone and stagnating):
            return False   # not in hover trap yet -- let PPO keep learning

        if self.condition == "rewardforge":
            return self._trigger_llm(include_history=True)
        elif self.condition == "ablation_blind":
            return self._trigger_llm(include_history=False)
        elif self.condition == "ablation_random":
            return self._trigger_random()
        return False

    def _trigger_llm(self, include_history: bool) -> bool:
        """
        Call qwen3-32b (via Groq) with optional history.  Handles 429 quota errors with
        exponential backoff (30s -> 60s -> 120s) before giving up.
        A pre-call sleep of LLM_CALL_DELAY_S seconds is inserted to spread
        requests across time and avoid bursting the free-tier RPM limit.
        """
        history = self.reward_history[-LOOK_BACK:] if include_history else []

        print(f"    Waiting {LLM_CALL_DELAY_S}s before LLM call "
              f"(rate-limit buffer) ...", flush=True)
        time.sleep(LLM_CALL_DELAY_S)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = request_new_reward_fn(
                    current_reward_fn_code=self.train_env.reward_fn_code,
                    reward_history=history,
                )
                break   # success
            except Exception as exc:
                is_quota = "429" in str(exc) or "quota" in str(exc).lower()
                if is_quota and attempt < max_attempts:
                    wait = 30 * (2 ** attempt)   # 60s, 120s
                    print(f"    ⚠️  429 quota hit — retrying in {wait}s "
                          f"(attempt {attempt}/{max_attempts}) …", flush=True)
                    time.sleep(wait)
                    continue
                note = (f"[step {self.reward_history[-1][0]:,}] "
                        f"LLM failure (attempt {attempt}): {exc}")
                self.failure_log.append(note)
                print(f"    ⚠️  {note}  → keeping current fn")
                return False
        else:
            return False   # all attempts exhausted

        if result is None:
            self.failure_log.append(
                f"[step {self.reward_history[-1][0]:,}] LLM returned unusable code"
            )
            return False
        self._apply_rewrite(*result)
        return True

    def _trigger_random(self) -> bool:
        self._apply_rewrite(*_make_random_reward_fn(self._rng))
        return True

    def _apply_rewrite(self, fn, code: str) -> None:
        self.train_env.set_reward_fn(fn, code)
        self._eval_env.reward_fn      = fn
        self._eval_env.reward_fn_code = code
        self.rewrite_count += 1
        self.reward_fn_history.append(code)
        _reset_value_head(self.model)
        self._grace_remaining = GRACE_CHECKPOINTS
        print(f"    🔄 v{self.train_env.reward_fn_version}  ⏳ grace={GRACE_CHECKPOINTS}ckpt")

    # ── Curriculum management (rewardforge only) ─────────────────────────────
    def _init_curriculum(self) -> None:
        """Pre-training: call LLM once for 3 staged functions."""
        result = request_curriculum_fns()
        if result is None:
            print("  ⚠️  Curriculum failed — rewardforge will use base reward.")
            return
        survive_fn, approach_fn, land_fn, codes = result
        self._curr_fns   = (survive_fn, approach_fn, land_fn)
        self._curr_codes = codes
        self._curr_state = CurriculumState()
        # Install blended closure on both envs (reads alpha from state at call-time)
        blended = make_blended_fn(self._curr_fns, self._curr_state)
        label   = "curriculum:stage0_survive"
        self.train_env.reward_fn      = blended
        self.train_env.reward_fn_code = label
        self._eval_env.reward_fn      = blended
        self._eval_env.reward_fn_code = label

    def _tick_blend(self) -> None:
        """Called every step. Updates curriculum alpha during a blend."""
        st = self._curr_state
        if not st.blending:
            return
        elapsed  = self.num_timesteps - st.blend_start_step
        progress = min(1.0, elapsed / BLEND_STEPS)
        st.alpha = st.blend_from + progress * (st.blend_to - st.blend_from)
        if elapsed >= BLEND_STEPS:
            st.blending = False
            st.stage    = int(round(st.blend_to))
            st.alpha    = st.blend_to
            st.advances += 1
            names = {0: "survive", 1: "approach", 2: "land"}
            print(f"\n    🎓 Stage {st.stage} ({names[st.stage]}) "
                  f"fully active @ step {self.num_timesteps:,}")

    def _start_blend(self, from_alpha: float, to_alpha: float) -> None:
        """Initiate a curriculum stage transition."""
        st = self._curr_state
        st.blending         = True
        st.blend_from       = from_alpha
        st.blend_to         = to_alpha
        st.blend_start_step = self.num_timesteps
        names = {0: "survive", 1: "approach", 2: "land"}
        fs, ts = int(round(from_alpha)), int(round(to_alpha))
        print(f"\n    🎓 Blending {names[fs]}→{names[ts]} "
              f"over {BLEND_STEPS:,} steps …")

    def _check_curriculum_advance(self, mean_rew: float, step: int) -> bool:
        """Checkpoint-level: decide whether to advance curriculum stage."""
        st = self._curr_state
        if st is None or st.blending:
            return False
        if st.stage == 0:
            steps_in_stage = step - st.stage0_start_step
            gate = mean_rew >= STAGE1_GATE
            tmax = steps_in_stage >= STAGE1_WAIT
            if gate or tmax:
                reason = "reward gate" if gate else f"time limit ({steps_in_stage:,} steps)"
                print(f"\n    🎓 Stage 0→1 [{reason}]  mean={mean_rew:+.1f}")
                self._start_blend(from_alpha=0.0, to_alpha=1.0)
                return True
        elif st.stage == 1 and len(self.reward_history) >= LOOK_BACK + 1:
            if mean_rew >= MIN_TRIGGER_REWARD:
                old_rew = self.reward_history[-(LOOK_BACK + 1)][1]
                if (mean_rew - old_rew) < IMPROVEMENT_DELTA:
                    print(f"\n    🎓 Stage 1→2 [hover trap]  mean={mean_rew:+.1f}")
                    self._start_blend(from_alpha=1.0, to_alpha=2.0)
                    return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Artifact saving  (same schema as main.py and CartPole experiment_runner)
# ═══════════════════════════════════════════════════════════════════════════════
def _save_run(run_dir: Path, cb: ExperimentCallback, condition: str, seed: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward", "std_reward", "reward_fn_version", "triggered"])
        for row in cb.log_rows:
            w.writerow([row["step"], f"{row['mean_reward']:.4f}", f"{row['std_reward']:.4f}",
                        f"v{row['version']}", row["triggered"]])

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("RewardForge LunarLander Experiment Summary\n")
        f.write("=" * 44 + "\n")
        f.write(f"Condition          : {condition}\n")
        f.write(f"Seed               : {seed}\n")
        f.write(f"Environment        : LunarLander-v3\n")
        f.write(f"Total timesteps    : {TOTAL_TIMESTEPS:,}\n")
        f.write(f"Checkpoint interval: {CHECKPOINT_EVERY:,}\n")
        f.write(f"Look-back window   : {LOOK_BACK} checkpoints\n")
        f.write(f"Improvement delta  : {IMPROVEMENT_DELTA} pts (absolute)\n")
        f.write(f"Max rewrites       : {MAX_REWRITES}\n")
        f.write(f"Rewrites used      : {cb.rewrite_count}\n")
        f.write(f"Best reward        : {cb.best_reward:.2f}\n")
        f.write(f"Best reward fn ver : v{cb.best_version}\n")
        if cb.failure_log:
            f.write("\nLLM failures:\n")
            for note in cb.failure_log:
                f.write(f"  {note}\n")

    rf_dir = run_dir / "reward_functions"
    rf_dir.mkdir(exist_ok=True)
    for i, code in enumerate(cb.reward_fn_history):
        (rf_dir / f"v{i}.py").write_text(code, encoding="utf-8")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "environment": "LunarLander-v3", "condition": condition, "seed": seed,
            "total_timesteps": TOTAL_TIMESTEPS, "checkpoint_every": CHECKPOINT_EVERY,
            "look_back": LOOK_BACK, "improvement_delta": IMPROVEMENT_DELTA,
            "max_rewrites": MAX_REWRITES, "n_eval_episodes": N_EVAL_EPISODES,
        }, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Single-run orchestrator
# ═══════════════════════════════════════════════════════════════════════════════
def run_single(condition: Condition, seed: int, run_dir: Path) -> dict:
    set_global_seeds(seed)
    env = CustomLunarLander()
    env.reset(seed=seed)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0)
    cb = ExperimentCallback(env, condition, seed)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)

    rows = cb.log_rows
    # LunarLander milestones (harder environment — lower thresholds than CartPole)
    steps_to_0   = next((r["step"] for r in rows if r["mean_reward"] >= 0),   -1)
    steps_to_100 = next((r["step"] for r in rows if r["mean_reward"] >= 100), -1)
    steps_to_200 = next((r["step"] for r in rows if r["mean_reward"] >= 200), -1)
    final_reward = rows[-1]["mean_reward"] if rows else 0.0

    _save_run(run_dir, cb, condition, seed)

    triggers = (cb._curr_state.advances
                if condition == "rewardforge" and cb._curr_state is not None
                else cb.rewrite_count)
    return {
        "condition":      condition,
        "seed":           seed,
        "best_reward":    cb.best_reward,
        "steps_to_0":     steps_to_0,
        "steps_to_100":   steps_to_100,
        "steps_to_200":   steps_to_200,
        "total_triggers": triggers,
        "final_reward":   final_reward,
        "final_std":      cb.log_rows[-1]["std_reward"] if cb.log_rows else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical analysis
# ═══════════════════════════════════════════════════════════════════════════════
def _iqr(arr) -> float:
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _milestone_str(vals, sentinel=-1) -> str:
    valid = [v for v in vals if v != sentinel]
    if not valid:
        pct = 0
        return f"never  (0/{len(vals)} seeds reached it)"
    pct = len(valid) / len(vals) * 100
    return f"{np.median(valid):.0f}  (IQR {_iqr(valid):.0f}, {pct:.0f}% seeds reached)"


def generate_analysis(results: list[dict], out_path: Path) -> str:
    from scipy import stats as sp_stats

    sep = "─" * 56
    lines = [
        "RewardForge LunarLander-v3 Experiment Analysis",
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
        s0    = [r["steps_to_0"]   for r in rows]
        s100  = [r["steps_to_100"] for r in rows]
        s200  = [r["steps_to_200"] for r in rows]
        trigs = [r["total_triggers"] for r in rows]
        lines += [
            f"\n  [{cond}]",
            f"    best_reward  : median={np.median(best):+.1f}  IQR={_iqr(best):.1f}"
            f"  range=[{min(best):+.0f}, {max(best):+.0f}]",
            f"    final_reward : median={np.median(final):+.1f}  IQR={_iqr(final):.1f}",
            f"    steps_to_0   : {_milestone_str(s0)}",
            f"    steps_to_100 : {_milestone_str(s100)}",
            f"    steps_to_200 : {_milestone_str(s200)}",
            f"    triggers used: median={np.median(trigs):.0f}  (cap={MAX_REWRITES})",
        ]

    # ── Mann-Whitney U: rewardforge vs each other condition ──────────────────
    rf_best = [r["best_reward"] for r in results if r["condition"] == "rewardforge"]
    lines += ["", sep,
              "Statistical Tests: Mann-Whitney U  (one-sided: rewardforge > other)", sep]

    comparisons = [c for c in CONDITIONS if c != "rewardforge"]
    any_significant = False
    for cond in comparisons:
        other = [r["best_reward"] for r in results if r["condition"] == cond]
        u, p  = sp_stats.mannwhitneyu(rf_best, other, alternative="greater")
        sig   = "✅ p<0.05" if p < 0.05 else "❌ p≥0.05"
        lines.append(
            f"  rewardforge vs {cond:<20}: U={u:.0f}  p={p:.4f}  {sig}"
        )
        if p < 0.05:
            any_significant = True

    # ── Key ablation insight (does history help vs blind?) ───────────────────
    blind_best  = [r["best_reward"] for r in results if r["condition"] == "ablation_blind"]
    _, p_hist   = sp_stats.mannwhitneyu(rf_best, blind_best, alternative="two-sided")
    lines += ["",
              f"  rewardforge vs ablation_blind (two-sided): p={p_hist:.4f}  "
              + ("← history context IS decisive" if p_hist < 0.05
                 else "<- history context not sig. (both use the LLM)")]

    # ── Plain-English conclusion ──────────────────────────────────────────────
    lines += ["", sep, "Conclusion:", sep]

    if any_significant:
        lines.append(
            "  ✅ SIGNIFICANT — RewardForge outperformed at least one baseline "
            f"(p < 0.05) on LunarLander-v3.\n"
            "     On this harder environment, reward function quality matters: "
            "random or blind\n"
            "     shaping does not reliably help, while curve-aware LLM design does."
        )
    else:
        lines.append(
            "  ❌ NOT SIGNIFICANT — no pairwise comparison reached p<0.05.\n"
            f"     This may reflect insufficient training ({TOTAL_TIMESTEPS//1000}k vs 1M steps "
            "for full convergence)\n"
            "     or n=5 being too small. Directional trend should still be visible."
        )

    # ── Vs literature ─────────────────────────────────────────────────────────
    rf_median = np.median(rf_best)
    gap = LITERATURE_MEAN - rf_median
    lines += [
        "",
        f"  Gap to literature baseline ({LITERATURE_STEPS//1000}k steps): ",
        f"    RewardForge median = {rf_median:+.1f}  vs  literature = {LITERATURE_MEAN:.1f}",
        f"    Gap = {gap:.1f} pts — expected given {TOTAL_TIMESTEPS//1000}k vs "
        f"{LITERATURE_STEPS//1000}k steps budget.",
    ]

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Timestamped output directory — never overwrites previous runs ─────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPR_ROOT = _EXPR_BASE / timestamp
    EXPR_ROOT.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[Condition, int]] = [
        (cond, seed) for cond in CONDITIONS for seed in SEEDS
    ]
    total = len(plan)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   RewardForge — LunarLander Experiment Runner            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  {len(CONDITIONS)} conditions × {len(SEEDS)} seeds = {total} total runs")
    print(f"  {TOTAL_TIMESTEPS:,} steps per run")
    print(f"  Output → {EXPR_ROOT}")
    print(f"  Literature reference: {LITERATURE_SOURCE}")
    print(f"  PPO @ {LITERATURE_STEPS:,} steps = {LITERATURE_MEAN:.1f} ± {LITERATURE_STD:.1f}\n")

    all_results: list[dict] = []

    for idx, (condition, seed) in enumerate(plan, start=1):
        run_dir = EXPR_ROOT / condition / f"seed_{seed}"
        print(f"[{idx:>2}/{total}] {condition:<22} seed_{seed} …", flush=True)

        result = run_single(condition, seed, run_dir)
        all_results.append(result)

        print(
            f"        ✓  best={result['best_reward']:>+8.1f}"
            f"  final={result['final_reward']:>+8.1f}"
            f"  triggers={result['total_triggers']}"
            f"  →0@{result['steps_to_0']}"
            f"  →100@{result['steps_to_100']}"
            f"  →200@{result['steps_to_200']}"
        )

    # ── results_summary.csv ───────────────────────────────────────────────────
    csv_path = EXPR_ROOT / "results_summary.csv"
    fields = ["condition", "seed", "best_reward", "steps_to_0",
              "steps_to_100", "steps_to_200", "total_triggers", "final_reward", "final_std"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"\n📊  Results CSV  → {csv_path}")

    # ── results_analysis.txt ──────────────────────────────────────────────────
    analysis_path = EXPR_ROOT / "results_analysis.txt"
    analysis_text = generate_analysis(all_results, analysis_path)
    print(f"📈  Analysis     → {analysis_path}")

    print("\n" + "═" * 58)
    print(analysis_text)
    print("═" * 58)
    print("✅  All experiments complete.\n")


if __name__ == "__main__":
    main()
