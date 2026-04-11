"""
RewardForge — LunarLander-v2 main entry point.

Trains PPO on LunarLander-v3 and dynamically rewrites the reward function
via the LLM (qwen3-32b on Groq) when learning stagnates.

╔══════════════════════════════════════════════════════════════════════════╗
║  DIFF vs CartPole main.py  (LunarLander-v3)                              ║
║                                                                          ║
║  Config:                                                                 ║
║    TOTAL_TIMESTEPS   2_000  →  100_000  (LunarLander needs much more)   ║
║    CHECKPOINT_EVERY    500  →   10_000  (longer checkpoint window)       ║
║    LOOK_BACK             2  →        3  (compare vs 3 ckpts ago)         ║
║    IMPROVEMENT_DELTA   n/a  →     20.0  (absolute reward gain, not %)   ║
║    N_EVAL_EPISODES      10  →       15  (higher variance env)            ║
║                                                                          ║
║  Stagnation logic:                                                       ║
║    CartPole used IMPROVEMENT_PCT (percentage change, works for +reward). ║
║    LunarLander uses IMPROVEMENT_DELTA (absolute Δ reward) because        ║
║    negative rewards make percentage comparisons misleading.              ║
║    Trigger fires if reward has not improved by ≥20 pts over last 3 ckpts ║
║                                                                          ║
║  Environment:                                                            ║
║    CustomCartPole  →  CustomLunarLander                                  ║
║    request_new_reward_fn imported from lunarlander.rewardforge_agent     ║
║                                                                          ║
║  Runs saved to:  runs/lunarlander/<timestamp>/                           ║
║  (separate from CartPole runs in runs/<timestamp>/)                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Run from the rewardforge/ directory:
    python lunarlander/main.py
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ── Path setup — allows running from rewardforge/ or from anywhere ────────────
_HERE = Path(__file__).parent.resolve()         # rewardforge/lunarlander/
_ROOT = _HERE.parent                            # rewardforge/
sys.path.insert(0, str(_ROOT))

from lunarlander.env_wrapper      import CustomLunarLander        # noqa: E402
from lunarlander.rewardforge_agent import request_new_reward_fn   # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# CHANGED FROM CARTPOLE: all values updated for LunarLander
TOTAL_TIMESTEPS   = 100_000
CHECKPOINT_EVERY  = 10_000
LOOK_BACK         = 5         # compare vs 5 ckpts ago (50k steps) for stable signal
IMPROVEMENT_DELTA = 25.0      # require 25 pts absolute gain over 50k steps
MAX_REWRITES      = 2         # fewer but better-timed rewrites
N_EVAL_EPISODES   = 15
GRACE_CHECKPOINTS = 3         # checkpoints to skip after each rewrite

# Guard rails: don't trigger LLM during the noisy early-training phase
MIN_TRIGGER_STEP   = 50_000   # in main.py (100k total), wait until half-way
MIN_TRIGGER_REWARD = -150.0   # don't fire if agent is still catastrophically bad


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helper  (IDENTICAL TO CARTPOLE VERSION)
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_mean_reward(model, eval_env, n_episodes: int = N_EVAL_EPISODES) -> float:
    """Roll out n_episodes on the isolated eval env and return mean reward."""
    rewards = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_episodes):
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
# Value-network reset  (IDENTICAL TO CARTPOLE VERSION)
# ═══════════════════════════════════════════════════════════════════════════════
def _reset_value_head(model: PPO) -> None:
    """Re-initialize the critic head so stale value estimates don't corrupt training."""
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
    print("  🧠  Value network head re-initialized")


# ═══════════════════════════════════════════════════════════════════════════════
# SB3 Callback
# ═══════════════════════════════════════════════════════════════════════════════
class RewardForgeCallback(BaseCallback):
    """
    Fires every CHECKPOINT_EVERY steps.

    CHANGED FROM CARTPOLE VERSION:
        Stagnation check uses IMPROVEMENT_DELTA (absolute reward gain) and
        LOOK_BACK (compare vs 3 checkpoints ago) instead of a 2-checkpoint
        percentage comparison.  Everything else is identical.
    """

    def __init__(self, env: CustomLunarLander, verbose: int = 0):
        super().__init__(verbose)
        self.env = env

        # Isolated eval env — not owned by SB3's training loop
        self._eval_env = CustomLunarLander()
        self._eval_env.reward_fn      = env.reward_fn
        self._eval_env.reward_fn_code = env.reward_fn_code

        self.reward_history:    list[tuple[int, float]] = []
        self.reward_fn_history: list[str]  = [env.reward_fn_code]
        self.log_rows:          list[dict] = []
        self.rewrite_count      = 0
        self.best_reward        = -float("inf")
        self.best_version       = 0
        self._grace_remaining   = 0
        self._header_printed    = False    # instance var avoids module-level global

    def _on_step(self) -> bool:
        if self.num_timesteps % CHECKPOINT_EVERY != 0:
            return True

        mean_rew = evaluate_mean_reward(self.model, self._eval_env)
        step = self.num_timesteps
        self.reward_history.append((step, mean_rew))

        if mean_rew > self.best_reward:
            self.best_reward  = mean_rew
            self.best_version = self.env.reward_fn_version

        triggered = self._maybe_trigger(mean_rew)

        row = {
            "step": step, "mean_reward": mean_rew,
            "version": self.env.reward_fn_version,
            "triggered": "YES" if triggered else "",
        }
        self.log_rows.append(row)
        self._print_row(row)
        return True

    # ── Stagnation check ─────────────────────────────────────────────────────
    def _maybe_trigger(self, mean_rew: float) -> bool:
        """
        Trigger the LLM only when PPO is genuinely stuck AND has had enough
        time to learn the basics (MIN_TRIGGER_STEP) AND is not in the
        catastrophic crash zone (MIN_TRIGGER_REWARD).
        """
        if self._grace_remaining > 0:
            self._grace_remaining -= 1
            print(f"  ⏳  Grace period — skipping trigger ({self._grace_remaining} left)")
            return False

        if len(self.reward_history) < LOOK_BACK + 1:
            return False
        if self.rewrite_count >= MAX_REWRITES:
            return False

        # Guard: wait until PPO has bootstrapped its own learning
        current_step = self.reward_history[-1][0]
        if current_step < MIN_TRIGGER_STEP:
            return False

        # Guard: don't fire if agent is still in the random-crash zone
        if mean_rew < MIN_TRIGGER_REWARD:
            return False

        # Stagnation: absolute improvement over last LOOK_BACK checkpoints
        old_rew  = self.reward_history[-(LOOK_BACK + 1)][1]
        curr_rew = self.reward_history[-1][1]
        if (curr_rew - old_rew) >= IMPROVEMENT_DELTA:
            return False   # improving fast enough

        self._trigger_rewrite()
        return True

    def _trigger_rewrite(self) -> None:
        """Call the LLM and hot-swap the reward function."""
        last_n = self.reward_history[-LOOK_BACK:]
        result = request_new_reward_fn(
            current_reward_fn_code=self.env.reward_fn_code,
            reward_history=last_n,
        )
        if result is not None:
            fn, code = result
            self.env.set_reward_fn(fn, code)
            self._eval_env.reward_fn      = fn
            self._eval_env.reward_fn_code = code
            self.rewrite_count += 1
            self.reward_fn_history.append(code)
            _reset_value_head(self.model)
            self._grace_remaining = GRACE_CHECKPOINTS
            print(f"🔄  Reward function → v{self.env.reward_fn_version}"
                  f"  ⏳ grace={GRACE_CHECKPOINTS}ckpt")
        else:
            print("⚠️  Keeping previous reward function (LLM returned unusable code).")

    # ── Printing helpers (instance methods — no module-level globals) ─────────
    def _print_row(self, row: dict) -> None:
        if not self._header_printed:
            print(f"\n{'Step':>8} | {'Mean Reward':>12} | {'Fn Ver':>8} | Triggered?")
            print("─" * 50)
            self._header_printed = True
        print(
            f"{row['step']:>8,} | "
            f"{row['mean_reward']:>+12.2f} | "
            f"v{row['version']:>7} | "
            f"{row['triggered']}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Run persistence  (IDENTICAL SCHEMA TO CARTPOLE — runs saved separately)
# ═══════════════════════════════════════════════════════════════════════════════
def save_run(callback: RewardForgeCallback, env: CustomLunarLander) -> None:
    """
    Save all run artefacts to  runs/lunarlander/<timestamp>/

    CHANGED FROM CARTPOLE:
        Output goes into runs/lunarlander/ subfolder so CartPole and
        LunarLander runs never collide.  File schema is IDENTICAL:
        training_log.csv, summary.txt, config.json, reward_functions/v*.py
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # CHANGED FROM CARTPOLE: "runs/<ts>" → "runs/lunarlander/<ts>"
    run_dir = _ROOT / "runs" / "lunarlander" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # training_log.csv
    with open(run_dir / "training_log.csv", "w", encoding="utf-8") as f:
        f.write("step,mean_reward,reward_fn_version,triggered\n")
        for row in callback.log_rows:
            f.write(
                f"{row['step']},{row['mean_reward']:.4f},"
                f"v{row['version']},{row['triggered']}\n"
            )

    # summary.txt
    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("RewardForge LunarLander Run Summary\n")
        f.write("=" * 42 + "\n")
        f.write(f"Timestamp          : {timestamp}\n")
        f.write(f"Environment        : LunarLander-v3\n")
        f.write(f"Total timesteps    : {TOTAL_TIMESTEPS:,}\n")
        f.write(f"Checkpoint interval: {CHECKPOINT_EVERY:,}\n")
        f.write(f"Look-back window   : {LOOK_BACK} checkpoints\n")
        f.write(f"Improvement delta  : {IMPROVEMENT_DELTA} pts (absolute)\n")
        f.write(f"Max rewrites       : {MAX_REWRITES}\n")
        f.write(f"Rewrites used      : {callback.rewrite_count}\n")
        f.write(f"Best reward        : {callback.best_reward:.2f}\n")
        f.write(f"Best reward fn ver : v{callback.best_version}\n")

    # reward_functions/v*.py
    rf_dir = run_dir / "reward_functions"
    rf_dir.mkdir(exist_ok=True)
    for i, code in enumerate(callback.reward_fn_history):
        (rf_dir / f"v{i}.py").write_text(code, encoding="utf-8")

    # config.json
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "environment":        "LunarLander-v3",
            "total_timesteps":    TOTAL_TIMESTEPS,
            "checkpoint_every":   CHECKPOINT_EVERY,
            "look_back":          LOOK_BACK,
            "improvement_delta":  IMPROVEMENT_DELTA,
            "max_rewrites":       MAX_REWRITES,
            "n_eval_episodes":    N_EVAL_EPISODES,
            "grace_checkpoints":  GRACE_CHECKPOINTS,
        }, f, indent=2)

    print(f"\n💾  Run saved → {run_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main  (IDENTICAL STRUCTURE TO CARTPOLE VERSION)
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("🚀  RewardForge — LunarLander-v3 + llama-3.3-70b reward shaping\n")
    print(f"    {TOTAL_TIMESTEPS:,} steps  |  checkpoint every {CHECKPOINT_EVERY:,}"
          f"  |  max {MAX_REWRITES} rewrites\n")

    env   = CustomLunarLander()
    model = PPO("MlpPolicy", env, verbose=0)
    cb    = RewardForgeCallback(env)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  TRAINING COMPLETE — LunarLander-v3")
    print("=" * 52)
    print(f"  Total timesteps     : {TOTAL_TIMESTEPS:,}")
    print(f"  Reward rewrites     : {cb.rewrite_count}")
    print(f"  Best reward achieved: {cb.best_reward:+.2f}"
          f"  (v{cb.best_version})")
    print(f"  Reference: solved ≥ 200 | random ≈ −200")
    print("=" * 52)

    save_run(cb, env)


if __name__ == "__main__":
    main()
