"""
RewardForge — main entry point.

Trains PPO on CartPole-v1 and dynamically rewrites the reward function
via Gemini when learning stagnates.
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env_wrapper import CustomCartPole
from rewardforge_agent import request_new_reward_fn


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS  = 20_000
CHECKPOINT_EVERY = 2_000     # log / check every N steps
IMPROVEMENT_PCT  = 10.0      # minimum % improvement expected
MAX_REWRITES     = 3         # cap on Gemini calls per run
N_EVAL_EPISODES  = 10        # episodes used to measure mean reward
GRACE_CHECKPOINTS = 1        # skip this many checkpoints after a rewrite before judging


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helper
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_mean_reward(model, env, n_episodes: int = N_EVAL_EPISODES) -> float:
    """Roll out *n_episodes* and return the mean total reward."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards))


# ═══════════════════════════════════════════════════════════════════════════
# Value-network reset — called after each reward function swap
# ═══════════════════════════════════════════════════════════════════════════
def _reset_value_head(model: PPO):
    """Re-initialize the value-function head of the PPO policy.

    After a reward function change the old value estimates are stale and will
    produce noisy advantage estimates.  Resetting only the value head lets the
    policy (actor) keep what it has learned while allowing the critic to
    recalibrate to the new reward scale.
    """
    policy = model.policy

    # mlp_extractor.value_net — hidden layers for the value branch
    for module in policy.mlp_extractor.value_net.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # value_net — final linear projection to scalar
    if isinstance(policy.value_net, torch.nn.Linear):
        torch.nn.init.orthogonal_(policy.value_net.weight, gain=1.0)
        if policy.value_net.bias is not None:
            torch.nn.init.zeros_(policy.value_net.bias)

    print("  🧠  Value network head re-initialized")


# ═══════════════════════════════════════════════════════════════════════════
# Custom SB3 callback — fires every CHECKPOINT_EVERY steps
# ═══════════════════════════════════════════════════════════════════════════
class RewardForgeCallback(BaseCallback):
    def __init__(self, env: CustomCartPole, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.checkpoint_every = CHECKPOINT_EVERY
        self.reward_history: list[tuple[int, float]] = []
        self.rewrite_count = 0
        self.log_rows: list[dict] = []
        self.best_reward = -float("inf")
        self.best_version = 0
        # Track every reward function version (code) for saving
        self.reward_fn_history: list[str] = [env.reward_fn_code]
        # Grace period: number of checkpoints to skip after a rewrite
        self._grace_remaining = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.checkpoint_every != 0:
            return True

        # ── Evaluate ─────────────────────────────────────────────────
        mean_rew = evaluate_mean_reward(self.model, self.env)
        step = self.num_timesteps
        self.reward_history.append((step, mean_rew))

        # Track best
        if mean_rew > self.best_reward:
            self.best_reward = mean_rew
            self.best_version = self.env.reward_fn_version

        # ── Decide whether to trigger RewardForge ────────────────────
        triggered = False
        if self._grace_remaining > 0:
            # Still in grace period after a recent rewrite — let the agent adapt
            self._grace_remaining -= 1
            print(f"  ⏳  Grace period active — skipping trigger ({self._grace_remaining} checkpoints left)")
        elif len(self.reward_history) >= 2 and self.rewrite_count < MAX_REWRITES:
            prev_rew = self.reward_history[-2][1]
            # Improvement check (handle zero/negative prev gracefully)
            if prev_rew <= 0 or (mean_rew - prev_rew) / abs(prev_rew) * 100 < IMPROVEMENT_PCT:
                triggered = True
                self._trigger_rewrite()

        # ── Log ──────────────────────────────────────────────────────
        row = {
            "step": step,
            "mean_reward": mean_rew,
            "version": self.env.reward_fn_version,
            "triggered": "YES" if triggered else "",
        }
        self.log_rows.append(row)
        _print_row(row)

        return True

    def _trigger_rewrite(self):
        """Call Gemini and swap the reward function if successful."""
        last_3 = self.reward_history[-3:] if len(self.reward_history) >= 3 else self.reward_history
        result = request_new_reward_fn(
            current_reward_fn_code=self.env.reward_fn_code,
            reward_history=last_3,
        )
        if result is not None:
            fn, code = result
            self.env.set_reward_fn(fn, code)
            self.rewrite_count += 1
            self.reward_fn_history.append(code)
            # Reset value network so old value estimates don't corrupt learning
            _reset_value_head(self.model)
            # Start grace period
            self._grace_remaining = GRACE_CHECKPOINTS
            print(f"🔄  Reward function updated → version {self.env.reward_fn_version}")
            print(f"  ⏳  Grace period started ({GRACE_CHECKPOINTS} checkpoint(s) before next trigger)")
        else:
            print("⚠️  Keeping previous reward function (Gemini returned unusable code).")


# ═══════════════════════════════════════════════════════════════════════════
# Pretty-printing helpers
# ═══════════════════════════════════════════════════════════════════════════
_HEADER_PRINTED = False

def _print_header():
    global _HEADER_PRINTED
    if not _HEADER_PRINTED:
        print(f"\n{'Step':>6} | {'Mean Reward':>12} | {'Reward Fn Ver':>14} | Triggered?")
        print("-" * 58)
        _HEADER_PRINTED = True


def _print_row(row: dict):
    _print_header()
    print(
        f"{row['step']:>6} | "
        f"{row['mean_reward']:>12.2f} | "
        f"v{row['version']:>13} | "
        f"{row['triggered']}"
    )


def _print_summary(callback: RewardForgeCallback):
    print("\n" + "=" * 58)
    print("TRAINING COMPLETE")
    print("=" * 58)
    print(f"  Total timesteps     : {TOTAL_TIMESTEPS}")
    print(f"  Reward rewrites     : {callback.rewrite_count}")
    print(f"  Best reward achieved: {callback.best_reward:.2f}  "
          f"(reward function version v{callback.best_version})")
    print("=" * 58)


# ═══════════════════════════════════════════════════════════════════════════
# Run persistence — save everything to a timestamped folder
# ═══════════════════════════════════════════════════════════════════════════
def save_run(callback: RewardForgeCallback, env: CustomCartPole):
    """Save all run artefacts to ``runs/<timestamp>/``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), "runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Training log as CSV
    csv_path = os.path.join(run_dir, "training_log.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,mean_reward,reward_fn_version,triggered\n")
        for row in callback.log_rows:
            f.write(
                f"{row['step']},{row['mean_reward']:.4f},"
                f"v{row['version']},{row['triggered']}\n"
            )

    # 2. Summary text
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("RewardForge Run Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp          : {timestamp}\n")
        f.write(f"Total timesteps    : {TOTAL_TIMESTEPS}\n")
        f.write(f"Checkpoint interval: {CHECKPOINT_EVERY}\n")
        f.write(f"Improvement thresh : {IMPROVEMENT_PCT}%\n")
        f.write(f"Max rewrites       : {MAX_REWRITES}\n")
        f.write(f"Rewrites used      : {callback.rewrite_count}\n")
        f.write(f"Best reward        : {callback.best_reward:.2f}\n")
        f.write(f"Best reward fn ver : v{callback.best_version}\n")

    # 3. Reward function versions (each as a .py file)
    reward_fns_dir = os.path.join(run_dir, "reward_functions")
    os.makedirs(reward_fns_dir, exist_ok=True)
    for i, code in enumerate(callback.reward_fn_history):
        fn_path = os.path.join(reward_fns_dir, f"v{i}.py")
        with open(fn_path, "w", encoding="utf-8") as f:
            f.write(code)

    # 4. Config snapshot as JSON
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_timesteps": TOTAL_TIMESTEPS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "improvement_pct": IMPROVEMENT_PCT,
            "max_rewrites": MAX_REWRITES,
            "n_eval_episodes": N_EVAL_EPISODES,
        }, f, indent=2)

    print(f"\n💾  Run saved to: {run_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("🚀  RewardForge — dynamic reward shaping with Gemini\n")

    # 1. Environment
    env = CustomCartPole()

    # 2. PPO agent
    model = PPO("MlpPolicy", env, verbose=0)

    # 3. Callback
    cb = RewardForgeCallback(env)

    # 4. Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)

    # 5. Summary
    _print_summary(cb)

    # 6. Save run to disk
    save_run(cb, env)


if __name__ == "__main__":
    main()
