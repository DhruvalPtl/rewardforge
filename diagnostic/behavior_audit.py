"""
diagnostic/behavior_audit.py — Behavioral autopsy for BipedalWalker-v3.

Run the current policy on the UNAUGMENTED environment and extract a rich
behavioral fingerprint.  Key insight: the LLM cannot observe what the agent
actually does, so we measure it and tell it precisely.

Metrics collected
─────────────────
  Locomotion quality
    mean_forward_velocity   obs[2] averaged across all steps
    mean_hull_angle_abs     |obs[0]|  — how upright is the agent?
    gait_rhythm_score       fraction of steps where obs[8] ≠ obs[13]
                            (alternating leg contact = natural gait)

  Failure analysis
    mean_survival_fraction  steps / 1600
    common_fail_hull_angle  |obs[0]| in last 5 steps of TERMINAL episodes
    velocity_collapse_det   True if vx drops below 0.1 in last 30 steps
    shuffle_detected        True if mean vx < 0.25 and gait < 0.45

  Bottleneck classification (automatic, used in prompt)
    - SHUFFLE   : agent moves tiny steps, no real forward progress
    - BALANCE   : agent falls from instability (high hull angle)
    - STALL     : velocity collapses just before fall (learned to stop)
    - GAIT      : agent moves ok but without alternating legs (inefficient)
    - MULTIPLE  : more than one issue
"""

from dataclasses import dataclass, field
from typing import Literal

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

_MAX_STEPS = 1600   # BipedalWalker default max episode length


# ── Data container ─────────────────────────────────────────────────────────────
@dataclass
class BehaviorReport:
    n_episodes:             int   = 0
    # Episode-level
    mean_episode_length:    float = 0.0
    std_episode_length:     float = 0.0
    mean_episode_reward:    float = 0.0
    # Per-step obs statistics
    mean_forward_velocity:  float = 0.0    # obs[2]
    mean_hull_angle_abs:    float = 0.0    # |obs[0]|
    gait_rhythm_score:      float = 0.0    # alternating-contacts fraction
    # Failure analysis
    mean_survival_fraction: float = 0.0
    common_fail_hull_angle: float = 0.0
    velocity_collapse_det:  bool  = False
    shuffle_detected:       bool  = False
    # Automatic bottleneck label
    bottleneck:             str   = "UNKNOWN"

    def diagnosis_str(self) -> str:
        """
        Human-readable multi-line report injected verbatim into the LLM prompt.
        """
        survival_pct = self.mean_survival_fraction * 100
        shuffle_tag  = "!! YES -- less than 0.25 avg velocity AND poor gait" \
                       if self.shuffle_detected else "No"
        vcollapse    = "!! YES -- velocity near zero at moment of fall" \
                       if self.velocity_collapse_det else "No"

        lines = [
            f"  n_episodes         : {self.n_episodes}",
            f"  mean surv. length  : {self.mean_episode_length:.0f} / {_MAX_STEPS} steps"
            f"  ({survival_pct:.1f}% of max episode)",
            f"  mean episode reward: {self.mean_episode_reward:+.1f}",
            "",
            "  Locomotion quality (per-step averages across all episodes):",
            f"    forward velocity obs[2] : {self.mean_forward_velocity:.4f}"
            f"  [GOOD if > 0.50]",
            f"    hull angle |obs[0]|     : {self.mean_hull_angle_abs:.4f}"
            f"  [GOOD if < 0.15]",
            f"    gait rhythm score       : {self.gait_rhythm_score:.2%}"
            f"  [GOOD if > 60% alternating steps]",
            "",
            "  Failure analysis:",
            f"    shuffle trap detected   : {shuffle_tag}",
            f"    velocity collapse       : {vcollapse}",
            f"    hull angle at fall      : {self.common_fail_hull_angle:.4f}",
            "",
            f"  AUTO-DIAGNOSED BOTTLENECK: {self.bottleneck}",
        ]
        return "\n".join(lines)


# ── Audit runner ───────────────────────────────────────────────────────────────
def run_audit(model: PPO, n_episodes: int = 20, seed_offset: int = 9_999) -> BehaviorReport:
    """
    Run *model* on vanilla BipedalWalker-v3 (no reward shaping) and return
    a BehaviorReport summarising the actual agent behaviour.

    Parameters
    ----------
    model       : trained PPO model to evaluate
    n_episodes  : number of rollout episodes
    seed_offset : seeds audit envs differently from training envs
    """
    env = gym.make("BipedalWalker-v3")

    ep_lengths, ep_rewards = [], []
    all_vx:   list[float] = []
    all_hull: list[float] = []
    all_gait: list[float] = []
    fail_angles:   list[float] = []
    fail_vx_last:  list[float] = []   # mean vx in last 30 steps of terminal eps

    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep_idx)
        total_r, steps = 0.0, 0
        ep_vx:   list[float] = []
        ep_hull: list[float] = []
        ep_gait: list[float] = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            steps   += 1

            ep_vx.append(float(obs[2]))
            ep_hull.append(abs(float(obs[0])))
            ep_gait.append(1.0 if bool(obs[8]) != bool(obs[13]) else 0.0)
            done = terminated or truncated

        ep_lengths.append(steps)
        ep_rewards.append(total_r)
        all_vx.extend(ep_vx)
        all_hull.extend(ep_hull)
        all_gait.extend(ep_gait)

        if terminated and steps >= 5:
            fail_angles.append(ep_hull[-1])
            last_n = min(30, len(ep_vx))
            fail_vx_last.append(float(np.mean(ep_vx[-last_n:])))

    env.close()

    r = BehaviorReport()
    r.n_episodes              = n_episodes
    r.mean_episode_length     = float(np.mean(ep_lengths))
    r.std_episode_length      = float(np.std(ep_lengths))
    r.mean_episode_reward     = float(np.mean(ep_rewards))
    r.mean_forward_velocity   = float(np.mean(all_vx))   if all_vx   else 0.0
    r.mean_hull_angle_abs     = float(np.mean(all_hull)) if all_hull else 0.0
    r.gait_rhythm_score       = float(np.mean(all_gait)) if all_gait else 0.0
    r.mean_survival_fraction  = r.mean_episode_length / _MAX_STEPS
    r.common_fail_hull_angle  = float(np.mean(fail_angles)) if fail_angles else 0.0
    r.velocity_collapse_det   = (
        len(fail_vx_last) >= 3 and float(np.mean(fail_vx_last)) < 0.12
    )
    r.shuffle_detected        = (
        r.mean_forward_velocity < 0.25 and r.gait_rhythm_score < 0.45
    )

    # Auto-classify dominant bottleneck
    issues: list[str] = []
    if r.shuffle_detected:
        issues.append("SHUFFLE")
    if r.mean_hull_angle_abs > 0.20 or r.common_fail_hull_angle > 0.30:
        issues.append("BALANCE")
    if r.velocity_collapse_det:
        issues.append("STALL")
    if r.gait_rhythm_score < 0.40 and not r.shuffle_detected:
        issues.append("GAIT")

    r.bottleneck = " + ".join(issues) if issues else "NONE (performing well)"
    return r
