"""
CustomCartPole — a gymnasium wrapper around CartPole-v1 whose reward
function can be hot-swapped at runtime by the RewardForge agent.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ── Default reward function (identity — returns the original reward) ─────────
def default_reward(obs, action, reward, terminated, info):
    """Standard CartPole reward: +1 for every step the pole stays upright."""
    return reward


class CustomCartPole(gym.Wrapper):
    """Thin wrapper that intercepts ``step()`` and applies a pluggable
    reward function.

    Attributes
    ----------
    reward_fn : callable
        ``(obs, action, reward, terminated, info) -> float``
    reward_fn_version : int
        Monotonically increasing counter; bumped every time ``set_reward_fn``
        is called.
    reward_fn_code : str
        Source code of the currently active reward function (for logging /
        sending to the LLM).
    """

    def __init__(self, **kwargs):
        env = gym.make("CartPole-v1", **kwargs)
        super().__init__(env)

        # pluggable reward
        self.reward_fn = default_reward
        self.reward_fn_version = 0
        self.reward_fn_code = _fn_source(default_reward)

    # ── public API ───────────────────────────────────────────────────────
    def set_reward_fn(self, fn, source_code: str):
        """Replace the active reward function.

        Parameters
        ----------
        fn : callable
            New reward function with signature
            ``(obs, action, reward, terminated, info) -> float``.
        source_code : str
            Python source of *fn* (kept for logging & LLM context).
        """
        self.reward_fn = fn
        self.reward_fn_version += 1
        self.reward_fn_code = source_code

    # ── gym.Wrapper overrides ────────────────────────────────────────────
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            shaped_reward = float(self.reward_fn(obs, action, reward, terminated, info))
        except Exception:
            # If the custom reward fn blows up, fall back to the original
            shaped_reward = reward
        return obs, shaped_reward, terminated, truncated, info


# ── helpers ──────────────────────────────────────────────────────────────────
def _fn_source(fn) -> str:
    """Best-effort source retrieval; falls back to repr."""
    import inspect
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return repr(fn)
