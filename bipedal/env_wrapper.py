"""
CustomBipedalWalker — gymnasium wrapper around BipedalWalker-v3 whose reward
function can be hot-swapped at runtime by RewardForge.

Observation space (24 floats)
────────────────────────────────────────────────────────────────────────────
  obs[0]   hull angle              [-pi, pi]   0 = upright
  obs[1]   hull angular velocity
  obs[2]   hull velocity x         target ≈ 1.0, range [-inf, inf]
  obs[3]   hull velocity y
  obs[4]   hip joint 1 angle
  obs[5]   hip joint 1 speed
  obs[6]   knee joint 1 angle      0=straight, negative=bent
  obs[7]   knee joint 1 speed
  obs[8]   leg 1 ground contact    0 or 1
  obs[9]   hip joint 2 angle
  obs[10]  hip joint 2 speed
  obs[11]  knee joint 2 angle
  obs[12]  knee joint 2 speed
  obs[13]  leg 2 ground contact    0 or 1
  obs[14-23] 10 lidar rangefinder readings (forward-facing, normalised 0-1)

Action space (4 continuous in [-1, 1])
────────────────────────────────────────────────────────────────────────────
  action[0]  hip joint 1 torque
  action[1]  knee joint 1 torque
  action[2]  hip joint 2 torque
  action[3]  knee joint 2 torque

Built-in reward
────────────────────────────────────────────────────────────────────────────
  + velocity_x  per step (forward progress)
  - 0.00025 * sum(action**2)  energy penalty each step
  - 100  if hull touches ground (fall; episode ends)
  Environment is "solved" at mean episode reward ≥ 300.
"""

import gymnasium as gym


def default_reward(obs, action, reward, terminated, info):
    """Pass the built-in BipedalWalker reward through unchanged."""
    return reward


class CustomBipedalWalker(gym.Wrapper):
    """
    Thin wrapper that intercepts step() and applies a pluggable reward function,
    allowing RewardForge to swap it at runtime.

    Attributes
    ----------
    reward_fn : callable
        ``(obs, action, reward, terminated, info) -> float``
    reward_fn_version : int
        Increments each time set_reward_fn() is called.
    reward_fn_code : str
        Source code of the currently active reward function.
    """

    def __init__(self, hardcore: bool = False, **kwargs):
        env = gym.make("BipedalWalker-v3", hardcore=hardcore, **kwargs)
        super().__init__(env)
        self.reward_fn         = default_reward
        self.reward_fn_version = 0
        self.reward_fn_code    = _fn_source(default_reward)

    # ── Public API ────────────────────────────────────────────────────────────
    def set_reward_fn(self, fn, source_code: str) -> None:
        """Replace the active reward function at runtime."""
        self.reward_fn          = fn
        self.reward_fn_version += 1
        self.reward_fn_code     = source_code

    # ── gym.Wrapper override ──────────────────────────────────────────────────
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            shaped = float(self.reward_fn(obs, action, reward, terminated, info))
        except Exception:
            shaped = reward   # fall back to original if custom fn crashes
        return obs, shaped, terminated, truncated, info


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fn_source(fn) -> str:
    """Best-effort source retrieval; falls back to repr()."""
    import inspect
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return repr(fn)
