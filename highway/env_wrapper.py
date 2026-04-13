"""
CustomHighwayEnv — gymnasium wrapper around highway-v0 (highway-env library)
with a hot-swappable reward function.

Observation: KinematicsObservation — shape (5, 5), dtype float64
  Each row = one vehicle.  Row 0 = ego.
  Columns:  [presence, x, y, vx, vy]  (relative to ego, normalised)

  obs[0]   = ego vehicle  —  always [1, 0, 0, vx_ego, 0]
  obs[1..4] = nearest other vehicles (presence=0 if none in range)
  obs[i][0]  presence  (0/1)
  obs[i][1]  x offset  (longitudinal / forward, normalised)
  obs[i][2]  y offset  (lateral, normalised)
  obs[i][3]  vx        (longitudinal speed, normalised)
  obs[i][4]  vy        (lateral speed, normalised)

Actions: Discrete(5)
  0 = LANE_LEFT    change to left lane
  1 = IDLE         maintain speed & lane
  2 = LANE_RIGHT   change to right lane
  3 = FASTER       accelerate
  4 = SLOWER       brake

Built-in reward (per step in highway-v0 default config):
  + 0.4  for high speed (proportional to normalised ego speed)
  + 0.1  for being in the right-most lane
  - 1.0  if crashed (terminal step)
  Episode max steps: 40  (configurable)
  "Solved": mean episode reward ≈ 20+ (40 steps × ~0.5/step without crashing)
"""

import gymnasium as gym
import highway_env   # registers highway-v0 and friends   # noqa: F401


def default_reward(obs, action, reward, terminated, info):
    """Pass the built-in highway reward through unchanged."""
    return reward


class CustomHighwayEnv(gym.Wrapper):
    """
    Thin wrapper that intercepts step() and applies a pluggable reward function.

    Attributes
    ----------
    reward_fn : callable
        ``(obs, action, reward, terminated, info) -> float``
    reward_fn_version : int
        Incremented each time set_reward_fn() is called.
    reward_fn_code : str
        Source code of the active reward function.
    """

    def __init__(self, **config_overrides):
        env = gym.make("highway-v0", render_mode=None)
        # Apply any config overrides (e.g. duration, lanes_count)
        if config_overrides:
            env.unwrapped.config.update(config_overrides)
            env.unwrapped.reset()
        super().__init__(env)
        self.reward_fn         = default_reward
        self.reward_fn_version = 0
        self.reward_fn_code    = _fn_source(default_reward)

    # ── Public API ────────────────────────────────────────────────────────────
    def set_reward_fn(self, fn, source_code: str) -> None:
        self.reward_fn          = fn
        self.reward_fn_version += 1
        self.reward_fn_code     = source_code

    # ── gym.Wrapper override ──────────────────────────────────────────────────
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            shaped = float(self.reward_fn(obs, action, reward, terminated, info))
        except Exception:
            shaped = reward
        return obs, shaped, terminated, truncated, info


def _fn_source(fn) -> str:
    import inspect
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return repr(fn)
