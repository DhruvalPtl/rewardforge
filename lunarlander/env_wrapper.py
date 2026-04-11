"""
CustomLunarLander — gymnasium wrapper around LunarLander-v3 whose reward
function can be hot-swapped at runtime by the RewardForge agent.

╔══════════════════════════════════════════════════════════════════════════╗
║  DIFF vs CartPole env_wrapper.py                                         ║
║  • gym.make target : "CartPole-v1"  →  "LunarLander-v3"                 ║
║  • Observation space: 4-D           →  8-D  (see table below)           ║
║  • Action space    : 2 discrete     →  4 discrete                       ║
║  • default_reward docstring updated to describe LunarLander semantics   ║
║  All other code is IDENTICAL to the CartPole version.                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Observation space (8 floats)
────────────────────────────────────────────────────
  obs[0]   x position          0 = landing-pad centre, ≈ [−1.5,  1.5]
  obs[1]   y position          0 = ground,             ≈ [ 0,    1.5]
  obs[2]   x velocity                                  ≈ [−2,    2  ]
  obs[3]   y velocity          negative = falling       ≈ [−2,    2  ]
  obs[4]   angle               0 = upright             ≈ [−π,    π  ]
  obs[5]   angular velocity                             ≈ [−5,    5  ]
  obs[6]   left  leg contact   0 or 1
  obs[7]   right leg contact   0 or 1

Action space (4 discrete actions)
────────────────────────────────────────────────────
  0  do nothing
  1  fire left-orientation engine   (turns right)
  2  fire main engine               (thrust upward)
  3  fire right-orientation engine  (turns left)

Built-in reward breakdown
────────────────────────────────────────────────────
  +10      per leg touching the ground (per step)
  +100…140 soft landing bonus (on termination)
  −100     crash penalty (on termination)
  −0.3     per step the main engine fires
  −0.03    per step a side engine fires
  Environment is "solved" when mean episode reward ≥ 200.
"""

import gymnasium as gym


# ── Default reward function — passes the built-in LunarLander reward through ─
def default_reward(obs, action, reward, terminated, info):
    """
    Standard LunarLander reward: return the env's built-in per-step reward
    unchanged.  Includes landing/crash bonus at terminal steps.

    CHANGED FROM CARTPOLE VERSION:
        CartPole default simply returns `reward` (+1/step to stay upright).
        LunarLander default also returns `reward` unchanged, but that value
        already encodes engine penalties, leg bonuses, and terminal signals —
        so the scale is very different (−200 random → +200 solved).
    """
    return reward


class CustomLunarLander(gym.Wrapper):
    """
    Thin wrapper that intercepts ``step()`` and applies a pluggable reward
    function, allowing RewardForge to swap it at runtime.

    Attributes
    ----------
    reward_fn : callable
        ``(obs, action, reward, terminated, info) -> float``
    reward_fn_version : int
        Monotonically increasing counter; bumped every time set_reward_fn()
        is called.
    reward_fn_code : str
        Source code of the currently active reward function (sent to Gemini /
        saved to disk).

    CHANGED FROM CARTPOLE VERSION:
        Wraps "LunarLander-v2" instead of "CartPole-v1".
        Everything else — pluggable reward pattern, fallback logic — is
        identical to CustomCartPole.
    """

    def __init__(self, **kwargs):
        env = gym.make("LunarLander-v3", **kwargs)   # ← CHANGED: env name
        super().__init__(env)

        self.reward_fn         = default_reward
        self.reward_fn_version = 0
        self.reward_fn_code    = _fn_source(default_reward)

    # ── Public API ───────────────────────────────────────────────────────────
    def set_reward_fn(self, fn, source_code: str) -> None:
        """Replace the active reward function at runtime."""
        self.reward_fn         = fn
        self.reward_fn_version += 1
        self.reward_fn_code    = source_code

    # ── gym.Wrapper override ─────────────────────────────────────────────────
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            shaped = float(self.reward_fn(obs, action, reward, terminated, info))
        except Exception:
            shaped = reward   # fall back to original if custom fn crashes
        return obs, shaped, terminated, truncated, info


# ── Helpers ──────────────────────────────────────────────────────────────────
def _fn_source(fn) -> str:
    """Best-effort source retrieval; falls back to repr()."""
    import inspect
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return repr(fn)
