"""
RewardForge Agent — LunarLander-v3 edition.

Calls Gemini to rewrite the reward function when the RL agent stagnates.

╔══════════════════════════════════════════════════════════════════════════╗
║  DIFF vs CartPole rewardforge_agent.py                                   ║
║  • _PROMPT_TEMPLATE: completely rewritten for LunarLander obs/action     ║
║    space, built-in reward semantics, and shaping guidance.               ║
║  • load_dotenv path: looks one directory up (lunarlander/ → rewardforge/)║
║  All Gemini call logic, safety exec, and smoke-test are IDENTICAL.       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import textwrap
import warnings
from pathlib import Path

# CHANGED FROM CARTPOLE: look one directory up for the shared .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

# ── Gemini configuration ─────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
_MODEL = genai.GenerativeModel("gemini-flash-latest")


# ── Prompt template ───────────────────────────────────────────────────────────
# CHANGED FROM CARTPOLE: entire prompt re-written for LunarLander-v2.
#   Key differences:
#     • 8-D observation space described in detail
#     • 4-action discrete space
#     • Built-in reward scale explained (−200 random → 0 learning → 200 solved)
#     • Shaping advice: distance-to-pad, angle penalty, soft-descent reward
#     • Terminal steps: DO NOT override — env already encodes landing/crash bonus
_PROMPT_TEMPLATE = textwrap.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: LunarLander-v3
Episode ends when: the lander lands/crashes, goes out of bounds, or 1000 steps pass.

Observation space (8 floats, obs[i]):
  obs[0]  x position          0 = landing-pad centre,  range ≈ [−1.5,  1.5]
  obs[1]  y position          0 = ground level,         range ≈ [ 0,    1.5]
  obs[2]  x velocity          positive = moving right,  range ≈ [−2,    2  ]
  obs[3]  y velocity          negative = falling,       range ≈ [−2,    2  ]
  obs[4]  angle               0 = perfectly upright,   range ≈ [−π,    π  ]
  obs[5]  angular velocity                              range ≈ [−5,    5  ]
  obs[6]  left  leg contact   0 = in air, 1 = touching ground
  obs[7]  right leg contact   0 = in air, 1 = touching ground

Action space (discrete — 4 actions):
  0 = do nothing
  1 = fire left-orientation engine  (rotates lander clockwise)
  2 = fire main engine              (thrust upward)
  3 = fire right-orientation engine (rotates lander counter-clockwise)

Built-in per-step reward (this is the `reward` argument you receive):
  • Small positive reward each step for moving closer to the landing pad
  • −0.3 per step the main engine fires
  • −0.03 per step a side engine fires
  • +10 per leg touching the ground (per step)
  • At termination: +100 to +140 for a soft landing, −100 for a crash
  • "Solved" = mean episode reward ≥ 200

Current reward function:
{current_reward_fn_code}

Training progress (mean reward per checkpoint):
{reward_history}

Reference scale: random agent ≈ −200, early learning ≈ −100 to 0, solved ≈ 200.

The agent is struggling. Rewrite the reward function to accelerate learning.

CRITICAL CONSTRAINTS — READ CAREFULLY:
- Use the `reward` argument (built-in per-step reward) as your BASE — add shaping on top.
- COEFFICIENT LIMITS: each shaping term must be ≤ 0.5 in absolute value per step.
  Total shaping must stay within [−1.0, +1.0] per step.
  Reason: the base reward is already in the range [−0.5, +0.5] per step during flight;
  large coefficients will completely overwhelm the signal and confuse the critic.
- CONCRETE EXAMPLE of acceptable coefficients:
    x_pen    = 0.3 * abs(obs[0])    # max ≈ 0.3×1.5 = 0.45  ✓
    ang_pen  = 0.2 * abs(obs[4])    # max ≈ 0.2×π  = 0.63 — reduce to 0.15  ✓
    shaping  = -(x_pen + ang_pen)   # total ≈ -0.5 to 0  ✓
- For TERMINAL steps: return `reward` unchanged — it already contains the
  ±100 landing/crash bonus. Do NOT add extra penalties at termination.
- No imports. Only standard Python arithmetic.
- Good shaping ideas (use SMALL coefficients, 0.1–0.3 range):
    • penalise horizontal distance:    −0.2 * abs(obs[0])
    • penalise tilt angle:             −0.15 * abs(obs[4])
    • reward gentle descent near pad:  +0.2 * max(0, −obs[3]) * (1 − obs[1])
    • small leg-touch bonus:           +0.3 if (obs[6] and obs[7])

Return ONLY valid Python code for a function with this exact signature:

def custom_reward(obs, action, reward, terminated, info):
    # obs: array of 8 floats (see above)
    # reward: the built-in LunarLander per-step reward (float)
    # terminated: True on landing, crash, or out-of-bounds
    # return: float
    return reward

No explanation. No markdown. Just the function.
""")


# ── Public API ────────────────────────────────────────────────────────────────
# IDENTICAL TO CARTPOLE VERSION — only the prompt template above changed.
def request_new_reward_fn(
    current_reward_fn_code: str,
    reward_history: list[tuple[int, float]],
) -> tuple[callable, str] | None:
    """
    Ask Gemini for a better LunarLander reward function.

    Parameters
    ----------
    current_reward_fn_code : str
        Source of the reward function currently in use.
    reward_history : list[(step, mean_reward)]
        Recent training checkpoints (can be empty for ablation_blind).

    Returns
    -------
    (fn, source_code) or None
    """
    history_str = "\n".join(
        f"  Step {step:>7,}: mean_reward = {mr:+.2f}"
        for step, mr in reward_history
    )

    prompt = _PROMPT_TEMPLATE.format(
        current_reward_fn_code=current_reward_fn_code,
        reward_history=history_str or "  (no history provided)",
    )

    print("\n🔮  Calling Gemini for a new reward function …")
    try:
        response = _MODEL.generate_content(prompt)
        raw = response.text.strip()
    except Exception as exc:
        print(f"⚠️  Gemini API error: {exc}")
        return None

    code = _strip_markdown_fences(raw)
    fn   = _safe_compile(code)
    if fn is None:
        return None

    print("✅  Gemini returned a new reward function:\n")
    print(textwrap.indent(code, "    "))
    print()
    return fn, code


# ── Helpers (IDENTICAL TO CARTPOLE VERSION) ──────────────────────────────────
def _strip_markdown_fences(text: str) -> str:
    """Remove ```python … ``` wrappers Gemini sometimes adds."""
    text = re.sub(r"^```(?:python)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _safe_compile(code: str):
    """
    exec() the code in a restricted namespace and return the
    ``custom_reward`` callable, or None on any error.
    Includes a smoke-test with a dummy 8-D obs vector.
    """
    namespace: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, namespace)  # noqa: S102
    except Exception as exc:
        print(f"⚠️  Failed to compile Gemini code: {exc}")
        return None

    fn = namespace.get("custom_reward")
    if fn is None or not callable(fn):
        print("⚠️  Gemini code did not define `custom_reward`.")
        return None

    # Smoke-test with a dummy 8-D obs vector          ← CHANGED FROM CARTPOLE
    # (CartPole used np.zeros(4); LunarLander needs 8-D)
    try:
        import numpy as np
        dummy_obs = np.array([0.0, 0.5, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
        result = fn(dummy_obs, 0, 1.0, False, {})
        float(result)   # must be numeric
    except Exception as exc:
        print(f"⚠️  Smoke test failed for new reward fn: {exc}")
        return None

    return fn
