"""
bipedal/bipedal_agent.py — LLM reward-shaping agent for BipedalWalker-v3.

Provides request_single_fn(): calls the LLM once before training with a
detailed BipedalWalker obs/action description and returns a compiled
custom_reward function + its source code.

Also exports make_single_blend_fn() and SingleFnState for the 20k-step
warmup blend used in the experiment runner.
"""

import re
import textwrap
import time

import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

import os
_CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL  = "llama-3.3-70b-versatile"

# ── Warmup blend constants ─────────────────────────────────────────────────────
BLEND_STEPS = 20_000   # alpha 0→1 linear warmup duration


# ── LLM prompt ────────────────────────────────────────────────────────────────
_PROMPT = textwrap.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: BipedalWalker-v3 (continuous control, episode up to 1600 steps)
The agent controls a 2D bipedal robot that must walk forward on flat terrain.

Observation space (24 floats):
  obs[0]  hull angle          [-pi, pi]  (0 = perfectly upright)
  obs[1]  hull angular vel
  obs[2]  hull velocity x     forward speed  (target approx 1.0, max ~3)
  obs[3]  hull velocity y     vertical speed
  obs[4]  hip joint 1 angle
  obs[5]  hip joint 1 speed
  obs[6]  knee joint 1 angle  (0=straight, negative=bent)
  obs[7]  knee joint 1 speed
  obs[8]  leg 1 ground contact   0 or 1
  obs[9]  hip joint 2 angle
  obs[10] hip joint 2 speed
  obs[11] knee joint 2 angle
  obs[12] knee joint 2 speed
  obs[13] leg 2 ground contact   0 or 1
  obs[14..23] 10 lidar rangefinder readings (forward-facing, normalised 0-1)

Action space: 4 floats in [-1, 1] — torques for [hip1, knee1, hip2, knee2]

Built-in reward per step:
  + forward velocity (obs[2])          main progress signal
  - 0.00025 * sum(action**2)           energy cost
  - 100 on fall (hull ground contact)  episode ends immediately

Common failure modes to fix:
  1. "Shuffle trap": tiny steps, barely positive reward, no real walking
  2. "Fall loop": repeated falling for -100, agent never learns balance

Write the SINGLE BEST reward shaping function for this environment.
Applied for the ENTIRE 500k-step run. Must encourage:
  1. Upright posture: reward when abs(obs[0]) is small
  2. Forward velocity: reward when obs[2] > 0.3 (shuffle threshold)
  3. Gait rhythm: reward when exactly ONE leg is on ground
      (obs[8] != obs[13] means alternating steps — natural gait)
  4. Do NOT re-penalise energy (already in built-in reward)

RULES:
1. Use reward as base. Return reward + shaping.
2. Total shaping bonus/penalty: [-0.5, +0.5] per step maximum.
3. If terminated: return reward unchanged (preserve -100 fall penalty).
4. obs and action are numpy arrays — use obs[i] indexing directly.
5. No imports. Pure Python arithmetic only (abs(), min(), max() are fine).

Return ONLY this function, no explanation, no markdown:

def custom_reward(obs, action, reward, terminated, info):
    return reward
""")


# ── Public API ─────────────────────────────────────────────────────────────────
def request_single_fn() -> tuple | None:
    """
    Ask the LLM for the best single BipedalWalker reward function.
    Returns (fn, code_str) or None on failure.
    """
    print(f"\n\U0001f52e  Calling Groq ({_MODEL}) for BipedalWalker reward function ?")
    try:
        time.sleep(10)
        response = _CLIENT.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _PROMPT}],
            max_tokens=512,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  \u26a0\ufe0f  LLM call failed: {exc}")
        return None

    code = _extract_fn(raw)
    fn   = _safe_compile(code)
    if fn is None:
        return None
    print("\u2705  BipedalWalker reward function ready:")
    print(textwrap.indent(code, "    "))
    return fn, code


class SingleFnState:
    """Mutable warmup state — closure reads alpha at call-time."""
    __slots__ = ("alpha", "blending", "blend_start_step", "advances")

    def __init__(self):
        self.alpha            = 0.0
        self.blending         = True    # warmup starts at step 0
        self.blend_start_step = 0
        self.advances         = 0       # set to 1 when warmup finishes


def make_single_blend_fn(fn, state: SingleFnState):
    """
    Return a closure: reward + alpha * shaping(fn).
    alpha warms 0→1 over BLEND_STEPS then stays at 1 permanently.
    Pure shaping delta extracted by passing reward=0.0.
    """
    def blended(obs, action, reward, terminated, info):
        if terminated:
            return reward                      # preserve fall penalty unchanged
        shaping = fn(obs, action, 0.0, False, info)
        return reward + state.alpha * shaping
    return blended


# ── Private helpers ────────────────────────────────────────────────────────────
def _extract_fn(raw: str) -> str:
    """Extract custom_reward function from LLM response (handles fences/prose)."""
    # strip thinking tags
    text = re.sub(r"<think>.*?</think>\s*",       "", raw, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)
    # strip markdown fences
    text = re.sub(r"^```(?:python)?\s*\n", "", text.strip())
    text = re.sub(r"\n```\s*$",            "", text)
    # find function
    m = re.search(r"(def custom_reward\b.*)", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        return re.sub(r"\n```.*$", "", code, flags=re.DOTALL).strip()
    return text.strip()


def _safe_compile(code: str):
    """exec() custom_reward, smoke-test it, return callable or None."""
    if not code:
        print("  \u26a0\ufe0f  Empty code from LLM.")
        return None
    namespace: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, namespace)   # noqa: S102
    except Exception as exc:
        print(f"  \u26a0\ufe0f  Compile error: {exc}")
        return None
    fn = namespace.get("custom_reward")
    if fn is None or not callable(fn):
        print("  \u26a0\ufe0f  custom_reward not found after exec.")
        return None
    try:
        # Plausible mid-walk state: upright, moving forward, right leg down
        dummy_obs    = np.array([0.02, 0.1, 0.8, 0.0,
                                 -0.1, 0.2, -0.3, 0.1, 1.0,
                                  0.1, 0.2, -0.3, 0.1, 0.0,
                                  0.9, 0.8, 0.7, 0.6, 0.5,
                                  0.4, 0.3, 0.2, 0.1, 0.05])
        dummy_action = np.array([0.1, 0.1, -0.1, 0.1])
        float(fn(dummy_obs, dummy_action, 1.5,   False, {}))  # walking step
        float(fn(dummy_obs, dummy_action, -100.0, True,  {})) # fall step
    except Exception as exc:
        print(f"  \u26a0\ufe0f  Smoke test failed: {exc}")
        return None
    return fn
