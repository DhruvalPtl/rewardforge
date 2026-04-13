"""
highway/highway_agent.py — LLM reward-shaping agent for autonomous driving (highway-v0).

Provides request_single_fn(): calls the LLM once pre-training with a detailed
description of the kinematic observation matrix and discrete action space.

The prompt targets the two main failure modes in highway-v0:
  1. Speed trap: agent learns IDLE (safe), never speeds up → low reward
  2. Collision loop: agent lane-changes recklessly, crashes frequently

Also exports SingleFnState and make_single_blend_fn for the 20k-step warmup.
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

BLEND_STEPS = 20_000   # warmup: alpha 0→1 over this many steps


# ── LLM prompt ────────────────────────────────────────────────────────────────
_PROMPT = textwrap.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: highway-v0 (highway-env library) — discrete action autonomous driving
The agent drives a car on a multi-lane highway and must drive fast without crashing.

Observation: numpy array of shape (5, 5) — kinematic matrix
  Each row = one vehicle. Row 0 = ego vehicle (always present).
  Columns: [presence, x, y, vx, vy]  (relative to ego, normalised)
    presence  1 if vehicle present, 0 otherwise
    x         longitudinal (forward) offset; positive = ahead of ego
    y         lateral offset; positive = to the left
    vx        longitudinal speed (normalised; ego speed ≈ 0.5-0.8 at cruise)
    vy        lateral speed (0 when not changing lanes)

  Examples:
    obs[0] = [1, 0,   0,   0.7, 0]   ego at origin, speed 0.7 (normalised)
    obs[1] = [1, 0.3, 0,   0.6, 0]   vehicle 30% ahead in same lane, slower
    obs[2] = [1, 0.1, 0.1, 0.5, 0]   vehicle slightly ahead, to the left, slow
    obs[3] = [0, 0,   0,   0,   0]   no vehicle detected
    obs[4] = [0, 0,   0,   0,   0]   no vehicle detected

Actions: Discrete integer
  0 = LANE_LEFT    1 = IDLE    2 = LANE_RIGHT    3 = FASTER    4 = SLOWER

Built-in reward (per step):
  + 0.4 * ego_speed_normalised  (proportional to how fast ego drives)
  + 0.1 if in rightmost lane
  - 1.0 if crashed (episode ends)
  Max episode: 40 steps.  Not crashing = +16+ per episode; crash = ~-1.

Common failure modes to fix:
  1. Speed trap: agent stays IDLE; never speeds up; gets reward ~0.2/step
  2. Reckless merging: agent lane-changes into occupied lane → crash loop

Write the SINGLE BEST reward shaping function for this environment.
Applied for the ENTIRE training run. Encourage:
  1. Safe following distance: penalise if obs[1][0]>0 and obs[1][1] < 0.2
     (front vehicle is close and present)
  2. Smooth driving: small penalty for unnecessary lane changes (action 0 or 2
     when no benefit), and for vy != 0 (lateral drift during lane change)
  3. Speed incentive: small bonus when obs[0][3] > 0.6 (ego going fast)
  4. Collision avoidance: small penalty when any obs[i][0]>0 and distance < 0.1

RULES:
1. Use reward as base. Return reward + shaping.
2. Total shaping: [-0.5, +0.3] per step.
3. If terminated: return reward unchanged (crash penalty preserved).
4. obs is a (5,5) numpy array; use obs[i][j] indexing.
5. action is an integer 0-4. No imports. Pure Python arithmetic.

Return ONLY this function, no explanation, no markdown:

def custom_reward(obs, action, reward, terminated, info):
    return reward
""")


# ── Public API ─────────────────────────────────────────────────────────────────
def request_single_fn() -> tuple | None:
    """Ask the LLM for the best single highway-v0 reward function.
    Returns (fn, code_str) or None on failure."""
    print(f"\n\U0001f52e  Calling Groq ({_MODEL}) for highway-v0 reward function …")
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
    print("\u2705  Highway reward function ready:")
    print(textwrap.indent(code, "    "))
    return fn, code


class SingleFnState:
    """Mutable warmup state; closure reads alpha at call-time."""
    __slots__ = ("alpha", "blending", "blend_start_step", "advances")

    def __init__(self):
        self.alpha            = 0.0
        self.blending         = True
        self.blend_start_step = 0
        self.advances         = 0


def make_single_blend_fn(fn, state: SingleFnState):
    """reward + alpha * shaping(fn). Alpha 0→1 over BLEND_STEPS."""
    def blended(obs, action, reward, terminated, info):
        if terminated:
            return reward
        shaping = fn(obs, action, 0.0, False, info)
        return reward + state.alpha * shaping
    return blended


# ── Private helpers ────────────────────────────────────────────────────────────
def _extract_fn(raw: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*",       "", raw, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:python)?\s*\n", "", text.strip())
    text = re.sub(r"\n```\s*$",            "", text)
    m = re.search(r"(def custom_reward\b.*)", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        return re.sub(r"\n```.*$", "", code, flags=re.DOTALL).strip()
    return text.strip()


def _safe_compile(code: str):
    """exec() custom_reward, smoke-test with a 2D obs, return callable or None."""
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
        # Plausible highway state: ego cruising, one car ahead in same lane
        dummy_obs = np.array([
            [1.0, 0.0,  0.0,  0.7,  0.0],   # ego
            [1.0, 0.25, 0.0,  0.55, 0.0],   # car ahead, same lane, slower
            [1.0, 0.05, 0.12, 0.6,  0.0],   # car to the left, close
            [0.0, 0.0,  0.0,  0.0,  0.0],   # no vehicle
            [0.0, 0.0,  0.0,  0.0,  0.0],   # no vehicle
        ])
        float(fn(dummy_obs, 1, 0.35, False, {}))    # IDLE step
        float(fn(dummy_obs, 3, 0.40, False, {}))    # FASTER step
        float(fn(dummy_obs, 0, -1.0, True,  {}))    # crash step
    except Exception as exc:
        print(f"  \u26a0\ufe0f  Smoke test failed: {exc}")
        return None
    return fn
