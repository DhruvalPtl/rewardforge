"""
curriculum_agent.py — v6 curriculum reward system for LunarLander-v3.

One pre-training LLM call produces three staged shaping functions:
  stage_1_survive  — stability (don't crash)
  stage_2_approach — proximity (fly toward pad)
  stage_3_land     — commitment (touch down)

The three functions are linearly blended by the experiment callback as
training progresses, controlled by a CurriculumState object.
"""

import re
import textwrap
import time

import numpy as np

# Re-use the Groq client and model from the main agent module
from lunarlander.rewardforge_agent import _CLIENT, _MODEL   # noqa: F401


# ── Stage-transition constants ────────────────────────────────────────────────
BLEND_STEPS  = 20_000   # linear-blend duration when advancing stages
STAGE1_GATE  = -50.0    # min mean_reward to advance stage 0 → 1 before deadline
STAGE1_WAIT  = 200_000  # force stage 0 → 1 advance even if gate never opens


# ── LLM prompt ────────────────────────────────────────────────────────────────
_CURRICULUM_PROMPT = textwrap.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: LunarLander-v3 (discrete actions, episode up to 1000 steps)

Obs (8 floats):
  obs[0] x: centre=0, +-1.5  |  obs[1] y: ground=0, 0-1.5
  obs[2] x-vel +-2            |  obs[3] y-vel +-2 (neg=falling)
  obs[4] angle +-pi (0=up)    |  obs[5] ang-vel +-5
  obs[6] left-leg (0/1)       |  obs[7] right-leg (0/1)
Actions: 0=nothing 1=left 2=main-engine 3=right
Built-in reward: pad proximity, -0.3/step main, +10/step per leg,
  +100/-100 landing/crash at termination.

Write THREE curriculum shaping functions, blended in sequence:

  stage_1_survive  -- Stability first. Penalise tilt (obs[4]), angular velocity
    (obs[5]), and high lateral speed (obs[2]).  Goal: stop crashing.

  stage_2_approach -- Proximity second. Penalise horizontal offset (obs[0]).
    Reward gentle altitude reduction toward the pad.  Goal: get above it.

  stage_3_land     -- Commitment last. Bonus proportional to obs[6]+obs[7]
    (leg contact). Bonus when obs[3]<0 AND obs[1]<0.5 (falling near ground).
    Goal: actually touch down.

Use EXACTLY these function names:

def stage_1_survive(obs, action, reward, terminated, info):
    return reward

def stage_2_approach(obs, action, reward, terminated, info):
    return reward

def stage_3_land(obs, action, reward, terminated, info):
    return reward

RULES (all three functions):
1. Use `reward` as base. Return reward + small shaping bonus.
2. Each shaping term: abs value <= 0.4/step. Total per fn: [-0.8, +0.8]/step.
3. If terminated: return `reward` unchanged.
4. No imports. Pure Python arithmetic only.

Return ONLY the three functions in order. No explanation, no markdown:
""")


# ── Public API ────────────────────────────────────────────────────────────────
def request_curriculum_fns() -> tuple | None:
    """
    Ask the LLM for three curriculum shaping functions in a single call.

    Returns
    -------
    (survive_fn, approach_fn, land_fn, codes_dict) or None on any failure.
    codes_dict = {"survive": str, "approach": str, "land": str}
    """
    print(f"\n\U0001f393  Calling Groq ({_MODEL}) for curriculum functions"
          " (survive / approach / land) …")
    try:
        time.sleep(10)   # rate-limit buffer before this pre-training call
        response = _CLIENT.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _CURRICULUM_PROMPT}],
            max_tokens=1024,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  \u26a0\ufe0f  Curriculum LLM call failed: {exc}")
        return None

    survive_code  = _extract_named_fn(raw, "stage_1_survive")
    approach_code = _extract_named_fn(raw, "stage_2_approach")
    land_code     = _extract_named_fn(raw, "stage_3_land")

    survive_fn  = _safe_compile_named(survive_code,  "stage_1_survive")
    approach_fn = _safe_compile_named(approach_code, "stage_2_approach")
    land_fn     = _safe_compile_named(land_code,     "stage_3_land")

    missing = [n for n, f in [("stage_1_survive",  survive_fn),
                               ("stage_2_approach", approach_fn),
                               ("stage_3_land",     land_fn)] if f is None]
    if missing:
        print(f"  \u26a0\ufe0f  Curriculum parse failed for stages: {missing}")
        return None

    print("\u2705  Curriculum functions ready:\n")
    for label, code in [("survive",  survive_code),
                        ("approach", approach_code),
                        ("land",     land_code)]:
        print(f"  [{label}]")
        print(textwrap.indent(code, "    "))
    print()
    return survive_fn, approach_fn, land_fn, {
        "survive":  survive_code,
        "approach": approach_code,
        "land":     land_code,
    }


def make_blended_fn(fns: tuple, state: "CurriculumState"):
    """
    Return a closure that reads state.alpha at call-time and blends the three
    stage functions linearly.

    Calling convention is identical to custom_reward so CustomLunarLander
    needs no changes.  The closure is installed once; alpha updates propagate
    automatically because state is captured by reference.

    alpha = 0.0 → pure survive
    alpha = 1.0 → pure approach
    alpha = 2.0 → pure land
    intermediate values → linear blend of adjacent stages
    """
    survive_fn, approach_fn, land_fn = fns

    def blended(obs, action, reward, terminated, info):
        if terminated:
            return reward                      # never add shaping at termination
        a = state.alpha
        if a <= 0.0:
            s = survive_fn(obs,  action, 0.0, False, info)
        elif a >= 2.0:
            s = land_fn(obs,    action, 0.0, False, info)
        elif a < 1.0:
            s = ((1.0 - a) * survive_fn(obs,  action, 0.0, False, info)
                 + a        * approach_fn(obs, action, 0.0, False, info))
        else:
            b = a - 1.0
            s = ((1.0 - b) * approach_fn(obs, action, 0.0, False, info)
                 + b        * land_fn(obs,    action, 0.0, False, info))
        return reward + s

    return blended


class CurriculumState:
    """
    Mutable state object shared between the curriculum closure and the callback.
    The closure reads .alpha at call-time so no reinstallation is needed.
    """
    __slots__ = ("alpha", "stage", "blending",
                 "blend_from", "blend_to", "blend_start_step",
                 "stage0_start_step", "advances")

    def __init__(self):
        self.alpha            = 0.0    # 0.0=survive … 2.0=land
        self.stage            = 0      # current integer stage
        self.blending         = False
        self.blend_from       = 0.0
        self.blend_to         = 1.0
        self.blend_start_step = 0
        self.stage0_start_step = 0
        self.advances         = 0      # how many stage transitions completed


# ── Private helpers ───────────────────────────────────────────────────────────
def _extract_named_fn(raw: str, fn_name: str) -> str:
    """Extract a named function block from an LLM response."""
    pattern = rf"(def {re.escape(fn_name)}\s*\(.*?)(?=\ndef |\Z)"
    m = re.search(pattern, raw, re.DOTALL)
    if not m:
        return ""
    code = m.group(1).strip()
    return re.sub(r"\n```.*$", "", code, flags=re.DOTALL).strip()


def _safe_compile_named(code: str, fn_name: str):
    """Compile a named function and smoke-test it. Returns callable or None."""
    if not code:
        print(f"  \u26a0\ufe0f  No code found for {fn_name}")
        return None
    namespace: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, namespace)   # noqa: S102
    except Exception as exc:
        print(f"  \u26a0\ufe0f  Compile error in {fn_name}: {exc}")
        return None
    fn = namespace.get(fn_name)
    if fn is None or not callable(fn):
        print(f"  \u26a0\ufe0f  {fn_name} not found after exec (got: {list(namespace)})")
        return None
    try:
        dummy = np.array([0.3, 0.6, -0.1, -0.4, 0.15, 0.05, 0.0, 0.0])
        float(fn(dummy, 0, 1.0,   False, {}))   # flight step
        float(fn(dummy, 0, 100.0, True,  {}))   # terminal step
    except Exception as exc:
        print(f"  \u26a0\ufe0f  Smoke test failed for {fn_name}: {exc}")
        return None
    return fn
