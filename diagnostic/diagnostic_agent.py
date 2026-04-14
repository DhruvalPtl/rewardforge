"""
diagnostic/diagnostic_agent.py — Behavioral-grounded LLM reward designer.

Unlike bipedal_agent.py (which prompts the LLM with only the environment
description), this agent sends a full BehaviorReport from actual agent rollouts.

The LLM therefore knows:
  ✅ What failure mode is actually occurring RIGHT NOW
  ✅ Which obs dimensions are already good (no need to re-reward those)
  ✅ Quantitative targets (e.g. "gait rhythm is 18%, needs to reach 60%")

This closes the loop: diagnose → prescribe → verify → re-diagnose.
"""

import os
import re
import textwrap
import time

import numpy as np
from dotenv import load_dotenv
from groq import Groq

from diagnostic.behavior_audit import BehaviorReport

load_dotenv()
_CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL  = "llama-3.3-70b-versatile"

BLEND_STEPS = 20_000   # warmup alpha 0→1 over this many steps after install

# ── Prompt ────────────────────────────────────────────────────────────────────
_ENV_HEADER = textwrap.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: BipedalWalker-v3 (continuous control, episode up to 1,600 steps)
The agent controls a 2D bipedal robot walking forward on flat terrain.

Key observation indices (all 24 floats):
  obs[0]  hull angle          0 = upright.  +/- = forward/backward lean
  obs[1]  hull angular vel
  obs[2]  hull forward velocity  (target for walking ≈ 0.5 – 1.0)
  obs[3]  hull vertical velocity
  obs[4..7]  hip/knee angles & speeds (leg 1)
  obs[8]  leg 1 ground contact   (0 or 1)
  obs[9..12] hip/knee angles & speeds (leg 2)
  obs[13] leg 2 ground contact   (0 or 1)
  obs[14-23] 10 lidar readings

Built-in reward: + forward velocity − energy cost.  −100 if hull hits ground.
""")

_DIAG_PROMPT_TEMPLATE = textwrap.dedent("""\
{env_header}
════════════════════════════════════════════════════════
BEHAVIORAL AUDIT — {n_episodes} real episodes after {train_steps:,} training steps
════════════════════════════════════════════════════════
{diagnosis_str}
════════════════════════════════════════════════════════

YOUR TASK — write a TARGETED reward shaping function:

  • Address ONLY the diagnosed bottleneck: {bottleneck}
  • Do NOT add rewards for things that are already working
    (e.g. if hull angle is fine, skip posture shaping)
  • Bottleneck-to-shaping mapping:
      SHUFFLE  → penalise vx < 0.25; bonus when vx > 0.5
      BALANCE  → penalise abs(obs[0]) > 0.2; bonus near 0
      STALL    → penalise prolonged low-vx stretches; obs[2] momentum bonus
      GAIT     → bonus when obs[8] != obs[13] (alternating contacts)
  • Total shaping capped at [−0.4, +0.4] per step
  • If terminated: return reward unchanged (preserves −100 fall penalty)

RULES:
  1. obs and action are numpy arrays — use obs[i] indexing.
  2. No imports. Pure Python arithmetic only.
  3. Return reward + shaping.

Return ONLY the function — no markdown, no explanations:

def custom_reward(obs, action, reward, terminated, info):
    return reward
""")


# ── Public API ─────────────────────────────────────────────────────────────────
def request_diagnostic_fn(report: BehaviorReport, train_steps: int) -> tuple | None:
    """
    Call the LLM with a behavioral audit report.
    Returns (fn, code_str) or None on failure.
    """
    prompt = _DIAG_PROMPT_TEMPLATE.format(
        env_header    = _ENV_HEADER,
        n_episodes    = report.n_episodes,
        train_steps   = train_steps,
        diagnosis_str = report.diagnosis_str(),
        bottleneck    = report.bottleneck,
    )

    print(f"\n  ?  Behavioral Audit ? LLM diagnostic call")
    print(f"      bottleneck     = {report.bottleneck}")
    print(f"      shuffle        = {report.shuffle_detected}")
    print(f"      velocity_coll  = {report.velocity_collapse_det}")
    print(f"      gait_rhythm    = {report.gait_rhythm_score:.1%}")
    print(f"      mean_vx        = {report.mean_forward_velocity:.4f}")
    print(f"      hull_abs       = {report.mean_hull_angle_abs:.4f}")

    try:
        time.sleep(10)
        response = _CLIENT.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  ??  Groq call failed: {exc}")
        return None

    code = _extract_fn(raw)
    fn   = _safe_compile(code)
    if fn is None:
        return None

    print("  ?  Diagnostic reward function ready:")
    print(textwrap.indent(code, "      "))
    return fn, code


# ── Warmup blend (same pattern as bipedal_agent / curriculum_agent) ───────────
class SingleFnState:
    """Mutable warmup state; closure reads alpha at call-time."""
    __slots__ = ("alpha", "blending", "blend_start_step", "advances")

    def __init__(self, start_step: int = 0):
        self.alpha            = 0.0
        self.blending         = True
        self.blend_start_step = start_step
        self.advances         = 0


def make_blend_fn(fn, state: SingleFnState):
    """Closure: reward + alpha * shaping(fn).  alpha 0→1 over BLEND_STEPS."""
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
    """exec(), smoke-test on plausible BipedalWalker states, return fn or None."""
    if not code:
        print("  ??  Empty code from LLM.")
        return None
    namespace: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, namespace)    # noqa: S102
    except Exception as exc:
        print(f"  ??  Compile error: {exc}")
        return None
    fn = namespace.get("custom_reward")
    if fn is None or not callable(fn):
        print("  ??  custom_reward not found after exec.")
        return None
    try:
        # walking state: upright, forward moving, right-leg down
        dummy_obs    = np.array([0.02, 0.1, 0.7, 0.0,
                                 -0.1, 0.2, -0.3, 0.1,  1.0,
                                  0.1, 0.2, -0.3, 0.1,  0.0,
                                  0.9, 0.8, 0.7, 0.6, 0.5,
                                  0.4, 0.3, 0.2, 0.1, 0.05])
        dummy_action = np.array([0.1, 0.1, -0.1, 0.1])
        float(fn(dummy_obs, dummy_action,   1.5, False, {}))   # normal step
        float(fn(dummy_obs, dummy_action, -100., True,  {}))   # fall step
    except Exception as exc:
        print(f"  ??  Smoke test failed: {exc}")
        return None
    return fn
