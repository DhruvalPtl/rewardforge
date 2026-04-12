"""Append llm_single API to curriculum_agent.py."""
from pathlib import Path

addition = r'''

# ── llm_single API (ablation: one pre-trained function, no curriculum stages) ─
import textwrap as _tw

_SINGLE_FN_PROMPT = _tw.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: LunarLander-v3 (discrete actions, episode up to 1000 steps)

Obs (8 floats):
  obs[0] x: centre=0, +-1.5  |  obs[1] y: ground=0, 0-1.5
  obs[2] x-vel +-2            |  obs[3] y-vel +-2 (neg=falling)
  obs[4] angle +-pi (0=up)    |  obs[5] ang-vel +-5
  obs[6] left-leg (0/1)       |  obs[7] right-leg (0/1)
Actions: 0=nothing 1=left 2=main-engine 3=right
Built-in reward: pad proximity, -0.3/step main, +10/step per leg, +100/-100 landing/crash.

Write the SINGLE BEST reward shaping function for LunarLander-v3.
Applied for the ENTIRE 500k-step run. Must handle stability, approach, AND landing.
Prioritise leg contact (obs[6]+obs[7]) and controlled descent near ground.

RULES:
1. Use reward as base. Return reward + shaping.
2. Total shaping: [-1.0, +1.0]/step.
3. If terminated: return reward unchanged.
4. No imports. Pure Python arithmetic.

Return ONLY this function, no explanation, no markdown:

def custom_reward(obs, action, reward, terminated, info):
    return reward
""")


class SingleFnState:
    """State for llm_single warmup blend. alpha: 0.0->1.0 over BLEND_STEPS."""
    __slots__ = ("alpha", "blending", "blend_start_step", "advances")

    def __init__(self):
        self.alpha            = 0.0
        self.blending         = True   # warmup starts immediately at step 0
        self.blend_start_step = 0
        self.advances         = 0      # set to 1 when warmup completes


def request_single_fn():
    """LLM call for llm_single condition. Returns (fn, code) or None."""
    print(f"\n\U0001f52e  Calling Groq ({_MODEL}) for single best reward function ...")
    try:
        time.sleep(10)
        response = _CLIENT.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": _SINGLE_FN_PROMPT}],
            max_tokens=512,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  \u26a0\ufe0f  llm_single LLM call failed: {exc}")
        return None

    from lunarlander.rewardforge_agent import _extract_python_function, _safe_compile
    code = _extract_python_function(raw)
    fn   = _safe_compile(code)
    if fn is None:
        return None
    print("\u2705  llm_single function ready:")
    print(_tw.indent(code, "    "))
    return fn, code


def make_single_blend_fn(fn, state: SingleFnState):
    """Closure: reward + alpha*shaping. alpha warms 0->1 over BLEND_STEPS."""
    def blended(obs, action, reward, terminated, info):
        if terminated:
            return reward
        shaping = fn(obs, action, 0.0, False, info)
        return reward + state.alpha * shaping
    return blended
'''

target = Path(__file__).parent / "lunarlander" / "curriculum_agent.py"
existing = target.read_text(encoding="utf-8")
target.write_text(existing.rstrip() + addition, encoding="utf-8")
print(f"Appended llm_single API to {target}")
