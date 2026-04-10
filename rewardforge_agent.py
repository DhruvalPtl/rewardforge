"""
RewardForge Agent — calls Gemini to rewrite the reward function when
the RL agent's learning stagnates.
"""

import os
import re
import textwrap

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── Gemini configuration ────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
_MODEL = genai.GenerativeModel("gemini-flash-latest")

# ── Prompt template ─────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = textwrap.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: CartPole-v1
Observation space: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
Action space: 0 (push left) or 1 (push right)
Max episode length: 500 steps

Current reward function:
{current_reward_fn_code}

Training progress (mean reward per checkpoint):
{reward_history}

The agent is struggling. Rewrite the reward function to improve learning.

CRITICAL CONSTRAINTS:
- The reward must stay centered around +1.0 per step for a surviving agent.
  The default CartPole reward is +1 per step, total ~500 for a perfect episode.
  Your function MUST preserve this approximate scale (return values roughly
  in the range [0.5, 1.5] for non-terminal steps). Do NOT return tiny decimals
  or huge penalties — the PPO value network is calibrated for ~+1/step.
- Use the `reward` argument (the original +1) as the base and ADD small
  shaping bonuses/penalties on top of it.
- For terminal steps you may return 0.0 but NEVER large negative penalties.
- Only use standard Python and math — no imports.

Return ONLY valid Python code for a function with this exact signature:

def custom_reward(obs, action, reward, terminated, info):
    # obs is array of 4 floats
    # reward is the original CartPole reward (+1.0 per step)
    # return a float
    return reward

No explanation. No markdown. Just the function.
""")


# ── Public API ───────────────────────────────────────────────────────────────
def request_new_reward_fn(
    current_reward_fn_code: str,
    reward_history: list[tuple[int, float]],
) -> tuple[callable, str] | None:
    """Ask Gemini for a better reward function.

    Parameters
    ----------
    current_reward_fn_code : str
        Source code of the reward function currently in use.
    reward_history : list[(step, mean_reward)]
        Recent training checkpoints.

    Returns
    -------
    (fn, source_code) or None
        The compiled function and its source, or *None* if Gemini returned
        something unparseable / dangerous.
    """
    history_str = "\n".join(
        f"  Step {step}: mean_reward = {mr:.2f}" for step, mr in reward_history
    )

    prompt = _PROMPT_TEMPLATE.format(
        current_reward_fn_code=current_reward_fn_code,
        reward_history=history_str,
    )

    print("\n🔮  Calling Gemini for a new reward function …")
    try:
        response = _MODEL.generate_content(prompt)
        raw = response.text.strip()
    except Exception as exc:
        print(f"⚠️  Gemini API error: {exc}")
        return None

    # ── Strip markdown fences if Gemini wraps its output ─────────────
    code = _strip_markdown_fences(raw)

    # ── Safety: compile in a restricted namespace ────────────────────
    fn = _safe_compile(code)
    if fn is None:
        return None

    print("✅  Gemini returned a new reward function:\n")
    print(textwrap.indent(code, "    "))
    print()
    return fn, code


# ── Helpers ──────────────────────────────────────────────────────────────────
def _strip_markdown_fences(text: str) -> str:
    """Remove ```python … ``` wrappers that Gemini sometimes adds."""
    text = re.sub(r"^```(?:python)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _safe_compile(code: str):
    """exec() the code string and return the ``custom_reward`` function,
    or *None* if anything goes wrong."""
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

    # Quick smoke test with dummy values
    try:
        import numpy as np
        result = fn(np.zeros(4), 0, 1.0, False, {})
        float(result)  # must be numeric
    except Exception as exc:
        print(f"⚠️  Smoke test failed for new reward fn: {exc}")
        return None

    return fn
