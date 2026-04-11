"""
RewardForge Agent — LunarLander-v3 edition  (Groq backend).

Calls an LLM via Groq Cloud to rewrite the reward function when training stagnates.

╔══════════════════════════════════════════════════════════════════════════╗
║  DIFF vs CartPole rewardforge_agent.py                                   ║
║  • Backend: google.generativeai  →  groq  (OpenAI-compatible)           ║
║  • Model: gemini-flash            →  llama-3.3-70b-versatile             ║
║  • _PROMPT_TEMPLATE: fully rewritten for LunarLander obs/action space    ║
║  • load_dotenv path: looks one directory up (lunarlander/ → rewardforge/)║
║  • Smoke-test obs: 4-D zeros  →  8-D plausible LunarLander state        ║
║  All call logic, safety exec, and retry handling are IDENTICAL.          ║
╚══════════════════════════════════════════════════════════════════════════╝

Why Groq?
  • Free tier: 1,000 requests/day   (Gemini free: 20/day  — 50× less)
  • llama-3.3-70b-versatile: excellent at structured code generation
  • OpenAI-compatible API → easy to swap models or providers later

Required .env key:
  GROQ_API_KEY=gsk_...
  (get one at https://console.groq.com/keys)
"""

import os
import re
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# CHANGED FROM CARTPOLE: look one directory up for the shared .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from groq import Groq   # noqa: E402  (loaded after dotenv)


# ── Groq client configuration ────────────────────────────────────────────────
# CHANGED FROM CARTPOLE: Groq SDK instead of google.generativeai
_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# llama-3.3-70b-versatile: 30 RPM, 1K RPD, 12K TPM, 100K TPD (free tier)
# No thinking overhead -- responses are compact, ~850 tokens total per call.
_MODEL = "llama-3.3-70b-versatile"


# ── Prompt template ───────────────────────────────────────────────────────────
# CHANGED FROM CARTPOLE: fully rewritten for LunarLander-v3:
#   • 8-D observation space with ranges
#   • 4-action discrete space
#   • Built-in reward scale (−200 random → 0 learning → 200 solved)
#   • Tight coefficient limits to prevent scale collapse (learned from v1 run)
#   • Terminal step handling: pass through unchanged
_PROMPT_TEMPLATE = textwrap.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: LunarLander-v3 (discrete actions, episode up to 1000 steps)

Obs (8 floats): x-pos, y-pos, x-vel, y-vel, angle, ang-vel, left-leg, right-leg
  obs[0] x: centre=0, range +-1.5  |  obs[1] y: ground=0, range 0-1.5
  obs[2] x-vel range +-2           |  obs[3] y-vel range +-2 (neg=falling)
  obs[4] angle range +-pi (0=up)   |  obs[5] ang-vel range +-5
  obs[6] left-leg contact (0/1)    |  obs[7] right-leg contact (0/1)

Actions: 0=nothing 1=left-engine 2=main-engine 3=right-engine

Built-in reward (= the `reward` arg): +/- for pad proximity, -0.3/step main,
  -0.03/step side, +10/step per leg contact, +100/-100 landing/crash at termination.
Scale: random=-200, learning=-100 to 0, solved=+200.

Current reward function:
{current_reward_fn_code}

Training progress:
{reward_history}

The agent is struggling. Write a new reward function to accelerate learning.

RULES:
1. Use `reward` as the base. Add shaping bonuses ON TOP.
2. Each shaping term: abs value <= 0.5/step. Total shaping: [-1.0, +1.0]/step.
   Safe example: shaping = -0.2*abs(obs[0]) - 0.15*abs(obs[4])  (total ~-0.5 max)
3. If terminated: return `reward` unchanged (it already has the +-100 landing bonus).
4. No imports. Pure Python arithmetic only.

Return ONLY the Python function. No explanation, no markdown:

def custom_reward(obs, action, reward, terminated, info):
    return reward
""")


# ── Public API ────────────────────────────────────────────────────────────────
def request_new_reward_fn(
    current_reward_fn_code: str,
    reward_history: list[tuple[int, float]],
) -> tuple[callable, str] | None:
    """
    Ask the LLM (via Groq) for a better LunarLander reward function.

    Parameters
    ----------
    current_reward_fn_code : str
        Source of the reward function currently in use.
    reward_history : list[(step, mean_reward)]
        Recent training checkpoints.  Pass [] for ablation_blind condition.

    Returns
    -------
    (fn, source_code) or None
        None means the LLM returned unusable code; caller keeps current fn.
    """
    history_str = "\n".join(
        f"  Step {step:>7,}: mean_reward = {mr:+.2f}"
        for step, mr in reward_history
    ) or "  (no history provided)"

    prompt = _PROMPT_TEMPLATE.format(
        current_reward_fn_code=current_reward_fn_code,
        reward_history=history_str,
    )

    print(f"\n🔮  Calling Groq ({_MODEL}) for a new reward function …")

    # CHANGED FROM CARTPOLE / GEMINI VERSION:
    # Groq uses OpenAI-compatible chat completions instead of generate_content()
    response = _CLIENT.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,     # llama has NO thinking overhead; code is ~100 tokens.
                            # 1024 is plenty and saves ~6000 tokens of TPD per call.
        temperature=0.2,    # low temperature -> deterministic, well-structured code
    )
    raw = response.choices[0].message.content.strip()
    # Use a robust extractor that handles <think> blocks, markdown fences,
    # and prose preamble regardless of model/provider formatting.
    code = _extract_python_function(raw)
    fn   = _safe_compile(code)
    if fn is None:
        return None

    print("✅  LLM returned a new reward function:\n")
    print(textwrap.indent(code, "    "))
    print()
    return fn, code


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_python_function(raw: str) -> str:
    """
    Robustly extract the ``custom_reward`` function body from any LLM response.

    Handles all known response formats:
      • <think>...<\think> blocks      (qwen3, deepseek-r1 inline thinking)
      • <thinking>...</thinking> blocks  (some anthropic-style models)
      • Markdown fences \`\`\`python ... \`\`\`
      • Explanatory prose before/after the function
      • Any combination of the above

    Strategy: strip all known wrappers, then find ``def custom_reward`` and
    return everything from that line onward (trimming trailing fences/prose).
    """
    text = raw

    # Guard: if <think> opened but was never closed (response cut by max_tokens),
    # jump straight to searching for the function -- the code won't be there, but
    # at least we don't try to exec() the thinking text.
    if "<think>" in text and "</think>" not in text:
        # Truncated thinking block -- look for the function after </think> fails.
        # Try finding the function anywhere in the raw text first.
        m_direct = re.search(r"(def custom_reward\b.*)", text, re.DOTALL)
        if m_direct:
            code = m_direct.group(1).strip()
            return re.sub(r"\n```.*$", "", code, flags=re.DOTALL).strip()
        # No code found -- response was entirely thinking, return empty string
        # so _safe_compile logs a clean 'no custom_reward defined' error.
        return ""

    # 1. Strip any thinking/reasoning blocks (multiple tag flavours)
    text = re.sub(r"<think>.*?</think>\s*",       "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)

    # 2. Strip outermost markdown code fences
    text = re.sub(r"^```(?:python)?\s*\n", "", text.strip())
    text = re.sub(r"\n```\s*$",            "", text)

    # 3. Find the function definition — discard any prose before it
    m = re.search(r"(def custom_reward\b.*)", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        # Trim anything after a trailing ``` or explanatory line
        code = re.sub(r"\n```.*$", "", code, flags=re.DOTALL).strip()
        return code

    # Fall-back: return whatever we have (safe_compile will catch real errors)
    return text.strip()


def _strip_thinking_tags(text: str) -> str:   # kept for backwards compatibility
    """Thin wrapper — prefer _extract_python_function for new code."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _strip_markdown_fences(text: str) -> str:   # kept for backwards compatibility
    """Remove ```python … ``` wrappers."""
    text = re.sub(r"^```(?:python)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$",            "", text)
    return text.strip()


def _safe_compile(code: str):
    """
    exec() the code in a restricted namespace, run a smoke-test, and
    return the callable — or None on any failure.

    Smoke-test uses a plausible mid-flight LunarLander state (8-D).
    CHANGED FROM CARTPOLE: CartPole used np.zeros(4); LunarLander needs 8-D.
    """
    namespace: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, namespace)   # noqa: S102
    except Exception as exc:
        print(f"⚠️  Failed to compile LLM code: {exc}")
        return None

    fn = namespace.get("custom_reward")
    if fn is None or not callable(fn):
        print("⚠️  LLM code did not define `custom_reward`.")
        return None

    try:
        import numpy as np
        # Plausible mid-flight state: lander slightly off-centre, tilted, descending
        dummy_obs = np.array([0.3, 0.6, -0.1, -0.4, 0.15, 0.05, 0.0, 0.0])
        result = fn(dummy_obs, 0, 1.0, False, {})
        float(result)   # must be numeric
    except Exception as exc:
        print(f"⚠️  Smoke test failed: {exc}")
        return None

    return fn
