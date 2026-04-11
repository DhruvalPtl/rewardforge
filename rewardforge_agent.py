"""
RewardForge Agent — CartPole-v1 edition  (Groq backend).

Calls an LLM via Groq Cloud to rewrite the reward function when training stagnates.

╔══════════════════════════════════════════════════════════════════════════╗
║  Migration note                                                          ║
║  Originally used google.generativeai (gemini-1.5-flash).               ║
║  Switched to Groq (llama-3.3-70b-versatile) because:                   ║
║    • Groq free tier: 1,000 RPD  vs  Gemini free: 20 RPD               ║
║    • No FutureWarning — google-generativeai package is deprecated       ║
║    • Groq uses OpenAI-compatible API — easy to swap models              ║
╚══════════════════════════════════════════════════════════════════════════╝

Required .env key:
  GROQ_API_KEY=gsk_...
"""

import os
import re
import textwrap

from dotenv import load_dotenv
load_dotenv()

from groq import Groq


# ── Groq client ───────────────────────────────────────────────────────────────
_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL  = "llama-3.3-70b-versatile"   # 30 RPM, 1K RPD, 100K TPD (free tier)


# ── Prompt template ───────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = textwrap.dedent("""\
You are a reward function engineer for Reinforcement Learning.

Environment: CartPole-v1
Episode ends when: pole falls beyond ±12°, cart moves beyond ±2.4, or 500 steps pass.

Observation space (4 floats, obs[i]):
  obs[0]  cart position        range ≈ [−2.4, 2.4]
  obs[1]  cart velocity        range ≈ [−3,   3  ]
  obs[2]  pole angle (radians) range ≈ [−0.21, 0.21]  (0 = upright)
  obs[3]  pole angular velocity range ≈ [−3,   3  ]

Action space: 0 = push left, 1 = push right

Default reward: +1 for every step the pole stays upright.
Maximum episode reward: 500.

Current reward function:
{current_reward_fn_code}

Training progress (mean reward per checkpoint):
{reward_history}

The agent is struggling. Rewrite the reward function to help it learn faster.

CRITICAL CONSTRAINTS:
- Keep total reward centred near +1.0 per step (PPO's value network expects this).
- Shaping terms must be small: total shaping ≤ ±0.5 per step.
  Example: −0.3 * abs(obs[2]) subtracts at most 0.063 (angle ≈ 0.21) — fine.
- No imports. Only standard Python arithmetic.

Return ONLY valid Python code for a function with this exact signature:

def custom_reward(obs, action, reward, terminated, info):
    # obs: array of 4 floats
    # reward: +1.0 if alive, 0.0 if terminated
    # return: float
    return reward

No explanation. No markdown fences. Just the function.
""")


# ── Public API ────────────────────────────────────────────────────────────────
def request_new_reward_fn(
    current_reward_fn_code: str,
    reward_history: list[tuple[int, float]],
) -> tuple[callable, str] | None:
    """
    Ask the LLM for a better CartPole reward function.

    Parameters
    ----------
    current_reward_fn_code : str
        Source of the currently active reward function.
    reward_history : list[(step, mean_reward)]
        Recent evaluation checkpoints. Pass [] for ablation_blind condition.

    Returns
    -------
    (fn, source_code) or None
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

    response = _CLIENT.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=6000,    # TPM ceiling for qwen3-32b free tier; model stops naturally ~3k tokens
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
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
    Robustly extract the ``custom_reward`` function from any LLM response.

    Handles: <think> blocks, <thinking> blocks, markdown fences, prose preamble.
    Strategy: strip wrappers, find ``def custom_reward``, return from there.
    """
    text = raw

    # Guard: if <think> opened but was never closed (response cut by max_tokens),
    # jump straight to searching for the function in the raw text.
    if "<think>" in text and "</think>" not in text:
        m_direct = re.search(r"(def custom_reward\b.*)", text, re.DOTALL)
        if m_direct:
            code = m_direct.group(1).strip()
            return re.sub(r"\n```.*$", "", code, flags=re.DOTALL).strip()
        return ""  # entirely thinking, no code

    # 1. Strip any thinking/reasoning blocks
    text = re.sub(r"^```(?:python)?\s*\n", "", text.strip())
    text = re.sub(r"\n```\s*$",            "", text)

    m = re.search(r"(def custom_reward\b.*)", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        code = re.sub(r"\n```.*$", "", code, flags=re.DOTALL).strip()
        return code

    return text.strip()


def _strip_thinking_tags(text: str) -> str:   # kept for backwards compatibility
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _strip_markdown_fences(text: str) -> str:   # kept for backwards compatibility
    text = re.sub(r"^```(?:python)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$",            "", text)
    return text.strip()


def _safe_compile(code: str):
    """
    exec() the code in a restricted namespace, smoke-test it, and return
    the callable — or None on any failure.
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
        dummy_obs = np.array([0.0, 0.0, 0.05, 0.0])   # slight tilt
        result = fn(dummy_obs, 0, 1.0, False, {})
        float(result)
    except Exception as exc:
        print(f"⚠️  Smoke test failed: {exc}")
        return None

    return fn
