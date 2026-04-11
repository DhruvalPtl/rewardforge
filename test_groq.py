"""
test_groq.py -- Diagnostic script for Groq API response format.

Shows exactly what qwen3-32b returns, raw and processed, so we can
see what's causing the compile failures in the main agent.

Run from rewardforge/:
    python test_groq.py
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "qwen/qwen3-32b"

# Minimal prompt -- same structure as the real agent
PROMPT = """\
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
def custom_reward(obs, action, reward, terminated, info):
    return reward

Training progress:
  Step  10,000: mean_reward = -433.98
  Step  20,000: mean_reward = -443.56
  Step  30,000: mean_reward = -702.05

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
"""

SEP  = "-" * 60
SEP2 = "=" * 60

print("\n" + SEP2)
print("  Groq API Diagnostic -- model: " + MODEL)
print(SEP2 + "\n")

# -- Make the raw API call ---------------------------------------------------
print("Sending request ...\n")
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_tokens=4500,   # prompt ~700tok + 4500 = 5200 < 6K TPM limit
    temperature=0.2,
)

# -- Inspect every field of the response ------------------------------------
choice = response.choices[0]
msg    = choice.message

print("FINISH REASON : " + str(choice.finish_reason))
print("MODEL         : " + str(response.model))
print()

# Check for separate reasoning_content field (some Groq models use this)
reasoning = getattr(msg, "reasoning_content", None)
print(SEP)
print("msg.reasoning_content  (separate field):")
print(SEP)
if reasoning:
    print(repr(reasoning[:500]))
    print("\n  ... (" + str(len(reasoning)) + " chars total)")
else:
    print("  <None -- model does NOT use a separate reasoning_content field>")

print()
print(SEP)
print("msg.content  (RAW -- exactly what we receive):")
print(SEP)
raw = msg.content.strip()
print(repr(raw[:1000]))      # repr() shows hidden chars, escape sequences, etc.
print("\n  ... (" + str(len(raw)) + " chars total)\n")

print(SEP)
print("msg.content  (PRINTED -- human-readable):")
print(SEP)
print(raw[:1000])
print()

# -- Check for known tags -----------------------------------------------
print(SEP)
print("Tag / keyword detection:")
print(SEP)
print("  Contains <think>        : " + str("<think>"        in raw))
print("  Contains </think>       : " + str("</think>"       in raw))
print("  Contains <thinking>     : " + str("<thinking>"     in raw))
print("  Contains </thinking>    : " + str("</thinking>"    in raw))
print("  Contains ```            : " + str("```"            in raw))
print("  Contains 'def '         : " + str("def "           in raw))
print("  Contains 'def custom_reward': " + str("def custom_reward" in raw))
print()

# -- Simulate each processing step ------------------------------------
print(SEP)
print("Processing pipeline simulation:")
print(SEP)

step = raw

# Step 1: strip <think> blocks
after_think = re.sub(r"<think>.*?</think>\s*", "", step, flags=re.DOTALL).strip()
print("\n[Step 1] After <think> strip  (" + str(len(step)) + " --> " + str(len(after_think)) + " chars):")
print(repr(after_think[:500]))

# Step 2: strip markdown fences
after_fences = re.sub(r"^```(?:python)?\s*\n", "", after_think.strip())
after_fences = re.sub(r"\n```\s*$", "", after_fences).strip()
print("\n[Step 2] After markdown fence strip  (" + str(len(after_think)) + " --> " + str(len(after_fences)) + " chars):")
print(repr(after_fences[:500]))

# Step 3: find def custom_reward
m = re.search(r"(def custom_reward\b.*)", after_fences, re.DOTALL)
extracted = ""
if m:
    extracted = m.group(1).strip()
    extracted = re.sub(r"\n```.*$", "", extracted, flags=re.DOTALL).strip()
    print("\n[Step 3] Extracted function  (" + str(len(extracted)) + " chars):")
    print(extracted)
else:
    print("\n[Step 3] FAIL -- 'def custom_reward' NOT FOUND after processing!")
    print("         Full post-step-2 content:")
    print(after_fences)

# -- Try exec() on the final result ------------------------------------
print()
print(SEP)
print("exec() test:")
print(SEP)
final_code = extracted if m else after_fences
namespace: dict = {}
try:
    exec(final_code, {"__builtins__": __builtins__}, namespace)   # noqa: S102
    fn = namespace.get("custom_reward")
    if fn and callable(fn):
        import numpy as np
        dummy_obs = np.array([0.3, 0.6, -0.1, -0.4, 0.15, 0.05, 0.0, 0.0])
        result = fn(dummy_obs, 0, 1.0, False, {})
        print("  PASS  exec() succeeded!")
        print("        fn(dummy_obs, 0, 1.0, False, {}) = " + str(result))
    else:
        print("  FAIL  exec() ran but custom_reward not found in namespace")
        print("        namespace keys: " + str(list(namespace.keys())))
except Exception as exc:
    print("  FAIL  exec() error: " + str(exc))
    print("\n  Failing code (each line numbered):\n")
    for i, line in enumerate(final_code.splitlines(), 1):
        print("  " + str(i).rjust(3) + ": " + line)

print("\n" + SEP2 + "\n")
