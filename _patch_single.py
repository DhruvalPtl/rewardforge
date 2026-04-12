"""Patch experiment_runner.py: insert _init_single/_tick_single, fix triggers, add analysis."""
import re
from pathlib import Path

src = Path("lunarlander/experiment_runner.py")
text = src.read_text(encoding="utf-8")

# ── 1. Insert _init_single / _tick_single before the module-level save section ──
single_methods = '''
    # ── llm_single management ─────────────────────────────────────────────────
    def _init_single(self) -> None:
        """Pre-training: call LLM once for the best single reward function."""
        result = request_single_fn()
        if result is None:
            print("  \\u26a0\\ufe0f  llm_single failed \\u2014 using base reward.")
            return
        fn, code = result
        self._single_code  = code
        self._single_state = SingleFnState()
        blended = make_single_blend_fn(fn, self._single_state)
        label   = "llm_single:warmup"
        self.train_env.reward_fn      = blended
        self.train_env.reward_fn_code = label
        self._eval_env.reward_fn      = blended
        self._eval_env.reward_fn_code = label
        print(f"  llm_single warmup: alpha 0->1 over {BLEND_STEPS:,} steps")

    def _tick_single(self) -> None:
        """Called every step. Warms up alpha 0->1 over BLEND_STEPS, then stays."""
        st = self._single_state
        if not st.blending:
            return
        elapsed  = self.num_timesteps - st.blend_start_step
        st.alpha = min(1.0, elapsed / BLEND_STEPS)
        if elapsed >= BLEND_STEPS:
            st.blending = False
            st.advances = 1
            print(f"\\n    llm_single: full shaping active @ step {self.num_timesteps:,}")

'''

anchor1 = "\n# \u2550" * 1  # won't match; use exact string below
anchor1 = "\n\n\n# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n# Artifact saving"

if anchor1 in text:
    text = text.replace(anchor1, single_methods + anchor1, 1)
    print("Inserted _init_single/_tick_single")
else:
    print("WARN: anchor1 not found, trying shorter match")
    # Try simpler anchor
    anchor1b = "\n# Artifact saving  (same schema as main.py and CartPole experiment_runner)"
    if anchor1b in text:
        text = text.replace(anchor1b, "\n" + single_methods.strip() + "\n\n" + anchor1b.strip(), 1)
        print("Inserted via anchor1b")
    else:
        print("WARN: could not find insertion anchor")

# ── 2. Fix triggers in run_single ──────────────────────────────────────────────
old_triggers = (
    "    triggers = (cb._curr_state.advances\n"
    "                if condition == \"rewardforge\" and cb._curr_state is not None\n"
    "                else cb.rewrite_count)\n"
)
new_triggers = (
    "    if condition == \"rewardforge\" and cb._curr_state is not None:\n"
    "        triggers = cb._curr_state.advances\n"
    "    elif condition == \"llm_single\" and cb._single_state is not None:\n"
    "        triggers = cb._single_state.advances\n"
    "    else:\n"
    "        triggers = cb.rewrite_count\n"
)
if old_triggers in text:
    text = text.replace(old_triggers, new_triggers, 1)
    print("Fixed triggers")
else:
    print("WARN: triggers anchor not found")

# ── 3. Add rewardforge vs llm_single to generate_analysis ──────────────────────
old_conclusion = "    # ── Plain-English conclusion"
new_analysis = (
    '    # ── Key ablation: does three-stage curriculum beat single LLM fn? ─────────\n'
    '    if "llm_single" in CONDITIONS:\n'
    '        single_best = [r["best_reward"] for r in results if r["condition"] == "llm_single"]\n'
    '        if single_best and rf_best:\n'
    '            _, p_struct = sp_stats.mannwhitneyu(rf_best, single_best, alternative="two-sided")\n'
    '            sig_struct  = "curriculum IS decisive" if p_struct < 0.05 else "curriculum not sig. vs single fn"\n'
    '            lines += ["",\n'
    '                      f"  rewardforge vs llm_single (two-sided): p={p_struct:.4f}"\n'
    '                      f"  <- {sig_struct}"]\n'
    '\n'
    '    # ── Plain-English conclusion'
)
if old_conclusion in text:
    text = text.replace(old_conclusion, new_analysis, 1)
    print("Added llm_single analysis comparison")
else:
    print("WARN: conclusion anchor not found")

src.write_text(text, encoding="utf-8")
print("Patch complete.")
