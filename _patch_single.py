"""Patch generate_analysis to add catastrophic failure rate and focused comparison table."""
from pathlib import Path

src = Path("lunarlander/experiment_runner.py")
text = src.read_text(encoding="utf-8")

old = (
    '    # \u2500\u2500 Key ablation: does three-stage curriculum beat single LLM fn? \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\r\n'
    '    if "llm_single" in CONDITIONS:\r\n'
    '        single_best = [r["best_reward"] for r in results if r["condition"] == "llm_single"]\r\n'
    '        if single_best and rf_best:\r\n'
    '            _, p_struct = sp_stats.mannwhitneyu(rf_best, single_best, alternative="two-sided")\r\n'
    '            sig_struct  = "curriculum IS decisive" if p_struct < 0.05 else "curriculum not sig. vs single fn"\r\n'
    '            lines += ["",\r\n'
    '                      f"  rewardforge vs llm_single (two-sided): p={p_struct:.4f}"\r\n'
    '                      f"  <- {sig_struct}"]\r\n'
    '\r\n'
)

new = (
    '    # \u2500\u2500 Key ablation: does three-stage curriculum beat single LLM fn? \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\r\n'
    '    if "llm_single" in CONDITIONS:\r\n'
    '        single_best  = [r["best_reward"]  for r in results if r["condition"] == "llm_single"]\r\n'
    '        single_final = [r["final_reward"] for r in results if r["condition"] == "llm_single"]\r\n'
    '        single_std   = [r["final_std"]    for r in results if r["condition"] == "llm_single"]\r\n'
    '        if single_best and rf_best:\r\n'
    '            _, p_struct = sp_stats.mannwhitneyu(rf_best, single_best, alternative="two-sided")\r\n'
    '            sig_struct  = "curriculum IS decisive" if p_struct < 0.05 else "curriculum not sig. vs single fn"\r\n'
    '\r\n'
    '            rf_final    = [r["final_reward"] for r in results if r["condition"] == "rewardforge"]\r\n'
    '            rf_std_vals = [r["final_std"]    for r in results if r["condition"] == "rewardforge"]\r\n'
    '            rf_failures     = sum(1 for x in rf_best     if x < 0)\r\n'
    '            single_failures = sum(1 for x in single_best if x < 0)\r\n'
    '\r\n'
    '            col_rf  = "rewardforge"\r\n'
    '            col_s   = "llm_single"\r\n'
    '            lines += [\r\n'
    '                "",\r\n'
    '                "  Focused comparison: curriculum RewardForge vs llm_single",\r\n'
    '                "  " + "-" * 52,\r\n'
    '                f"  {chr(32)*25} {col_rf:>14}  {col_s:>12}",\r\n'
    '                f"  {\'median best_reward\':<25} {np.median(rf_best):>+14.1f}  {np.median(single_best):>+12.1f}",\r\n'
    '                f"  {\'median final_reward\':<25} {np.median(rf_final):>+14.1f}  {np.median(single_final):>+12.1f}",\r\n'
    '                f"  {\'median final_std\':<25} {np.median(rf_std_vals):>14.1f}  {np.median(single_std):>12.1f}",\r\n'
    '                f"  {\'catastrophic fails (< 0)\':<25} {rf_failures:>14d}  {single_failures:>12d}",\r\n'
    '                f"  {\'n seeds\':<25} {len(rf_best):>14d}  {len(single_best):>12d}",\r\n'
    '                "",\r\n'
    '                f"  Mann-Whitney U (two-sided): p={p_struct:.4f}  <- {sig_struct}",\r\n'
    '            ]\r\n'
    '\r\n'
)

if old in text:
    text = text.replace(old, new, 1)
    print("Patched generate_analysis with failure rate reporting")
else:
    print("WARN: anchor not found - trying without \\r")
    old2 = old.replace('\r\n', '\n')
    new2 = new.replace('\r\n', '\n')
    if old2 in text:
        text = text.replace(old2, new2, 1)
        print("Patched (LF-style)")
    else:
        print("ERROR: could not find anchor")
        raise SystemExit(1)

src.write_text(text, encoding="utf-8")
print("Done.")
