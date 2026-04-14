"""Strip non-ASCII from print lines in all relevant agent/runner files."""
from pathlib import Path

targets = [
    "bipedal/bipedal_agent.py",
    "diagnostic/diagnostic_agent.py",
    "diagnostic/behavior_audit.py",
    "diagnostic/experiment_runner.py",
]

for rel in targets:
    p = Path(rel)
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    cleaned = []
    changed = 0
    for line in lines:
        if "print(" in line:
            new = line.encode("ascii", errors="replace").decode("ascii")
            if new != line:
                changed += 1
            line = new
        cleaned.append(line)
    p.write_text("".join(cleaned), encoding="utf-8")
    print(f"  {rel}: {changed} lines changed")

print("All done.")
